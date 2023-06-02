import math
import sys
import time

import torch

import train_utils.distributed_utils as utils
from .coco_eval import EvalCOCOMetric


# def calculate_accuracy(outputs, targets, K=5):
#     predicted_labels = []
#     total = 0
#     correct = 0
#     topk_correct = 0
#
#     for i, output in enumerate(outputs):
#         target = targets[i]["labels"]
#         scores = output["scores"]
#         labels = output["labels"]
#
#         if len(labels) == 0:
#             predicted_labels.append(-1)
#             continue
#
#         max_idx = torch.argmax(scores).item()
#         predicted_label = labels[max_idx].item()
#
#         predicted_labels.append(predicted_label)
#
#         if predicted_label == target.item():
#             correct += 1
#
#         _, pred_topk = scores.topk(k=K)
#         if torch.any(torch.eq(labels[pred_topk], target.item())):
#             topk_correct += 1
#
#         total += 1
#
#     acc = float(correct) / total
#     topk_acc = float(topk_correct) / total
#
#     return acc, topk_acc

def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('acc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # metric_logger.add_meter('topk', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses

    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict, preds = model(images, targets)
            # print(preds)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # 反向传播和优化
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()


        # acc, topk = calculate_accuracy(model, data_loader, device)


        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
        # metric_logger.update(lr=now_lr, acc=1, topk=2)

    return mloss, now_lr



@torch.no_grad()
def evaluate(model, data_loader, device):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    det_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="bbox", results_file_name="det_results.json")
    seg_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="segm", results_file_name="seg_results.json")
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        det_metric.update(targets, outputs)
        seg_metric.update(targets, outputs)
        metric_logger.update(model_time=model_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 同步所有进程中的数据
    det_metric.synchronize_results()
    seg_metric.synchronize_results()

    if utils.is_main_process():
        coco_info = det_metric.evaluate()
        seg_info = seg_metric.evaluate()
    else:
        coco_info = None
        seg_info = None

    return coco_info, seg_info