````
## 文件结构：
```
  ├── backbone: 特征提取网络
  	├──feature_pyramid_network.py:特征金字塔网络
  	├──resnet101_fpn_model.py:主干网络部分
  	
  ├── network_files: Mask R-CNN网络
  	├──boxes.py
  	├──det_utils.py:
  	├──faster_rcnn_framework.py
  	├──image_list.py
  	├──mask_rcnn.py
  	├──roi_head.py
  	├──rpn_function.py
  	├──transform.py
  	
  ├── train_utils: 训练验证相关模块（包括coco验证相关）
  	├──coco_eval.py
  	├──coco_utils.py
  	├──distributed_utils.py
  	├──group_by_aspect_ratio.py
  	├──train_eval_utils.py
  	
  ├── my_dataset_coco.py: 自定义dataset用于读取COCO2017数据集
  ├── train.py: 训练脚本
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测
  ├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
  └── transforms.py: 数据预处理（随机水平翻转图像以及bboxes、将PIL图像转为Tensor）
```
````

### 代码运行:

train.py进行训练，训练完成的model放在了save_weights中，每一次epoch都有记录。并且生成了有关loss与lr的图像和mAP的图像、val验证集里面的每个epoch中的txt文件。

predict.py文件对自己的图片（img.png）进行分析并得到相应的分割图。



### 总结：

参考代码：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/mask_rcnn

基于faster_rcnn中的网络框架：

![fasterRCNN.png](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_object_detection/faster_rcnn/fasterRCNN.png?raw=true)



mask_rcnn是在faster_rcnn的基础上再添加一个额外的mask分支，对检测框中的图像进行分割。

* 代码修改：

​		在GitHub源码中使用的是resnet50_fpn_model，通过修改可以得到resnet101的代码；

​		关于acc和topk，在代码中，我尝试在每个epoch中计算得到，但是问题很多，我通过搜索发现acc和topk分别为分类准确率（classification accuracy）和精度（top-k accuracy)，主要用在分类当中，但是当我在修改源码过程中，对每一张图进行预测时（在train_one_epoch中添加相应的代码），发现有些图没有预测信息（有可能是因为刚开始，导致的预测效果差，我并没有往下测试），而我对这个acc和topk的确有些不懂的地方，通过对targets和outputs中的“labels”标签作比较，然后带入相应的计算公式，这个应该就是思路，但是我没有办法完成这个任务。（可能是我的这个思路错了）

* 改进方案

  我只训练了10个epoch，可以增加epoch的个数来提升分割的效果，还可以更改学习率以及学习的衰减、调整batch（我用的为10）大小以及不同的优化器和激活函数。
