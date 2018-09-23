# MTCNN-tensorflow-caffe-win7
## 1.项目描述
###  
     项目重现MTCNN算法，使用多任务级联神经网络同时实现人脸检测和校正。
## 2.环境和数据需求
###
    1.在win7系统环境下tensorflow GPU版本，使用anaconda配置tensorflow-gpu的环境，此外还需要配置CUDA和cudnn加速驱动用来训练模型。
    2.需要下载WIDER Face 和 CelebA这两个数据集，其中WIDER Face用来实现人脸检测，CelebAs数据集用来预测人脸关键点预测。
    2.1 WIDER Face：
         数据集包括不同场景中包含有不同带数目人脸的图片，每张照片都有对应的标注文件，对应图片中人脸数目，人脸图框位姿，还有其他参数。
         属性描述如下：
          附加属性名称和标签值之间的映射。

          模糊：
            clear-> 0
             正常模糊 - > 1
             沉重的模糊 - > 2

          表达：
             典型的表达 - > 0
             夸大表达 - > 1

          照明：
             正常照度 - > 0
             极端照明 - > 1

          闭塞：
             没有遮挡 - > 0
             部分遮挡 - > 1
             重度闭塞 - > 2

          姿势：
             典型的姿势 - > 0
             非典型姿势 - > 1

          无效：
             false-> 0（有效图像）
             true-> 1（图像无效）

          txt地面真相的格式。
          文件名
          边界框数量
          x1，y1，w，h，模糊，表情，光照，无效，遮挡，姿势
      2.2 CelebA: 
         CelebFaces属性数据集（CelebA）是一个大型人脸属性数据集，拥有超过200K名人图像，每个图像都有40个属性注释。
         此数据集中的图像覆盖了大的姿势变化和背景杂乱。 CelebA具有大量的多样性，大量的数量和丰富的注释，包括
          10,177个人信息，
          202,599个脸图像，和 5关键点位置信息（眼睛，鼻子和嘴角）。
          可用作以计算机视觉任务的训练和测试集：面部属性识别，面部关键点检测。
## 3.网络模型训练
###
     1.下载Wider Face 数据集。
     2.运行prepare_data / gen_12net_data.py为PNet生成训练数据（人脸检测部分）。
     3.运行gen_landmark_aug_12.py为PNet生成训练数据（面部标记检测部分）。
     4.运行gen_imglist_pnet.py以合并两部分训练数据。
     5.运行gen_PNet_tfrecords.py为PNet生成tfrecord。
     6.训练PNet后，运行gen_hard_example为RNet生成训练数据（人脸检测部分）。
     7.运行gen_landmark_aug_24.py为RNet生成训练数据（面部标志检测部分）。
     8.运行gen_imglist_rnet.py以合并训练数据的两部分。
     9.运行gen_RNet_tfrecords.py为RNet生成tfrecords。（你应该运行这个脚本四次，分别生成neg，pos，part和landmark的tfrecords）
     10.训练RNet后，运行gen_hard_example以生成ONet的训练数据（人脸检测部分）。
     11.运行gen_landmark_aug_48.py为ONet生成训练数据（面部标记检测部分）。
     12.运行gen_imglist_onet.py以合并训练数据的两部分。
     13.运行gen_ONet_tfrecords.py为ONet生成tfrecords。（你应该运行这个脚本四次，分别生成neg，pos，part和landmark的tfrecords）
     一些细节
 ## 4.文件详解
 ## 5.细节
 ## 6.结果
