# MTCNN-tensorflow-caffe-win7
## 1.项目描述
###  
     1.描述
     项目重现MTCNN算法，使用多任务级联神经网络同时实现人脸检测和校正。
     2.应用背景
     人脸检测是对人脸进行识别和处理的第一步，主要用于检测并定位图片中的人脸，返回高精度的人脸框坐标及人脸特征点坐标。人脸识别会进一步      提取每个人脸中所蕴涵的身份特征，并将其与已知的人脸进行对比，从而识别每个人脸的身份。目前人脸检测/识别的应用场景逐渐从室内演变到室      外，从单一限定场景发展到广场、车站、地铁口等场景，人脸检测/识别面临的要求也越来越高，比如：人脸尺度多变、数量冗大、姿势多样包括俯      拍人脸、戴帽子口罩等的遮挡、表情夸张、化妆伪装、光照条件恶劣、分辨率低甚至连肉眼都较难区分等。随着深度学习的发展，基于深度学习技      术的人脸检测/识别方法取得了巨大的成功。
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
     在训练PNet时，我将四个部分的数据（pos，part，landmark，neg）合并为一个tfrecord，因为它们的无线电总数几乎是1：1：1：3。但是当      训练RNet和ONet时，我生成四个tfrecords，由于它们的总数不平衡。在训练期间，我从pos，part和landmark tfrecord中读取64个样本，并       从neg tfrecord读取192个样本以构建小批量。

     对于PNet和RNet来说，保持高召回率是非常重要的。当使用训练有素的PNet为RNet生成训练数据时，我可以获得14w + pos样本。当使用训练有       素的RNet为ONet生成训练数据时，我可以获得19w + pos样本。
      由于MTCNN是一个多任务网络，我们应该注意训练数据的格式。格式是：
     [图像路径] [cls_label] [bbox_label] [landmark_label]
     对于pos样本，cls_label = 1，bbox_label（计算），landmark_label = [0,0,0,0,0,0,0,0,0,0,0]。
     对于部分样本，cls_label = -1，bbox_label（计算），landmark_label = [0,0,0,0,0,0,0,0,0,0,0]。
     对于标志性样本，cls_label = -2，bbox_label = [0,0,0,0]，landmark_label（计算）。
     对于neg样本，cls_label = 0，bbox_label = [0,0,0,0]，landmark_label = [0,0,0,0,0,0,0,0,0,0,0]。
     由于地标的训练数据较少。我使用变换，随机旋转和随机翻转来进行数据增强（地标检测的结果不是那么好）。
 ## 6.结果
 ## 7.参考资料
 ###
    1. 如何应用MTCNN和FaceNet模型实现人脸检测及识别http://www.uml.org.cn/ai/201806124.asp?artid=20840
    2.
