## 项目结构：
- src：源码目录
- src/model：基于Keras Applications的卷积神经网络model，与杨博士提出使用的
卷积神经网络结构(sunspot_private_conv_model.py)
- src/data_factory.py : 输入png数据文件目录产出Tensorflow Dataset格式训练集、验证集、
全量数据集。
- src/predict_pic.py : 完成比赛测试数据预测与写出结果文件。
- src/test_dataset_trans_png.py : 将比赛所用fits数据文件转换为png图片文件。
- src/train_data_trans_png.py : 将比赛所用测试fits数据文件转换为png图片文件。

## 目前思路与进度：
1: 由于该比赛训练与推导的过程完全在线下进行，对算力及速度没有要求，所以我认为可以不考虑网络
参数数量，输入模型的数据大小尽可能大一点。然后选取了在Keras  
Applications中Top-1 Accuracy与Top-5 Accuracy两项指标Top5的Model。这5个Model在src/model
目录下有具体的实现。

2: 第一次提交的结果是基于杨博士提出的卷积神经网络在为进行数据增强的数据不平衡状态下，仅使用
2/3的白光图完成模型训练，最后打出了Beta、Betax、Alpha三类的F1-Score0.65  0.61  0.56
的成绩。

3: 目前思路与做法：已在src/train_data_trans_png.py文件中转换png图像时进行数据增强，数据
增强的方式为：betax所有图片均进行水平镜面转换，对角线转换。Alpha图像每隔一张图像进行一次水平
镜像转换。数据增强后三类别数据量均衡。使用data_factory中切分训练集与验证集的方法切分的结果为：

训练Alpha： 4612， 训练Beta： 4966， 训练Betax： 4653，

测试Alpha： 2451， 测试Beta： 2387 测试Betax： 2568。

并且使用src/model/sunspot_nas_net_large.py完成三个Epoch的训练之后


||precision|recall|f1-score|support|
|---|-----|-----|----|----|
|Alpha|0.96|0.90|0.93|2449|
|Beta|0.74|0.88|0.80|2385|
|BetaX|0.91|0.80|0.85|2558|

通过对F1-Score的观测我认为和比赛测试集在结果当中反应的结果类似，Beta类F1-socre最小，Alpha类F1-score最大。可以认为当前验证集可以类比测试集对模型进行验证。

## 所以我目前的策略为：

- 使用数据增强之后的数据
- 1:分别使用磁图的白光图完成5个model的训练。使用投票的方法进行模型融合，如果出现5:5的投票情况，对投票的softmax的数值求和取大的一类作为融合结果。
- 2:同时将磁图与白光图输入模型，训练5个model，使用投票方法进行模型融合，完成预测。
