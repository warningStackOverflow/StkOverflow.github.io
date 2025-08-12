# 第11周实习记录
## 6月23日
### er建模
阅读ESMM方法论文,该论文是一个经典的多任务学习网络，根据ctcvr = ctr*cvr 的关系，学习ctcvr ctr间接估计cvr。对比目前任务，
er = ecr*cr 的关系，可以通过er = ecr * cr的方式来实现er建模。其中预测ecr的部分还可以加上对dp的单调约束。
### uplift建模
检查DESCN在训练时第二轮loss为nan的问题, 发现问题在计算cross response loss处，按照treatment分组导致尺寸不匹配，尝试解决。

## 6月24日
### er建模
实现esmm论文中的结构用于er建模。参考原文设置损失函数为cr损失加上ecr损失。ecr部分暂时未加约束。

## 6月25日
### er建模
训练ESMM_without_constraint模型，查看训练效果。在测试集上auc=0.733，和直接预测er模型（auc=0.7281）相比效果略好。
询问得知测试集不要用新给定的数据，还是照常使用原来数据进行train-test切分后进行评估。
目前评估er模型包括： simple-dnn直接预测er；esmm 预测ecr和cr，计算er = ecr * cr ；esmm直接预测er；esmm多任务学习主任务预测er 

### uplift建模
1.DESCN模型debug完成，之前nan的原因是计算cross response loss时，需要计算交叉熵损失，当tensor尺寸为0时会导致nan。现在已经修复，模型可以正常训练。

## 6月26日
### er建模
在全国数据集上。各个模型的按照er预测分组和dp分组的图存在线上cooper中 <br>
各个模型对er的预测效果如下：<br>
对照基线方法：<br>
XGB: auc=0.7731, mae=0.2953, wmape=1.2590 <br>
DNN: auc=0.7618, mae=0.3038, wmape=1.2954 <br>
无约束的ESMM方法：<br>
1. 双头输出ecr和cr，乘算er，以er为监督信号 <br>
ESMM_er: auc=0.7731, mae=0.2922, wmape=1.2462 <br>
2. 双头输出ecr和cr，分别作为监督信号，预测ecr和cr，计算er <br>
ESMM_ecrcr: auc=0.7717, mae=0.2917, wmape=1.2438 <br>
3. 双头输出ecr和cr，损失前各自加上可学习参数：
ESMM_multi2: auc=0.7713, mae=0.2900, wmape=1.2376 <br>
4. 多头输出ecr，er，cr，损失函数为L_er+ alpha * L_ecr + beta * L_cr （似乎没有收敛）<br>
ESMM_multi3: auc=0.7704, mae=0.2916, wmape=1.2432 <br>

## 6月27日
突发恶疾请假一天

## 本周总结
1， er建模：完成了多种方式的er预测，包括直接预测er，esmm预测ecr和cr，计算er = ecr * cr，esmm直接预测er，esmm多任务学习主任务预测er等。
2. uplift建模：DESCN模型debug完成，模型可以正常训练。