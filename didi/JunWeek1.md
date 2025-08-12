# 动调第8周实习记录
## 本周计划
1. 完成uplift建模的基础方法、评价指标等，做好文档记录。（已完成）
2. 尽量完成深度学习方法，在本机训练。
## 6月2日
端午节假期，休息。

## 6月3日
1. 用树模型重新从原始数据中选择特征，因为pca降维可解释性没有用原来的特征好。
2. multi x-learner实现和评估

## 6月4日
训练五个不同的深度模型，对send/total; accept/send; finish/send; accept/total; finish/total进行预测，并且进行一些数据拼接工作。<br>
* 记住训练深度模型时一定要顺便存一下standardscaler。<br>

## 6月5日
todo: <br>
1. 继续完善多treatment的uplift模型。目前已经实现了x-learner,但是效果看上去比较奇怪。<br>
2. 尝试使用dml/causalforest-dml将多值treatment视为离散的treatment进行建模。<br>
3. 在本机上下载需要的包，熟悉mac训练的流程。<br>

## Double Machine Learning(dml)方法学习：
原论文[Double/Debiased Machine Learning for Treatment and Causal Parametershttps](arxiv.org/abs/1608.00060) <br>
[可参考1](https://blog.csdn.net/taozibaby/article/details/140591792) <br>
[可参考2](https://zhuanlan.zhihu.com/p/751459729) <br>
### 1. 概念和基本假设（注意和一般causal inference的区别）
Y(outcome), T(treatment), X(features) 特征变量，低维，用于学习异质性处理效应。这三个和一般因果推断类似。<br> 
W(Confounder): 干扰因子，既影响Y又影响X,高维，控制干扰因子以剔除其对Y和T的影响，得到无偏的处理效应. X与W的区别：在nuisance model拟合的时候，X和W是一起用来与T和Y拟合去残差的，首先他们都是控制变量；然而在final model中，我们只考虑在特征X上学习异质性。<br>
基本假设：无干扰因素，定义：在估计T的处理效应时，所有会影响T和Y的干扰因子都是可观测的（即这些特征都要找全），即：在有效控制W的情况下，T取0/1是独立于Y的。$T_i \perp (Y_i(0),Y_i(1)) | W$
可以解决的问题：包含混淆变量的因果推断问题。<br>
### 2. 模型，两步走
(1). nuisance model：使用X和W拟合T和Y的残差，这里可以用任意的ml模型，得到$R_T$和$R_Y$。<br>
(2). final model：使用$R_T$和$R_Y$拟合T和Y的残差，$R_Y = \theta (X) R_T + \epsilon $; 用任意回归求解 $\theta (X)$，得到处理效应的估计值。<br>

## 6月6日
1. 尝试在本机进行模型训练，发现lightgbm包无法安装，因为需要的libomp.dylib 库未找到。libomp 是 OpenMP 的实现，在 macOS 上需要它来支持多线程。查询
后发现有两个处理办法，一是用brew安装libomp，但是brew需要安装，过程中出现github time out；二是用conda下载lightgbm，发现conda好像需要滴滴正版化申请。<br>
2. 本机torch安装没有问题，因此可以后期的深度学习模型在本机跑，基于xgb的meta-learner仍然在服务器跑，可以暂时规避上面的问题。
3. 完成了s-learner多treatment建模，和之前的x-learner一起可以算是作为基础机器学习的基线方法了。<br>

## 本周总结
1. 完成了多treatment的uplift建模和评估，将x-learner和s-learner作为基线方法。
2. 验证了深度学习模型在ar cr ecr era erf五个指标上的对dynamic time的单调性。
3. 配置完成本机的torch环境，之后深度学习模型可以在本机跑，下周开始写深度学习模型。