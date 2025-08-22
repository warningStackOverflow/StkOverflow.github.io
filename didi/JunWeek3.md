---
layout: default
---

# 第 10 周实习记录
## 6月16日
1. 对causal forest方法进行评估：
将数据按照高低dp分组，分别用causal forest方法建模，性能较差 <br>
2. DESCN结构：
包括一个输出倾向分的网络，一个输出tr和cr的网络（论文中用的x-network），输入先通过一个shared network，再分别进入这两个网络。
本质上借鉴了x-learner方法：.
We can see that our X-network is similar to X-learner in that
both try to directly learn the counterfactual treatment effect. In
X-learner, ITE is learned based on the results from base learners,
and its performance is heavily subject to that of the base models.
By contrast, in X-network, ITE is learned together with the base
learners in an integrated 
3. 因为接下来要上深度学习模型，所以目前200+的特征可能还是有点多，需要在特征选择一下，进行特征选择，顺便用之前的meta-learner验证选择的特征

## 6月17日, 6月18日

实现descn：已有的代码用不了，只能借鉴思路自行完成
### step 1 实现针对 multi-treatment 的 esn
论文中的针对bi-treatment的esn建模思路： <br>
1. 用一个倾向分网络在全样本上预测二分类的倾向分，即$ \pi(x) = P(T=1|X)$，再结合一个已有的可以输出
treatment response和control response的uplift模型（这里的模型可以是很多，如因果森林，tarnet，xnet等）输出 $TR = P(Y|T=1,X)$ 和 $CR = P(Y|T=0,X)$ <br>
2. 计算论文里提到的estr和escr,即全样本空间上的tr和cr $estr = tr*\pi(x); escr = cr*(1-\pi(x))$ <br>
这样就可以在所有数据上学习estr和escr，又因为可以在所有数据上学习 pi_t pi_c,所以最后得到的个体提升效果$ite =\mu_1 - \mu_0 = estr/ \pi + escr/(1- \pi)$ 
就是无偏的，解决了treatment bias issue。
### step 2 DESCN 训练


## 6.19
### 数据处理
发现原始数据集有些问题，包括一些rate计算远大于1，和特征分布不合理，这个不合理对于uplift建模和ecr建模都有影响，所以先对原先的这个4gb的数据集进行处理，以备后用。(完成)<br>
### er模型
用xgb， dnn和新的multi-head dnn([阿里2018的esmm](https://arxiv.org/abs/1804.07931) )对er进行建模，并且画出按照dp分组和按照预估er分组的曲线 <br>
1. xgb模型(完成) <br>
2. 简单的dnn模型（完成） <br>
3. esmm模型 <br>

## 6.20
### uplift模型
完成descn模型的实现，上午debug并且在本机上跑通，但是效果不是很好，
下午请假半天
### ecr模型
esmm的论文看完

## 本周总结
uplift方面，本周完成并且验证了因果森林方法，效果不佳；完成了DESCN方法。
er模型方面，本周完成了并且验证了xgb和dnn模型，得到er随着dp的关系；下周尝试esmm方法。