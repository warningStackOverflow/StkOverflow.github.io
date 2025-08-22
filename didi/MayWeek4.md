---
layout: default
---

# 动调第7周记录
## 本周计划
1. 结束dl模型的优化(上周工作收尾)
2. 在新数据集上完成ar（is_send==1） 的 uplift任务：Dptimes作为treatment，is_accept作为outcome（本周工作）
3. 在新数据集上对两种预测方法进行对比：分开建模ecr和ar的方法 ｜｜ 直接建模er的方法（如果快的话）

## 5月26日
### 上周收尾工作：auto-dis 性能分析
采用autodis结构对输入的连续特征进行处理，模型性能没有显著提升。<br>
### 学习关于多treatment的uplift模型 [此篇文章](https://zhuanlan.zhihu.com/p/13394677406)
Meta-learner：将treatment作为特征（S-learner）、或根据不同的treatment搭建多个模型（T-learner），这个已有许多py包可以实现。<br>
[TarNet](arxiv.org/abs/1810.00656) 提到了使用CFRNet/TarNet对多treatment进行建模，共享层表征X后，根据treatment的个数分到多头。<br>
[DRNet](arxiv.org/abs/1902.00981) 提到了对于连续treatment的建模方法，DRNet是一个二维、多值干预模型，常用场景为“给一个用户发什么类型的优惠券、发券金额是多少”的二维多值干预。<br>
## 5月27日
### 关于上周收尾任务：画出每个模型的qini曲线
### 关于本周计划2的todolist
1. 尝试使用meta-learner思路(s-learner, t-learner, x-learner)对多treatment建模，作为基线方法.(已完成) <br>
2. 实现uplift树模型. <br>
3. 实现uplift深度学习模型（tarnet，drnet，descn等 <br>
4. 将uplift模型的训练和预测过程封装好
## 5月28日
1. 按照供需比dsr分组，查看real ar和dpt的关系 <br>
2. 拿到了新数据集，大小约为400w x 400，先进行初步的数据清洗和预处理。（注意如果is_send==0,那么orderid和is_accept, is_finish以及后面的一系列和出行相关的都是nan）<br>

## 5月29日
1. 完成特征选择。将400+个特征筛选到200个左右，主要是相关性分析（已完成）。<br>
2. 在demo数据（20000条）上跑通单treatment的uplift模型，为在gpu上跑做准备（已完成）。<br>

## 5月30日
1. 在demo数据上跑通多treatment的uplift模型。<br>

## 本周总结
本周完成了对多treatment的uplift模型的初步实现，主要是meta-learner思路的s-learner和x-learner。<br>
还把数据集的特征选择和预处理完成了。<br>
第一次接触uplift model，处理进度有些慢。<br>