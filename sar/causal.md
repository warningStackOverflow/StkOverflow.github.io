### 学习关于多treatment的uplift模型 [此篇文章](https://zhuanlan.zhihu.com/p/13394677406)
Meta-learner：将treatment作为特征（S-learner）、或根据不同的treatment搭建多个模型（T-learner），这个已有许多py包可以实现。<br>
[TarNet](arxiv.org/abs/1810.00656) 提到了使用CFRNet/TarNet对多treatment进行建模，共享层表征X后，根据treatment的个数分到多头。<br>
[DRNet](arxiv.org/abs/1902.00981) 提到了对于连续treatment的建模方法，DRNet是一个二维、多值干预模型，常用场景为“给一个用户发什么类型的优惠券、发券金额是多少”的二维多值干预。<br>