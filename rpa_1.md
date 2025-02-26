# 序.0.自动驾驶系统的基本框架/决策控制算法框架/决策规划总览
## 分类一：[sequential planning、behavior-aware planning、和end-to-end planning](https://blog.csdn.net/CV_Autobot/article/details/139016301)
## 分类二：[模块化和端到端](https://zhuanlan.zhihu.com/p/713880302)
大意就是分为传统的分模块的和端到端省去中间层的。 

如模块化：

输入：lidar\camera\radar 

中间算法：perception\localization\planning\control 

输出：steering\acceleration\brake 

优点：过程可控，方便理解。 

缺点：长尾问题不好解决，随着长尾问题的解决，架构变得冗余。 

如端到端神经网络一锅端： 

输入输出不变，中间过程用神经网络来代替。 

优点：信息的无损传递 

缺点：每个子任务传递的信息不可解释，训练要求的样本数量多，如果场景没有训练，其泛化性不确定。 

