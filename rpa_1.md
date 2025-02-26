# 序.0.自动驾驶系统的基本框架/决策控制算法框架/决策规划总览
## 0.1 业内对框架的分类
分类一：[sequential planning、behavior-aware planning、和end-to-end planning](https://blog.csdn.net/CV_Autobot/article/details/139016301) 
分类二：[模块化和端到端](https://zhuanlan.zhihu.com/p/713880302)
其实两类意思差不多，大意就是分为传统的分模块的和端到端省去中间层的。 未来将是局部ruled-based，但是大势所趋端到端。

如模块化/sequential planning：
输入：lidar\camera\radar 
中间算法：perception\localization\planning\control 
输出：steering\acceleration\brake 
优点：过程可控，方便理解（感知、决策、控制三个方面层次清晰）。 
缺点：长尾问题不好解决，随着长尾问题的解决，架构变得冗余。 

如端到端（神经网络DRL，DL一锅端）： 
输入输出不变，中间过程用神经网络来代替。
优点：信息的无损传递 
缺点：每个子任务传递的信息不可解释，训练要求的样本数量多，如果场景没有训练，其泛化性不确定。 
## 0.2 常见路径规划控制算法（传统规划算法）
包括： 
PRM、RRT为代表的基于采样的算法 
以为A* 、D* 代表的基于搜索的算法 
以β样条曲线为代表的基于插值拟合的轨迹生成算法 
以PID 、MPC为代表的用于局部路径规划的最优控制算法。
## 0.3 常见决策类算法框架（待补中）
