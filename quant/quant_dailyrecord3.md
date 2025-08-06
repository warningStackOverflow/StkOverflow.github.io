# AugWeek1
## 2025/08/04
1. 继续复现alphagen： <br>
目前基本架构已经完成
正在写和公司数据对接的部分。目前公司的数据不支持读取dt1到dt2的时间段，需要自己手动拼接，
也不支持直接读as tensor,虽然点进去数据库里todo写着下一步他们就要实现这个功能。。。
2. RLgen架构：<br>
```
RLgen
|- rl
    |- alphapool：因子池，通过计算ic、 rank ic选因子
    |- data：用来生成表达式，按照expression -> token -> tree的流程生成逆波兰表达式
    |- env：实现gym.Env接口，包括env.py和wrapper.py
    |- higgs_database：主要实现class StockData，用蔷薇的数据取代qlib
    |- utils：一些工具函数，直接抄的alphagen
    |- policy.py: 策略网络实现，transformer结构
|- scripts：一些脚本 
    |- rlgen_train.py: 训练脚本
    |- rlgen_test.py: 测试脚本(没写）
|- configs.py：一些全局参数
```

## 2025/08/05
1. stable-baselines3库线上环境里没有安装（已解决，装了）<br>
公司的公共库里，没有gym，没有sb3等一系列主流强化学习库，需要装一下。

2. stable-baselines3库:一个封装良好的强化学习库，有一个简易入门demo可以参考[这里](https://zhuanlan.zhihu.com/p/406517851)
3. tensorboardX库: 可以参考[这里](https://zhuanlan.zhihu.com/p/220403674)
4. 参考的alphagen的代码时间颗粒度为日线，蔷薇数据的时间颗粒度为分钟，需要修改一下表达式中涉及时间的部分到min (已解决，修了) <br>
5. alphagen的收益率是从数据中计算得到，具体在calculator中实现；蔷薇有现成的收益率数据，所以需要修改一下alphapool类
6. 初步完成rlgen，提交slurm任务时报错gcc版本问题

## 2025/08/06
### 解决gcc版本问题：<br>
在gpu节点运行rlgen会报错如下，在调用matplotlib->kiwisolver时,显示缺少GLIBCXX_3.4.29版本的libstdc++.so.6库；
```
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /opt/share/linux-rocky8-x86_64/gcc-12.2.0/anaconda3-2023.03/lib/python3.10/site-packages/kiwisolver/_cext.cpython-310-x86_64-linux-gnu.so)
```
之前在cpu节点进行debug时不报这个错，
检查了cpu节点的libstdc++.so.6，其中有GLIBCXX_3.4.29版本，因此不报错；最后直接重新下载了一个kiwisolver覆盖了anaconda的kiwisolver，解决了问题。
### RLgen的多卡训练加速问题
在单张RTX3090上训练RLgen，训练速度不快，一轮差不多8-11s，这样的话完整一轮下来需要20天，这是不可行的。需要进行加速优化。
使用的stable-baselines3库不支持多卡训练，他们的理由是RL任务都很简单，[不需要多卡](https://blog.csdn.net/javastart/article/details/130531185)，需要自己实现多卡训练。
