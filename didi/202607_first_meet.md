# 11会议简纪

## todolist
### 1 优化目前的dp8策略中的两处涉及模型的部分（模型优化）：
```go
1. DemandSupplyFilter -> 出一个grid-wise的filterdptimes（如1.5），在其基础上加上范围，用来给司机看（看到1.2-1.8），调节司机：
    1. 供需平衡+pid反馈出一个dp1（这个不太好，希望这个权重低些）
    2. ar trigger， 根据dp-ar间关系（此处ar模型预测，参考最新发的DIPN），预测一个期望ar对应的dp2
    3. filterdp = max(dp1, dp2)

2. 在估算GMV时的er： GMV = baseprice * DP * er（此处的er=ecr*cr，模型预测）
```
### 2 升级BR MOTO业务
目前的国际化出行业务分很多：</br>
BR -(POP快车 dp8 + odsurge策略， FLEX议价车，MOTO) </br>
现在BR的moto用的策略版本还是dp7，把它升级到dp8（有社招人在做，可能赛马）</br>

### 3 RL application
尝试RL在策略中的应用，目前两条线
```go
1. 巴西快车在原先的dp8基础上，多了一个ODSurge策略，该策略中 score = GMV + \lambda TCH(eta*er)
即不仅要优化gmv，还要提升总的tch，司机尽可能都在开车，使用rl来调节此处的\lambda 
2. 在DemandSupplyFilter中，目前是一两个管线取max出基础dp，修改为rl直出（这个比较直观，但是出的dp只是作为给司机显示的，对后续的优化不显著，可能拿不到收益），目前是校企合作，我跟着听听

```
### 4 时间分配
三块目标同步推进，时间分配40 40 20 

## 一些定价业务的特殊处
1. 做ab-test时，ct两组不能和sra业务一样按照流量直接分，而是按照时间切片分（6点-8点 c；8点-10点t）。这样会造成ct组间有gap
2. 做ab-test时，由于巴西地区原因，按照小城-中城-大城推实验，流程慢，其次有些实验要和运营沟通，且抽成率要保持稳定。
3. 由于存在运力共享，实验周期较慢，和sra不一样。
4. 整个业务策略部分比搜推要复杂，搜推更多的是直接上模型，这件事有好有坏。
5. 目前组内无人做强化学习，需要抓住机会。

## 以前实习时记的东西：
ar模型，cr模型和ecr模型在整体策略中的作用
详细框架见动调串讲文档中的两幅图，我们负责的部分在策略中有两处。<br> 
1. DemandSupplyFliterNew -> dp处理 -> DpTimes计算的多种方式中 -> 使用dp倍数-ar模型计算 (func UsePredArElas in strategy/dp/demand_supply_filter_ar_help.go) <br>
2. optsolve -> 拉格朗日松弛求解的参数获取 -> 使用cr和ecr模型获取参数 (func GetEcrCrModelPredictConstElas in strategy/dp/dp_8_0.go) <br>