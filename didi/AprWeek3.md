# 动调实习第一周记录
## 2025.4.14
### 1. 开通各类权限，学习数据安全流程
### 2. 本地配置趁手工具，IDE,copilot 等

## 2025.4.15
### 1. 通过落地文档了解业务
落地文档里有完整的司机和乘客的价格策略的流程图，大概了解了上下游的业务关系，并且明确了我做的部分应该是DP部分。
### 2. 了解了动调的基本概念

## 2025.4.16；
### 0.第一次交流后记录
上午看动调代码，下午和带教交流，晚上看模型训练的代码 <br>
后面工作计划 <br>
1、格子AR、冒泡AR模型优化
（1）基本训练流程，特征 样本 lgb训练 评估 (jupyter中)
（2）熟悉DNN训练框架，尝试特征、结构的优化、多任务联合训练
（3）uplift建模调研及训练 <br>
2、强化学习调研和落地
（1）方案调研和设计
（2）样本构建、模型训练及评估
### 1. 常用简写
dp：dynamic pricing 动态定价，动调; cp：calibration pricing 基价； rp：random pricing 随机价； FP： fixed一口价 <br>
ar: accept rate 接单率; $ar = \frac{accept}{call}$; dar：司机接单率 $dar = \frac{accept}{broadcast}$； cr：完单率；ecr：有效呼叫率 $ecr = \frac{call}{bubbles}$; <br>
未知的缩写：bcr $ bcr = 1 - (1- dar)^{h} $ 看注释说是司机播单率，又有说是冒泡完成率的。<br>
gmv：gross merchandise volume，交易总额；tch: total charge time 总收费时间；这两个貌似是策略的优化目标。<br>
### 2. ar模型，cr模型和ecr模型在整体策略中的作用，具体见4
详细框架见动调串讲文档中的两幅图，我们负责的部分在策略中有两处。<br> 
1. DemandSupplyFliterNew -> dp处理 -> DpTimes计算的多种方式中 -> 使用dp倍数-ar模型计算 (func UsePredArElas in strategy/dp/demand_supply_filter_ar_help.go) <br>
2. optsolve -> 拉格朗日松弛求解的参数获取 -> 使用cr和ecr模型获取参数 (func GetEcrCrModelPredictConstElas in strategy/dp/dp_8_0.go) <br>
### 3. AR模型场景和作用 && CR/ECR模型场景和作用(格子ar && 冒泡ar)
对于每一个六边形的cell， 其中的供需关系都是不一样的， ar模型通过建立cell中的dptimes和ar的关系，输出dp倍数，对基础价格进行修正，达到利用不同cell价格调节供需，提升乘客体验的目的。<br>
基准dp倍数小于2时的场景比较常见，数据样本较多，可以使用统计学习模型建立ar-dp倍数的关系； 大于2时，样本较少，通过规则+线性差值近似找出ar-dp倍数的关系。<br>
同样对于cell中，每一个cell中会有多个可能潜在的od需求，称为一个bubble，complete/bubble 表示完单率.对于每一个bubble进行optsolve，这是一个凸优化问题，目标函数是最大化gmv，在松弛的时候需要用完单率作为参数。和ar模型类似，都是算对dp倍数的弹性。<br>
### 4. func UsePredArElas 的具体逻辑(func GetEcrCrModelPredictConstElas类似)
```
INPUT: dp_candidate, ar_hat， realtimeAR, realtimeDSR， ArSetpoint

dp_candidate < 2:
    dpc , ar_hat = [dpc_i],[ar_hat_i] # 外部传入 ar_hat是统计学习模型预测得出
    elas = d(ar_hat)/d(dpc) # 用 model.LeastSquares 算出弹性
    elas *= defactor # 乘一个衰减系数 
    
dp_candidate >2: # 样本少所以用规则+线性差值计算ar的预测值
    dpcs = [2.0 : 3.0 : 0.1]
    for dpc in dpcs:
        new defactor # 一个不一样的衰减系数，和每一个dp_candidate 对应
        elas *= defactor
        elas = max(elas, elas_lowbound) # 下限
        d(ar) = elas * (dpc- 1.9) # dpc-1.9在0，1
        ar_hat = ar_1.9 + d(ar) # ar_1.9是上一段模型预测的ar的最后一位，也是这里插值起点

# 上述结束后获得两个向量 dp_candidate, ar_hat， dp<2 的 ar的弹性elas
arDiff = ArSetpoint - realtimeAR # 确保上下界的省略
d(dp) = arrDiff / elas 
dp_times = valid_dp_times + d(dp) 

OUTPUT: dp_times
```
## 2025.4.17
因为学校导师线上召开组会，请假一天，无进展。

## 2025.4.18
### 1. 配置环境
拉取项目组 /最优化ECR-AR模型.ipynb 并且在本地配置环境。本地配置环境拉取包的速度太慢，线上使用已有的env运行代码发现总是报内核正忙的错误。接下来需要找出原因。
### 2. 熟悉jupyter notebook操作，试图在本地ide远程连接项目组的托管仓库
没有成功，只能读取，尝试修改ssh配置，还是不行，遂放弃，ide不一样不影响写代码，习惯了就行。

## 总结
第一周主要工作量：<br>
1. 阅读业务文档，熟悉实习环境和所用工具，熟悉动调在定价中的位置，熟悉动调算法中模型。 <br>
2. 配置各类环境，阅读最优化ar/ecr模型的代码，尝试发掘优化的部分。<br>
下一周的计划：<br>
1. 使用当前离线数据复现出最优化ar/ecr模型的结果，理清模型的训练流程。<br>
2. 寻找可能的模型优化点，继续熟悉格子ar和冒泡ar的模型。