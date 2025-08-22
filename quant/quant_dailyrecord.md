---
layout: default
---

# JulWeek3
## 0714
### 入职，熟悉环境，装好配置，得到一个小demo：
```
1. 生成20250401-20250430全市场的1分钟频率 sign(最近1minKbar的Vwap 对数收益率) * Volume_t 因子（完成）
2. 计算IC(pearson correlation)，rankIC (spearman correlation)
	i) 所有股票所有时间（overall）的IC，rankIC （完成）
	ii) 逐股平均 （完成）
	iii) 逐天逐股 （完成） 
     然后分析overall的IC是否与逐股平均有差别，哪个高，为什么：（完成） 
     IC和rankIC是否有差别，哪个高，为什么 （完成） 
3. 分别计算因子和label的自相关系数和偏自相关系数，多少阶是显著的，哪个高？  如何匹配因子和label的自相关（EMA平滑？MA平滑？）（完成） 
4. 因子和label平稳吗，平均来看，在日内的早盘（开盘后前30分钟）和其余时间，因子和label的分布是否有差异？label的标准差是否有差异？
5. 尝试改进这个因子，提高其overall IC或rankIC。
6. 如果做的快的话完成：https://zhuanlan.zhihu.com/p/616022509 中提到的内容。
```

公司使用hpc集群，平时写代码在登陆节点编辑，运行用slurm，anaconda用module的方式直接load <br>
环境在 module load anaconda3/2023.03 <br>
得到token：660144fd0257a45ea8247d209eff689251f66a2f0cc083a3 <br>

本地测试代码时，具体使用方法参考：https://github.higgsasset.com/HiggsIT/HPC-Introduction/wiki/Modules-Guide <BR>
线上实验使用slurm，具体参考：https://github.higgsasset.com/HiggsIT/HPC-Introduction/wiki/Slurm-Guide <br>
### 理解和复现demo
1. shannon_store，higgsboom, gauss库的安装（已完成）
higgsboom中的一些数据： https://github.higgsasset.com/HiggsIT/Boom-Introduction/tree/main/%E6%96%B0%E7%89%88Boom/data/market
逐笔数据格式参考：https://github.higgsasset.com/Shannon/shannon_innovation
2. 中频1min和高频1s数据的表头：
中频1min主要是k线数据，包含开收高低，成交量 <BR>
```
rtn_1s：最近1秒的收益率（通常为对数收益率）
volume_1s：最近1秒的成交量
amount_1s：最近1秒的成交额
VWAP：该分钟的成交量加权平均价（Volume Weighted Average Price）
TWAP：该分钟的时间加权平均价（Time Weighted Average Price）
OPEN：该分钟的开盘价
CLOSE：该分钟的收盘价
HIGH：该分钟的最高价
LOW：该分钟的最低价
VOLUME：该分钟的总成交量
AMT：该分钟的总成交额
RecID：记录编号或唯一标识
SliceIndex：时间片索引（如第几分钟）
StockUpdateTime：股票数据更新时间（通常为该分钟的时间戳）
MidPrice：中间价（买一和卖一的均价）
AskPrice：卖一价
BidPrice：买一价
AskVol：卖一挂单量
BidVol：买一挂单量
```
高频1s是逐笔数据，包含每一笔交易的详细信息，通常包括以下字段：
```
rtn_1s：最近1秒的收益率
volume_1s：最近1秒的成交量
amount_1s：最近1秒的成交额
VWAP：该分钟的成交量加权平均价（Volume Weighted Average Price）
TWAP：该分钟的时间加权平均价（Time Weighted Average Price）
OPEN：该分钟的开盘价
CLOSE：该分钟的收盘价
HIGH：该分钟的最高价
LOW：该分钟的最低价
VOLUME：该分钟的总成交量
AMT：该分钟的总成交额
RecID：记录编号或唯一标识
SliceIndex：时间片索引（如第几分钟）
StockUpdateTime：股票数据更新时间（通常为该分钟的时间戳）
MidPrice：中间价（买一和卖一的均价）
AskPrice：卖一价
BidPrice：买一价
AskVol：卖一挂单量
BidVol：买一挂单量
```

## 0715
### 复现demo 
1. 计算得到因子，做出IC和rankIC的图表，分析结果。<br>
over_all_ic <BR>
```
ovreall ic
log_rtn_vwap	0.011680
factor_demo	-0.003471

over_all_rankic
log_rtn_vwap	-0.013958
factor_demo	-0.014757

```
ic by_stock <BR>
```
log_rtn_vwap	0.016343
factor_demo	0.000295
```
2. 对因子的ic和rank ic的分析
发现overall ic和逐股平均ic相比，overall ic略低。且两种ic的abs都在1e-3左右，说明因子效果较差。<br>

## 0716
### 因子的评价指标- ic, rank_ic, icir
1. 计算因子和y之间的ic，rank_ic，这两个指标应该是同号的，且一般更要关注rank_ic，因为它对极端值不敏感。<br>
2. ic和rank_ic可以逐股计算,即计算每个股票上的ic再做简单的平均，$ic_s = \frac{\sum_{t=1}^{T} (f_{s,t} - \bar{f_s})(y_{s,t} - \bar{y_s})}{\sqrt{\sum_{t=1}^{T} (f_{s,t} - \bar{f_s})^2 \sum_{t=1}^{T} (y_{s,t} - \bar{y_s})^2}}$, $ic = 1/n \sum_{s=1}^{N} ic_s$<br>
3. ic和rank_ic可以在市场上所有数据上计算，即计算所有股票在所有时间上的ic和rank_ic，$ic = \frac{\sum_{s=1}^{N} \sum_{t=1}^{T} (f_{s,t} - \bar{f})(y_{s,t} - \bar{y})}{\sqrt{\sum_{s=1}^{N} \sum_{t=1}^{T} (f_{s,t} - \bar{f})^2 \sum_{s=1}^{N} \sum_{t=1}^{T} (y_{s,t} - \bar{y})^2}}$<br>
4. icir（信息比率）是ic的时间序列的平均值除以标准差，$icir = \frac{\bar{ic}}{\sigma_{ic}}$ , icir衡量因子对y的预测的稳定性，icir越大，说明因子对y的预测越稳定。<br>
5. 总结：ic和rank_ic需要同号且abs越大越好，由于下游用所有数据一起炼，我们更关心overall ic/rank_ic，但是逐股ic/rank_ic也需要算，如果发现逐股ic/rank_ic和overall ic/rank_ic差别很大，最好查出原因。<BR>

### 分析当前因子
当前因子 $f_{s,t} = sign(rtn_{s,t}) * volume_{s,t}$，其中$rtn_{s,t}$是最近1分钟的对数收益率，$volume_{s,t}$是最近1分钟的成交量。<br>
我的改进逻辑：因子中包含了量的信息volume和价格信息rtn，算是一个简单的量价因子。但是其 fct_p 只有1，-1两个信号，且只有1min的频率，没有什么信息量。<br>
成交量反应市场活跃度，但是不考虑后市多空；因此需要在价格方面进行信息的补充。<br>
### 改进因子v0.1
1 在价格方面，使用vwap的对数收益率，不加符号函数，得到连续值
2 为防止log_rtn的极端值，tanh缩放到[-1, 1]区间
$ \mathbb{I}(A) = \begin{cases} 1, & \text{若条件}~A~\text{成立} \ 0, & \text{否则} \end{cases}$

其中 (\mathbb{I}(A)) 表示当条件 (A) 成立时取1，否则取0。

## 0717
### 改进因子v0.1
根据昨天的思路得到新的fct_p和fct_v,将它们乘为新的量价因子fct_pv,并且查看fct_p,fct_v和fct_pv的ic和rank_ic。<br>

| 因子名                                  | IC        | overall IC  |
|:-----------------------------------------|----------:|------------:|
| softsign_log_rtn_vwap                    | 0.011680  | -0.013961   |
| log_rtn_vwap                             | 0.011678  | -0.013961   |
| softsign_log_rtn_vwap_volume_avg_pre20   | 0.002102  | -0.013529   |
| log_rtn_vwap_volume_avg_pre20            | 0.002101  | -0.013529   |
| softsign_log_rtn_vwap_volume_avg_pre10   | 0.002055  | -0.013319   |
| log_rtn_vwap_volume_avg_pre10            | 0.002054  | -0.013319   |
| softsign_log_rtn_vwap_volume_avg_pre5    | 0.001541  | -0.013474   |
| log_rtn_vwap_volume_avg_pre5             | 0.001541  | -0.013474   |
| sign_log_rtn_vwap_volume_avg_pre10       | 0.000059  | -0.011766   |
| sign_log_rtn_vwap_volume_avg_pre20       | -0.000408 | -0.012015   |
| sign_log_rtn_vwap_volume_avg_pre5        | -0.000630 | -0.012159   |
| sign_log_rtn_vwap                        | -0.000868 | -0.012723   |
| log_rtn_vwap_VOLUME                      | -0.003200 | -0.015148   |
| softsign_log_rtn_vwap_VOLUME             | -0.003201 | -0.015148   |
| sign_log_rtn_vwap_VOLUME                 | -0.003496 | -0.014771   |
| volume_avg_pre20                         | -0.003778 | -0.008751   |
| VOLUME                                   | -0.006925 | -0.008791   |
| volume_avg_pre10                         | -0.007507 | -0.009947   |
| volume_avg_pre5                          | -0.008806 | -0.010720   |

第一轮得到的这些因子，按价格和量分开看：<br> 
$f_p$的ic和rankic都在0.01左右，除了sign(log_rtn)的ic很低接近0，因为sign函数只取1和-1两个值，算ic时会低，rankic同理会略低一些。<br>
$f_v$的ic和rankic都在1e-3左右，说明量的因子只包含成交量信息的话并不能有效预测后市。，需要试着加入量的变化率信息。<br>
结论：
对于价格因子，单纯的sign()处理会导致信息损失，不加为好，<BR>
对于量的因子，最好需要加入vol的变化率信息，最终得到的因子应该能描述放量和缩量的情况。<br>

### 改进因子v0.2
在量因子中加入vol的变化率。若$vol_t$是当前时刻的成交量，
变化率为$d = \frac{vol_t - vol_{t-n}}{vol_{t-n}}$，其中n为时间窗口大小。<br>
另外尝试了将vol拆为ask和bid两部分，分别计算他们的1min,5min,10min,20min的变化率。<br>
结果：
vol_d5, vol_d10, vol_d20（vol的前5，10，20分钟的变化率）的ic -0.000159， -0.004783， -0.011868； rank_ic 0.001844, -0.004838, -0.008150 是逐渐提升的。

对于rank ic，表现较好如下：

| 因子名                                   | overall rank IC |
|:------------------------------------------|----------------:|
|softsign_log_rtn_vwap_VOLUME_d5	|       -0.011122 |
|log_rtn_vwap_VOLUME_d5	|       -0.011122 |
|log_rtn_vwap_BidVol|      	-0.012232 |
|softsign_log_rtn_vwap_BidVol|      	-0.012232 |
|log_rtn_vwap_AskVol	|       -0.012777 |
|softsign_log_rtn_vwap_AskVol|      	-0.012777 |
|softsign_log_rtn_vwap|      	-0.013961 |
|log_rtn_vwap	|       -0.013961 |
|log_rtn_vwap_VOLUME|      	-0.015148 |
|softsign_log_rtn_vwap_VOLUME	|       -0.015148 |

## 0718
### 改进因子v0.3
在v0.2的得到的结果中。因子前向20分钟的vol变化率 vol_d20的ic和rank_ic较好。将volume拆分为askvol和bidvol的效果无明显提升。<br>
参考聚宽的日频因子的构造方式，进行改进，计算vol和十分钟，五分钟平均交易量的差，vold5, vold10,得到新的因子如下：
目前表现最好的如下，并且它们的 cum_icir 都很单调

| 因子名                                   | overall rankic |
|:--------------------------------------|:---------------|
| log\_rtn\_vwap\_VOLUME\_d10           | 0.011750       |
| softsign\_log\_rtn\_vwap\_VOLUME\_d10 | 0.011750       |
| log\_rtn\_vwap\_VOLUME\_d5            | 0.011258       |
| softsign\_log\_rtn\_vwap\_VOLUME\_d5  | 0.011258       |
| sign\_log\_rtn\_vwap\_VOLUME\_d10     | 0.010120       |
| softsign_log_rtn_vwap_VOLUME	         | -0.015148；     |
| sign_log_rtn_vwap_VOLUME	             | -0.014771      |


## 本周总结
1. landing和熟悉环境，熟悉hpc集群的调度。
2. 基于一个简单的量价因子评估流程，知道了工作链路
3. 进行了一些改进，目标是优化新因子的ic rankic等指标