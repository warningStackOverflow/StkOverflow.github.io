# JulWeek3
## 0714
### 入职，熟悉环境，装好配置，得到一个小demo：
1. 生成20250401-20250430全市场的1分钟频率 sign(最近1minKbar的Vwap 对数收益率) * Volume_t 因子 // 先一周，沪深300
2. 计算IC(pearson correlation)，rankIC (spearman correlation)
	i) 所有股票所有时间（overall）的IC，rankIC
	ii) 逐股平均
	iii) 逐天逐股
     然后分析overall的IC是否与逐股平均有差别，哪个高，为什么
     IC和rankIC是否有差别，哪个高，为什么
3. 分别计算因子和label的自相关系数和偏自相关系数，多少阶是显著的，哪个高？如何匹配因子和label的自相关（EMA平滑？MA平滑？）
4. 因子和label平稳吗，平均来看，在日内的早盘（开盘后前30分钟）和其余时间因子和label的分布是否有差异？label的标准差是否有差异？
5. 尝试改进这个因子，提高其overall IC或rankIC。
6. 如果做的快的话完成：https://zhuanlan.zhihu.com/p/616022509 中提到的内容。

公司使用hpc集群，平时写代码在登陆节点编辑，运行用slurm，anaconda用module的方式直接load <br>
环境在 module load anaconda3/2023.03 <br>
得到token：660144fd0257a45ea8247d209eff689251f66a2f0cc083a3 <br>

本地测试代码时，具体使用方法参考：https://github.higgsasset.com/HiggsIT/HPC-Introduction/wiki/Modules-Guide <BR>
线上实验使用slurm，具体参考：https://github.higgsasset.com/HiggsIT/HPC-Introduction/wiki/Slurm-Guide <br>
### 理解和复现demo
1. shannon_store，higgsboom, gauss库的安装（已完成）
higgsboom中的一些数据： https://github.higgsasset.com/HiggsIT/Boom-Introduction/tree/main/%E6%96%B0%E7%89%88Boom/data/market
2. 确认数据的格式，数据量，和计算的时间复杂度。
3. 复现demo，熟悉代码结构。
4. 了解IC和rankIC的计算方法，如何使用pearson和spearman correlation