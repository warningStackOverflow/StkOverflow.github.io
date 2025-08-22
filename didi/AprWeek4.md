---
layout: default
---

# 动调实习第2周记录

## 4月21日
### 理解并复现组内已有的AR-ECR模型训练、测试流程
参考caixinjun代码：https://dintl-ml-s2-8077.intra.didiglobal.com/tree/caixinjun/BR/code/最优化ECR、AR模型.ipynb <br>
今天看明白它的逻辑，已经解决了基本训练流程，特征 样本 lgb训练 评估 (jupyter中)。<br>
预计明天看一看各个列的具体含义，包括grid和call的线上sql和通过计算得到的其他几样指标 <br>
OptmizARECR.py代码逻辑：<br>
```pseudo
1. 读取配置文件，预处理，调包等
    略
    
2. 用到的各类函数
    2.1 划分数据集
    func data_preparation(df_train, df_test)-> X_train, Y_train, X_test, Y_test
        分类特征onehot处理，对齐test train尺寸,输出训练/测试的x/y
        
    2.2-2.4 时间str处理函数，返回int;
    func get_day_of_week_from_date: 将日期字符串转换为星期几(0-6),int
    func get_week_of_month_from_date: 计算给定日期是所在月的第几周,int
    func get_hour_from_date: 从日期时间字符串中提取小时值(24),int
    
    2.5 concat
    func concat_data(bubble_data, rider_data, grid_data) -> df
        将格子数据、冒泡数据、司机数据拼接成一个df，返回df; no usage，在主函数里又写了一次这个逻辑
    
    2.6-2.9 加入特征列的几个函数
    func add_time_features(df) -> df：调用上面三个函数，加三行特征列
    func one_hot_enciding(df, columns) -> df: 
        对指定列(w, m, h)进行onehot编码
    func dummy_time_features(df) -> df: 
        和上面的区别是编码后将原先的列删除，加入新的列；；上面是不改变原先的列，只返回新的列
    func split_train_test(bubble_rider_grid)-> df_train, df_test:
        基于日期划分train test，其中test包含最近test_days天的数据，train包含之前的数据
    
    2.10-2.12 gb模型相关的参数传递，用到了再细看
    func get_lgb_params() -> params: 
    func get_ar_lgb_params() -> params:
    func get_psm_ar_lgb_params() -> params:
    
    2.13 模型训练
    func train_model(X_train, Y_train, X_test, Y_test, model_type) -> model, y_test_pred, y_train_pred
        训练模型，返回模型和预测值; model_type:ar, ecr, cr； 支持样本加权的boost；训练boost轮数和早断设置写在函数内，自己写的时候记得挖出来new func
    
    2.14，2.15 两个可视化展示函数
    func ar_sort_qcnt && func ecr_sort_qcnt
        相似逻辑，预测值排序；计算每组真实值、预测值；计算mae，auc；plot
    
    2.16 func getK：算斜率，可是为什么要算斜率，先疑惑一下
    
3. ar模型训练/测试
    3.1 获取训练数据
        获取格子/冒泡 grid_data:df , call_data:df 
        
# 4. ecr模型训练/测试
```
## 4月22日
### ecr模型训练/测试
上午： ar模型和ecr模型的大体逻辑都一致，但是一直前者可以运行后者报错，对比调试无果，遂先吃饭 <br>
下午： 继续调试ecr模型，发现是前后对catagorical特征的处理不一致。在预处理时，day week hour等特征应该进行哑元变量处理，ar模型跳过了直接训练。当然统计学角度看这模型练的也有问题。ecr模型只进行了哑元变量处理，但是特征列用的还是老的day week hour，于是对不上遂报错。<br>
思考：树模型对于这种标签变量到底要不要哑元处理，按道理应该要，明天带上李航书来细看一下。
### ecr预测值情况
全体数据真实ecr:0.159284 <br>
全体数据预测ecr:0.156985 <br>
AUC: 0.807 <br>
iteration: 1150 <br>
ecr的预测效果应该还可以增加。

## 4月23日
### 1. 特征变量工程
虽然ar/ecr模型的预测问题是针对结构化数据的，但是目前使用的列有160+，其中不少列之间是correlated的，比如3日期和7日期gmv等，尝试通过降维方法减少特征列数量，提速、提高解释性和性能。 <br>
在原先的原始数据集call_data,grid_data上，使用主成分分析和因子分析的方法，进行降维，逻辑如下： <br>
```pseudo
1. 获取grid_data, call_data
2. 获取特征列，只需要数值型的列，id，时间等去掉
3. pca 输出前n个主成分，作图
```
原始数据集12.6G，超过了numpy.array的限制，做了一些尝试减少内存开销，包括将int64和float64改为int32和float32，以及随机抽数据进行训练。以下为根据特征值排序的前3主成分。<br>
* 主成分 1(解释方差比: 0.5672):
* grid_cnt_bubble_dest_14: 0.1042
* grid_cnt_bubble_dest_21: 0.1041
* grid_cnt_bubble_dest_28: 0.1041
* grid_cnt_bubble_dest_7: 0.1041
* grid_cnt_bubble_origin_7: 0.1040
* 
* 主成分 2(解释方差比: 0.1334):
* grid_ar_origin_14: -0.1631
* grid_ar_origin_21: -0.1627
* grid_avg_dynamic_times_order_dest_21: -0.1617
* grid_cr_origin_21: -0.1613
* grid_avg_dynamic_times_order_dest_28: -0.1613
* 
* 主成分 3(解释方差比: 0.0326):
* grid_driver_cut_origin_3: -0.1914
* grid_gmv_origin_3: -0.1909
* grid_gmv_dest_3: 0.1874
* grid_driver_cut_dest_3: 0.1858
* grid_driver_cut_origin_7: -0.1651
* ......
* 
从grid data的主成分分析可以看出，其中有不少特征列是高相关的，在巨量数据下的三日七日十四日特征有冗余，对于某些格子而言，其打车的供需关系并没有随时间突变，因此不同滑动时间窗的均值类似。可以做出特征间协方差矩阵并且画出热力图验证此猜想。<br>
从相关系数热力图可以看出不少变量确实是高相关，预计明天进行数据清洗，去掉一些冗余变量。<br>
## 4月24日
计划降维处理后，用lgb进行训练。对于grid data和call data，计算出特征列间相关系数，并且用并查集方法将高相关（>0.85）的列进行去重。降维后主成分分析发现基本没有冗余变量。<br>
详细的实验过程可以参考我的线上仓库：https://dintl-ml-s2-8077.intra.didiglobal.com/notebooks/YangX/ecr.ipynb <br>

## 4月25日
### 对原模型进行降维处理
1. 微调和修改ar模型，原始ar模型的性能如下：<br>
* mae:0.009
* auc:0.731
* wmape:0.012
* iteration: 740
2. 对原始模型特征进行降维，做出前后的pca的图（见线上文档）<br>
3. 使用并查集检查特征列间的相关性，去掉了部分冗余变量（没用无监督方法因为基于规则的更快些，线性复杂度）<br>
4. 再次进行训练。在不影响模型性能的情况下，加速了模型训练速度。新的模型性能如下：<br>
* mae:0.011
* auc:0.730
* wmape:0.015
* iteration:460
5. 降维前后的总结：<br>
降维后选择新特征，使用树模型进行训练，模型训练速度加快了40%。但是模型性能没有显著变化。这说明树模型在建树的时候，冗余变量只是加大了建树深度，导致模型训练速度可以继续被优；而对于各个特征间的隐藏关系的挖掘能力，其实降维前后的方法是没有提升的。

