---
layout: default
---

# 动调实习第 5 周记录
## 本周计划
1. 继续学习uplift
2. 继续优化dl模型，期望超过原有树模型的0.73
3. 学习一些推荐系统中的ctr模型的知识，这和预测ar模型很像

## 5月12日
### 深度学习中的unbalanced （已有结论，不需要做平衡，采样分布和真实分布一致模型表现好）
1. 目前措施：weighted loss
由于训练数据集中的不平衡，需要在训练的时候进行样本加权，目前发现正负样本在3:1左右时表现较好，auc = 0.707 <br>
缺点（来自知乎）：这种re-weight 做法，基本假设是「weight normalisation crucially relies on the weight norms being smaller for rare classes」，但是这种假设对于优化器（optimiser）的选择是十分敏感的。<br>
2. 可能其他措施：adjustment logit loss
Long-Tail Learning via Logit Adjustment ICLR2021； 提到了一种调整交叉熵的新的loss方法，可以参考一下。 <br>
3. 可能其他措施： smote进行人工数据补强

### 是否需要L1正则化
目前的训练特征是主成分分析后的特征，经过了降维处理，特征数目已经不多了，所以L1正则化的必要性应该不大，而且L1和L2正则项的尺度目前无法确定是否一致。 <br>

### one epoch过拟合现象
特征稀疏性与one-epoch现象密切相关。目前模型似乎没有出现此现象

## 5月13日
### time embedding后模型性能提升 （已完成）
目前的模型使用了时间变量hour的embedding，模型性能到了auc = 0.721左右。训练关键参数如下：<br>
lr = 0.001; lambda l1 =1e-4; lambda l2 = 1e-4; batch_size = 4096 <br>
embedding net 的结构
```python
class EmbeddingNet(nn.Module):
    def __init__(self, numerical_dim, catalog_dim, embedding_dim):
        super(EmbeddingNet, self).__init__()
        # embedding
        self.typ = 'embeddingnet'
        
        self.embedding = nn.Embedding(catalog_dim, embedding_dim)
        self.numerical = nn.Linear(numerical_dim, 64)
        
        self.fc_output = nn.Linear(embedding_dim+64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        
    def forward(self, x_numerical, x_catalog):
        x_catalog_embedded = self.embedding(x_catalog).squeeze(1)
        x_numerical_processed = F.elu(self.numerical(x_numerical))
        
        x_combined = torch.cat([x_numerical_processed, x_catalog_embedded], dim=1)
        x = F.elu(self.fc_output(x_combined))
        x = F.elu(self.fc2(x)) 
        x = torch.sigmoid(self.fc3(x))  
        
        return x
```
### 采用embedding思路后，模型可能的优化方向（拟定周四，周五完成）
1. 考虑对连续变量进行embedding处理
https://zhuanlan.zhihu.com/p/480030543 参考这篇博客可以对重要程度较高的连续变量进行embedding处理
2. 采用deep&cross network
也是一个经典的ctr模型，可以加一点高维特征进去。

## 5月14日
### 验证模型弹性发现目前网络模型的输出分布不合理（or置信度过高，自信）(已解决)
模型输出的pdf应该是以E(ar)为中心的单峰值的曲线，目前模型倾向于将所有结果预测到0或1上。可能原因和解决方案如下：<br>
1. 模型使用的交叉熵损失函数，交叉熵损失会特别关注预测概率低的样本，导致模型过于自信，提出解决方案是使用一个修改后的bce作为损失函数,参见[https://spaces.ac.cn/archives/9526]
2. 树模型为单峰是因为bagging的时候相当于投票，会筛去极端情况；其他分类器可能会导致模型输出偏向0和1。该文章中以naive bayes为例,提到重复输入会导致模型自信，参见[https://zhuanlan.zhihu.com/p/453713117]
3. 参见[https://blog.csdn.net/ytusdc/article/details/128503206], 这篇文章提到了神经网络预测极端的原因，并且使用label smoothing解决问题，虽然他说的是多分类场景，但是，和上述文章表达的是一致的，如下：<br>
* 网络学习过程中，鼓励模型预测为目标类别的概率趋近1，非目标类别的概率趋近0，即最终预测的logits向量（logits向量经过softmax后输出的就是预测的所有类别的概率分布）中目标类别 zi 的值会趋于无穷大，使得模型向预测正确与错误标签的logit差值无限增大的方向学习，而过大的logit差值会使模型缺乏适应性，对它的预测过于自信。<br>
* soft label 可行的另一个人的论据：二分类的softmax其实等价于sigmoid，如果一直给one-hot的标签，在训练集几乎都能分正确的前提下，网络趋向于将feature的norm无限拉长，sigmoid越来越接近0-1的阶跃函数，几乎所有样本的输出就都在接近0和接近1的位置，中间态几乎没有。<br>

4. [https://www.zhihu.com/question/362870151 ]有人系统整理了解决方案，和上述一致。<br>

可能解决方案：<br>
1. 画出现有的xgb和dnn的calibration curve，查看模型的输出分布是否合理，是不是这个问题。理想情况应该是xgb是正常的，目前dnn不太正常。<br>
2. 修改损失函数，不用bce，（有说法说bce+softmax+2dim ｜｜ binaryloss+sigmoid+1dim） <br>
3. 使用label smoothing用soft label <br>
4. 待定,现有数据有900k条，按照100一组加权当成回归来做。每组里的 num(ar==1)/100可以看成是预测概率的一个估计。<br>

解决进展：<br>
1. 画出了xgb和dnn的calibration curve，发现xgb的calibration curve
2. 损失函数的修改方法，改用bce而不是bcewithlogits，问题解决，目前神经网络模型的auc，mae，wmape都和树模型非常接近。
3. 发现之前模型标准化时是train和test分别做标准化，这有可能导致模型各种东西不合理，重写之后重新训练。
4. dp的弹性的画图函数封装了一下，毕竟以后总用。

一些问题： <br>
1. 不管是xgb还是dnn，对于dynamictime的弹性都有些问题,按道理当动调倍数越高，ar应该越大，但是目前模型在dp=2之后ar没有变化，趋于平缓。猜测可能是模型的原因，也可能是样本分布中高dp的较少，可以对dp分桶验证一下
2. 等频分组离散化后，dnn模型总是比真实值略小，不分组或等宽分组均无此现象，原因未知 #todo

## 5月15日和16日
### todo1:对数据进行分桶离散化后，训练模型，查看性能 （已完成,auc会有小提升，大概0.002-0.003)
对几个连续变量按bin=10等宽分桶离散化，各自做 input=10，embedding=10的嵌入，只加入时间变量embed的模型性能如下：<br>
* auc= 0.7206735428343035
* mae= 0.22709697
* wmape= 0.1928232

等频分桶后，模型的性能如下：<br>
* mae:0.007 
* auc:0.7226 
* wmape:0.009

等宽分桶后，模型性能如下：<br>
* mae:0.006
* auc:0.723
* wmape:0.008

### todo2: dynamic time和模型输出ar应该是严格单调增的关系 (已完成,新模型可以确保dpt单调递增)
深度学习模型在dpt>2后，预测的ar反而下降了。
https://arxiv.org/pdf/2002.05515 可参考这篇文章搭建网络

### todo3:将目前深度学习模型全流程封装好

## 本周总结：
1. 对深度学习模型进行了更细的修改
2. 确保dpt和ar是单调递增的关系
3. 目前最好的模型结构应该是等频分桶+embedding+monotonicnet
