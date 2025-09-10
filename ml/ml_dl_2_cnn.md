---
layout: default
---

从循环神经网络RNN开始，介绍GRU, LSTM等处理序列数据的网络结构，以及接下来的[transformer](./ml_transformer.md)架构。<br>
关于这条技术路线的发展，可以参考[这篇文章](https://zhuanlan.zhihu.com/p/5910889223)，从直觉到数学推导到代码实现. 本文档只记录数学原理和部分关键伪代码. <br>
# 2. 循环神经网络
## 2.1 Recurrent Neural Network (RNN)
循环神经网络及其变体可以有效处理序列数据，如交通流量，自然语言序列等。LSTM和GRU是RNN的两个重要变体，用于解决长期依赖问题。
和全连接网络比，RNN引入了共享的时间维度，使得网络可以处理任意长度的序列数据。一个典型的RNN结构如下