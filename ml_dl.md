# 神经网络和深度学习
## 1. 神经网络基本概念，包括前向传播，反向传播，激活函数等
我个人倾向于把神经网络和其中的传播过程当作水流一样，神经网络本质就是一个双向管道系统，不同激活函数都是不同的水阀，权重就是管道的粗细。输入前向传播，误差根据梯度反向传播。
### 1.1 全连接网络结构
全连接网络包括输入层、隐藏层和输出层，每一层的神经元与下一层的神经元全部相连。对于第i层的神经元，其输出为：$a_i = \sigma(W_i a_{i-1} + b_i)$，其中$W_i$为第i层的权重矩阵，$b_i$为第i层的偏置，$\sigma$为激活函数。
* 这里的偏置项$b_i$起到两个作用，第一个是**加速收敛，增加模型表达能力，本质上是多了一个参数**。第二个是设置了一个阈值-b，当$Wx+b>0$相当于激活，否则不激活。
* 有时候会把偏置项和权重项合并到一起，相当于权重矩阵多一个维度，即$W_i = [W_i, b_i]$，方便计算，但是自己别这么写。
### 1.2 前向传播 Forward Propagation
```
# 个人项目里dueling DQN的VAnet，注意其中pytorch实现的forward方法，本质就是水管套娃
class VAnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        # self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc0 = torch.nn.Linear(state_dim, 128) # 4 modes
        self.fc1 = torch.nn.Linear(128, hidden_dim)
        self.fcA = torch.nn.Linear(hidden_dim, action_dim)
        self.fcV = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        A = self.fcA(F.relu(self.fc1(x)))
        V = self.fcV(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)
        return Q
```

**前向传播的是数据输入流，从输入层开始，通过激活函数传播到输出层，得到预测值**。对于第i层的神经元，其输入为$i-1$层的输出： <br>
**$z_i = W_i a_{i-1} + b_i$ <br>
$a_i = \sigma(z_i)$ <br>**
其中$z_i$为第i层的输入，$a_i$为第i层的输出，$\sigma$为激活函数。其实就是简单复读上述的全连接网络结构。<br>


### 1.3 激活函数 Activation Function
从特征映射角度看激活函数对于输入输出之间，构建了**一个非线性映射**，使得神经网络可以拟合非线性函数。如果不加激活函数那么不管叠多少层都是线性变换。 <br>
激活函数按照映射关系可分为**饱和（Sigmoid, tanh等，一般有一个$e^{-x}$）、非饱和(各种Rectified Linear Unit，ReLU, Leaky ReLU. etc)**。饱和激活函数在abs(x)很大时，由于函数收敛，梯度接近于0，**容易出现梯度消失问题**。（sigmoid和tanh）因为有**指数运算计算开销大**。 

**线性激活函数**：$f(x) = x$，就是没有激活。 <br>
**Sigmoid**：$f(x) = \frac{1}{1+e^{-x}}$，常用于**二分类**，输出在(0,1)之间，函数**曲线光滑平缓，可以防止梯度突变**；但是在两端梯度接近于0，且导数$f'(x) =f(x)(1-f(x))= e^{-x}/(1+e^{-x})^2$在0到0.25之间，**容易出现梯度消失问题**。 <br>
**tanh**：$f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$，常用于**多分类**，输出在(-1,1)之间，函数**曲线光滑平缓，可以防止梯度突变**；但是在两端梯度接近于0，且导数$f'(x)=1 - f(x)^2$在0到1之间，**容易出现梯度消失问题**。 <br>
* 其实这里可以看出这两个，tanh是sigmoid的线性变换:$tanh(x)=2Sigmoid(2x)-1$，所以问题都是共有的。导数性质不够好，导致梯度消失。
* tanh还有一个优势是输出均值为0，而且不变号，sigmoid是0.5。所以练着练着sigmoid会导致输出偏移，权重更新会变慢。 


**ReLU**：$f(x) = max(0,x)$，订书机函数，负数部分梯度为0，**容易出现神经元死亡问题**。 <br>
**Leaky ReLU**：$f(x) = max(0.01x,x)$，大开口订书机，其中负数部分的线性权重可以调整,若自动学习权重就变为PReLU。**在负数部分梯度不为0，解决了神经元死亡问题**。 <br>
**PReLU**：$f(x) = max(\alpha x,x)$ 其中$\alpha$为学习参数。 <br>
**ELU**：$f(x) = \begin{cases} x & x>0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$，ELU会让输出均值接近0（因为负数部分梯度绝对值在0附近），这样有利于梯度接近自然梯度，**但是有指数计算，计算量大**。 <br>
**SELU**：$f(x) = \lambda \begin{cases} x & x>0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$，其中$\lambda$和$\alpha$为超参数。 可学习版ELU <br>
**Softmax**：$f(x)_i = \frac{e^{x_i}}{\sum_{j}e^{x_j}}$，常用于**多分类**，输出在(0,1)之间，（公式和离散选择模型是一样的，可以看作不同神经元的概率）常用于最后的输出层。 $\sum_{i}f(x)_i = 1$ <br>
* 激活函数优缺点和原因速记：$e^{-x}$导致平滑，饱和，梯度消失，计算量大；ReLU中的梯度=0导致神经元死亡，LeakyReLU，ELU和SELU解决问题。
### 1.4 反向传播 Backward Propagation
在神经网络训练过程中（以监督学习为例），对于输入特征x，输出Y，记神经网络作为一个泛函为$f(x;\omega)$，其中$\omega$为每一层权重/模型参数，损失函数为$L(Y,f(x))$，**反向传播只是求解损失函数对于模型参数梯度的一个方法**，求出梯度后再进行梯度下降来更新模型，无奈总有公司要手推反向传播，以下是推导过程： <br>
* 假设我们需要对网络模型$Y = f(X;\omega)$优化，根据梯度下降，需要计算损失函数$L(Y,f(X))$（[计算方法在这里](\ml_traintest.md)）对于模型参数$\omega$的梯度$\nabla_{\omega}L(Y,f(X))$，如果按照正常思路求导，那么可能要导晕了。于是通过从后往前的链式法则，将复杂度降到O(n)。 <br>
* 损失函数简写为L，对$\omega$的梯度更新公式为：$\omega = \omega - \eta \nabla_{\omega}L$，其中$\eta$为学习率。接下来求解$\nabla_{\omega}L$ 其实就是求解 **$\frac{\partial L}{\partial \omega}$和$\frac{\partial L}{\partial b}$**，**损失函数对权重和偏置的梯度** <br>
* $\frac{\partial L}{\partial \omega}$和$\frac{\partial L}{\partial b}$无法直接求，使用链式法则（类似递归），第i层的梯度$\frac{\partial L}{\partial \omega_l} = \frac{\partial L}{\partial a_l} \frac{\partial a_l}{\partial z_l} \frac{\partial z_l}{\partial \omega_l}$，其中$a_l$为第l层的输出，$z_l$为第l层的输入，$\omega_l$为第l层的权重，（前向传播反过来）。 <br>
* 同理，$\frac{\partial L}{\partial b_l} = \frac{\partial L}{\partial a_l} \frac{\partial a_l}{\partial z_l} \frac{\partial z_l}{\partial b_l}$，其中$b_l$为第l层的偏置。 <br>
* 接下来为简化运算，定义$\delta_l = \frac{\partial L}{\partial z_l}$，即第l层的误差项，等于损失函数对于第l层输入的梯度，用来衡量在第l层的神经元对输出误差的贡献。 将其展开，$\delta_l = \frac{\partial L}{\partial a_l} \frac{\partial a_l}{\partial z_l} = \frac{\partial L}{\partial a_l} \sigma'(z_l)$，其中$\sigma'$为激活函数的导数。即**第i层偏差 = 损失函数随着第i层输出的变化率 * 第i层激活函数随$z_i$的变化率/激活函数的导数**。 <br>
* 将误差项带入，$\frac{\partial L}{\partial \omega_l} = \delta_l a_{l-1}$，$\frac{\partial L}{\partial b_l} = \delta_l$，即**第i层权重更新 = 第i层误差项 * 第i-1层输出**，**第i层偏置更新 = 第i层误差项**。求偏导即可得到 <br>
* 于是一个从前向后导晕过去的问题，变成了一个O(n^2)的计算误差项的问题，已知第l层误差，求第l-1层误差的反向传播为：<br>
* $\delta_{l-1} = \frac{\partial L}{\partial z_{l-1}} = \frac{\partial L}{\partial a_{l-1}} \frac{\partial a_{l-1}}{\partial z_{l-1}} = \delta_l W_l \sigma'(z_{l-1})$，即**第i-1层误差项 = 第i层误差项 * 第i层权重 * 第i-1层激活函数的导数**。 <br>
* 记住前向，记住w a z 和误差定义，慢慢导即可。 <br>
* 可能会问pytorch的backward方法，其实就是自动求导，只要定义好网络结构，然后定义好损失函数，调用backward方法，pytorch会自动计算梯度，然后调用optimizer.step()更新参数。 <br>
* 可能会问反向传播过程中的哈达玛积，即对应元素相乘为什么没体现，只能说，纸上推导写乘法，代码用torch里的广播机制，真问这么细那别实习了。 <br>
### 1.5 梯度消失和梯度爆炸
同一个问题的一体两面，梯度消失是指在反向传播过程中，梯度逐渐变小，导致模型参数无法更新，梯度爆炸是指梯度逐渐变大，导致模型参数更新过大，两者都会导致模型训练困难。 <br>
解决方法：<br>
* 选用合适的激活函数，如ReLU，Leaky ReLU，ELU等，避免梯度消失；用饱和激活函数，如tanh，sigmoid，避免梯度爆炸。 <br>
* 使用Batch Normalization，对每一层的输入进行归一化，使得梯度更加稳定。 <br>
* 使用梯度裁剪，设置一个阈值，当梯度超过阈值时，将梯度裁剪到阈值以内。 <br>

## 2. 循环神经网络（包括RNN，LSTM和GRU）
## 3. 图卷积网络
## 4. Attention机制，Transformer（其他和时空特征有关的模型，高德估计会问）
## 5. 深度学习训练技巧
## 6，大模型有关，如BERT（了解，表示我紧跟技术）
