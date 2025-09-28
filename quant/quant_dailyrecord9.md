# 第九周
1. 关于网络优化：
理由 - 原有的PPO-MASK算法，使用了LSTM网络，LSTM处理时间序列的表现不一定比transformer好，transformer的多头可以并行计算，而且k-v的形式
可以让模型更好地学习到生成的符号序列之间的关系。<br>
修改 - 将网络改成了transformer，替代了之前的lstm，多头设为6个，原论文是8个，但是8个在1800票上跑会爆显存。<br>
结果 - 使用transformer网络在1800票上训练，在全市场上测试，得到的因子如下：

2. st股和新股的处理：
在和强化学习因子生成框架对接的higgsdb类中添加了st和新股的处理，保证训练数据不含这些，但是用目前的程序，tensor-wise地生成测试集的逐分钟的因子值时，如果剔除st和新股，会导致
会导致tensor维度不一致，暂时不能解决。目前的解决方法是直接用jupyter脚本手动写这些因子，在脚本里可以有剔除的逻辑。

3. 关于llm在因子挖掘中的应用
使用qwen替代原有的gpt，qwen的具体调用分为流式输出和非流式输出两种，先用后者实现一次完整的通信。<br>
具体的方法参考qwen的[技术文档](https://www.alibabacloud.com/help/zh/model-studio/compatibility-of-openai-with-dashscope?spm=a2c63.p38356.help-menu-2400256.d_2_10_0.1f9d665aoelQzs)<br>
简单的流程就是：（1）获取api-key （2）安装openai包（3）构建messages（4）贴上api-key，调用 <br>

4. qwen的副作用
上述的 2 中提到了st股和新股的处理，原有的higgsdb类中是没有这些处理的，导致训练时会报错，于是可以直接通过调用LLM的方式，实现从逆波兰表达式到正常df['col]的转换，目前已经把这个功能作为一个新的proj实现了(在qwen帮助下）。<br>
简单来说新的proj架构如下:

`factor-agent/
│
├── agent.py                  # 主程序：构建 prompt + 调用 LLM + 生成代码
├── requirements.txt          # 依赖包
├── prompts/
│   └── system_prompt.txt     # 标准化提示词模板（可热更新）
├── output/
│   └── fetch_factors.py      # 自动生成的因子函数文件（每次覆盖或版本化）
├── examples/
│   └── sample_exprs.json     # 示例因子表达式集合
└── README.md                 # 使用说明
`