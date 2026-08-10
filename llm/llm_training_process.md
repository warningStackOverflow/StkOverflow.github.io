训练流程
在大语言模型（LLM）的整个生命周期中，大模型主流的训练方法可以划分为 预训练/后预训练、监督微调（SFT）、偏好对齐（Preference Alightment） 以及 自演化/自修正（Self-Improvement） 四大类：

# 一、 预训练与领域增强（Pre-training & Continual Training）
1. 基础自回归语言建模（Next-Token Prediction / Causal LM）
原理：大模型最底层的“筑基”方法。给定前文 $t_1, \dots, t_{k-1}$，预测下一个词 $t_k$ 的概率分布。采用交叉熵损失函数（Cross-Entropy Loss）。
作用：让模型在海量文本数据中吸收人类世界的知识、语言语法结构和基础逻辑推导能力。
2. 持续预训练 / 领域增量预训练（Continual Pre-training）
原理：在基座模型（Base Model）训练完成后，使用特定领域（如法律、医学、交通交通仿真代码等）的大规模未标注文本继续进行 Next-Token Prediction 训练。
作用：注入特定的领域知识库与专业词汇表，避免直接微调导致模型失去泛化能力的“灾难性遗忘”。
# 二、 监督微调（SFT, Supervised Fine-Tuning）
1. 全参数 SFT（Full Fine-Tuning）
原理：使用高质量的“指令-回答”对（Instruction-Response Pairs），更新模型的所有参数。
作用：将“续写文本”的基座模型转化为“理解并遵循人类指令”的对话助手。
2. 参数高效微调（PEFT, Parameter-Efficient Fine-Tuning）
为了降低显存和算力开销，不更新全量参数，而是仅更新极少部分参数：
LoRA (Low-Rank Adaptation)：
原理：冻结原模型权重 $W_0$，在旁边并行引入两个低秩矩阵 $A$ 和 $B$（即 $\Delta W = B \cdot A$）。只更新 $A$ 和 $B$ 的参数。
作用：将可训练参数量减少到全量微调的 0.01%~1%，极大降低微调门槛，且支持多任务 Adapters 的快速切换。
QLoRA：结合 4-bit 量化（NormalFloat4）与 LoRA，进一步将显存占用降低近 60%~70%，使得单卡消费级显卡（如 RTX 4090）微调大模型成为可能。
Prefix Tuning / P-Tuning：在 Transformer 的输入或注意力层隐状态前拼接入可学习的连续 Virtual Tokens（虚拟提示词）。
# 三、 偏好对齐（Preference Alignment）
1. DPO（Direct Preference Optimization，直接偏好优化）
原理：通过数学变换，将基于 Bradley-Terry 模型的 RM 损失函数直接表达为“当前策略模型 $\pi_\theta$”与“参考模型 $\pi_{\text{ref}}$”的概率比值。
损失核心：提高优胜回答 $y_w$ 的相对生成概率，压低落败回答 $y_l$ 的相对生成概率。
优点：结构极其简单（就是一个分类 Loss），不需要采样生成、不需要运行 RM，训练稳定且高效。
2. IPO（Identity Preference Optimization）
原理：针对 DPO 容易对偏好数据过拟合（导致 Likelihood Collapse）的问题，在 Loss 中加入了二次方归一化惩罚项。
作用：控制策略模型偏离 Reference Model 的程度，保证输出多样性。
3. KTO（Kahneman-Tversky Optimization）
原理：灵感源自心理学的前景理论（Prospect Theory）。它不需要成对的对比数据 $(y_w, y_l)$，只需要单条回答及一个二元标签（好/坏，True/False）。
作用：极大降低了偏好数据的收集难度（现实中收集单个回答的好坏比收集两两比较容易得多）。
# 四、 自演化与自修正训练（Self-Improvement & Reasoning Training）
随着 Reasoning 模型（如 OpenAI o1, DeepSeek-R1）的兴起，不依赖人类标注、依靠模型自身生成与验证的训练方法成为当前研究最火热的前沿方向。
1. STaR（Self-Taught Reasoner，自教推理者）
原理：让模型尝试生成带 CoT（思维链）的解答；如果答案正确，就将该 CoT 加入微调数据集；如果答案错误，提供正确提示后再让模型重试，成功后加入数据集。然后用这些过滤出的高质量样本重新 SFT 模型。
作用：通过 Bootstrapping（自举）循环不断提升模型的逻辑推理能力。
2. Rejection Sampling Fine-Tuning（拒绝采样微调 / RFT）
原理：针对每个 Prompt，用模型批量采样生成多个回答，利用可验证规则（如代码编译器、数学计算器）或强 RM 进行过滤，仅保留得分最高的回答，然后对其进行常规的 SFT。
作用：在进行复杂的 RL 之前，快速提升模型在特定任务上的输出质量基线。
3. ORPO（Odds Ratio Preference Optimization）
原理：将 SFT 与偏好对齐（Preference Optimization）融合成单阶段训练。在标准 SFT 损失的基础上，增加了一个几率比（Odds Ratio）惩罚项，在教模型“如何回答”的同时直接惩罚不好的回答。
作用：无需分成“先 SFT，后 RLHF/DPO”两步，节省训练流程。

# GRPO
GRPO（Group Relative Policy Optimization，组相对策略优化） 是一种专门针对大语言模型（LLM）强化学习对齐（RLHF）高效设计的策略梯度算法。它最早由 DeepSeek 团队在 DeepSeekMath 中提出，并在 DeepSeek-R1 等推理大模型的训练中发挥了核心作用。
传统的强化学习对齐算法（如 PPO）通常需要占用大量的算力和显存，而 GRPO 成功破解了这一瓶颈，成为大模型 Reasoning（复杂逻辑推理、数学、代码）训练的行业新标准之一。
## 一、 GRPO 解决的核心痛点
在传统的 PPO（Proximal Policy Optimization） 训练框架中，系统需要同时维持多个模型：
Actor Model（策略模型）：负责生成文本（就是我们想训练的目标 LLM）。
Critic Model（价值模型）：用于评估生成文本的预期收益，计算“基线（Baseline）”，从而得出 Advantage（优势值）。
Ref Model（参考模型） & Reward Model（奖励模型）。
传统 PPO 的致命伤在于 Critic 模型：
为了能准确评估动辄几百亿/千亿参数的 Actor 模型生成的文本，Critic 模型通常需要和 Actor 保持同等规模。这意味着显存开销和计算资源直接翻倍，导致训练成本极高、部署难度极大。
GRPO 的核心突破：斩掉 Critic 模型！
GRPO 摒弃了独立的 Critic 参数网络，转而利用同组内采样结果的相对比较来隐式计算 Baseline，从而极大地节省了显存，使得更大参数规模的强化学习训练成为可能。
## 二、 GRPO 的工作流程与运行机制
GRPO 的核心逻辑可以拆解为以下 4 个关键步骤：
`Prompt (问题 q)
  └──> Policy 生成一组回答 (Group Outputs: o1, o2, ..., oG)
         └──> Reward/Rules 计算各自打分 (r1, r2, ..., rG)
                └──> 组内标准化计算 Advantage (Ai)
                       └──> 更新 Policy 参数 (PPO-Style Clip)`

1. 组采样（Group Sampling）
对于输入的一个提示词或问题 $q$，模型（旧策略 $\pi_{\theta_{old}}$）不会只生成一次回答，而是并发地采样生成一组（$G$ 个）候选回答 $\{o_1, o_2, \dots, o_G\}$。
2. 奖励打分（Reward Scoring）
每一个回答 $o_i$ 会被送入评估系统，获得一个标量奖励得分 $r_i$。奖励可以来自：
基于规则的检验（Rule-based Reward）：如数学题答案是否正确、代码能否跑通编译、输出格式是否满足要求（在 DeepSeek-R1 中被大量使用）。
神经网络奖励模型（RM-based Reward）：传统偏好打分模型。
3. 组内相对优势计算（Group Advantage Estimation）
这是 GRPO 最精妙的地方。它不依赖独立的 Critic 网络预测价值，而是将这 $G$ 个回答的得分在组内进行标准化：
计算这组得分的均值 $\text{mean}(r)$ 和标准差 $\text{std}(r)$；
计算每个回答的优势值（Advantage） $A_i$：
$$A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G\})}{\text{std}(\{r_1, r_2, \dots, r_G\})}$$
原理本质：如果一个回答得分大于组内平均水平（$A_i > 0$），它就会得到正向鼓励；反之（$A_i < 0$）受到惩罚。这种“组内卷”的方式天然消除了全局量纲差异，构成了稳定的参照系。
4. 策略更新与 KL 散度约束（PPO-style Update）
得到了优势值 $A_i$ 后，GRPO 使用类似 PPO 的裁剪目标函数（Clipped Objective）来更新 Actor 模型，确保模型参数不会一次性改变过多，并引入 KL 散度约束（防止偏离原始模型太远造成崩塌）。
# 三、 GRPO 在大模型训练中的关键作用
1. 极大地降低显存开销与训练成本
通过省去 Critic 模型，显存消耗可以减少 30%~50%，使得开源社区和中小型团队能够在有限的 GPU 资源上，针对超大规模模型（如 70B+）实施强化学习对齐。
2. 激发大模型的“思维链（CoT）”与自省反思能力
在 DeepSeek-R1 的训练中，GRPO 被配合“可验证奖励（RLVR，如数学/代码真值校验）”使用。由于无需人工标注思考步骤，模型在尝试获得更高组内相对奖励的过程中，自主演化出了长文本推理、自我纠错（Self-Correction）、多视角验证等高级思维特性。
3. 支持高并发采样，加速收敛
大模型训练中生成文本通常慢于反向传播。GRPO 天生需要对同一 Prompt 生成多份回答，这极其契合现代推理引擎（如 vLLM、SGLang）的并行批处理（PagedAttention/Continuous Batching）机制，极大提升了吞吐效率。
