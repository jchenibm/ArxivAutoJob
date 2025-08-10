# Arxiv Paper Summary - 2025-08-10

## [Towards Generalizable Safety in Crowd Navigation via Conformal Uncertainty Handling](https://arxiv.org/pdf/2508.05634v1)

**ID**: `2508.05634v1`

### 摘要 (Detailed Summary)
该论文提出了一种基于强化学习的机器人人群导航方法，旨在提高在分布外(OOD)场景中的泛化能力。研究指出，传统强化学习方法在面对与训练数据不同的新场景时，性能会显著下降。该方法通过自适应共形推理(ACI)量化行人轨迹预测的不确定性，并将其融入到机器人观测中。然后，利用约束强化学习(CRL)引导机器人行为，使其能够适应分布变化，从而提高导航策略的鲁棒性。实验结果表明，该方法在分布内(in-distribution)场景中取得了优于现有方法的成功率，并显著减少了碰撞和侵入行人轨迹的次数。在三个OOD场景（速度变化、策略变化和群体行为）中，该方法也表现出更强的鲁棒性。最终，该方法在真实机器人上进行了部署验证，证明了其在稀疏和密集人群中安全导航的有效性。

### 主要贡献 (Contributions)
- 提出了一种基于不确定性量化的人群导航框架，通过自适应共形推理(ACI)估计行人轨迹预测的不确定性。
- 利用约束强化学习(CRL)将不确定性估计融入到决策过程中，引导机器人行为，提高对分布变化的适应性。
- 设计了一种基于累积侵入不确定性区域的约束，从而有效解决稀疏约束反馈的问题，提高安全性。
- 在多个OOD场景中验证了该方法的鲁棒性，证明其在速度变化、策略变化和群体行为等情况下仍能保持高性能。
- 在真实机器人上部署了该方法，验证了其在实际人群环境中的可行性和有效性。

### 技术方法 (Methods)
- **自适应共形推理 (ACI):** 使用动态调整的自适应共形推理 (DtACI) 量化预测不确定性。DtACI 通过同时运行多个预测误差估计器，并根据历史性能自适应地选择最佳输出，从而有效适应分布变化。
- **约束强化学习 (CRL):** 使用 CRL 引导智能体的行为，利用不确定性量化结果。通过对智能体侵入其他智能体的不确定性区域的累积量施加约束，而不是直接约束碰撞率，从而提供行为层面的指导，并解决稀疏约束反馈问题。
- **策略网络结构:** 将不确定性量化结果与预测轨迹连接，然后输入到包含注意力层的策略网络中。网络包含人-人注意力 (H-H attention) 和人-机器人注意力 (H-R attention)，以捕捉智能体之间的交互。

### 主要结论 (Conclusions)
该论文提出了一种基于强化学习的轨迹规划框架，该框架将共形不确定性集成到 CRL 方案中，以减轻 OOD 性能下降。与过度拟合并在分布变化下失败的传统 RL 规划器不同，该方法动态地利用不确定性估计来适应速度变化、策略变化以及从个体动力学到群体动力学的转变。 广泛的仿真证明了各种 OOD 场景下的稳健稳定性，而实际试验证实了该方法的实际有效性。

### GitHub链接 (GitHub Links)
- https://gen-safe-nav.github.io/

---

## [论SFT的泛化性：一种基于奖励修正的强化学习视角](https://arxiv.org/pdf/2508.05629v1)

**ID**: `2508.05629v1`

### 摘要 (Detailed Summary)
该论文提出了一种针对大型语言模型（LLM）的监督式微调（SFT）的改进方法，旨在解决SFT相对于强化学习（RL）泛化能力有限的问题。通过数学分析，论文揭示了标准SFT梯度隐含地编码了一种有问题的奖励结构，这严重限制了模型的泛化能力。为了纠正这一点，论文提出了动态微调（DFT），通过动态地重新调整目标函数与每个token的概率，来稳定每个token的梯度更新。DFT在一系列具有挑战性的基准测试和基础模型上显著优于标准SFT，表现出大大提高的泛化能力。此外，DFT在离线RL设置中也表现出有竞争力的结果，提供了一种有效但更简单的替代方案。这项工作将理论见解和实践解决方案联系起来，大大提高了SFT的性能。

### 主要贡献 (Contributions)
- 从理论上将LLM SFT确定为策略梯度空间中的一种特殊RL，并精确指出SFT泛化能力有限的根本原因，并推导出一种改进它的方法。
- 提出了动态微调（DFT），通过token概率动态地重新调整SFT损失，有效地抵消了导致意外奖励结构和无界方差的反概率加权。
- 实验表明，DFT这种简单的解决方案（只需一行代码），可以显著提高LLM SFT在各种任务和模型上的性能和泛化能力。
- DFT在离线强化学习环境中也表现出最佳性能，优于离线（RFT、DPO）和在线（PPO、GRPO）基线，证明了其作为一种简单而有效的微调策略的效率和优势。
- 分析了训练前后token概率分布的变化，发现SFT倾向于均匀地增加token概率，而DFT则对一部分token概率进行抑制，可能有利于模型关注实质性内容而非语法功能。

### 技术方法 (Methods)
- **数学分析：** 将SFT梯度重写为策略梯度，通过重要性采样，揭示了SFT可以被视为一种特殊的策略梯度方法，其奖励函数是指示函数，但受到重要性权重1/πθ的偏差。
- **动态微调（DFT）：** 通过将标准SFT目标与token概率进行重新缩放，中和了导致意外奖励结构和无界方差的反概率加权。具体来说，对于每个token，将SFT目标乘以token概率。
- **公式化：** 将修正后的SFT损失定义为L_DFT(θ) = E(x,y⋆)∼D [sg(πθ(yt⋆|x)) log πθ(yt⋆|x)]，其中sg(·)表示停止梯度运算符。

### 主要结论 (Conclusions)
该论文通过理论分析和实验验证，表明DFT是一种简单而有效的方法，可以显著提高SFT的性能和泛化能力，缩小了与更复杂的RL方法之间的性能差距。DFT通过动态地重新调整SFT损失与token概率，纠正了SFT梯度中隐含的ill-posed奖励结构，从而稳定了学习过程，促进了更好的泛化。

### GitHub链接 (GitHub Links)
- https://github.com/yongliang-wu/DFT

---

## [H-NET ++: Hierarchical Dynamic Chunking for Tokenizer-Free Language Modelling in Morphologically-Rich Languages](https://arxiv.org/pdf/2508.05628v1)

**ID**: `2508.05628v1`

### 摘要 (Detailed Summary)
该论文提出了一种名为H-NET++的层级动态分块模型，用于在形态丰富的语言（MRLs）中实现无分词器的语言建模。该模型旨在解决传统分词器在处理MRLs时遇到的问题，如词汇量爆炸、空格不一致以及零宽度非连接符（ZWNJ）等正字法伪影。H-NET++通过端到端训练学习语言相关的分段，其关键创新包括：轻量级的Transformer上下文混合器用于跨块注意力；用于文档级别一致性的双层潜在超先验；针对波斯语ZWNJ等正字法伪影的专门处理；以及带有分阶段序列长度的基于课程的训练。在14亿token的波斯语语料库上，H-NET++实现了最先进的结果，包括降低bits-per-byte（BPB）、在ParsGLUE上获得更高的准确率、提高对ZWNJ损坏的鲁棒性，并在黄金形态边界上获得更高的F1分数。该模型学习到的chunks与波斯语形态对齐，无需显式监督，证明层级动态chunking为MRLs提供了一种有效的无分词器解决方案，同时保持了计算效率。

### 主要贡献 (Contributions)
- **新型架构（H-NET++）。** 具有Transformer增强的层级路由器，带有用于形态丰富语言的潜在超先验。
- **课程优化。** 一种分阶段的AdamW训练方案，可稳定长序列字节级别训练。
- **鲁棒性评估套件。** 字符级别的噪声鲁棒性基准和一个新的波斯语黄金分割数据集。
- **最先进的性能。** 在BPB、下游准确性和鲁棒性方面实现了领先的结果。
- **无监督形态切分。** 通过动态分块学习，在没有显式形态监督的情况下，对波斯语形态学进行有意义的切分。

### 技术方法 (Methods)
- **层级路由器：** 使用L层的双向GRU和边界预测器，每一层逐步对字节进行分组。

- **Transformer上下文混合器：** 使用一个单层多头自注意力模块，使chunks能够彼此关注，捕捉长距离依赖。

- **双层潜在超先验：** 使用全局潜在向量来捕获文档级别的形态一致性。

- **ZWNJ感知字节嵌入：** 针对ZWNJ字符引入特殊的嵌入方式，使其能够学习ZWNJ特定的模式。

- **课程学习：** 通过逐渐增加序列长度的三阶段课程策略来稳定优化。
- **Gumbel Softmax：** 通过Gumbel Softmax来采样边界gates，并用温度退火来鼓励离散决策。
- **损失函数：** 损失函数结合了语言建模、KL正则化和形态对齐。

### 主要结论 (Conclusions)
该论文得出结论：H-NET++是一种层级动态分块模型，能够成功消除形态丰富的语言的分词瓶颈，同时保持计算效率。通过对波斯语的系统评估，证明了学习到的分割可以超越精心设计的标记器，并且优于基于BPE的模型，同时在没有形态监督的情况下学习与语言语素对齐的片段。

---

## [Simulating Human-Like Learning Dynamics with LLM-Empowered Agents](https://arxiv.org/pdf/2508.05622v1)

**ID**: `2508.05622v1`

### 摘要 (Detailed Summary)
该论文提出了LearnerAgent，一个基于大型语言模型（LLMs）的多智能体框架，用于模拟真实的教学环境，并探索类似人类的学习动态。该框架构建了具有不同心理学特征的学习者，例如深度学习者、表面学习者和懒惰学习者，以及一个无角色设定的通用学习者，以观察基础LLM的默认行为。通过每周的知识获取、每月的策略选择、定期的测试和同伴互动，该框架能够跟踪个体学习者在为期一年的学习过程中的动态学习进展。研究发现深度学习者能够实现持续的认知增长，而表面学习者则表现出知识的脆弱性。此外，学习者的自我概念会随着时间的推移而发生变化，通用学习者尽管认知能力有限，但自我效能感却出人意料地高。更重要的是，基础LLM的默认状态是一种“勤奋但脆弱的表面学习者”，即模仿优秀学生的行为，但缺乏真正的、可泛化的理解。

### 主要贡献 (Contributions)
- 提出了LearnerAgent框架，这是一个新颖的多智能体系统，能够模拟真实的教学环境，并可用于研究不同学习者的学习行为。
- 通过构建具有不同心理学特征的学习者（深度学习者、表面学习者、懒惰学习者和通用学习者），成功模拟了不同学习者的学习策略、推理和认知努力。
- 通过纵向分析，揭示了不同学习者的学习动态，发现只有深度学习者才能实现可持续的认知增长，而表面学习者的知识则显得脆弱。
- 观察到学习者的自我概念会随着时间的推移而动态演变，通用学习者（基础LLM）的自我效能感会逐渐增强。
- 发现基础LLM的默认状态是一种“勤奋但脆弱的表面学习者”，即能够模仿理想学生的行为，但缺乏更深层次的理解能力。

### 技术方法 (Methods)
- **角色设定：** 构建教师Agent和多种学习者Agent（深度学习者、表面学习者、懒惰学习者、通用学习者）。
- **学习与改进周期：** 设计为期一年的学习周期，包括每周学习和策略选择、每月复习和评估、同伴互动和辩论。
- **记忆机制：** 采用短期记忆（维护对话上下文）和长期记忆（存储完整的学习历史）机制，并使用上下文相关的检索策略。
- **能力评估：** 综合评估策略，涵盖学业表现（通过各类考试和陷阱问题）和心理变化（通过自我概念评估）。

### 主要结论 (Conclusions)
该论文介绍了LearnerAgent，一个旨在模拟和分析基于教育心理学的人类学习复杂动态的多智能体框架。通过为期一年的模拟，我们成功地跟踪了不同学习者的各种学习行为、发展轨迹和社会互动。研究工作得出了三个主要见解。首先，角色驱动的智能体可以高保真地复制细致入微的人类学习行为。其次，我们发现了智能体之间明显的学习缺陷，这些智能体表现出相似的短期表现，但在长期泛化方面存在显著差异——这一发现通过暴露更深层次脆弱性的表现“陷阱”来说明。第三，无角色设定的基础LLM的默认涌现行为，被称为“勤奋但脆弱的表面学习者”，反映了一个掌握表面能力但缺乏强大、可泛化理解的智能体。研究还表明，膨胀的自我概念会导致过度自信，最终阻碍增长。

---

## [The Missing Reward: Active Inference in the Era of Experience](https://arxiv.org/pdf/2508.05619v1)

**ID**: `2508.05619v1`

### 摘要 (Detailed Summary)
这篇论文探讨了主动推理（Active Inference，AIF）在构建自主AI智能体中的关键作用，特别是在无需持续人工奖励工程的情况下，如何使智能体从经验中学习。随着高质量训练数据逐渐耗尽，AI系统越来越依赖大量的人工奖励设计，这给当前模式的可扩展性带来了挑战，阻碍了真正自主智能的发展。论文指出，尽管“经验时代”的提议（即智能体从自我生成的数据中学习）是一个有希望的方向，但它仍然依赖于大量人工设计的奖励函数，从而将瓶颈从数据策展转移到奖励策展。论文提出了“具身智能差距”（grounded-agency gap）的概念，即现有AI系统无法自主地制定、调整和追求目标以应对不断变化的环境。论文论证了AIF可以通过用内在的自由能最小化驱动来取代外部奖励信号，从而弥合这一差距，允许智能体通过统一的贝叶斯目标自然地平衡探索和利用。通过整合大型语言模型（LLMs）作为生成世界模型，并结合AIF的原则性决策框架，可以创建能够从经验中高效学习且与人类价值观对齐的智能体。这种结合为构建能够自主发展且遵守计算和物理约束的AI系统提供了一条有吸引力的路径。

### 主要贡献 (Contributions)
- 指出了当前AI发展中的“具身智能差距”（grounded-agency gap），即AI系统无法自主地形成、评估和调整目标。
- 论证了主动推理（AIF）可以通过内在的自由能最小化来提供自主学习的动力，从而弥合“具身智能差距”，无需持续的人工奖励工程。
- 提出了将大型语言模型（LLMs）作为生成世界模型整合到AIF决策框架中的新方法，结合了深度学习的可扩展性和自由能原理的理论严谨性。
- 分析了AIF在热力学上的优势，认为自由能最小化不仅具有计算优势，而且可能是可持续AI发展的热力学必要条件。
- 阐述了将AIF应用于解决AI在数据耗尽和外部认知依赖方面的问题，提供了一种更具可持续性和伦理性的AI发展路径。

### 技术方法 (Methods)
- **主动推理（Active Inference, AIF）：** 将智能视为一个贝叶斯推断过程，通过最小化自由能来统一感知和行动。
- **自由能最小化（Free Energy Minimization）：** 使用自由能作为内在的驱动力，替代外部奖励信号，使智能体能够自主地平衡探索和利用。
- **大型语言模型（Large Language Models, LLMs）作为生成世界模型：** 利用LLMs学习到的世界知识，构建智能体的生成模型，用于预测和推理。
- **LLM-AIF架构：** 将LLMs与AIF控制循环整合，实现高效的经验学习、透明的推理和基于常识的判断。
- **贝叶斯推断：** 使用贝叶斯推断更新智能体的信念，根据观察到的数据调整对世界状态的理解。

### 主要结论 (Conclusions)
论文的核心结论是：

1.  当前AI发展面临数据资源饱和和外部认知依赖的挑战，需要一种新的方法来实现真正的自主智能。
2.  主动推理（AIF）可以通过内在的自由能最小化来提供自主学习的动力，从而弥合“具身智能差距”。
3.  将大型语言模型（LLMs）作为生成世界模型整合到AIF决策框架中，是一种有希望的解决方案，可以结合深度学习的可扩展性和自由能原理的理论严谨性。
4.  AIF在热力学上具有优势，可能是可持续AI发展的必要条件。

---

## [TRAJEVO: Trajectory Prediction Heuristics Design via LLM-driven Evolution](https://arxiv.org/pdf/2508.05616v1)

**ID**: `2508.05616v1`

### 摘要 (Detailed Summary)
该论文提出了一种名为TRAJEVO的新框架，利用大型语言模型（LLM）自动设计轨迹预测启发式算法。轨迹预测在社交机器人和自动驾驶等安全关键领域至关重要。传统启发式方法精度和泛化性不足，深度学习方法计算成本高、可解释性差且泛化能力弱。TRAJEVO采用进化算法从历史轨迹数据中生成和优化预测启发式算法。论文提出了两个关键创新：跨代精英抽样（Cross-Generation Elite Sampling）以鼓励种群多样性，以及统计反馈循环（Statistics Feedback Loop）使LLM能够分析和改进预测结果。实验结果表明，TRAJEVO在多个真实世界数据集上优于现有的启发式方法，并在未见过的分布外（OOD）数据集上显著超越启发式和深度学习方法，具有快速、可解释和可泛化的优势。论文开源了代码，以促进未来研究。

### 主要贡献 (Contributions)
- 提出了TRAJEVO框架，该框架集成了LLM和进化算法，用于自动发现和设计快速、可解释且鲁棒的轨迹预测启发式算法，适用于真实世界的应用。
- 引入了跨代精英抽样策略（Cross-Generation Elite Sampling）来维持种群多样性，以及统计反馈循环（Statistics Feedback Loop），使LLM能够分析启发式性能并基于历史轨迹数据指导生成改进的候选方案。
- 实验证明，TRAJEVO生成的启发式算法在公开的真实世界数据集上显著优于现有的启发式方法，并在未见过的OOD数据集上实现了超过20%的性能提升，同时保持了计算速度和可解释性。
- TRAJEVO是第一个将LLM驱动的进化算法应用于轨迹预测领域的框架。
- TRAJEVO通过自动化启发式设计过程，解决了手动设计启发式算法的局限性，能够发现适用于真实部署的新型高性能启发式算法。

### 技术方法 (Methods)
- **进化框架：** 利用LLM作为核心遗传算子，通过迭代生成、评估和改进预测启发式算法。
- **初始种群：** 使用LLM生成包含N个多样化启发式算法的初始种群，种子启发式算法可以是恒定速度模型(CVM)。
- **交叉选择：** 从当前种群中成功执行的启发式算法中选择父代进行交叉，兼顾探索和利用。70%的父代从成功运行的候选者中随机选择，30%从精英执行者（目标函数J最低的那些）中选择。
- **反馈：** 使用LLM进行短期反馈（比较交叉父代的性能）和长期反馈（跨代积累见解），以指导后代的生成。
- **交叉：**  通过组合两个父代启发式算法的代码来创建新的后代。LLM在短期反馈的指导下，混合父代的有效“基因”。
- **精英变异：** 变异算子变异最优（目前发现的最佳）启发式算法。在TRAJEVO中，这涉及到生成器LLM修改选择的精英启发式算法。这个变异步骤由通过长期反馈收集的见解提供信息。
- **跨代精英抽样（CGES）：** 维护一个跨过去所有代的高性能启发式算法的历史档案，以增强探索能力，避免陷入局部最优。
- **统计反馈循环（SFL）：** 分析启发式算法生成的20个不同轨迹预测集的贡献，利用统计分布反馈到反射器和变异LLM，以确定哪些策略对性能贡献最大。

### 主要结论 (Conclusions)
该论文介绍了TRAJEVO，一个利用大型语言模型和进化算法自动设计轨迹预测启发式算法的框架。实验表明，TRAJEVO生成的启发式算法不仅优于标准基准测试中的传统方法，而且表现出卓越的分布外泛化性能，显著超过了未见数据的深度学习模型，同时保持了快速性和可解释性。TRAJEVO代表了自动发现高效、可解释和可泛化的轨迹预测启发式算法的重要一步，为传统的黑盒模型提供了一种实用且强大的替代方案。

### GitHub链接 (GitHub Links)
- https://github.com/ai4co/trajevo

---

## [Test-Time Reinforcement Learning for GUI Grounding via Region Consistency](https://arxiv.org/pdf/2508.05615v1)

**ID**: `2508.05615v1`

### 摘要 (Detailed Summary)
This paper tackles the challenge of GUI grounding, which is the task of mapping natural language instructions to precise screen coordinates, crucial for autonomous GUI agents. Current methods rely heavily on supervised training or reinforcement learning with labeled rewards, facing limitations in terms of data cost and annotation availability. The authors observe that spatial overlap patterns in multiple predictions from the same GUI element can reveal implicit confidence signals. To leverage this, they propose GUI-RC (Region Consistency), a test-time scaling method that constructs spatial voting grids from multiple sampled predictions to identify consensus regions. Furthermore, they introduce GUI-RCPO (Region Consistency Policy Optimization), which transforms these consistency patterns into rewards for test-time reinforcement learning, enabling models to refine outputs on unlabeled data during inference. Experiments demonstrate that GUI-RC and GUI-RCPO improve grounding accuracy across different architectures and benchmarks, offering a promising path toward more robust and data-efficient GUI agents.

### 主要贡献 (Contributions)
- Propose GUI-RC, a test-time scaling method for GUI grounding that uses spatial voting across multiple predictions to enhance localization accuracy without additional training data.
- Introduce GUI-RCPO, a test-time reinforcement learning method that leverages region consistency as a self-supervised reward signal, enabling models to improve grounding capabilities through policy optimization on unlabeled GUI screenshots.
- Demonstrate consistent improvements across multiple benchmarks and model architectures. GUI-RC improves accuracy by 2-3% on average, and GUI-RCPO achieves further gains of 4-5% on average through label-free optimization.
- Reveal that applying GUI-RC after GUI-RCPO yields additional performance gains, demonstrating a progressive, self-bootstrapping improvement process without external supervision, providing a complementary alternative to train-time optimization for GUI automation.

### 技术方法 (Methods)
- **GUI-RC (GUI Region Consistency):**
-   - **Multi-Sample Generation:** Sample K predictions from the model using temperature-based sampling.
-   - **Spatial Voting Mechanism:** Construct a spatial voting grid where each sampled prediction contributes votes to the grid based on the predicted region or expanded point.
-   - **Consensus Extraction:** Identify the maximum vote count and extract the largest contiguous region with the maximum vote count as the final consensus region.
- **GUI-RCPO (GUI Region Consistency Policy Optimization):**
-   - **Region Consistency as Reward:** Compute a reward for each sampled prediction based on the average vote density within the predicted region.
-   - **Policy Optimization:** Formulate GUI grounding as a reinforcement learning problem and optimize the expected region consistency reward using Group Relative Policy Optimization (GRPO).

### 主要结论 (Conclusions)
The paper concludes that GUI-RC and GUI-RCPO are effective methods for improving GUI grounding performance. GUI-RC leverages region consistency across multiple predictions to enhance model performance without additional training, while GUI-RCPO transforms region consistency into a self-supervised reward signal for test-time reinforcement learning. The methods consistently improve grounding performance and generalize well to out-of-distribution scenarios, highlighting the potential of test-time training for GUI agents.

### GitHub链接 (GitHub Links)
- https://github.com/zju-real/gui-rcpo

---

## [OMNI EAR: B ENCHMARKING A GENT R EASONING IN E MBODIED T ASKS](https://arxiv.org/pdf/2508.05614v1)

**ID**: `2508.05614v1`

### 摘要 (Detailed Summary)
本文提出了一个名为OmniEAR的综合框架，用于评估大型语言模型在具身任务中进行推理的能力，特别是关于物理交互、工具使用和多智能体协作。与现有基准测试不同，OmniEAR要求智能体动态地获取能力，并根据任务需求自主决定协作策略。该框架通过基于文本的环境表示，对连续的物理属性和复杂的空间关系进行建模，涵盖家庭和工业领域的1500个场景。评估结果表明，当模型必须从约束条件进行推理时，性能会严重下降。虽然在明确指令下能达到85-96%的成功率，但工具推理的性能下降到56-85%，隐式协作的性能下降到63-85%，复合任务的失败率超过50%。令人惊讶的是，完整的环境信息反而降低了协作性能，表明模型无法过滤任务相关的约束。微调可以显著提高单智能体任务的性能(0.6%到76.3%)，但对多智能体任务的提升很小(1.5%到5.5%)，暴露了基本的架构限制。该研究表明，具身推理提出了与当前模型能够解决的根本不同的挑战，并将OmniEAR确立为一个严格的基准，用于评估和推进具身人工智能系统。

### 主要贡献 (Contributions)
- 提出了OmniEAR框架，该框架通过要求智能体理解物理属性如何决定行动、能力和协作需求的情景来评估具身推理，解决了当前评估方法中的根本差距。
- 开发了EAR-Bench，这是一个包含1500个场景的基准，具有连续的物理属性和动态能力，由EAR-Sim和一个自动生成管道支持。
- 提供了经验证据，表明当前的语言模型缺乏核心的具身推理能力，从明确的指令到具身推理，性能下降超过60%，揭示了推进具身AI的关键要求。

### 技术方法 (Methods)
- **环境表示：** 将具身环境形式化为有向图 Gt = (Vt, Et, At)，其中Vt包含空间节点（房间和区域）、物体节点和智能体节点。每个节点维护一个属性字典At，存储连续的物理属性，如重量、温度、材料成分和几何尺寸。Et编码空间关系，通过静态的包含关系（如“在…中”、“在…上”）和动态的邻近关系Enear，跟踪哪些物体在智能体的交互范围内。
- **任务形式化：** 每个评估任务被定义为一个元组T = (Sinit, I, Ggoal, Atask)，其中Sinit指定初始环境状态，I提供自然语言指令，Ggoal通过逻辑谓词定义成功条件，Atask识别参与的智能体。评估目标是评估智能体是否可以生成一个动作序列Π = (π1, . . ., πT)，将环境从Sinit转换为满足Ggoal中所有谓词的终端状态Sfinal。
- **EAR-Sim：** 采用基于文本的环境建模来实现大规模的高效模拟。图结构Gt通过拓扑连接而不是连续坐标来维护空间关系，避免了昂贵的碰撞检测，同时保留了必要的空间约束。状态更新遵循增量方法，其中动作仅修改直接受影响的节点和边。工具对象维护一个capability属性，指定它启用的操作。当代理掌握一个工具时，系统动态地将相关能力绑定到代理的动作集。释放工具时，这些能力会自动解除绑定。
- **自动化基准生成：** 利用一个四阶段的流程，结合了大型语言模型的创造能力和基于规则的一致性检查，包括：从互联网语料库生成场景，通过技能抽样生成任务，提取评估逻辑，以及通过人工验证生成专家轨迹。

### 主要结论 (Conclusions)
当从明确的指令转向基于约束的推理时，当前的语言模型表现出严重的性能下降，工具使用和协调任务的性能从85%以上降至65%以下。结果表明，具身推理需要与当前语言模型不同的计算机制。OmniEAR为这些局限性提供了系统的诊断，并为开发下一代具身人工智能系统提供了一个严谨的平台。

### GitHub链接 (GitHub Links)
- https://github.com/ZJU-REAL/OmniEmbodied

---

## [COOPER: CO-OPTIMIZING POLICY AND REWARD MODELS IN REINFORCEMENT LEARNING FOR LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2508.05613v1)

**ID**: `2508.05613v1`

### 摘要 (Detailed Summary)
这篇论文提出了一个名为Cooper的强化学习（RL）框架，用于联合优化策略模型和奖励模型，以提高大型语言模型（LLMs）在推理任务中的能力。该框架旨在解决当前奖励机制（包括基于模型和基于规则的奖励）的局限性。Cooper利用基于规则的奖励在高精度识别正确响应方面的优势，并动态构建和选择正负样本对，以持续训练奖励模型，从而增强模型的鲁棒性并减轻奖励劫持的风险。为了支持Cooper，作者还引入了一种混合标注策略，用于高效准确地生成奖励模型的训练数据。此外，论文还提出了一个基于参考答案的奖励建模范式，训练了一个名为VerifyRM的奖励模型，该模型在VerifyBench上实现了比同等规模的其他模型更高的准确率。实验结果表明，Cooper不仅减轻了奖励劫持，还提高了端到端RL的性能。

### 主要贡献 (Contributions)
- 提出了一个奖励建模数据集，该数据集使用混合标注策略进行标注，该策略结合了基于规则的验证和LLM作为评判的验证，从而实现了高效且可靠的正确性监督。在该数据集上训练的奖励模型在VerifyBench上达到了89.42%的准确率，超过了现有同等规模的奖励模型。
- 基于规则的奖励在高精度识别正确答案方面的优势，提出了Cooper，这是一个强化学习框架，可同时共同优化策略模型和奖励模型。该框架缓解了奖励模型基于RL中常见的奖励劫持问题，并提高了整体训练性能。
- 研究表明，在RL训练过程中动态调整奖励模型的参数可以有效地减轻奖励劫持现象，从而为研究界如何更好地在强化学习中利用奖励模型提供了宝贵的见解。

### 技术方法 (Methods)
- **VerifyRM的训练：**包括数据收集和标注策略，以及奖励模型的训练过程。数据包括问题、参考答案、模型生成的完成以及指示完成是否正确的标签。使用混合标注方法（基于规则的验证器和LLM作为评判）进行正确性标注。
- **Cooper框架：** Cooper框架由两个阶段组成：策略模型优化和奖励模型优化。策略模型优化遵循GRPO范式，使用参考感知奖励模型对响应进行采样和评分，并基于组内归一化优势和KL正则化执行策略更新。奖励模型优化通过对比学习不断完善奖励模型，使用高精度基于规则的信号识别的正样本，以及通过使用辅助LLM将正确的响应转换为不正确的响应而生成的负样本。

### 主要结论 (Conclusions)
Cooper通过共同训练策略模型和奖励模型，结合了基于规则的奖励的高精度和基于模型的奖励的鲁棒性，有效减轻了奖励劫持问题，并实现了比单独使用任一类型奖励更好的性能。VerifyRM通过混合标注方法，在VerifyBench基准测试中优于现有同等规模的模型。动态更新奖励模型对于对抗奖励劫持是有效的。

### GitHub链接 (GitHub Links)
- https://github.com/zju-real/cooper
- https://github.com/huggingface/Math-Verify

---

## [使用Kolmogorov-Arnold网络 (KANs) 优化物联网威胁检测](https://arxiv.org/pdf/2508.05591v1)

**ID**: `2508.05591v1`

### 摘要 (Detailed Summary)
该论文探讨了Kolmogorov-Arnold网络（KANs）在物联网（IoT）网络入侵检测中的应用潜力，旨在寻找传统机器学习模型的替代方案。面对物联网设备日益增长的安全威胁，传统入侵检测系统难以适应动态变化的攻击模式。KANs通过采用可学习的激活函数，能够动态适应复杂的数据模式，优于传统的MLP网络。研究表明，KANs在实现与Random Forest和XGBoost等先进模型相当甚至更优的准确率的同时，还提供了卓越的可解释性。通过特征选择和KANs的结合，该研究旨在优化物联网环境中检测性能和计算效率。该研究结果表明KANs有潜力提升物联网安全，并为未来的研究方向提供启示。

### 主要贡献 (Contributions)
- 将KANs应用于CIC IoT 2023数据集，展示了边缘的可学习激活函数如何提高模型的准确性和可解释性。
- 将KANs与传统模型（例如，Random Forest、XGBoost）进行评估，以证明它们在物联网入侵检测中的竞争性能和适用性。
- 通过符号公式生成展示了KANs的可解释性，从而在安全关键型物联网系统中实现透明的决策制定。
- 通过实验证明，KANs在捕获复杂非线性关系方面优于传统机器学习模型。
- 强调了优化特征选择的重要性，从而提高了模型性能，减少了训练时间和计算开销，从而有助于在资源受限的物联网系统中实现实时应用。

### 技术方法 (Methods)
- **数据预处理：** 使用 Pandas、PyTorch 和 Scikit-Learn 等库加载 CIC IoT 2023 数据集，并将数据集分为训练集（67%）和测试集（33%）。数据集包含1048575个样本，每个样本有47个特征。
- **特征选择：** 使用随机森林模型评估特征的重要性，并选择前10个最重要的特征进行分析。
- **模型构建：** 构建 KAN 模型，包括一个输入层（神经元数量等于所选特征的数量）、两个隐藏层（分别有 16 和 8 个神经元）以及一个输出层（有 2 个神经元，代表“良性流量”和“恶意流量”分类）。
- **模型训练：** 使用 Adam 优化器和 CrossEntropyLoss 训练 KAN 模型，训练过程迭代 114680 次。学习率设置为 0.001，批量大小设置为 128，训练周期设置为 20。
- **模型评估：** 评估模型的性能，包括准确率、损失降低和计算效率（训练和预测时间）。

### 主要结论 (Conclusions)
研究表明，Kolmogorov-Arnold网络（KANs）在捕获物联网环境中的复杂非线性关系方面非常有效，明显优于传统的机器学习模型。该研究强调了优化特征选择的关键重要性，这不仅通过减少变量数量来提高模型性能，还最大限度地减少了训练时间和计算开销，从而有助于在资源受限的物联网系统中实现实时应用。KANs与可学习激活函数的集成代表了物联网安全框架领域中的一项重大进步。这种集成提供了一种强大的解决方案，该方案提高了网络流量分类的准确性和可解释性，这对于保护敏感数据免受不断演变的网络威胁至关重要。

---

## [Enhancing PyKEEN with Multiple Negative Sampling Solutions for Knowledge Graph Embedding Models](https://arxiv.org/pdf/2508.05587v1)

**ID**: `2508.05587v1`

### 摘要 (Detailed Summary)
这篇论文提出了一种针对知识图谱嵌入（KGE）模型中负采样方法的增强方案。知识图谱嵌入模型依赖于正样本和负样本进行训练，而负样本通常需要通过各种负采样策略人工生成。本文旨在填补现有KGE框架（如PyKEEN）在高级负采样策略上的不足。该论文的核心是为PyKEEN框架开发一个扩展，该扩展集成了一套先进的负采样器，包括静态和动态的负例生成方法，并在一致的模块化架构中运作，能够生成更有意义的负样本，同时保持与现有PyKEEN工作流的兼容性。通过这种扩展，PyKEEN的功能得到增强，并且更容易开发和定制嵌入方法。论文还通过全面的实验研究，验证了扩展的有效性，并分析了不同负采样策略对链接预测任务性能的影响，为设计更有效的负采样策略提供了有用的见解。

### 主要贡献 (Contributions)
- 为PyKEEN框架开发了一个模块化的扩展，集成了多种高级负采样技术（包括静态和动态策略）。
- 该扩展采用一致的模块化架构，易于集成和定制，同时保持与现有PyKEEN工作流程的兼容性。
- 实现了五种新的静态负采样器（Corrupt, Typed, Relational），它们利用结构和语义标准来选择实体。
- 实现了两种新的动态负采样器（Nearest Neighbour, Adversarial），利用预训练的辅助模型来指导负样本的选择。
- 对所开发的扩展进行了全面的实证研究，分析了不同负采样策略对不同嵌入方法在链接预测任务上的性能影响。
- 提供了详细的文档和示例，以促进代码重用和社区采用。

### 技术方法 (Methods)
- **静态负采样（Static Corruption）:**
-   - **随机抽样（Random Sampling）:** 从整个实体集中随机选择实体。
-   - **伯努利抽样（Bernoulli Sampling）:** 根据关系连接实体的概率分布，不对称地破坏头实体或尾实体。
-   - **损坏抽样（Corrupt Sampling）:** 基于出现在每个关系中的头实体或尾实体来定义负例池。
-   - **类型抽样（Typed Sampling）:** 利用知识图谱中的类型信息，根据关系的域和范围类来定义负例池。
-   - **关系抽样（Relational Sampling）:** 假设每个头尾对只参与一个关系，基于不同的关系来定义负例池。
- **动态负采样（Dynamic Corruption）:**
-   - **最近邻（Nearest Neighbor）:** 使用预训练的辅助模型生成实体向量表示，并选择与头实体或尾实体向量最接近的k个实体作为负例。
-   - **对抗抽样（Adversarial Sampling）:** 与最近邻方法类似，但使用模型预测的向量空间，而不是实体嵌入。

### 主要结论 (Conclusions)
该论文介绍并验证了PyKEEN框架的一个模块化扩展，该扩展旨在为采用KGE方法时提供广泛的标准负采样器实现覆盖，从而填补了更高级的负采样器实现可用性的一个重要空白。具体来说，我们提供了一个完全兼容的五种负采样器实现，包括静态和动态损坏策略。通过一系列实验，我们展示了该扩展的实际效用，展示了该扩展如何无缝集成到KGE的训练、评估和超参数优化管道中。此外，我们还提供了四种常用数据集的负例池统计数据，突出了不同条件下各种采样策略的操作约束和行为。通过遵守PyKEEN架构和设计标准，所提出的扩展支持可重复性、模块化和易于实验。通过减少开发和评估新采样策略的障碍，我们的目标是培养一个更加统一和高效的研究生态系统。

### GitHub链接 (GitHub Links)
- https://github.com/ivandiliso/refactor-negative-sampler/
- https://ivandiliso.github.io/refactor-negative-sampler

---

## [使用大型语言模型迭代学习治疗耐药性高血压的可计算表型](https://arxiv.org/pdf/2508.05581v1)

**ID**: `2508.05581v1`

### 摘要 (Detailed Summary)
这篇论文探索了大型语言模型（LLM）在生成可解释的可计算表型（CPs）方面的潜力，特别是在治疗耐药性高血压（TRH）的背景下。研究提出了一种*合成、执行、调试、指导*（SEDI）的迭代策略，利用LLM生成和改进CPs，并通过数据驱动的反馈进行优化。研究评估了LLM在无需训练（zero-shot）下的表现，以及通过SEDI策略迭代改进CPs的效果。结果表明，LLM结合迭代学习能够生成具有合理准确性的可解释程序，其性能接近最先进的机器学习方法，同时所需的训练样本显著减少。这项研究对于自动化CPs的生成和在不同临床实践中应用具有重要意义，并有望提高高血压患者的治疗效果。

### 主要贡献 (Contributions)
- 提出了一种*合成、执行、调试、指导*（SEDI）的迭代策略，用于利用LLM生成和改进可计算表型（CPs）。
- 验证了LLM结合SEDI策略能够生成具有合理准确性的可解释程序，用于识别高血压及其亚型患者。
- 证明了LLM生成的CPs在性能上接近最先进的机器学习方法，同时所需的训练样本显著减少。
- 展示了LLM在自动化CPs生成方面的潜力，可以减少手动特征工程的需求，并提高CPs的生成效率和适应性。
- 提供了一个公开可用的SEDI框架，可以用于开发其他疾病的可计算表型。

### 技术方法 (Methods)
- **零样本提示（Zero-shot prompts）：** LLM在没有反馈的情况下生成Python函数，基于可用特征预测表型概率。
- **SEDI提示：** 遵循*合成-执行-调试-指导*（SEDI）循环，迭代接收关于CP在训练数据集上的性能反馈。如果CP执行失败，LLM会收到包含错误回溯的消息（调试）。如果CP执行成功，LLM会收到性能指标以及假阳性（FP）和假阴性（FN）的例子，并被指示改进其表型定义以提高程序的性能（指导）。

### 主要结论 (Conclusions)
这项研究表明，即使提供简单的提示，最先进的LLM也能为高血压表型生成相当准确和简洁的CPs。当提供详细和集中的提示，并配备数据驱动的迭代反馈（即SEDI）时，LLM生成的CPs可以与使用监督ML训练的CPs相媲美。传统的利用图表审查示例的监督ML方法仍然优于LLM衍生的CPs，但产生的模型更大，并且需要访问更大的专家标记数据集。该研究还验证了LLM迭代学习以可理解的Python代码表达的CPs的潜力。

### GitHub链接 (GitHub Links)
- https://github.com/cavalab/htn-phenotyping-with-llms

---

## [2508.05567v1](https://arxiv.org/pdf/2508.05567v1)

**ID**: `2508.05567v1`

---

## [MV-Debate: Multi-view Agent Debate with Dynamic Reflection Gating for Multimodal Harmful Content Detection in Social Media](https://arxiv.org/pdf/2508.05557v1)

**ID**: `2508.05557v1`

### 摘要 (Detailed Summary)
本文提出了一种名为MV-Debate的多视角Agent辩论框架，用于统一的多模态有害内容检测。该框架旨在解决社交媒体中存在的跨模态矛盾、快速变化的文化和社会语境以及细微的语用线索给有害信息识别带来的挑战。MV-Debate由四个互补的辩论Agent组成：表面分析师、深度推理者、模态对比者和社会语境主义者，从不同的解释角度分析内容。通过迭代辩论和反思，Agent们在∆-gain准则下不断改进回应，确保准确性和效率。在三个基准数据集上的实验结果表明，MV-Debate显著优于强大的单模型和现有的多Agent辩论基线。这项工作突出了多Agent辩论在推进安全关键在线环境中可靠的社会意图检测方面的潜力。

### 主要贡献 (Contributions)
- 提出了MV-Debate，一个多Agent辩论框架，引导Agent采用不同的推理视角来检测社交媒体中的多模态有害内容。
- 设计了四个具有特定视角的辩论Agent，并采用动态反射门控机制以提高性能。
- 在多个多模态有害内容基准数据集上，通过实验验证了所提出方法的有效性。
- MV-Debate 通过分配专门的角色给辩论Agent来增强推理视角的多样性，不同于单一视角的prompt，该设计整合了表面层面、深层语义、跨模态和社会文化分析，从而降低了遗漏隐式或特定上下文的有害线索的风险。
- Top-k ∆ -reflection gating机制增强了可靠性，同时保持了效率。 通过仅在预期有实质性改进时才自适应地触发reflection，该框架避免了冗余计算，但实现了与无条件reflection相当或更好的准确性。

### 技术方法 (Methods)
- **多视角辩论：** 设计了四个具有特定视角的辩论Agent，包括表面分析师（SA）、深度推理者（DR）、模态对比者（MC）和社会语境主义者（SC），分别从不同的角度分析内容。
- **动态反射门控：** 引入Top-k ∆-reflection gating策略，仅当反射增益超过预定义的阈值时才触发反射，以减少计算开销并提高效率。
- **迭代辩论循环：** 从第二轮开始，将上一轮得分最高的响应纳入每个Agent的历史记录中。在接下来的回合中，每个Agent利用这些推理轨迹和解决方案作为额外的输入，选择性地从不同的角度提取有用的信息，以完善自己的答案。
- **∆-gain 准则：** 通过比较有无reflection feedback的Agent得分来评估reflection的预期效用。只有当∆-gain超过预定义阈值τ时，才会采用新的响应。
- **角色特定Prompt：** 为每个Agent设计角色特定的Prompt，严格执行不同的分析视角，确保整体推理过程的多样性和互补性。

### 主要结论 (Conclusions)
本文提出的MV-Debate框架通过整合不同的推理视角和自适应反射，实现了在多模态有害内容检测方面的卓越准确性和效率。该框架为扩展多Agent辩论方法到更广泛的安全关键多模态推理任务奠定了基础。

---

## [Adapting Vision-Language Models Without Labels: A Comprehensive Survey](https://arxiv.org/pdf/2508.05547v1)

**ID**: `2508.05547v1`

### 摘要 (Detailed Summary)
这篇论文全面综述了视觉-语言模型（VLMs）在无标签数据下的自适应方法。尽管VLMs在各种任务中表现出卓越的泛化能力，但直接应用于特定下游场景时，通常需要进行自适应调整才能达到最佳性能。为了提高VLMs的实用性并保持数据效率，论文重点关注不需要标签数据的无监督自适应方法。论文提出了一个基于无标签视觉数据可用性和性质的分类法，将现有方法分为四种主要范式：无数据迁移、无监督领域迁移、情景测试时自适应和在线测试时自适应。该综述分析了每种范式的核心方法和自适应策略，旨在系统地理解该领域，并回顾了不同应用中的代表性基准，突出了未来的开放挑战和有希望的研究方向。

### 主要贡献 (Contributions)
- 提出了一个基于无标签视觉数据可用性的新的分类法，用于组织和理解现有的无监督 VLM 自适应方法。这种分类法包括四种范式：Data-Free Transfer (无数据), Unsupervised Domain Transfer (大量数据), Episodic Test-Time Adaptation (批量数据), 和 Online Test-Time Adaptation (流数据)。
- 对每种范式下的核心方法和自适应策略进行了详细的分析和总结，建立了一个系统的理解框架。
- 回顾了不同应用领域的代表性基准，提供了更广阔的视角来看待这些方法在实际应用中的意义和价值。
- 识别了该领域存在的开放性挑战，并提出了未来有前景的研究方向，为后续研究提供了指导。
- 提供了一个积极维护的相关文献仓库，方便研究人员查找相关资源。

### 技术方法 (Methods)
- **Data-Free Transfer:** 依赖于文本增强（如使用LLMs生成更丰富的描述）、图像利用（检索或生成相关图像）以及网络修改等策略。
- **Unsupervised Domain Transfer:** 采用自训练（利用伪标签）、熵优化（最小化样本熵和最大化类别熵）以及外部资源利用（如MLLMs和知识蒸馏）等技术。
- **Episodic Test-Time Adaptation:** 主要策略包括熵最小化（调整模型参数以降低预测的不确定性）、反馈信号（利用扩散模型或CLIP模型的反馈）、分布对齐（将测试样本分布与源数据对齐）和自监督学习（使用对比学习等方法）。
- **Online Test-Time Adaptation:** 使用伪标签、记忆机制（如动态键值缓存）和分布建模（如高斯分布估计）等方法，以适应不断变化的输入数据流。

### 主要结论 (Conclusions)
该综述全面地概述了无监督视觉-语言模型自适应这一快速发展的领域。通过引入一个新颖的分类方法，将现有方法基于无标签视觉数据的可用性进行分类，为理解每个场景中固有的独特挑战和假设提供了一个系统的框架。在这一结构中，分析了核心方法并回顾了具有代表性的基准，对当前技术水平提出了一个整体的视角。最后，确定了未来研究的几个关键挑战和方向，包括开发理论分析、处理开放世界场景和隐私考虑，以及进一步探索新的下游任务和应用领域。

### GitHub链接 (GitHub Links)
- https://github.com/tim-learn/Awesome-LabelFree-VLMs

---

## [2508.05544v1](https://arxiv.org/pdf/2508.05544v1)

**ID**: `2508.05544v1`

---

