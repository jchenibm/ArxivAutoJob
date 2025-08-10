# Arxiv Paper Summary - 2025-08-10

## [Towards Generalizable Safety in Crowd Navigation via Conformal Uncertainty Handling](https://arxiv.org/pdf/2508.05634v1)

**ID**: `2508.05634v1`

### 摘要 (Detailed Summary)
This paper tackles the challenge of performance degradation in reinforcement learning-based crowd navigation robots when faced with out-of-distribution scenarios. The core idea is to improve the robot's ability to generalize by explicitly accounting for uncertainties in pedestrian behavior predictions. The proposed method leverages adaptive conformal inference (ACI) to quantify prediction uncertainty, generating prediction sets that contain the true future positions with a specified probability. These uncertainty estimates are then integrated into a constrained reinforcement learning (CRL) framework to guide the agent's behavior and ensure robustness to distribution shifts. Experiments demonstrate that the system achieves state-of-the-art performance in in-distribution settings and exhibits significantly improved robustness in out-of-distribution scenarios involving velocity variations, policy changes, and group dynamics. The method is also validated on a real robot, showcasing its effectiveness in safe and robust navigation among both sparse and dense crowds.

### 主要贡献 (Contributions)
- Proposes a novel learning-based framework that explicitly reasons about prediction uncertainty in RL-based crowd navigation to improve generalizability.
- Applies adaptive conformal inference (ACI) to quantify the uncertainty of predicted human trajectories, providing a prediction set that contains the true future position with a user-defined coverage probability.
- Employs constrained reinforcement learning (CRL) to introduce effective controllability into the decision-making system, using uncertainty estimates to guide both the learning process and the agent's behavior.
- Achieves state-of-the-art performance in safety metrics and demonstrates much smaller performance drops in OOD settings compared to existing methods.
- Demonstrates successful deployment of the learned policy on a real Mecanum-wheel robot, showcasing safe and robust navigation in both sparse and dense crowds.

### 技术方法 (Methods)
- **Adaptive Conformal Inference (ACI):** Used to quantify the uncertainty of predicted human trajectories. ACI dynamically adjusts its parameters to maintain coverage in an online and distribution-free manner, making it suitable for time-sequential applications.
- **Constrained Reinforcement Learning (CRL):** Employed to guide the agent's behavior using uncertainty estimates. CRL incorporates constraints on the agent's actions to ensure safety, maximizing rewards while satisfying cost constraints.
- **Policy Network with Attention Mechanism:** A policy network with a combined human-human (H-H) and human-robot (H-R) attention mechanism is used to process uncertainty information along with other features. This allows the RL agents to account for prediction uncertainty in their decision-making process.
- **PPO Lagrangian:** The PPO Lagrangian method is used for optimization. Two critics are used to compute the state value for reward and the state value for cost.

### 主要结论 (Conclusions)
The paper presents an RL-based trajectory-planning framework that integrates conformal uncertainty into a CRL scheme to mitigate OOD performance degradation. By dynamically leveraging uncertainty estimates, the method adapts to velocity variations, policy changes, and transitions from individual to group dynamics. The method achieves robust stability across diverse OOD scenarios and practical effectiveness in real-world trials.

### GitHub链接 (GitHub Links)
- https://gen-safe-nav.github.io/

---

## [KuaiLive: A Real-time Interactive Dataset for Live Streaming Recommendation](https://arxiv.org/pdf/2508.05633v1)

**ID**: `2508.05633v1`

### 摘要 (Detailed Summary)
本文介绍了一个名为KuaiLive的大规模、实时的直播推荐数据集，该数据集来源于中国领先的直播平台快手，拥有超过4亿日活跃用户。该数据集记录了23772个用户和452621个主播在21天内的互动日志。KuaiLive相对于现有数据集的优势在于，它包含了精确的直播间开始和结束时间戳，多种类型的实时用户互动（点击、评论、点赞、送礼），以及用户和主播的丰富附加信息。这些特性使得能够更真实地模拟动态候选项目，并更好地建模用户和主播的行为。通过对KuaiLive的多角度分析，并评估了在该数据集上的几种代表性推荐方法，为未来的研究建立了一个强大的基准。KuaiLive支持直播领域的广泛任务，如Top-K推荐、点击率预测、观看时长预测和礼物价格预测。此外，其细粒度的行为数据也支持对多行为建模、多任务学习和公平感知推荐的研究。该数据集和相关资源已公开。

### 主要贡献 (Contributions)
- 提出了 KuaiLive，一个大规模、实时的直播推荐数据集，来源于中国领先的直播平台快手。
- KuaiLive 包含精确的直播间开始和结束时间戳，允许研究人员模拟真实的直播推荐设置，其中候选项目受到时间约束并动态变化。
- KuaiLive 记录了多种用户行为（例如，点击、评论、点赞、送礼），可用于研究多任务学习和多行为建模。
- KuaiLive 保留了每个交互的时间顺序，支持对用户行为轨迹的细粒度分析。
- KuaiLive 不仅包括用户和项目 ID，还包括丰富的附加信息特征，例如人口统计数据和属性，这有助于进行特征感知建模。
- 对数据集进行了全面的分析，揭示了直播场景的关键特征和模式
- 在 KuaiLive 上评估了代表性的推荐方法，为未来的研究建立了基准。

### 技术方法 (Methods)
- **用户抽样:** 随机抽取约25000名活跃用户，这些用户在2025年5月5日至2025年5月25日期间在快手直播领域参与了所有四种类型的互动（点击、评论、点赞和送礼）。
- **互动收集:** 收集了21天内（2025年5月5日至2025年5月25日）来自直播服务日志的多种类型的细粒度用户互动，包括点击、评论、点赞和送礼四种行为类型。
- **附加信息收集:** 收集了所有三个核心实体（用户、主播和直播间）的广泛附加信息。
- **匿名化:** 应用了严格的匿名化程序，以确保符合数据发布政策并保护用户隐私。

### 主要结论 (Conclusions)
KuaiLive是一个有价值的资源，可以促进直播推荐的研究。对数据集的分析揭示了直播场景的独特特征，为设计更有效的推荐模型提供了见解。在KuaiLive上评估的代表性推荐和CTR预测方法建立了一个可靠且可重复的基准，为未来的研究奠定了基础。

### GitHub链接 (GitHub Links)
- https://imgkkk574.github.io/KuaiLive
- https://github.com/THUwangcy/ReChorus

---

## [- ON THE GENERALIZATION OF SFT: A REINFORCEMENT LEARNING PERSPECTIVE WITH REWARD RECTIFICATION](https://arxiv.org/pdf/2508.05629v1)

**ID**: `2508.05629v1`

### 摘要 (Detailed Summary)
该论文提出了一种改进监督微调（SFT）的方法，称为动态微调（DFT），旨在解决SFT在大语言模型（LLM）中泛化能力有限的问题。通过数学分析，作者发现标准SFT的梯度隐式地编码了一种有问题的奖励结构，可能会严重限制模型的泛化能力，尤其是在模型对专家行为分配较低概率时。DFT通过动态地重新调整每个token的目标函数，利用该token的概率来稳定梯度更新，从而纠正了这个问题。这种方法有效地消除了导致意外奖励结构和无界方差的逆概率加权。实验结果表明，DFT在多个具有挑战性的基准测试和基础模型中显著优于标准SFT，并且在离线强化学习设置中也表现出竞争优势，为SFT性能的提升提供了一种有效且简单的替代方案。

### 主要贡献 (Contributions)
- 从理论上分析了LLM SFT，将其视为策略梯度空间中的一种特殊的RL，并指出了SFT泛化能力有限的根本原因。
- 提出了动态微调（DFT），一种通过token概率动态重加权SFT损失的简单有效的方法，以解决由不合理的隐式奖励结构引起的泛化问题。
- 实验证明，DFT只需一行代码的修改，就能显著提高LLM SFT在各种任务和模型上的性能和泛化能力。
- 展示了DFT在数学推理基准测试上的一致性和显著改进，包括在标准SFT性能下降的具有挑战性的数据集上。
- 探索了DFT在离线RL环境中的适用性，并表明它优于其他离线RL方法，甚至与在线RL方法相比也具有竞争力。

### 技术方法 (Methods)
- **数学分析：** 将SFT梯度重写为策略梯度，并通过重要性采样进行分析，揭示SFT梯度可以被解释为带有特定隐式奖励结构的策略梯度方法。分析表明，这种隐式奖励是极其稀疏的，并且与策略分配给专家行为的概率成反比。
- **动态微调（DFT）：** 提出了一种动态重加权SFT损失的方法，通过token概率来校正SFT的隐式奖励结构。对于每个token，DFT通过token概率来重新调整标准SFT目标，有效地消除了导致意外奖励结构和无界方差的逆概率加权。
- **公式推导：** 通过公式推导，展示了DFT如何将不稳定的、有偏差的、依赖于概率的梯度估计器转换为稳定的、均匀加权的更新过程。
- **token级别应用：** 为了避免数值不稳定性，在token级别应用重要性采样，最终DFT损失函数是在token级别上重新加权的交叉熵损失。

### 主要结论 (Conclusions)
该论文提出了一种名为动态微调（DFT）的简单而有效的方法，通过动态地重新调整每个token的目标函数来改进监督微调（SFT），从而解决了SFT在大语言模型（LLM）中泛化能力有限的问题。DFT在多个具有挑战性的基准测试和基础模型中显著优于标准SFT，并且在离线强化学习设置中也表现出竞争优势，为SFT性能的提升提供了一种有效且简单的替代方案。

### GitHub链接 (GitHub Links)
- https://github.com/yongliang-wu/DFT

---

## [H-N ET ++: Hierarchical Dynamic Chunking for Tokenizer-Free Language Modelling in Morphologically-Rich Languages](https://arxiv.org/pdf/2508.05628v1)

**ID**: `2508.05628v1`

### 摘要 (Detailed Summary)
这篇论文提出了一种名为H-NET++的层次动态分块模型，用于在形态丰富的语言（MRLs）中进行无分词器的语言建模。H-NET++通过端到端训练学习具有语言学知识的分词方法。该模型的主要创新包括：一个用于跨块注意力的轻量级Transformer上下文混合器（1.9M参数），一个用于文档级别一致性的两级潜在超先验，对波斯语ZWNJ等正字法伪像的专门处理，以及使用分阶段序列长度的基于课程的训练。在1.4B tokens的波斯语语料库上，H-NET++实现了最先进的结果，包括相对于基于BPE的GPT-2-fa的0.159 BPB降低（压缩率提高12%），ParsGLUE提升5.4pp，对ZWNJ损坏的鲁棒性提高53%，以及在黄金形态边界上达到73.8%的F1分数。该模型学习到的块与波斯语形态对齐，无需显式监督，表明层次动态分块为MRLs提供了一种有效的无分词器解决方案，同时保持了计算效率。

### 主要贡献 (Contributions)
- **新颖的架构（H-NET++）**：一个带有潜在超先验的Transformer增强层次路由器，专为形态丰富的语言设计。
- **课程优化**：一个分阶段的AdamW训练方案，用于稳定长序列byte-level训练。
- **鲁棒性评估套件**：字符级别噪声鲁棒性基准和一个新的波斯语黄金分词数据集。
- **最先进的性能**：在BPB、下游准确性和鲁棒性方面取得了领先的结果。

### 技术方法 (Methods)
- **层次路由器**：使用多层双向GRU和边界预测器，动态地将字节组合成语言学上有意义的块。
- **Transformer上下文混合器**：使用一个轻量级的Transformer自注意力模块，允许块之间进行信息交互，捕获长距离依赖关系。
- **两级潜在超先验**：使用变分推理学习文档级别的全局隐变量，捕捉文档级别的形态一致性。
- **ZWNJ感知字节嵌入**：专门处理波斯语中的零宽度非连接符（ZWNJ），学习其特有模式，避免与可见字符混淆。
- **课程学习**：使用三阶段课程学习策略，逐步增加序列长度，稳定训练过程。

### 主要结论 (Conclusions)
H-NET++成功地消除了形态丰富的语言的分词瓶颈，同时保持了计算效率。实验结果表明，学习到的分词可以超越精心设计的传统tokenizer，在多个维度上，包括困惑度、下游任务性能、鲁棒性和形态有效性。此外，实验还证明Transformer混合器和文档级别的超先验对于捕获形态一致性至关重要。

---

## [How Do LLMs Persuade? Linear Probes Can Uncover Persuasion Dynamics in Multi-Turn Conversations](https://arxiv.org/pdf/2508.05625v1)

**ID**: `2508.05625v1`

### 摘要 (Detailed Summary)
这篇论文研究了大型语言模型（LLMs）如何说服人类，并提出使用线性探针来分析多轮对话中的说服动态。研究的背景是LLMs开始展现出说服人类的能力，但我们对这种动态的理解还很有限。论文受到认知科学的启发，训练了三种探针来捕捉说服的不同方面：说服成功、被说服者的性格和说服策略。实验结果表明，这些探针能够有效地识别对话中说服发生的时间点，以及在整个数据集中说服成功的普遍规律。此外，探针在揭示说服策略方面，与基于提示的方法相比，表现同样出色甚至更好。这表明探针是一种有前途的研究其他复杂行为（如欺骗和操纵）的途径，尤其是在多轮设置和大规模数据集分析中。

### 主要贡献 (Contributions)
- 提出了一个使用线性探针分析LLM驱动的对话中说服动态的框架。该框架设计了轻量级、高效的探针，能够捕捉说服的关键方面，从而实现细粒度的、轮次级别的分析。
- 证明了线性探针可以通过训练LLM的激活值，来准确地识别说服成功或失败发生的位置，检测说服者使用的修辞策略，并估计被说服者在对话中的人格。
- 揭示了说服线索在人类对话的中间轮次集中，但在LLM生成的对话中转移到最后的一两个轮次，揭示了自然数据和合成数据之间说服展开方式的系统性差异。
- 通过关联探针的输出，揭示了外向性等人格特质会调节不同修辞策略的有效性（例如，可信度或情感诉求），从而提供了LLMs如何调整说服策略的细致图景。

### 技术方法 (Methods)
- **线性探针 (Linear Probes):** 使用多类逻辑回归，通过最小化经验风险在冻结的LLM激活值上进行训练。
- **经验风险最小化 (Empirical Risk Minimization):** 通过训练集中的激活值和整数标签，最小化线性探针的损失函数。
- **交叉熵损失 (Cross-Entropy Loss):** 使用交叉熵损失函数来优化线性探针的权重和偏置。
- **梯度下降 (Gradient Descent):** 使用梯度下降法来更新线性探针的参数。

### 主要结论 (Conclusions)
论文的核心结论是，线性探针可以有效地用于理解LLM如何在多轮对话中进行说服。探针能够揭示说服对话中有意义的特征，例如被说服者未被说服的时间点，以及策略和人格对说服的交互影响。研究还发现，在合成数据集中，说服的成功仅限于最后的一两个回合，而在人类数据集中，说服的成功在对话的中点附近达到顶峰。这些发现表明，线性探针是一种很有前景的方法，可以用于理解LLM中的其他抽象行为，例如欺骗和操纵，尤其是在多轮设置和大规模数据集分析中。

### GitHub链接 (GitHub Links)
- https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE
- https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/USE_POLICY.md

---

## [Simulating Human-Like Learning Dynamics with LLM-Empowered Agents](https://arxiv.org/pdf/2508.05622v1)

**ID**: `2508.05622v1`

### 摘要 (Detailed Summary)
本文提出了一种名为LearnerAgent的新型多智能体框架，该框架基于大型语言模型（LLMs），旨在模拟真实的教学环境，从而捕捉人类的学习行为。研究的重点在于探索不同学习者（包括深度学习者、表面学习者和懒惰学习者）在长期学习过程中的学习动态。通过每周的知识获取、每月的策略选择、周期性测试和同伴互动，LearnerAgent能够追踪个体学习者在一整年学习过程中的动态学习进度。研究发现，只有深度学习者能够实现持续的认知增长，而表面学习者则表现出知识的脆弱性。此外，学习者的自我概念会随着时间的推移而发生变化，通用学习者（即基准LLM）虽然认知有限，但会发展出令人惊讶的高自我效能感。最关键的发现是，基准LLM的默认行为模式是“勤奋但脆弱的表面学习者”，它模仿好学生的行为，但缺乏真正的、可泛化的理解。实验结果表明，LearnerAgent能够很好地模拟真实场景，为理解LLMs的行为提供更深入的见解。

### 主要贡献 (Contributions)
- 提出了LearnerAgent框架，该框架能够模拟具有不同心理学特征的学习者在真实教学环境中的学习过程。
- 通过长期分析，揭示了不同学习者的学习动态，例如只有深度学习者能够实现持续的认知增长，而表面学习者则表现出知识的脆弱性。
- 发现学习者的自我概念会随着时间的推移而发生变化，通用学习者（即基准LLM）虽然认知有限，但会发展出令人惊讶的高自我效能感。
- 揭示了基准LLM的默认行为模式是“勤奋但脆弱的表面学习者”，它模仿好学生的行为，但缺乏真正的、可泛化的理解。
- 构建了一个新的测试套件，用于模拟为期一年的英语语法学习过程，包括知识点和测试问题，并设计了陷阱问题来评估学习者的深层理解。

### 技术方法 (Methods)
- **角色扮演模拟：** LearnerAgent通过角色扮演模拟构建了一个真实的学习环境，模拟了基于教育心理学和认知科学理论的为期一年的学习过程。
- **多智能体框架：** 该框架由教师智能体和多个学习者智能体组成，每个学习者智能体都有不同的配置文件。
- **配置文件构建：** 学习者根据学习动机、初始自我概念得分和发展策略等多个维度进行分类，包括深度学习者、表面学习者、懒惰学习者和通用学习者。
- **学习和改进机制：** 通过设计结构化的学习、改进和评估周期，包括每周学习和策略选择、每月回顾和评估以及同伴互动和辩论，来探索学习者的认知发展过程。
- **记忆机制：** 采用短期记忆和长期记忆机制，以实现有效的知识获取、评估和反思。
- **情境依赖检索策略：** 根据当前时间步或学习阶段动态地为智能体提供最相关的记忆片段。
- **综合评估策略：** 采用综合评估策略，涵盖学术表现和心理变化，包括陷阱问题来评估更深层次的理解。

### 主要结论 (Conclusions)
本文介绍了LearnerAgent，这是一个多智能体框架，旨在基于教育心理学模拟和分析人类学习的复杂动态。通过为期一年的模拟，我们成功地跟踪了不同学习者的多样化学习行为、发展轨迹和社会互动。我们的工作产生了三个关键见解。首先，角色驱动的智能体能够以高保真度复制细致入微的人类学习行为。其次，我们识别出在短期内表现相似但在长期泛化方面存在显著差异的智能体之间的明显学习缺陷——这一发现通过暴露更深层脆弱性的表现“陷阱”来说明。第三，没有角色特征的基础LLM的默认涌现行为，称为“勤奋但脆弱的表面学习者”，反映了一种掌握表面能力但缺乏强大、可泛化理解的智能体。我们进一步表明，膨胀的自我概念会导致过度自信，最终阻碍成长。

---

## [The Missing Reward: Active Inference in the Era of Experience](https://arxiv.org/pdf/2508.05619v1)

**ID**: `2508.05619v1`

### 摘要 (Detailed Summary)
本文探讨了主动推理（AIF）在开发能够从经验中学习而无需持续人工奖励工程的自主AI代理方面的关键作用。随着AI系统开始耗尽高质量的训练数据并依赖于日益庞大的人力来进行奖励设计，当前的范式面临着严重的可扩展性挑战，这可能会阻碍真正自主智能的发展。作者认为，代理人从自我生成的数据中学习的“经验时代”是一个有希望的进步。然而，这种愿景仍然依赖于大量的人工奖励函数工程，从而有效地将瓶颈从数据管理转移到奖励管理。文章强调了当前AI系统自主制定、适应和追求目标的“基础代理差距”。作者提出，AIF可以通过用最小化自由能量的内在驱动来取代外部奖励信号来弥合这一差距，从而使代理人可以通过统一的贝叶斯目标来自然地平衡探索和利用。通过将大型语言模型（LLM）作为生成世界模型与AIF的原则性决策框架相结合，可以创建能够从经验中有效学习同时保持与人类价值观一致的代理。这种综合提供了一条通往AI系统的引人注目的道路，这些系统可以在遵守计算和物理约束的同时自主开发。

### 主要贡献 (Contributions)
- 识别了当前AI系统中的“基础代理差距”，即无法自主形成、评估和调整目标。
- 论证了主动推理（AIF）可以通过内在的自由能量最小化来弥合这一差距，从而消除对持续奖励工程的需求。
- 提出了一个新颖的集成方案，其中大型语言模型（LLM）充当主动推理决策框架中学习的生成世界模型，结合了现代深度学习的可扩展性和自由能原理的理论严谨性。
- 论证了自由能最小化的能量效率不仅在计算上有利，而且可能是可持续AI进展的热力学必要条件。
- 详细阐述了一个自主实验室助手的例子，展示了LLM-AIF框架如何在没有外部奖励工程的情况下实现自主、安全和适应性行为。

### 技术方法 (Methods)
- **主动推理（AIF）**：AIF将智能视为一个统一的贝叶斯推理过程，感知和行动的目标是最小化变分自由能（VFE），而不是最大化外部奖励。VFE包括模型复杂度和预测精度，代理通过平衡探索（寻求信息以减少不确定性）和利用（采取行动使观察结果与期望状态匹配）来最小化预期自由能（EFE）。
- **大型语言模型（LLM）作为世界模型**：利用LLM作为AIF代理的生成世界模型，利用它们通过大量文本训练获得的常识知识和推理能力。LLM能够创建和管理AIF代理的生成模型的组件，实现更结构化的关于不确定性和因果关系的推理。
- **LLM-AIF架构**：该架构集成了LLM世界模型、AIF控制循环和在线细化。LLM的内部状态充当变分后验的充分统计量，AIF指导探索、学习和通过自由能最小化选择行动，代理通过经验不断更新其世界模型。

### 主要结论 (Conclusions)
论文得出结论，主动推理（AIF）为开发能够从经验中学习而无需持续人工奖励工程的自主AI代理提供了关键的基础。通过利用LLM作为生成世界模型，并将它们与AIF的原则性决策框架相结合，可以创建能够有效学习并与人类价值观保持一致的AI系统。论文还强调，AIF的能量效率可能不仅仅是一种计算优势，而是一种可持续AI进展的热力学必要条件。

---

## [T RAJ E VO : Trajectory Prediction Heuristics Design via LLM-driven Evolution](https://arxiv.org/pdf/2508.05616v1)

**ID**: `2508.05616v1`

### 摘要 (Detailed Summary)
该论文介绍了一种名为T RAJ E VO的新框架，它利用大型语言模型（LLMs）自动设计轨迹预测启发式方法。轨迹预测在社会机器人和自动驾驶等安全关键领域至关重要，但传统方法缺乏准确性和泛化性，而深度学习方法计算成本高且可解释性差。T RAJ E VO采用进化算法从过去的轨迹数据中生成和改进预测启发式方法。该框架引入了两个关键创新：交叉世代精英采样（Cross-Generation Elite Sampling）以鼓励种群多样性，以及统计反馈循环（Statistics Feedback Loop）使LLM能够分析和改进替代预测。实验结果表明，T RAJ E VO在多个真实世界数据集中优于现有的启发式方法，并且在推广到未见过的分布外（OOD）真实世界数据集时，明显优于启发式和深度学习方法。T RAJ E VO标志着在自动设计快速、可解释和可泛化的轨迹预测启发式方法方面迈出了有希望的一步，并且开源了代码以促进未来的研究。

### 主要贡献 (Contributions)
- 提出了T RAJ E VO框架，该框架将LLM与进化算法相结合，用于自动发现和设计快速、可解释和鲁棒的轨迹预测启发式方法，适用于现实世界的应用。
- 引入了交叉世代精英采样（Cross-Generation Elite Sampling）策略，以维持种群多样性。
- 引入了统计反馈循环（Statistics Feedback Loop），使LLM能够分析启发式方法的性能，并根据过去的轨迹数据指导改进候选方法的生成。
- 实验结果表明，T RAJ E VO生成的启发式方法在公共开放的真实世界数据集上显著优于现有的启发式方法，并且具有显著的泛化能力，在未见过的OOD数据集上，其性能比传统启发式方法和深度学习方法都提高了20%以上。
- 生成的启发式方法计算速度快且可解释性强。

### 技术方法 (Methods)
- **进化框架:** 利用LLM作为核心遗传算子，通过迭代生成、评估和优化启发式方法。
- **初始种群:** 使用包含问题细节、输入/输出格式和目标函数的基本启发式方法（如恒定速度模型）来初始化LLM。
- **交叉选择:** 从当前种群中成功执行的启发式方法中选择父代进行交叉，选择过程兼顾探索和利用。
- **反思（Reflections）:** 使用短期反思和长期反思来指导LLM生成过程。
- **交叉:** 通过组合两个父代启发式方法的代码来创建新的后代，LLM被提示混合它们的有效“基因”。
- **精英变异:** LLM修改精英（迄今为止发现的最好）启发式方法，这个变异步骤受到通过长期反思收集的见解的指导。
- **交叉世代精英采样（CGES）：** 维护一个跨越所有过去世代的高性能启发式方法的历史档案，从而改进探索能力，促进逃离局部最优解和发现更稳健的启发式方法。
- **统计反馈循环（SFL）：** 分析由启发式方法生成的不同轨迹预测集的贡献，为LLM提供关键见解，并识别哪些预测策略是有效的，并根据观察到的组成策略的有效性，对启发式方法的多重预测生成逻辑进行具体改进。

### 主要结论 (Conclusions)
T RAJ E VO 代表着在自动发现高效、可解释和可泛化的轨迹预测启发式方法方面迈出了重要一步，为传统的黑盒模型提供了一种实用且强大的替代方案。

### GitHub链接 (GitHub Links)
- https://github.com/ai4co/trajevo

---

## [Test-Time Reinforcement Learning for GUI Grounding via Region Consistency](https://arxiv.org/pdf/2508.05615v1)

**ID**: `2508.05615v1`

### 摘要 (Detailed Summary)
这篇论文提出了一种名为GUI-RC (GUI Region Consistency) 的测试时缩放方法，用于提高图形用户界面（GUI）grounding的准确性。GUI grounding 是将自然语言指令映射到屏幕上的精确坐标的关键任务，对于自主GUI代理至关重要。论文观察到，当模型对同一GUI元素生成多个预测时，空间重叠模式揭示了隐含的置信度信号。GUI-RC通过从多个采样预测中构建空间投票网格，识别模型高度一致的共识区域，从而利用这些信号。论文还提出了GUI-RCPO (GUI Region Consistency Policy Optimization)，将一致性模式转化为测试时强化学习的奖励，使模型能够在推理过程中迭代地优化其在未标记数据上的输出。实验表明，GUI-RC和GUI-RCPO在ScreenSpot基准测试上显著提高了各种架构的准确性，无需任何训练或额外标注数据，为更强大和数据高效的GUI代理开辟了道路。

### 主要贡献 (Contributions)
- 提出了GUI-RC，一种测试时缩放方法，通过在多个预测中利用空间投票来提高GUI grounding的定位精度，无需额外的训练或标注数据。
- 引入了GUI-RCPO，一种测试时强化学习方法，使用区域一致性作为自监督奖励信号，使模型能够通过在未标记的GUI屏幕截图上进行策略优化来提高grounding能力。
- 在多个基准测试和模型架构上展示了一致的改进。GUI-RC平均提高准确率2-3%，而GUI-RCPO通过无标签优化平均获得4-5%的进一步提升。
- 揭示了在GUI-RCPO之后进一步应用GUI-RC可以产生额外的性能提升，表明我们的方法支持渐进式的自举式改进，无需外部监督，并为GUI自动化提供了一种与训练时优化互补的替代方案。

### 技术方法 (Methods)
- **Multi-Sample Generation (多样本生成):** 使用基于温度的采样从模型中生成K个预测。
- **Spatial Voting Mechanism (空间投票机制):** 构建一个与屏幕截图分辨率匹配的空间投票网格，每个采样预测都为该网格贡献选票，用于量化一致性。
- **Consensus Extraction (共识提取):** 通过识别整个网格中的最大投票数来提取共识区域，该最大投票数代表预测之间最高级别的同意，并选择具有最大面积的区域作为最终的预测结果。
- **Region Consistency as Reward (区域一致性奖励):**  基于每个预测区域内的平均投票密度，计算区域一致性奖励。
- **Policy Optimization (策略优化):**  将GUI grounding 任务形式化为强化学习问题，使用GRPO（Group Relative Policy Optimization）算法优化预期区域一致性奖励。

### 主要结论 (Conclusions)
这篇论文提出了GUI-RC和GUI-RCPO，分别作为测试时缩放和测试时强化学习方法，用于GUI grounding任务。通过利用区域一致性来提高模型性能，且无需额外的训练数据，该方法在多个模型和基准测试中表现出良好的一致性和泛化性，为更强大和数据高效的GUI自动化系统提供了一个有希望的方向。

### GitHub链接 (GitHub Links)
- https://github.com/zju-real/gui-rcpo

---

## [OMNI EAR: B ENCHMARKING A GENT R EASONING IN E MBODIED T ASKS](https://arxiv.org/pdf/2508.05614v1)

**ID**: `2508.05614v1`

### 摘要 (Detailed Summary)
该论文提出了OmniEAR，一个综合性的框架，用于评估大型语言模型在具身任务中关于物理交互、工具使用和多智能体协作的推理能力。与现有基准测试提供预定义的工具集或明确的协作指令不同，OmniEAR要求智能体动态地获取能力，并根据任务需求自主地确定协作策略。通过基于文本的环境表示，论文模拟了跨越家庭和工业领域的1500个场景中连续的物理属性和复杂的空间关系。系统评估揭示了当模型必须从约束中进行推理时，性能会严重下降：在明确的指令下达到85-96%的成功率，但对于工具推理，性能下降到56-85%，对于隐式协作，性能下降到63-85%，而复合任务的失败率超过50%。令人惊讶的是，完整的环境信息会降低协作性能，表明模型无法过滤任务相关的约束。微调可以显著提高单智能体任务的性能(0.6%到76.3%)，但对多智能体任务的收益最小(1.5%到5.5%)，暴露了根本的架构限制。这些发现表明，具身推理提出了与当前模型可以解决的问题截然不同的挑战，从而将OmniEAR确立为评估和推进具身AI系统的严格基准。

### 主要贡献 (Contributions)
- 提出了OmniEAR框架，该框架通过需要智能体理解物理属性如何决定动作、能力和协作需求的场景来评估具身推理，解决了当前评估方法中的根本缺陷。
- 开发了EAR-Bench，一个包含1500个场景的基准测试，具有连续的物理属性和动态能力，由EAR-Sim和一个自动生成管道支持。
- 提供了经验证据，表明当前的语言模型缺乏核心的具身推理能力，当从明确的指令转移到具身推理时，性能下降超过60%，揭示了推进具身AI的关键需求。
- 设计了EAR-Sim，可以捕获详细的物体属性和空间关系，同时支持通过工具获取实现动态能力演变。
- 构建了自动化pipeline，生成多样化的场景，这些场景的解决方案自然取决于对具身原理的理解。

### 技术方法 (Methods)
- **环境表示：** 使用有向图 *G* *t* = (*V* *t* *, E* *t* *, A* *t* )来形式化具身环境，其中 *V* *t* 包含空间节点、物体节点和智能体节点三种实体类型，*A* *t* 存储连续的物理属性，*E* *t* 编码空间关系。
- **任务形式化：** 每个评估任务定义为一个元组 T = (*S* init *, I, G* goal *,* A task )，其中 *S* init 指定初始环境状态，*I* 提供自然语言指令，*G* goal 通过逻辑谓词定义成功条件，A task 标识参与的智能体。评估目标是评估智能体是否可以生成一个动作序列 Π = (*π* 1 *, . . ., π* *T* )，将环境从 *S* init 转换为满足 *G* goal 中所有谓词的终端状态 *S* final。
- **分层任务分类：** 任务按照两个正交维度组织：智能体配置（单智能体 vs. 多智能体）和认知复杂性（L1: 基础, L2: 中级, L3: 高级）。
- **EAR-Sim环境模拟器：** 使用基于文本的环境建模来实现大规模高效模拟。图结构 *G* *t* 通过拓扑连接维护空间关系，而不是连续坐标，从而避免了昂贵的碰撞检测，同时保留了必要的空间约束。状态更新采用增量方法，其中动作仅修改直接受影响的节点和边。
- **动态能力管理：** 通过动态工具-能力绑定系统，当智能体抓住一个工具时，系统会动态地将相关能力绑定到智能体的动作集；释放工具时，这些能力会自动解除绑定。
- **涌现协作：** EAR-Sim支持从物理约束中涌现的协作。当智能体尝试对属性超出个体能力的物体执行动作时，系统会启用协作请求机制。
- **自动化基准生成：** 通过四阶段pipeline，结合LLM和基于规则的验证，自动生成多样化、物理一致的场景。包括场景生成、任务生成、评估逻辑提取和专家轨迹生成。

### 主要结论 (Conclusions)
该论文提出的OmniEAR基准测试表明，当前的语言模型在具身推理方面存在严重不足，尤其是在需要从物理约束进行推理的任务中。研究结果揭示了维持多步骤计划的关键参数阈值，环境信息对协作的矛盾影响，以及微调无法解决多智能体推理差距的问题。具身推理需要与当前语言模型不同的计算机制。OmniEAR为诊断这些局限性并开发下一代具身AI系统提供了一个严谨的平台。

### GitHub链接 (GitHub Links)
- https://github.com/ZJU-REAL/OmniEmbodied

---

## [COOPER: CO-OPTIMIZING POLICY AND REWARD MODELS IN REINFORCEMENT LEARNING FOR LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2508.05613v1)

**ID**: `2508.05613v1`

### 摘要 (Detailed Summary)
这篇论文提出了名为**Cooper**的强化学习框架，旨在解决大型语言模型(LLMs)在推理任务中使用强化学习(RL)时，奖励函数设计上的局限性。当前的奖励机制主要分为基于模型和基于规则两种，前者容易受到奖励黑客攻击，后者缺乏鲁棒性。Cooper通过联合优化策略模型和奖励模型，利用规则奖励识别正确答案的高精度优势，并动态构建和选择正负样本对以持续训练奖励模型，从而提高鲁棒性并减轻奖励黑客的风险。为了支持Cooper，论文还提出了一种混合标注策略，高效准确地生成奖励模型的训练数据。此外，还提出了基于参考答案的奖励建模范式，并基于此训练了一个名为VerifyRM的奖励模型，该模型在VerifyBench上实现了比同等规模的其他模型更高的准确率。实验结果表明，Cooper不仅缓解了奖励黑客问题，还提高了端到端RL的性能。

### 主要贡献 (Contributions)
- 提出了一个奖励建模数据集，该数据集使用混合标注策略进行标注，结合了基于规则的验证和基于LLM的验证，从而实现了高效可靠的正确性监督。在该数据集上训练的奖励模型在VerifyBench上达到了89.42%的准确率，超过了现有同等规模的奖励模型。
- 基于规则奖励在识别正确答案时的高精度，提出了**Cooper**，一个同时共同优化策略模型和奖励模型的强化学习框架。该框架缓解了基于奖励模型的RL中常见的奖励黑客问题，并提高了整体训练性能。
- 研究表明，在RL训练过程中动态调整奖励模型的参数可以有效地缓解奖励黑客现象，为研究界如何更好地在强化学习中利用奖励模型提供了有价值的见解。

### 技术方法 (Methods)
- **VerifyRM的训练配方:**
    *   **数据准备:** 收集包含推理问题、参考答案和模型生成的补全的数据三元组。使用7个常用的数学推理数据集和11个主流LLM生成了65K个问题-参考-补全三元组。
    *   **正确性的混合标注:** 结合基于规则的验证器（Math-verify）和LLM-as-a-judge（Qwen3-4B）进行自动标注，只保留两种方法都同意正确性标签的样本，得到58.7K个训练样本。
    *   **奖励模型训练:** 将奖励模型建模为文本分类器，将参考答案纳入奖励模型的输入。使用二元交叉熵损失训练模型。

- **带有Cooper的强化学习:**
    *   **策略模型优化:** 遵循GRPO范式，对每个训练样本，使用策略采样一组响应，然后奖励模型评估每个rollout并生成分数。对这些奖励进行组内归一化以计算优势估计，然后用于通过策略梯度更新策略。为了规范探索并确保训练稳定性，在强化学习期间加入了KL散度惩罚。
    *   **奖励模型优化:** 使用对比学习优化奖励模型，对于给定的问题，参考答案和一对候选响应，目标是最大化RM分配给正确响应和不正确响应之间的分数差异。
        *   **正样本选择:** 从一组响应中，随机选择一个被规则判断为正确的样本，并将其视为正样本。
        *   **负样本生成:** 使用辅助LLM将正确的推理过程转换为最终产生不正确答案的过程，由精心设计的prompt指导。通过规则进行验证，若不被识别为不正确，则重复该过程，直到获得有效的负样本。

### 主要结论 (Conclusions)
论文得出以下结论：
1.  **奖励黑客攻击是静态奖励模型的基本问题**: 固定的奖励模型存在固有的缺陷，模型会利用奖励模型的弱点进行攻击。
2.  **高精度信号的关键作用**: 基于规则的验证器具有高精度但低召回率的不对称特性，可以作为选择正训练样本的优势。
3.  **协同优化是解决奖励模型限制的有效方法**: Cooper通过协同优化策略模型和奖励模型，提高了LLMs在推理任务中的性能，缓解了奖励黑客攻击，并提高了训练的稳定性。

### GitHub链接 (GitHub Links)
- https://github.com/zju-real/cooper
- https://github.com/huggingface/Math-Verify

---

## [Shuffle-R1: Efficient RL framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle](https://arxiv.org/pdf/2508.05612v1)

**ID**: `2508.05612v1`

### 摘要 (Detailed Summary)
这篇论文提出了Shuffle-R1，一种用于多模态大型语言模型（MLLM）的强化学习（RL）框架，旨在提高RL微调的效率。论文指出现有的RL流程存在两个主要问题：优势崩溃（Advantage Collapsing），即大多数优势值集中在零附近，以及回滚沉默（Rollout Silencing），即对梯度有贡献的回滚比例随时间减少。为了解决这些问题，Shuffle-R1引入了成对轨迹采样（Pairwise Trajectory Sampling），选择具有较大优势对比的高对比度轨迹，以提高梯度信号质量；以及基于优势的批次打乱（Advantage-based Batch Shuffle），通过策略性地重新组织批次，增加有价值的回滚的曝光度。实验结果表明，Shuffle-R1在多个推理基准测试中优于现有的强化学习基线，且计算开销最小。该研究强调了以数据为中心的适应性对于提高MLLM中RL训练效率的重要性。

### 主要贡献 (Contributions)
- 揭示了RL微调中训练效率的两个关键但未被充分探索的限制：优势崩溃（Advantage Collapsing）和回滚沉默（Rollout Silencing）。
- 提出了Shuffle-R1，一种新型自适应RL框架，可动态选择高对比度轨迹并重新组织训练批次，以突出有用的样本。
- 通过跨模型规模以及领域内和领域外基准的广泛实验，证明了该框架的有效性和泛化性。
- 提出了一种轻量级、模块化的框架，可以与现有的RL算法无缝集成，并且计算开销最小。
- 在具有挑战性的多模态推理任务中，Shuffle-R1能够显著提高模型性能，甚至在MathVerse和MathVista上超过GPT4o和Claude-3.7。

### 技术方法 (Methods)
- **成对轨迹采样 (Pairwise Trajectory Sampling, PTS):** 从扩展的回滚池中选择具有大幅度优势的高对比度轨迹对，将学习信号集中，以减轻优势崩溃问题。通过“最大-最小”原则，将最高优势的轨迹与最低优势的轨迹配对，形成信息丰富的“正-负”对，并仅保留优势对比最大的轨迹对用于训练。
- **基于优势的批次打乱 (Advantage-based Batch Shuffle, ABS):** 动态重塑训练批次，以优先处理和加强高价值样本，解决回滚沉默问题。ABS根据每个轨迹对的绝对优势总和分配重要性权重，并根据采样概率对原始批次进行子采样，从而增加高优势轨迹的更新频率，同时通过重复曝光保持多样性。

### 主要结论 (Conclusions)
论文提出了Shuffle-R1，一种简单而有效的框架，可提高多模态大型语言模型强化学习的训练效率。通过成对轨迹采样和基于优势的批次打乱，该框架在领域内和领域外任务中均显着优于代表性算法和模型，证明了以数据为中心的自适应设计的价值。

### GitHub链接 (GitHub Links)
- https://github.com/XenoZLH/Shuffle-R1

---

## [Non-omniscient backdoor injection with a single poison sample: Proving the one-poison hypothesis for linear regression and linear classification](https://arxiv.org/pdf/2508.05600v1)

**ID**: `2508.05600v1`

### 摘要 (Detailed Summary)
这篇论文研究了机器学习模型中的后门注入攻击，这种攻击通过在训练数据中插入恶意样本，使模型在特定触发条件下产生恶意行为。论文提出了“单毒药假设”，即一个具有少量背景知识的攻击者，仅需一个毒药样本就能成功注入后门，且不会对良性学习任务产生显著影响。论文针对线性回归和线性分类证明了这个假设。研究表明，如果毒药样本的方向与良性数据分布未使用的方向一致，那么模型与排除该毒药样本的模型功能等效。此外，论文还基于统计后门学习的先有研究，证明在其他情况下，毒药样本对良性学习任务的影响仍然有限。最后，通过在现实基准数据集上的实验验证了理论结果。

### 主要贡献 (Contributions)
- 证明了对于线性分类和线性回归模型，在对训练数据知之甚少的情况下，只需一个中毒样本就可以成功植入后门，且攻击误差几乎为零。
- 证明了如果良性数据分布的所有样本在某个方向上的投影幅度为零，那么当攻击者选择该方向作为其单个毒药样本时，干净模型和中毒模型对于任何干净数据样本都是功能等效的。
- 基于Wang等人的先前工作，针对分类问题，并将他们的工作扩展到回归问题，表明在所有其他情况下，毒药样本对良性学习任务的影响仍然有限。
- 通过在现实基准数据集上进行评估，验证了理论结果。
- 研究结果表明，即使攻击者对训练数据了解有限，机器学习模型也可能受到单个恶意数据点的攻击，这突显了开发有效防御机制的重要性。

### 技术方法 (Methods)
- 理论证明：使用数学方法，针对线性回归和线性分类问题，证明了单毒药假设的成立性。证明过程包括构建攻击者模型，分析损失函数的梯度，并利用Chebyshev不等式等数学工具，推导出毒药样本的注入条件。
- 功能等价性分析：针对特定情况（良性数据在特定方向上的投影幅度为零），证明了中毒模型和干净模型在功能上的等价性，即对良性样本的预测结果一致。这部分主要使用向量空间和正交性的相关理论。
- 统计风险分析：基于Wang等人的工作，扩展了统计风险分析方法，用于量化毒药样本对良性学习任务的影响。通过分析中毒数据分布和良性数据分布之间的差异，以及模型在两种分布上的统计风险，给出了良性学习任务性能下降的界。
- 实验验证：在真实数据集上进行实验，验证理论分析的正确性。实验包括训练干净模型和中毒模型，评估它们在良性任务和后门任务上的性能，并计算相关指标（如MSE、准确率等）。

### 主要结论 (Conclusions)
论文证明了线性回归和线性分类的单毒药假设。研究表明，通过投毒单个数据点，并仅需了解少量其他数据点的信息，就可以成功攻击这些模型。论文中的界限经过正式证明，适用于真实世界的实例大小，并且通过实验验证。

---

## [使用 Kolmogorov-Arnold 网络 (KANs) 优化物联网威胁检测](https://arxiv.org/pdf/2508.05591v1)

**ID**: `2508.05591v1`

### 摘要 (Detailed Summary)
这篇论文探讨了Kolmogorov-Arnold Networks (KANs)在物联网 (IoT) 网络入侵检测中的应用潜力，作为传统机器学习模型的替代方案。物联网的快速发展导致了安全问题日益严重，物联网网络已成为网络攻击的主要目标。KANs 采用可学习的激活函数，优于传统的多层感知器 (MLP)，并且与随机森林 (Random Forest) 和 XGBoost 等最先进的模型相比，KANs 能够获得具有竞争力的准确性，同时为物联网网络中的入侵检测提供卓越的可解释性。该研究表明，KANs 可以有效地检测物联网网络中的入侵，通过可学习的激活函数实现对复杂数据模式的动态适应，并通过特征选择优化检测性能和计算效率，满足实时物联网环境的需求。

### 主要贡献 (Contributions)
- 将 KAN 应用于 CIC IoT 2023 数据集，展示了边缘上可学习的激活函数如何提高模型的准确性和可解释性。
- 评估 KAN 与传统模型（例如，随机森林、XGBoost），以证明其在物联网入侵检测中的竞争性能和适用性。
- 通过符号公式生成展示 KAN 的可解释性，从而在安全关键型物联网系统中实现透明的决策。
- 通过优化特征选择，在不牺牲精度的前提下降低计算开销
- 为在物联网环境中实时应用入侵检测系统开辟了新的途径

### 技术方法 (Methods)
- **数据预处理：** 使用 Pandas, PyTorch 和 Scikit-Learn 等库加载和处理 CIC IoT 2023 数据集。将数据集划分为训练集（67%）和测试集（33%），并对目标变量进行二值化处理，将正常流量标记为 1，攻击流量标记为 0。
- **特征选择：** 使用 Random Forest 模型评估特征重要性，并选择前 10 个最重要的特征。
- **模型构建：** 构建 KAN 模型，包括输入层（神经元数量等于选择的特征数量）、两个隐藏层（分别有 16 和 8 个神经元）以及输出层（2 个神经元，分别代表正常流量和恶意流量）。KAN 模型使用 MultiKAN 架构，允许特征之间的加性和乘性交互。
- **模型训练：** 使用 Adam 优化器和 CrossEntropyLoss 损失函数训练 KAN 模型。训练过程迭代多次，通过调整模型参数最小化损失函数。训练过程共进行 114680 次迭代。
- **模型评估：** 使用精确率、召回率和 F1 值评估模型的分类性能。同时测量训练时间和预测时间，评估计算效率。
- **基线模型：** 使用 Scikit-learn 库实现 Logistic Regression, Random Forest, Decision Trees, K-Nearest Neighbors (KNN), Gradient Boosting, XGBoost, Naive Bayes, Multi-Layer Perceptron (MLP), 和 AdaBoost 等基线模型，并使用默认参数。

### 主要结论 (Conclusions)
研究表明，Kolmogorov-Arnold 网络 (KAN) 在捕获物联网环境中复杂的非线性关系方面非常有效，明显优于传统的机器学习模型。研究强调了优化特征选择的关键重要性，优化特征选择不仅可以通过减少变量数量来提高模型性能，还可以最大限度地减少训练时间和计算开销，从而有助于在资源受限的物联网系统中实现实时应用。将 KAN 与可学习的激活函数相结合，代表了物联网安全框架领域的一项重大进步。这种集成提供了一种强大的解决方案，可以提高网络流量分类的准确性和可解释性，这对于保护敏感数据免受不断演变的网络威胁至关重要。作为未来的工作，通过硬件加速（GPU）优化 KAN 可以弥合训练效率差距，使其适用于大规模物联网部署。将可解释的符号层与更快的检测后端相结合的混合方法可以允许在检测性能和透明度之间取得平衡。

---

## [增强 PyKEEN：用于知识图嵌入模型的多种负采样解决方案](https://arxiv.org/pdf/2508.05587v1)

**ID**: `2508.05587v1`

### 摘要 (Detailed Summary)
这篇论文旨在解决知识图嵌入（KGE）模型中负采样策略的不足问题。KGE 模型依赖于正负样本进行训练，但负样本通常需要人工生成。现有的 KGE 库通常只支持基础的负采样策略，缺乏高级解决方案。为了填补这一空白，作者为 PyKEEN 框架开发了一个扩展，集成了多种高级负采样器（包括静态和动态的 corruption 策略）。该扩展采用一致的模块化架构，生成有意义的负样本，同时与现有的 PyKEEN 工作流程兼容。该扩展不仅增强了 PyKEEN 本身，还简化了嵌入方法的开发和定制。论文还通过一个全面的实证研究，展示了该扩展对不同嵌入方法在链接预测任务上的性能影响，并为设计更有效的策略提供了有用的见解。

### 主要贡献 (Contributions)
- 开发了 PyKEEN 的一个模块化扩展，集成了多种高级负采样技术，包括静态和动态 corruption 策略。
- 该扩展采用一致的模块化架构，易于集成到现有的 PyKEEN 工作流程中。
- 提供了五种新的静态负采样器：“Corrupt”、“Typed”（带领域和范围类型）、“Typed”（仅带实体类型）和“Relational”。
- 集成了两种动态负采样器：“Nearest Neighbor”和“Adversarial”，利用预训练的辅助模型来指导负样本的选择。
- 通过全面的实证研究，评估了所开发扩展对链接预测任务的影响，并分析了不同负采样策略在不同数据集上的性能。

### 技术方法 (Methods)
- **静态负采样**：通过定义一个标准来选择候选实体的子集作为负样本池，然后从中抽取实体。包括 Random Sampling、Bernoulli Sampling、Corrupt Sampling、Typed Sampling 和 Relational Sampling。
- **动态负采样**：利用预训练的辅助模型来指导信息量更大的负样本的选择。包括 Nearest Neighbor 和 Adversarial Sampling。
- **SubsetNegativeSampler 类**：该类继承自 PyKEEN 的 NegativeSampler 抽象基类，用于处理特定负样本池的底层功能。
- **generate_subset() 方法**：该方法处理采样的任何预计算，例如按类型、频率或其他条件进行过滤。
- **strategy_negative_pool() 方法**：此方法指定如何计算给定三元组、损坏目标和预先计算的子集的实际候选负数集。

### 主要结论 (Conclusions)
这项工作介绍并验证了 PyKEEN 框架的模块化扩展。该扩展旨在为采用 KGE 方法时提供广泛的标准负采样器实现覆盖，从而填补了更多高级实现的负采样器可用性的重要空白。我们特别提供了与静态和动态损坏策略兼容的五个负采样器的完全兼容实现。通过一系列实验，我们展示了其实际效用，展示了如何将扩展无缝集成到 KGE 的训练、评估和超参数优化管道中。此外，我们还提供了四个常用数据集的负样本池统计数据，突出了不同条件下各种采样策略的操作约束和行为。通过遵守 PyKEEN 架构和设计标准，所提出的扩展支持可重复性、模块化和易于实验。通过减少开发和评估新采样策略的障碍，我们的目标是培养一个更加统一和高效的研究生态系统。

### GitHub链接 (GitHub Links)
- https://github.com/ivandiliso/refactor-negative-sampler/
- https://ivandiliso.github.io/refactor-negative-sampler

---

## [使用大型语言模型迭代学习治疗耐药性高血压的可计算表型](https://arxiv.org/pdf/2508.05581v1)

**ID**: `2508.05581v1`

### 摘要 (Detailed Summary)
本研究探讨了大型语言模型（LLMs）在生成可解释的可计算表型（CPs）方面的潜力，特别是在高血压管理方面。论文提出了一种*合成、执行、调试、指导*（SEDI）的迭代学习策略，利用LLMs生成CPs，并通过数据驱动的反馈进行迭代优化。研究评估了LLMs在零样本学习情景下的表现，并将其与传统的机器学习方法进行了比较。结果表明，LLMs结合迭代学习能够生成既可解释又具有相当准确性的程序，其性能接近最先进的机器学习方法，同时显著减少了所需的训练样本数量。该研究强调了LLMs在自动化CP生成和临床决策支持方面的潜力，并讨论了其在医疗保健领域应用的一般性见解。

### 主要贡献 (Contributions)
- 提出了一种新的基于LLM的CP生成方法，利用自然语言描述自动引导CP构建，减少了对大量标注数据的依赖。
- 设计并验证了一种迭代学习策略（SEDI），通过合成、执行、调试和指导循环，不断优化LLM生成的CPs。
- 对多种LLM模型在不同提示细节程度和特征数量下生成CPs的能力进行了全面评估，并与传统的机器学习方法进行了比较。
- 证明了LLM生成的CPs在特定情况下（如难治性高血压）可以达到与最先进的机器学习方法相媲美的性能，同时保持了更高的可解释性。
- 强调了LLMs在自动化CP生成和临床决策支持方面的潜力，并讨论了其在医疗保健领域应用的一般性见解。

### 技术方法 (Methods)
- 利用LLMs生成Python程序形式的CPs，这些程序可以根据患者的电子健康记录数据预测特定表型的概率。
- 设计了零样本提示和SEDI提示两种模式。零样本提示直接让LLM生成CPs，而SEDI提示则通过迭代地接收CP在训练数据集上的表现反馈来不断改进CP。
- 在SEDI循环中，如果CP执行失败，LLM会收到错误回溯信息；如果CP执行成功，LLM会收到性能指标以及假阳性和假阴性病例的示例，并被指示改进其表型定义以提高程序性能。

### 主要结论 (Conclusions)
最先进的LLMs能够为高血压表型生成相当准确和简洁的CPs，即使只提供简单的提示。当提供详细和集中的提示，并配备数据驱动的迭代反馈（即SEDI）时，LLM生成的CPs可以与使用监督ML训练的CPs相媲美。传统的监督ML方法利用图表审查的例子仍然优于LLM衍生的CPs，但产生的模型更大，需要访问更大的专家标记数据集。该研究还确立了LLMs迭代学习以可理解的Python代码表示的CPs的潜力。

### GitHub链接 (GitHub Links)
- https://github.com/cavalab/htn-phenotyping-with-llms
- https://GitHub.com/FacebookResearch/Nevergrad

---

## [Fairy ±i : the First 2-bit Complex LLM with All Parameters in {± 1, ±i}](https://arxiv.org/pdf/2508.05571v1)

**ID**: `2508.05571v1`

### 摘要 (Detailed Summary)
这篇论文提出了Fairy ±i，一种新颖的用于复值LLM的2位量化框架。该框架旨在通过提高全精度模型的精度上限，而非仅仅减少量化误差，来突破现有量化方法的精度限制。Fairy ±i利用复数域的表示优势来提升全精度模型的准确性。具体来说，它将权重映射到单位的第四个根{±1, ±i}，形成一个完美的对称和信息论上最优的2位表示。这种表示的每个量化权重要么具有零实部，要么具有零虚部，从而实现仅使用加法和元素交换的无乘法推理。实验结果表明，Fairy ±i在PPL和下游任务方面优于现有2位量化方法的精度上限，同时保持严格的存储和计算效率。这项工作为构建在极低比特约束下高度准确和实用的LLM开辟了一个新的方向。

### 主要贡献 (Contributions)
- 提出了一种低比特量化的新视角：通过提高全精度模型的精度上限来提高量化模型的精度。
- 设计了一种复值LLM架构，该架构利用复数域的表示优势，而无需增加参数存储。
- 设计了一种2位量化方案，该方案将复数权重映射到单位的第4个根{±1, ±i}，充分利用比特容量，同时保留对称性和稀疏性等关键属性。
- 实验结果表明，该量化模型在PPL和下游理解任务方面优于现有2位量化方法的上限。

### 技术方法 (Methods)
- **Complex-Valued Transformer Backbone**: 将标准的LLaMA风格架构适配到复数域，重新设计了核心组件，例如嵌入层、自注意力层、语言模型头和前馈网络，使用了ComplexLinear模块来处理复值参数和激活。
- **PhaseQuant Quantization**: 一种确定性的方法，基于其在复平面中的相位，将每个全精度复数权重映射到单位的第四个根{±1, ±i}之一。
- **Complex-Valued Activation Quantization**: 采用对称的per-token INT8量化方案，独立处理激活的实部和虚部。
- **Efficient Complex-Valued Self-Attention**: 使用厄米特内积的实部作为注意力分数，并将复值计算重塑为更大的实值矩阵乘法，以便使用高度优化的实值FlashAttention内核。

### 主要结论 (Conclusions)
论文提出了Fairy ±i，这是第一个参数全部为{±1, ±i}的2位复数LLM。通过将复值表示集成到Transformer中，并通过提出的PhaseQuant将权重量化为单位的第四个根{±1, ±i}，Fairy ±i充分利用了2位空间，同时保留了对称性、效率和硬件兼容性。实验结果表明，在同等模型大小下，Fairy ±i在困惑度和任务准确性方面优于所有现有量化方法的精度上限。

### GitHub链接 (GitHub Links)
- https://github.com/PKULab1806/Fairy-plus-minus-i

---

## [具有 Richardson-Romberg 外推的马尔可夫 LSA 的高阶误差界限](https://arxiv.org/pdf/2508.05570v1)

**ID**: `2508.05570v1`

### 摘要 (Detailed Summary)
本文研究了马尔可夫噪声下具有 Polyak-Ruppert (PR) 平均的线性随机逼近 (LSA) 算法的偏差和高阶误差界限。文章重点研究了常步长 α 的算法版本，并提出了一种通过线性化技术分解偏差的新方法。分析了偏差的结构，表明主导项是 α 的线性函数，并且不能通过 PR 平均消除。为了解决这个问题，文章应用了 Richardson-Romberg (RR) 外推程序，有效地消除了主导偏差项。文章推导了 RR 迭代的高阶矩界限，并表明主导误差项与原始平均 LSA 迭代的渐近最优协方差矩阵对齐。

### 主要贡献 (Contributions)
- 提出了一种新的技术来量化 θn(α) 的渐近偏差。该方法考虑了联合马尔可夫链 {(θk(α), Zk+1)}k∈N 的极限分布 Πα，并分析了偏差 Πα(θ0) - θ⋆。然后，应用来自 (Aguech, Moulines, and Priouret 2000) 的 θk(α) 的线性化方法。这允许研究组件的极限分布，其平均值显示为按 α 的幂排序。
- 建立了 Richardson-Romberg 方法的高阶矩误差界限，其中主导项与渐近最优协方差 Σ∞ 对齐。分析了它对步数 n、步长 α 和混合时间 tmix 的依赖性。
- 详细的偏差分解，明确识别了误差界限的主导项系数，并分析了其对混合时间等参数的依赖性。
- 对 Markovian LSA 进行了高阶矩误差分析，这是以前工作中未明确强调的。
- 通过数值实验验证了理论结果，并验证了所提出的误差界限的准确性。

### 技术方法 (Methods)
- **线性化技术:** 通过对 LSA 迭代进行线性化，将偏差分解为一系列项，其中每一项对应于步长 α 的不同幂次。
- **Richardson-Romberg 外推:** 应用 Richardson-Romberg 外推法来消除偏差分解中的主导项，从而提高算法的收敛速度。
- **Wasserstein 距离:** 使用 Wasserstein 距离来分析 Markov 链的收敛性，并证明算法的遍历性。
- **Rosenthal 不等式:** 利用 Rosenthal 不等式来建立高阶矩误差界限。
- **耦合技术:** 使用耦合技术来分析 Markov 链的性质，并推导误差界限。

### 主要结论 (Conclusions)
论文研究了马尔可夫线性随机逼近中 Richardson-Romberg 外推法的高阶误差界限。通过应用一种新的偏差表征技术，文章得以获得与渐近最优协方差矩阵 Σ∞ 对齐的主导项。进一步的工作将考虑将获得的结果推广到非线性马尔可夫 SA 和具有状态依赖噪声的 SA 设置。

---

## [X-VFL: A New Vertical Federated Learning Framework with Cross Completion and Decision Subspace Alignment](https://arxiv.org/pdf/2508.05568v1)

**ID**: `2508.05568v1`

### 摘要 (Detailed Summary)
该论文提出了一种新的垂直联邦学习框架X-VFL，旨在解决VFL中两个关键挑战：一是需要所有客户端的样本完全对齐（不允许缺失特征），二是在联合协同推断/预测中需要所有客户端参与（不支持单个客户端的本地独立推断）。X-VFL通过引入Cross Completion (XCom)和Decision Subspace Alignment (DS-Align)两个模块来解决这些问题。XCom模块利用其他客户端的信息来补全非对齐样本中缺失的特征。DS-Align模块将本地特征与跨客户端的全局特征在决策子空间内对齐，从而支持每个客户端的本地独立推断。该论文还提供了X-VFL训练中使用的不同算法的收敛性定理，证明了SGD类型算法的收敛速度为O(1/√T)，PAGE类型算法的收敛速度为O(1/T)。在真实数据集上的大量实验表明，X-VFL明显优于现有方法，例如在CIFAR-10图像数据集上实现了15%的准确率提升，在MIMIC-III医疗数据集上实现了43%的提升。

### 主要贡献 (Contributions)
- 提出了X-VFL，一种新型VFL框架，旨在处理具有（部分）缺失特征的非对齐数据样本，并支持每个客户端对新数据样本进行本地独立推理。X-VFL引入了两个关键模块：Cross Completion (XCom) 和 Decision Subspace Alignment (DS-Align)。
- 据我们所知，我们是第一个在VFL中引入具有*部分*缺失特征的实际设置，其中客户端可以保留一些本地特征，而不是完全缺失非对齐数据样本的所有本地特征。我们引入了*缺失率*，用 R_miss 表示，其中客户端可能丢失 R_miss * (m/2) 个本地特征。 R_miss = 1 恢复了完全缺失的情况。
- 为X-VFL训练中使用的算法提供了理论收敛性定理，证明了SGD类型算法的收敛速度为O(1/√T)，PAGE类型算法的收敛速度为O(1/T)，其中T表示训练更新步骤的数量。
- 在真实世界的数据集上进行了广泛的实验，证明了 X-VFL 显著优于现有的 VFL 方法，例如，在 CIFAR-10 上提高了 15% 的准确率，在 MIMIC-III 医疗数据集上提高了 43% 的准确率。

### 技术方法 (Methods)
- Cross Completion (XCom)：该模块旨在利用其他客户端的信息来完成/重建非对齐数据样本的缺失特征。通过建立不同客户端提供的分离局部特征之间的交叉互补依赖关系，本地客户端可以通过XCom完成其缺失的特征，从而有效地增加可用于训练和推理的数据量。
- Decision Subspace Alignment (DS-Align)：该模块旨在决策子空间内对齐所有客户端的特征，以支持每个客户端的本地独立推理，同时保持与涉及所有客户端的协作推理相当的性能。

### 主要结论 (Conclusions)
该论文提出了X-VFL，一种新型VFL框架，通过有效地处理具有部分缺失特征的数据集，并在每个客户端启用本地独立推理，从而解决了传统VFL的关键挑战。特别是，X-VFL引入了两个关键模块：Cross Completion（XCom）和Decision Subspace Alignment（DS-Align）。在实际数据集上的大量实验表明，X-VFL显著优于现有的VFL方法，验证了其在解决缺失特征、本地独立推理和数据不平衡等关键挑战方面的实际有效性和优越性。

---

## [L1 正则化函数支持向量机](https://arxiv.org/pdf/2508.05567v1)

**ID**: `2508.05567v1`

### 摘要 (Detailed Summary)
该论文研究了函数数据分析中具有多元函数协变量的二元分类问题，填补了该领域的一个空白。论文提出了一种 L1 正则化函数支持向量机 (L1-fSVM) 用于二元分类，并开发了一种算法来拟合该分类器。通过施加 L1 惩罚，该算法能够识别二元响应的相关函数协变量，从而实现特征选择。数值模拟和实际应用的结果表明，该分类器在预测和特征选择方面都具有良好的性能。该研究为处理具有多个函数协变量的二元分类问题提供了一种有效的方法。

### 主要贡献 (Contributions)
- 提出了 L1 正则化函数支持向量机 (L1-fSVM)，用于处理具有多元函数协变量的二元分类问题。
- 开发了一种迭代更新算法来拟合 L1-fSVM 分类器，该算法可以分别更新系数函数和 SVM 中的向量。
- 通过施加 L1 惩罚，该算法能够识别二元响应的相关函数协变量，从而实现特征选择。
- 在模拟研究中，L1-fSVM 在预测准确性和特征选择方面略优于一些现有的函数分类器。
- 将所提出的分类器应用于脑电图 (EEG) 数据集，以预测酒精依赖状态，并识别与酒精依赖相关的脑电通道。

### 技术方法 (Methods)
- L1 正则化函数支持向量机 (L1-fSVM)：将 L1 正则化引入到函数支持向量机中，用于特征选择。
- B-样条表示：使用 B-样条基函数来近似系数函数，从而将无限维问题转换为有限维问题。
- 坐标下降算法：开发了一种坐标下降算法来迭代更新 L1-fSVM 的参数，包括系数函数和 SVM 中的向量。
- 梯度下降：使用梯度下降法来更新系数函数。

### 主要结论 (Conclusions)
该论文提出了一种 L1 正则化函数支持向量机用于函数数据分类。此外，开发了一种基于坐标下降的算法来估计函数系数并选择与二元响应相关的函数协变量。与文献中的现有方法相比，所提出的分类器在模拟设置中实现了更好的预测精度，并在特征选择方面表现出相当的性能。然后将所提出的分类器应用于 EEG 数据集以预测酒精依赖状态。研究识别了头皮上的几个可能与酒精依赖状态相关的通道（区域）。

---

## [On the Design of Expressive and Trainable Pulse-based Quantum Machine Learning Models](https://arxiv.org/pdf/2508.05559v1)

**ID**: `2508.05559v1`

### 摘要 (Detailed Summary)
这篇论文探讨了脉冲量子机器学习 (QML) 模型的设计，旨在使其既具有表达性又易于训练。脉冲 QML 模型由于其硬件效率而成为量子人工智能领域的新范式。先前的研究表明，在动态对称性下的脉冲模型可以有效地训练，因为其损失函数没有贫瘠高原。然而，当模型设计不充分时，由此产生的不受控性可能会损害表达性。本文研究了脉冲 QML 模型在保持可训练性的同时，具备表达性所需满足的条件。论文提出了一个关于系统初始状态、测量算符和潜在动态对称李代数的必要条件，并通过数值模拟验证。研究结果为设计实用的脉冲 QML 模型奠定了基础，从而平衡了表达性和可训练性。

### 主要贡献 (Contributions)
- 提出了脉冲 QML 模型同时具备表达性和可训练性的设计框架。
- 研究了动态对称性下脉冲 QML 模型的表达性要求。
- 提出了关于系统初始状态、测量算符和潜在动态对称李代数的必要条件。
- 利用戴森级数展开分析了脉冲模型的表达性，并开发了相应的李代数工具。
- 通过数值模拟验证了所提出的理论框架，并展示了表达性和可训练性之间的平衡。

### 技术方法 (Methods)
- 李代数理论：利用李代数框架评估由动态对称性引起的不可控脉冲 QML 模型的可训练性。
- 戴森级数展开：使用戴森级数展开（也称为控制理论中的 Fliess 级数）分析脉冲 QML 模型的表达性，并推导出模型输出函数的多项式级数。
- 数值模拟：通过数值实验验证脉冲 QML 模型在选定的动态对称性下的表达性和可训练性。

### 主要结论 (Conclusions)
论文建立了一个设计实用脉冲 QML 模型的综合框架，该模型既具有表达性又易于训练。该设计结合了戴森多项式级数展开以及现有的与量子系统中表现出动态对称性的贫瘠高原现象相关的李代数理论。理论分析和数值模拟表明，可以利用动态对称性来构建适用于 NISQ 设备上硬件高效部署的、具有表达性和可训练性的脉冲模型。

---

## [MV-Debate: Multi-view Agent Debate with Dynamic Reflection Gating for Multimodal Harmful Content Detection in Social Media](https://arxiv.org/pdf/2508.05557v1)

**ID**: `2508.05557v1`

### 摘要 (Detailed Summary)
这篇论文提出了一个名为MV-Debate的多视角代理辩论框架，用于在社交媒体中检测多模态有害内容。该框架旨在解决由于跨模态矛盾、快速文化变化和微妙语用线索导致识别讽刺、仇恨言论或虚假信息等有害意图的难题。MV-Debate由四个互补的辩论代理组成：表面分析师、深度推理者、模态对比者和社会语境主义者，从不同的解释角度分析内容。通过迭代辩论和反思，代理在∆-gain准则下改进响应，确保准确性和效率。在三个基准数据集上的实验表明，MV-Debate显著优于强大的单模型和现有的多代理辩论基线。这项工作突出了多代理辩论在推进安全关键在线环境中可靠的社会意图检测方面的潜力。

### 主要贡献 (Contributions)
- 提出了MV-Debate，一个多代理辩论框架，它引导代理使用不同的推理视角来检测社交媒体中的多模态有害内容。
- 设计了四个具有特定视角的辩论代理，并采用动态反思门控机制来提高性能。
- 在多个多模态有害内容基准上验证了所提出的方法的有效性。
- 通过分配专门的角色给辩论代理，MV-Debate 能够有效结合表面层面、深度语义、跨模态和社会文化分析，降低了遗漏隐含或与上下文相关的有害线索的风险。
- Top-k Δ-reflection gating 机制增强了可靠性，同时保持了效率。 通过仅在预期有重大改进时才自适应地触发反思，该框架避免了冗余计算，但实现了与无条件反思相当或更好的准确性。

### 技术方法 (Methods)
- **表面分析师代理 (SA)**：此代理专门关注显式的文本和视觉线索来检测有害内容。
- **深度推理者代理 (DR)**：此代理揭示隐含的含义和隐藏的意图来检测有害内容。
- **模态对比代理 (MC)**：此代理评估文本和视觉模态之间的一致性或矛盾来检测有害内容。
- **社会语境主义者代理 (SC)**：此代理利用外部文化和社会语境知识来检测有害内容。
- **判断代理**：这个代理评估由辩论代理生成的论点。 它根据逻辑连贯性、一致性和合理性分配分数，其中更好的响应会获得更高的分数。
- **反思代理**：此代理生成结构化反馈，突出显示逻辑缺陷和改进建议。
- **总结代理**：此代理汇总辩论历史并提供最终预测。
- **Top-k ∆-Reflection Gating**：为了减少计算开销，引入了一种 Top-k ∆ -反思门控策略。 在每一轮中，反思代理都会收到所有辩论代理的回复，并检查每个代理的推理过程。 然后，它会指出推理错误并提供修改建议。 接下来，选择由判断代理评分的最高的 top k 个回复。 然后，每个选定的原始辩论代理都会使用查询实例、初始回复和修改建议生成一个新的回复ˆri,1。 之后，判断代理将重新对新回复进行评分，表示为ˆsi,1。

### 主要结论 (Conclusions)
这项工作介绍了一种新的多视角辩论框架 MV-Debate，用于社交媒体上的多模态有害内容检测。通过协调四个具有互补推理策略的特定视角代理和一个动态反思门控机制，MV-Debate 有效地整合了跨模态证据和语境线索，以识别复杂的社会意图，例如讽刺、仇恨言论和虚假信息。在多个基准上的广泛实验证实了其相对于强大基线的卓越准确性、效率和可解释性。除了性能提升之外，MV-Debate 还生成透明的辩论记录，支持模型调试、审计和用户信任。展望未来，该框架为将多代理辩论方法扩展到更广泛的安全关键型多模态推理任务奠定了基础。

---

## [Adapting Vision-Language Models Without Labels: A Comprehensive Survey](https://arxiv.org/pdf/2508.05547v1)

**ID**: `2508.05547v1`

### 摘要 (Detailed Summary)
这篇论文全面调查了视觉-语言模型 (VLM) 的无监督自适应方法。虽然 VLM 在各种任务中表现出卓越的泛化能力，但在应用于特定下游场景时，未经特定任务的自适应，其性能通常欠佳。为了在保持数据效率的同时增强 VLM 的效用，近期的研究越来越关注不依赖于标记数据的无监督自适应方法。论文提出了一个基于无标记视觉数据的可用性和性质的分类法，将现有方法分为四个关键范式：无数据迁移（没有数据）、无监督领域迁移（有大量数据）、情景式测试时自适应（批量数据）和在线测试时自适应（流式数据）。在这一框架内，论文分析了与每种范式相关的核心方法和自适应策略，旨在建立对该领域的系统性理解。此外，论文还回顾了跨各种应用的代表性基准，并强调了开放的挑战和未来研究的有希望的方向。论文提供了一个GitHub链接，其中维护着与无标签VLM相关的文献。

### 主要贡献 (Contributions)
- 提出了一个基于无标签视觉数据可用性的 VLM 无监督自适应方法分类法，包括无数据迁移、无监督领域迁移、情景式测试时自适应和在线测试时自适应四个范式。
- 对每个范式中的核心方法和自适应策略进行了详细分析，系统地梳理了现有方法。
- 回顾了各种应用的代表性基准，提供了实践角度的分析。
- 总结了该领域的新兴趋势，并指出了关键的科学问题，为未来的研究提供了方向。
- 提供了一个维护好的相关文献的GitHub仓库。

### 技术方法 (Methods)
- **数据自由迁移:** 依赖于LLM进行文本增强，利用外部数据集检索图像，或者修改网络结构。
- **无监督领域迁移:** 使用自训练方法（self-training），使用熵优化方法(entropy optimization)，利用外部资源（external resource utilization）。
- **情景式测试时自适应:** 最小化熵(entropy minimization)，使用反馈信号(feedback signal)，对齐分布(distribution alignment)，自监督学习(self-supervised learning)。
- **在线测试时自适应:** 伪标签(pseudo-labeling)，记忆机制(memory mechanism)，分布建模(distribution modeling)。

### 主要结论 (Conclusions)
论文总结了视觉-语言模型 (VLMs) 无监督自适应领域的研究进展，提出了一个基于无标签视觉数据可用性的新颖分类法。论文认为，现有的方法在特定场景下取得了一定的成功，但仍然面临着理论分析不足、开放世界场景适应性差、对抗鲁棒性弱、隐私安全隐患、推理效率低等挑战。未来的研究应该关注这些挑战，并探索新的下游任务和应用领域。

### GitHub链接 (GitHub Links)
- https://github.com/tim-learn/Awesome-LabelFree-VLMs

---

## [Conformal Sets in Multiple-Choice Question Answering under Black-Box Settings with Provable Coverage Guarantees](https://arxiv.org/pdf/2508.05544v1)

**ID**: `2508.05544v1`

### 摘要 (Detailed Summary)
这篇论文提出了一种在黑盒环境下，基于频率的不确定性量化方法，用于提升大型语言模型（LLMs）在多项选择题回答（MCQA）任务中的可靠性。该方法利用共形预测（CP）框架，通过对模型输出分布进行多次独立采样，并以最频繁的样本作为参考来计算预测熵（PE）。这种基于频率的PE能够有效区分正确和错误的预测，并通过控制经验误覆盖率来保证用户指定风险水平下的覆盖率。实验结果表明，在MedMCQA、MedQA、MMLU和MMLU-Pro等数据集上，该方法优于基于logits的PE，为MCQA中的不确定性量化提供了一种可靠且模型无关的框架，从而增强了LLMs在实际应用中的可信度。

### 主要贡献 (Contributions)
- 提出了一种新颖的基于频率的不确定性量化方法，用于解决黑盒LLMs在MCQA任务中的不确定性问题。
- 该方法通过多次独立采样模型输出分布，并以最频繁的样本作为参考来计算预测熵（PE），从而有效区分正确和错误的预测。
- 验证了频率可以作为黑盒场景中logit概率的可行替代品，为无法访问内部logits的LLMs提供了一种有效的不确定性量化手段。
- 结合共形预测（CP）框架，构建了具有可证明覆盖保证的预测集合，增强了LLMs在实际应用中的可靠性。
- 通过在多个数据集和模型上的实验，证明了该方法在AUROC和经验误覆盖率控制方面的优越性。

### 技术方法 (Methods)
- **频率-预测熵（Frequency-based PE）:** 对每个多项选择题，进行M次独立采样，得到候选答案集合E。计算每个候选答案的频率P^(a)，选择频率最高的候选答案。
- **共形预测（Conformal Prediction, CP）:** 构建一个预测集合，该集合包含真实值的概率不低于预先设定的置信水平1-α。对于校准集中的每个样本xi，定义其非一致性得分si = 1 - F(xi)y*i，其中F(xi)y*i是模型对于真实标签y*i的输出得分。
- **风险控制:** 通过CP确保预测集合以不低于1-α的概率覆盖真实标签，从而实现风险控制。

### 主要结论 (Conclusions)
本研究侧重于黑盒设置下MCQA任务中llm的不确定性量化问题。它提出了一种基于频率的预测熵(PE)方法，通过对模型的输出分布进行多次独立采样，以最频繁的样本作为参考来量化不确定性。结合CP框架，该方法构建了旨在确保预测集具有可证明覆盖保证的预测集。

实验结果表明，在六个模型和四个数据集(MedMCQA、MedQA、MMLU和MMLU-Pro)中，频率PE在不确定性量化性能方面通常优于以AUROC衡量的基于logit的方法。此外，它可以有效地控制各种用户指定的风险水平下的经验性错误覆盖率。这证实了采样频率可以作为黑盒llm中基于logit的概率的可行替代方法，为无法访问内部logit的场景提供了有效的不确定性量化方法。

---

## [Tractable Sharpness-Aware Learning of Probabilistic Circuits](https://arxiv.org/pdf/2508.05537v1)

**ID**: `2508.05537v1`

### 摘要 (Detailed Summary)
这篇论文研究了概率电路（PCs）在数据有限时容易过拟合的问题。论文指出，PC过拟合通常是由于模型收敛到log-likelihood landscape中的尖锐最优解，这些解泛化能力差。为了解决这个问题，论文受到神经网络中sharpness aware minimization的启发，提出了一个基于Hessian矩阵的正则化方法来训练PC。关键贡献在于，论文证明了对于PCs，log-likelihood的Hessian矩阵的迹（trace）可以高效地计算出来，这个迹可以作为sharpness的代理指标。最小化Hessian迹能够产生一个基于梯度范数的正则化项，该正则化项可以无缝地集成到EM算法和基于梯度的学习方法中。实验结果表明，该方法能够引导PCs收敛到更平坦的极小值，从而提高泛化性能。

### 主要贡献 (Contributions)
- 推导了树结构PC的log-likelihood的精确完整Hessian矩阵的闭式表达式，并证明了它可以被有效地计算。
- 对于一般的（DAG结构的）PC，论文确定虽然完整的Hessian矩阵可能是难以处理的，但它的迹（trace）仍然可以在时间和参数数量上线性计算，为大规模PC提供第一个可用的曲率度量。
- 提出了一个新的sharpness-aware正则化项，用于学习PC，该正则化项是从Hessian矩阵的迹推导出来的。
- 论文表明，虽然直接通过EM算法最小化Hessian矩阵的迹会导致三次更新方程，但可以把这个目标重新表述为一个等价的梯度范数最小化问题，从而得到一个具有闭式参数更新的二次方程。
- 在多个合成数据集和真实世界数据集上进行了详尽的实验，以表明论文提出的正则化项强制收敛到更平坦的最优值，并有助于减少过拟合，尤其是在有限数据设置中。

### 技术方法 (Methods)
- Hessian矩阵分析：利用Hessian矩阵量化损失函数的平坦性，其特征值捕获沿不同方向的曲率。
- Hessian迹（Trace）正则化：引入Hessian迹作为衡量整个log-likelihood曲面曲率的标量指标，并设计一个正则化项。
- 梯度范数最小化：将Hessian迹的最小化问题转化为等价的梯度范数最小化问题，简化了计算并推导出闭式参数更新。
- EM算法集成：将Hessian迹正则化项集成到期望最大化（EM）算法的M步中，通过约束总和节点上的梯度范数来寻找更平坦的解。

### 主要结论 (Conclusions)
这项工作引入了一个新的方向，通过log-likelihood表面几何的视角来研究PC的训练。论文推导了树结构PC中log-likelihood的精确完整Hessian的闭式表达式，并证明了其可处理性。对于一般DAG结构的PC，论文表明，虽然完整的Hessian可能是难以处理的，但它的迹仍然可以精确且高效地计算——为训练大型PC提供了第一个可扩展的曲率度量。在此基础上，论文设计了一种新颖的正则化项，其等效梯度范数公式产生闭式二次更新，从而实现高效优化。实验证实，该方法将训练引导到更平坦的最小值并减少过拟合，尤其是在低数据状态下。

---

