# Meta-Analysis of Recent LLM Reasoning Post-Training Papers (2025-2026)

Mini Project 1 - Large Language Models  
Theme: reasoning-focused post-training, reinforcement learning, and test-time scaling  
Date: May 20, 2026

## 1. Introduction

Large language models (LLMs) have moved from simple next-token prediction systems toward models that can solve multi-step problems in mathematics, coding, science, and planning. In 2023 and 2024, much of the public research conversation centered on instruction tuning, RLHF, DPO-style preference optimization, and efficient fine-tuning. In 2025 and 2026, a newer research wave focused on reasoning: how to make models search, verify, reflect, use longer chains of thought, and improve through reinforcement learning or inference-time compute.

This meta-analysis compares five recent papers on reasoning-focused post-training. The connecting theme is not general chatbot alignment, but how LLMs acquire stronger reasoning behavior after pretraining. The papers study different levers: pure RL, large-scale multimodal RL, small curated reasoning demonstrations, test-time compute control, and off-policy data reuse.

Selected papers:

1. DeepSeek-AI et al. (2025/2026), "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", Nature 2025 / arXiv v2 2026.
2. Kimi Team et al. (2025), "Kimi k1.5: Scaling Reinforcement Learning with LLMs", arXiv technical report.
3. Muennighoff et al. (2025), "s1: Simple test-time scaling", EMNLP 2025.
4. Ye et al. (2025), "LIMO: Less is More for Reasoning", COLM 2025.
5. Wan et al. (2026), "Buffer Matters: Unleashing the Power of Off-Policy Reinforcement Learning in Large Language Model Reasoning", ICLR 2026.

## 2. Paper Summaries

### 2.1 DeepSeek-R1

Full citation: DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q., Xu, R., Zhang, R., Ma, S., Bi, X., Zhang, X., Yu, X., Wu, Y., and others. (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." Nature, 645, 633-638. arXiv:2501.12948, revised 2026.

Research problem: Strong reasoning often depends on expensive human-written chain-of-thought demonstrations. DeepSeek-R1 asks whether reasoning can emerge from reinforcement learning with verifiable rewards, without manually labeling reasoning trajectories.

Proposed solution: The paper first trains DeepSeek-R1-Zero from DeepSeek-V3-Base using Group Relative Policy Optimization (GRPO), skipping the usual supervised fine-tuning warm-up. Rewards are rule-based: accuracy rewards check whether math, coding, or logic answers are correct, and format rewards enforce answer structure. The later DeepSeek-R1 pipeline adds cold-start long-CoT data, additional RL, rejection sampling, supervised fine-tuning, and distillation into smaller dense models.

Datasets, architecture, and metrics: The base is DeepSeek-V3-Base, a large mixture-of-experts LLM. Training uses verifiable reasoning prompts from math, coding, and logical reasoning. Evaluation includes AIME 2024, math benchmarks, coding competitions, and STEM tasks. Metrics include pass@1, self-consistency, benchmark accuracy, and comparisons against supervised baselines.

Main results: DeepSeek-R1-Zero improves from 15.6% to 77.9% pass@1 on AIME 2024, and reaches 86.7% with self-consistency. The model develops longer reasoning, verification, reflection, and strategy adaptation. The main limitations are readability problems, language mixing in the zero model, and the fact that verifiable rewards work best in domains with objective answers.

### 2.2 Kimi k1.5

Full citation: Kimi Team, Du, A., Gao, B., Xing, B., Jiang, C., Chen, C., Li, C., Xiao, C., Du, C., Liao, C., Tang, C., Wang, C., Zhang, D., Yuan, E., Lu, E., Tang, F., Sung, F., and others. (2025). "Kimi k1.5: Scaling Reinforcement Learning with LLMs." arXiv:2501.12599.

Research problem: Pretraining scale is constrained by high-quality data availability. Kimi k1.5 asks whether reinforcement learning can become another scaling axis by letting LLMs explore solution paths and learn from reward signals.

Proposed solution: Kimi k1.5 uses a long-context RL recipe for a multimodal LLM. It combines curated RL prompts, long-CoT supervised warm-up, improved policy optimization, length penalties, curriculum and prioritized sampling, and infrastructure for partial rollouts. The model scales RL context to 128k tokens and avoids heavier methods such as Monte Carlo tree search, value functions, and process reward models.

Datasets, architecture, and metrics: The model is multimodal, using text and image-text reasoning data. The RL prompt set covers STEM, coding, competitions, and general reasoning. Evaluation includes AIME, MATH 500, Codeforces, MathVista, and LiveCodeBench.

Main results: Kimi k1.5 reports 77.5 on AIME, 96.2 on MATH 500, 94th percentile on Codeforces, and 74.9 on MathVista, matching OpenAI o1-level reasoning in the authors' comparisons. Its long2short methods also transfer long-CoT gains to shorter outputs. Limitations include limited public detail on proprietary training data, very large infrastructure requirements, and dependence on verifiable or judgeable tasks.

### 2.3 s1: Simple test-time scaling

Full citation: Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L., Hajishirzi, H., Zettlemoyer, L., Liang, P., Candes, E., and Hashimoto, T. (2025). "s1: Simple test-time scaling." Proceedings of EMNLP 2025, 20275-20321.

Research problem: Proprietary reasoning models showed that using more inference-time compute can improve answers, but their methods were not public. s1 asks how simple a public reproduction of test-time scaling can be.

Proposed solution: The authors curate s1K, a dataset of 1,000 questions with reasoning traces chosen for difficulty, diversity, and quality. They fine-tune Qwen2.5-32B-Instruct and add "budget forcing": if the model tries to stop too early, inference appends "Wait" to make it continue checking its reasoning; if it thinks too long, the process can be terminated.

Datasets, architecture, and metrics: The model is Qwen2.5-32B-Instruct fine-tuned into s1-32B. The main dataset is s1K. Evaluation focuses on competition math benchmarks such as MATH and AIME24, with open-source code, model, and data.

Main results: s1 exceeds o1-preview on competition math questions by up to 27% on MATH and AIME24. Budget forcing improves AIME24 from 50% to 57%. Its limitation is narrowness: the method is elegant and reproducible, but results are strongest on math-like tasks where extra reasoning tokens are useful and evaluation is objective.

### 2.4 LIMO: Less is More for Reasoning

Full citation: Ye, Y., Huang, Z., Xiao, Y., Chern, E., Xia, S., and Liu, P. (2025). "LIMO: Less is More for Reasoning." COLM 2025.

Research problem: Many reasoning systems assume that strong mathematical reasoning requires huge post-training datasets. LIMO tests whether a small number of strategically designed examples can unlock reasoning already latent in a pretrained model.

Proposed solution: LIMO uses simple supervised fine-tuning on a small set of carefully curated reasoning examples. The paper proposes the Less-Is-More Reasoning Hypothesis: if pretraining already encodes enough domain knowledge, a few examples can act as cognitive templates that teach the model how to use that knowledge.

Datasets, architecture, and metrics: The paper focuses on mathematical reasoning and compares against earlier fine-tuned models trained on far more examples. Evaluation includes AIME24, MATH500, and out-of-distribution reasoning benchmarks. Metrics include accuracy and cross-benchmark generalization.

Main results: LIMO achieves 63.3% on AIME24 and 95.6% on MATH500 while using about 1% of the training data required by prior approaches. It also reports a 45.8% absolute improvement across diverse benchmarks. The main limitation is that the approach depends on extremely high-quality example selection and may not transfer equally to domains where the pretrained model lacks the needed knowledge.

### 2.5 Buffer Matters / BAPO

Full citation: Wan, X., Wang, Y., Huang, W., and Sun, M. (2026). "Buffer Matters: Unleashing the Power of Off-Policy Reinforcement Learning in Large Language Model Reasoning." ICLR 2026.

Research problem: On-policy RL with verifiable rewards can waste experience: older rollouts are discarded, and many examples become uninformative when all sampled answers receive similar rewards. BAPO asks whether off-policy reuse can make reasoning post-training more data-efficient.

Proposed solution: Batch Adaptation Policy Optimization (BAPO) dynamically builds training batches by re-evaluating historically difficult samples and reusing high-quality samples from a buffer. It delays rollout policy updates in a controlled way and includes a lower-bound argument for policy improvement.

Datasets, architecture, and metrics: Experiments cover mathematics, planning, and visual reasoning. The paper uses backbones such as DeepSeek-R1-Distill 1.5B, Qwen2.5-Math 1.5B/7B, and Qwen2.5-VL 3B/7B. Training/evaluation includes DeepScaleR-Preview, AIME24, AMC23, MATH500, Minerva Math, OlympiadBench, Countdown tasks, and Geometry3K. Metrics include benchmark accuracy, convergence speed, rollout efficiency, and comparison to GRPO-style baselines.

Main results: BAPO reports an average 12.5% improvement over GRPO across math, planning, and visual reasoning, and solves 40.7% of problems that base models consistently fail to solve. Its limitation is that buffer management adds algorithmic complexity, and the method still relies on tasks with reliable verifiers.

## 3. Comparative Analysis

| Aspect | DeepSeek-R1 | Kimi k1.5 | s1 | LIMO | BAPO |
|---|---|---|---|---|---|
| Main objective | Elicit reasoning via RL | Scale RL for multimodal reasoning | Reproduce test-time scaling simply | Show small data can unlock reasoning | Improve RLVR data efficiency |
| Core method | GRPO with rule rewards, then multi-stage post-training | Long-context RL, sampling strategies, long2short | SFT on s1K plus budget forcing | SFT on curated cognitive templates | Off-policy batch adaptation and replay |
| Model scale | DeepSeek-V3/R1 plus distilled models | Large multimodal Kimi model | Qwen2.5-32B-Instruct | Reasoning-focused fine-tuned LLM | 1.5B to 7B text/VL backbones |
| Data strategy | Verifiable math, coding, logic prompts | Curated STEM/coding/general and image-text prompts | 1,000 high-quality reasoning traces | Very small curated math reasoning set | Reuse difficult and high-quality historical samples |
| Evaluation | AIME, coding, STEM, self-consistency | AIME, MATH500, Codeforces, MathVista | MATH, AIME24 | AIME24, MATH500, OOD benchmarks | Math, planning, visual reasoning |
| Strength | Shows reasoning can emerge from incentives | Demonstrates RL scaling and long context | Simple, open, reproducible | Strong data-efficiency argument | More efficient use of RL rollouts |
| Limitation | Objective rewards do not cover all tasks | Proprietary scale and data opacity | Narrow math-centric evaluation | Depends on very careful data selection | More complex than on-policy RLVR |

The papers share a move away from classic chatbot alignment and toward reasoning behavior. DeepSeek-R1 and Kimi k1.5 argue that reinforcement learning can train models to search through solution space. s1 and LIMO argue that small, carefully chosen supervised data can trigger strong reasoning, especially when paired with inference-time compute. BAPO focuses on improving the efficiency of the RL loop itself.

The key methodological split is between training-time and test-time compute. DeepSeek-R1, Kimi k1.5, LIMO, and BAPO change the model through post-training. s1 changes both the model and the decoding process: the model is fine-tuned, but the main novelty is controlling how much it thinks during inference. This is important because reasoning quality increasingly depends not only on parameters, but also on how much search or verification the model performs at test time.

Another difference is reward structure. DeepSeek-R1 and BAPO rely on verifiable rewards, which are clean and scalable for math and coding. Kimi k1.5 also emphasizes reliable verification and prompt filtering. s1 and LIMO reduce reliance on RL by using curated reasoning traces. This suggests that the field is exploring two complementary paths: learn from outcomes through RL, or learn from a small number of high-quality reasoning demonstrations.

Reproducibility varies. s1 is especially reproducible because the authors release data, model, and code. LIMO also emphasizes open resources and a compact dataset. DeepSeek-R1 released models and distilled variants, but the full training pipeline is massive. Kimi k1.5 is highly informative as a technical report, but proprietary scale and training data make exact reproduction difficult. BAPO is algorithmically clearer and evaluated on smaller backbones, making it easier to study in academic settings.

## 4. Insights and Reflection

The strongest trend is that reasoning has become a post-training problem. Pretraining supplies broad knowledge, but post-training teaches models how to deploy that knowledge through search, verification, reflection, and longer deliberation. The 2025-2026 papers are less focused on making a model answer politely and more focused on making it solve harder problems.

The most promising approaches are hybrid. DeepSeek-R1 shows the power of RL incentives. LIMO and s1 show that small but excellent demonstrations can be surprisingly effective. Kimi k1.5 shows that context length and infrastructure matter when scaling RL. BAPO shows that the efficiency of experience reuse matters. A practical future recipe may combine all of these: curated reasoning examples, verifiable RL, replay of difficult samples, and controllable test-time budgets.

The papers also reveal common limitations. First, most results are strongest in domains with objective answers: math, coding, STEM, planning, and geometry. That is useful, but it does not fully solve open-ended reasoning, social judgment, or safety. Second, longer reasoning can create cost and latency problems. Third, benchmark gains can hide failure modes such as overthinking, brittle formatting, reward hacking, or answers that look reflective but are wrong.

Future research should focus on broader verifiers, better process supervision, robust safety evaluation for reasoning models, and ways to control reasoning length without hurting accuracy. Another important direction is transparency: if reasoning models are trained with private data and massive infrastructure, the field needs smaller reproducible studies like s1, LIMO, and BAPO to separate real scientific insight from scale effects.

## 5. Conclusion

This meta-analysis shows that modern LLM research in 2025 and 2026 is strongly shaped by reasoning post-training. DeepSeek-R1 demonstrates that pure reinforcement learning can induce reflection and verification. Kimi k1.5 shows how long-context RL and multimodal data can scale reasoning. s1 shows that a minimal open recipe can reproduce test-time scaling effects. LIMO shows that carefully chosen examples can unlock strong mathematical reasoning with very little data. BAPO shows that off-policy replay can make RL post-training more efficient.

Together, these papers suggest that the field is evolving from instruction following toward deliberate problem solving. The most important open question is not only how to make models think longer, but how to make them think usefully, efficiently, safely, and reproducibly.

## References

1. DeepSeek-AI et al. (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." Nature, 645, 633-638. https://arxiv.org/abs/2501.12948
2. Kimi Team et al. (2025). "Kimi k1.5: Scaling Reinforcement Learning with LLMs." https://arxiv.org/abs/2501.12599
3. Muennighoff et al. (2025). "s1: Simple test-time scaling." EMNLP 2025. https://aclanthology.org/2025.emnlp-main.1025/
4. Ye et al. (2025). "LIMO: Less is More for Reasoning." COLM 2025. https://openreview.net/forum?id=T2TZ0RY4Zk
5. Wan et al. (2026). "Buffer Matters: Unleashing the Power of Off-Policy Reinforcement Learning in Large Language Model Reasoning." ICLR 2026. https://openreview.net/forum?id=RduOiisl1S
