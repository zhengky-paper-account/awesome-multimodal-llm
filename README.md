# ⭐ Kaiyang Zheng的大语言模型与多模态大模型知识库

---

# 📚 大语言模型

## 🧱 第一部分：基础组件（从最小单元开始）

1. [整体架构](./1.%20Architecture%20大模型整体架构/1.%20Architecture%20大模型整体架构.md)
2. [Tokenization / 分词](./1.1%20Tokenization%20分词/1.1%20Tokenization%20分词.md)
3. [Embedding / 词嵌入](./1.2%20Embedding%20词嵌入/1.2%20Embedding%20词嵌入.md)
4. [Attention / 注意力](./1.3%20Attention%20注意力/1.3%20Attention%20注意力.md)
5. [FFN、残差、层归一化](./1.4%20FFN%20%26%20Add%20%26%20LN%20前馈层、残差、层归一化/1.4%20FFN%20%26%20Add%20%26%20LN%20前馈层、残差、层归一化.md)
6. [Positional Encoding / 位置编码](./1.5%20Positional%20Encoding%20位置编码/1.5%20Positional%20Encoding%20位置编码.md)

---

## 🏗️ 第二部分：训练范式（从表示学习到能力对齐）

1. [Pre-training / 预训练](./2.%20Pre-training%20预训练/2.%20Pre-training%20预训练.md)
2. [预训练定义](./2.1%20预训练定义/2.1%20预训练定义.md)
3. [预训练数据](./2.2%20预训练数据/2.2%20预训练数据.md)
4. [预训练流程](./2.3%20预训练流程/2.3%20预训练流程.md)
5. [预训练评估](./2.4%20预训练评估.md)
6. [Post-training / 后训练](./3.%20Post-training%20后训练/3.%20Post-training%20后训练.md)
7. [SFT 监督微调](./3.1%20SFT%20监督微调/3.1%20SFT%20监督微调.md)
8. [RL 强化学习基础](./3.2%20RL%20强化学习基础/3.2%20RL%20强化学习基础.md)
9. [RLHF](./3.3%20RLHF%20基于人类反馈的强化学习/3.3%20RLHF%20基于人类反馈的强化学习.md)
10. [DPO 直接偏好优化](./3.4%20DPO%20直接偏好优化/3.4%20DPO%20直接偏好优化.md)
11. [PEFT 参数高效微调](./3.5%20PEFT%20参数高效微调/3.5%20PEFT%20参数高效微调.md)
12. [大模型自动评估](./3.6%20大模型自动评估.md)

---

## 🤖 第三部分：模型谱系（从经典 LLM 到 MLLM）

1. [Common Models / 常见模型](./4.%20Common%20Models%20常见模型/4.%20Common%20Models%20常见模型.md)
2. [BERT 及变体](./4.1%20BERT及变体/4.1%20BERT及变体.md)
3. [PaLM 系列](./4.2%20PaLM系列/4.2%20PaLM系列.md)
4. [GPT 系列](./4.3%20GPT系列/4.3%20GPT系列.md)
5. [LLama 系列](./4.4%20LLama系列/4.4%20LLama系列.md)
6. [GLM 系列](./4.5%20GLM系列/4.5%20GLM系列.md)
7. [Qwen 系列](./4.6%20Qwen系列/4.6%20Qwen系列.md)
8. [DeepSeek 系列](./4.7%20Deepseek系列/4.7%20Deepseek系列.md)
9. [MoE 系列](./4.8%20MOE系列/4.8%20MOE系列.md)
10. [其他系列](./4.9%20其他系列.md)
11. [多模态大模型发展脉络](./docs/多模态大模型发展脉络.md)
12. [多模态大模型核心能力边界](./docs/多模态大模型核心能力边界.md)

---

## ⚙️ 第四部分：训练与推理系统（从模型到系统）

1. [训练推理优化](./5.%20Training%20%26%20Inferring%20训练推理优化/5.%20Training%20%26%20Inferring%20训练推理优化.md)
2. [显存占用分析](./5.1%20训练推理显存占用分析.md)
3. [FlashAttention 原理](./5.2%20FlashAttention原理/5.2%20FlashAttention原理.md)
4. [PageAttention 原理](./5.3%20PageAttention原理/5.3%20PageAttention原理.md)
5. [训练框架](./5.4%20训练框架/5.4%20训练框架.md)
6. [推理框架](./5.5%20推理框架/5.5%20推理框架.md)
7. [推理耗时及优化](./5.6%20推理耗时及优化/5.6%20推理耗时及优化.md)
8. [Packing 技巧](./5.7%20大模型的packing技巧/5.7%20大模型的packing技巧.md)
9. [训练与推理扩展的转变](./docs/训练与推理扩展的转变.md)

---

## 🎥 第五部分：多模态与应用能力（从单模型到任务系统）

1. [Application / 大模型应用](./6.%20Application%20大模型应用/6.%20Application%20大模型应用.md)
2. [Prompt 技术](./6.1%20Prompt%20Tech%20提示技术/6.1%20Prompt%20Tech%20提示技术.md)
3. [LLM-based Agent](./6.2%20LLM-based%20Agent%20基于大模型的智能体/6.2%20LLM-based%20Agent%20基于大模型的智能体.md)
4. [RAG 检索增强生成](./6.3%20RAG%20检索增强生成/6.3%20RAG%20检索增强生成.md)
5. [RAG 优化工作](./6.4%20RAG优化工作推荐/6.4%20RAG优化工作推荐.md)
6. [Deep Research](./6.5%20Deep%20Research工作梳理及推荐/6.5%20Deep%20Research工作梳理及推荐.md)
7. [利用大模型进行数据打标](./6.6%20利用大模型进行数据打标.md)
8. [多模态融合方法](./docs/多模态融合方法.md)
9. [基于知识图谱的多模态数据分析](./docs/基于知识图谱的多模态数据分析.md)
10. [多模态大模型幻觉](./docs/多模态大模型幻觉.md)
11. [多模态大模型可解释性](./docs/多模态大模型可解释性.md)
12. [MLLM 的计算瓶颈](./docs/MLLM的计算瓶颈.md)

---

## 🗜️ 第六部分：模型压缩与部署（从能力到成本）

1. [Compression / 模型压缩](./7.%20Compression%20模型压缩（了解）/7.%20Compression%20模型压缩（了解）.md)
2. [Quantization / 模型量化](./7.1%20Quantization%20模型量化/7.1%20Quantization%20模型量化.md)
3. [Pruning / 模型剪枝](./7.2%20Pruning%20模型剪枝/7.2%20Pruning%20模型剪枝.md)
4. [Knowledge Distillation / 知识蒸馏](./7.3%20Knowledge%20Distillation%20知识蒸馏/7.3%20Knowledge%20Distillation%20知识蒸馏.md)
5. [Low-Rank Factorization / 低秩分解](./7.4%20Low-Rank%20Factorization%20低秩分解.md)

---

## 📄 第七部分：论文阅读与技术跟踪（从体系到前沿）

1. [论文阅读笔记](./8.%20Papers%20论文阅读笔记.md)
2. [技术报告详解](./8.1%20技术报告详解.md)
3. [o1 技术路线论文](./8.2%20o1技术路线论文.md)
4. [Prompt 相关论文](./8.3%20Prompt技术.md)
5. [自建评估体系](./8.4.1%20自建评估体系.md)
6. [自训 Critical 模型](./8.4.2%20自训Critical模型.md)
7. [调用强基座模型](./8.4.3%20调用强基座模型.md)
8. [拓展阅读](./8.5%20拓展阅读.md)
9. [工作推荐系列](./8.6%20工作推荐系列.md)
10. [25年5月LLM reasoning工作推荐](./8.6.1%2025年5月LLM%20reasoning工作推荐.md)

---

## 🛠️ 第八部分：实践与复现（从理解到落地）

1. [Practice / 大模型实践内容](./9.%20Practice%20大模型实践内容.md)
2. [nanoGPT：从头训练一个 GPT](./9.1%20nanoGPT：从头训练一个GPT.md)
3. [OpenRLHF 微调教程及代码解析](./9.2%20OpenRLHF微调教程及代码解析.md)
4. [从零开始训练大模型（预训练篇）](./9.4%20从零开始训练大模型（预训练篇）.md)
5. [从零开始训练大模型（ReFT篇）](./9.5%20从零开始训练大模型(ReFT篇).md)
6. [从零开始训练大模型（SFT篇）](./9.6%20从零开始训练大模型（SFT篇）.md)
7. [从零开始训练大模型（DPO篇）](./9.7%20从零开始训练大模型（DPO篇）.md)
8. [从零开始训练大模型（蒸馏篇）](./9.10%20从零开始训练大模型（蒸馏篇）.md)
9. [白盒蒸馏 DeepSeek R1 32B](./9.11%20白盒蒸馏DeepSeek%20R1%2032B.md)
10. [Qwen2.5-Math-PRM 代码复现](./9.3%20Qwen2.5-Math-PRM代码复现.md)
11. [基于MCP实现的企业内部工具agent工作流](./9.12%20基于MCP实现的企业内部工具agent工作流.md)
12. [基于vllm+fschat的数据合成Agent框架实现](./9.13%20基于vllm+fschat的数据合成Agent框架实现.md)

---

## 🧩 第九部分：附录（面试 / 八股 / 入门导航）

1. [Interview / 大模型面试八股](./10.%20Interview%20大模型面试八股.md)
2. [Python 相关八股](./10.1%20Python相关八股.md)
3. [实习校招秋招春招面经合集](./10.2%20实习校招秋招春招面经合集/10.2%20实习校招秋招春招面经合集.md)
4. [Transformer 量化分析计算](./10.3%20Transformer量化分析计算/10.3%20Transformer量化分析计算.md)
5. [八股面经（含答案）](./10.4%20八股面经（含答案）/10.4%20八股面经（含答案）.md)
6. [面试手撕专题](./10.5%20面试手撕专题/10.5%20面试手撕专题.md)
7. [机器学习 / 深度学习八股](./10.6%20机器学习深度学习八股（初稿）/10.6%20机器学习深度学习八股（初稿）.md)
8. [AGENT 专项](./11.%20AGENT专项.md)
9. [tool-use 数据合成](./11.1%20tool-use数据合成/11.1%20tool-use数据合成.md)
10. [Agentic 能力优化策略](./11.2%20Agentic能力优化策略/11.2%20Agentic能力优化策略.md)
11. [Agentic 入门项目推荐](./11.3%20Agentic入门项目推荐.md)



# 📚 多模态大模型
1. [BEiT](./BEiT/BEiT.md)
2. [BEiT-2](./BEiT-2/BEiT-2.md)
3. [BEiT-3](./BEiT-3/BEiT-3.md)
4. [BLIP](./BLIP/BLIP.md)
5. [BLIP-2](./BLIP-2/BLIP-2.md)
6. [CLIP](./CLIP/CLIP.md)
7. [Deepseek-VL](./Deepseek-VL/Deepseek-VL.md)
8. [Deepseek-VL2](./Deepseek-VL2/Deepseek-VL2.md)
9. [EVA](./EVA/EVA.md)
10. [EVA-CLIP](./EVA-CLIP/EVA-CLIP.md)
11. [Flamingo](./Flamingo/Flamingo.md)
12. [ImageBind](./ImageBind/ImageBind.md)
13. [InstructBLIP](./InstructBLIP/InstructBLIP.md)
14. [Janus](./Janus/Janus.md)
15. [Kimi-VL](./Kimi-VL/Kimi-VL.md)
16. [LLaMA-VID](./LLaMA-VID/LLaMA-VID.md)
17. [LLaVA](./LLaVA/LLaVA.md)
18. [LLaVA-1.5](./LLaVA-1.5/LLaVA-1.5.md)
19. [LLaVA-Next](./LLaVA-Next/LLaVA-Next.md)
20. [LLaVA-Video](./LLaVA-Video/LLaVA-Video.md)
21. [MVP](./MVP/MVP.md)
22. [Qwen-VL](./Qwen-VL/Qwen-VL.md)
23. [Qwen2-VL](./Qwen2-VL/Qwen2-VL.md)
24. [Qwen2.5-VL](./Qwen2.5-VL/Qwen2.5-VL.md)
25. [Qwen2.5-Omni](./Qwen2.5-Omni/Qwen2.5-Omni.md)
26. [SigLIP](./SigLIP/SigLIP.md)
27. [Video-LLaMA](./Video-LLaMA/Video-LLaMA.md)
28. [ViT](./ViT/ViT.md)
29. [从零构建多模态RAG对话系统](./从零构建多模态RAG对话系统/从零构建多模态RAG对话系统.md)
30. [从零开始训练多模态大模型（微调篇）](./从零开始训练多模态大模型（微调篇）/从零开始训练多模态大模型（微调篇）.md)
31. [大模型笔记代码及简历模板下载](./大模型笔记代码及简历模板下载/大模型笔记代码及简历模板下载.md)
32. [多模态综述](./多模态综述/多模态综述.md)
