<img src="https://zzh123-1325455460.cos.ap-nanjing.myqcloud.com/%E4%B8%AA%E4%BA%BA%E7%AE%80%E5%8E%86_01.png"/>



Hey，我是周子涵（Summer），就读于浙江大学智慧能源专业。

这里会存放我的学习笔记，记录一些有趣的想法和尝试。欢迎你常来玩~

## 参与项目

### AI+能源: "能小问"综合能源系统领域教学大模型

项目描述：针对综合能源系统的复杂性与多学科交叉特性，开发面向综合能源系统的大模型研究助手能小问（EnerAgentic），为该领域的复杂数据分析、知识检索和仿真建模提供全流程的智能辅助。

<img src="https://zzh123-1325455460.cos.ap-nanjing.myqcloud.com/%E6%8A%80%E6%9C%AF%E8%B7%AF%E7%BA%BF%E5%9B%BE2_02.png"/>

- **数据集构建**：使用**信息计量学方法**收集学术文献、专业教材和仿真代码，并设计生成器、检验器智能体自动化构建了一个包含约**5.6w条**高质量样本的微调指令数据集。
- **模型微调**：使用 LoRA 对 Qwen3-14B-Base 基座模型微调训练，使用 **NEFTune + Dropout** 方法提高模型泛化能力，使用 **预热与余弦退火** 相结合确保训练稳定性。
- **检索增强生成**：使用GraphRAG方法建立领域知识图谱，结合PAPTOR递归检索技术搭建RAG框架，增强回复的准确性、可追溯性，提升复杂查询的检索效果。

相关成果以第一作者发表在SCI期刊 *International Journal of General Systems* 上，[Eneragentic: multi-agent large language models for assisting scientific research tasks in integrated energy systems](https://www.tandfonline.com/doi/full/10.1080/03081079.2026.2663918)，并获得首届全国大学生“启真问智”人工智能模型&智能体大赛特等奖、第一届全国“AI+能源”大学生科技创新竞赛一等奖。未来将进一步研究综合能源系统领域的多智能体架构等。

[模型开源访问地址](http://zjua4e.com:3000/)

### AI+能源: 能源领域前沿技术检索项目

当前能源领域的前沿资料主要依赖手动搜索，覆盖范围有限，尤其缺乏国外专家与机构的信息，体系化程度不高。我们希望利用大模型技术构建知识库，整合访谈数据与公开信息，形成支持智能分析的能源战略研究平台。

本产品具备以下功能：
1、自动化网络检索，补充被遗漏的专家与单位；
2、构建知识图谱或思维导图，形成结构化知识体系；
3、对访谈内容进行文本分析与特征提取，生成人才画像；
4、支持自然语言问答，便于直接利用研究成果。
此举可改变传统的研究模式，提升效率与检索效果。


<img src="https://zzh123-1325455460.cos.ap-nanjing.myqcloud.com/%E5%B7%A5%E9%99%A2AI%E7%9F%A5%E8%AF%86%E6%A3%80%E7%B4%A2%E9%A1%B9%E7%9B%AEdemo.gif"/>



### CCUS项目: 多级溶剂循环工艺增强 SO₂和 CO₂协同捕集

以第六作者身份在 **Chemical Engineering Journal** 发表[论文](https://www.sciencedirect.com/science/article/abs/pii/S1385894724087679) **Enhanced SO2 and CO2 synergistic capture with reduced NH3 emissions using multi-stage solvent circulation process** 

![CCUS](https://zzh123-1325455460.cos.ap-nanjing.myqcloud.com/20250831161341.png)

创新点:
- 创新使用**MSC**工艺实现多环节协同捕集; 传统氨法捕集将 SO₂去除、CO₂吸收、氨控制视为独立环节，易导致氨逃逸、溶剂浪费和能耗叠加, 而本研究设计的MSC 工艺通过 “功能分区 + 溶液循环” 实现了多环节协同：
- 通过**溶液循环与参数调控**降低氨逃逸, 提升SO2捕集效率; 
- 结合**模拟与中试设计**，兼顾实验室性能与工业化落地, 提升能耗经济性; 
- 选用**NH₃/K₂CO₃混合溶剂**替代单一氨溶液或胺类溶剂, 降低溶液粘度, 提升CO₂传质效率; 降低所需氨浓度，从源头减少氨逃逸; 混合溶剂的循环容量更大，解析能耗更低. 

## 学生工作经历

- 浙江大学能源工程学院第四十九届学生会主席团成员
- 浙江大学学业指导与促进中心学业发展部部长