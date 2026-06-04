<img src="https://zzh123-1325455460.cos.ap-nanjing.myqcloud.com/%E4%B8%AA%E4%BA%BA%E7%AE%80%E5%8E%86_01.png"/>

Hey, I'm Zihan Zhou (Summer), a Smart Energy student at Zhejiang University.

This is where I share my study notes, interesting ideas, and experiments. Feel free to stop by anytime~

## Projects

### AI + Energy: "EnerAgentic," an Educational LLM for Integrated Energy Systems

**Project overview:** Integrated energy systems are complex and highly interdisciplinary. To support research in this field, we developed EnerAgentic, a large language model research assistant that provides end-to-end intelligent support for complex data analysis, knowledge retrieval, and simulation modeling.

<img src="https://zzh123-1325455460.cos.ap-nanjing.myqcloud.com/%E6%8A%80%E6%9C%AF%E8%B7%AF%E7%BA%BF%E5%9B%BE2_02.png"/>

- **Dataset construction:** Used **informetric methods** to collect academic literature, professional textbooks, and simulation code. We then designed Generator and Validator agents to automatically build a fine-tuning instruction dataset containing approximately **56,000 high-quality samples**.
- **Model fine-tuning:** Fine-tuned the Qwen3-14B-Base model with LoRA, improved generalization using **NEFTune + Dropout**, and combined learning-rate warmup with cosine annealing to ensure stable training.
- **Retrieval-augmented generation:** Used GraphRAG to build a domain knowledge graph and incorporated RAPTOR recursive retrieval into the RAG framework, improving response accuracy and traceability as well as retrieval performance for complex queries.

The project resulted in my first-author paper in the SCI-indexed journal *International Journal of General Systems*: [Eneragentic: multi-agent large language models for assisting scientific research tasks in integrated energy systems](https://www.tandfonline.com/doi/full/10.1080/03081079.2026.2663918). It also received the Grand Prize at the inaugural National College Student "Qizhen Wenzhi" AI Model & Agent Competition and First Prize at the inaugural National "AI + Energy" College Student Science and Technology Innovation Competition. Future work will further explore multi-agent architectures for integrated energy systems.

<img src="https://zzh123-1325455460.cos.ap-nanjing.myqcloud.com/%E6%96%B0%E5%BB%BA%20Microsoft%20PowerPoint%20%E6%BC%94%E7%A4%BA%E6%96%87%E7%A8%BF_02.png"/>

[Access the open-source EnerAgentic model](http://zjua4e.com:3000/)

<img src="https://zzh123-1325455460.cos.ap-nanjing.myqcloud.com/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202026-06-04%20171458.png"/>

### AI + Energy: Frontier Technology Retrieval for the Energy Sector

Research on frontier energy technologies currently relies heavily on manual searches, which offer limited coverage and often overlook international experts and institutions. We aim to build a large language model-powered knowledge base that integrates interview data with public information, forming an energy strategy research platform capable of intelligent analysis.

The platform provides the following capabilities:

1. Automated web retrieval to identify overlooked experts and organizations
2. Knowledge graph and mind map generation to create a structured knowledge system
3. Text analysis and feature extraction from interviews to generate expert profiles
4. Natural-language question answering for direct access to research findings

The platform is designed to improve conventional research workflows, search coverage, and efficiency.

<img src="https://zzh123-1325455460.cos.ap-nanjing.myqcloud.com/%E5%B7%A5%E9%99%A2AI%E7%9F%A5%E8%AF%86%E6%A3%80%E7%B4%A2%E9%A1%B9%E7%9B%AEdemo.gif"/>

### CCUS: Enhanced Synergistic SO₂ and CO₂ Capture Using a Multi-Stage Solvent Circulation Process

I am the sixth author of the paper [Enhanced SO2 and CO2 synergistic capture with reduced NH3 emissions using multi-stage solvent circulation process](https://www.sciencedirect.com/science/article/abs/pii/S1385894724087679), published in *Chemical Engineering Journal*.

![CCUS](https://zzh123-1325455460.cos.ap-nanjing.myqcloud.com/20250831161341.png)

**Key innovations:**

- Introduced a **multi-stage solvent circulation (MSC)** process for coordinated capture across multiple stages. Conventional ammonia-based capture treats SO₂ removal, CO₂ absorption, and ammonia control as separate steps, which can lead to ammonia slip, solvent waste, and cumulative energy consumption. Our MSC process coordinates these steps through functional zoning and solvent circulation.
- Reduced ammonia slip and improved SO₂ capture efficiency through **solvent circulation and parameter optimization**.
- Combined **simulation and pilot-scale design** to balance laboratory performance with industrial deployment and improve energy efficiency and economic viability.
- Replaced conventional ammonia solutions or amine solvents with an **NH₃/K₂CO₃ blended solvent**, reducing viscosity, improving CO₂ mass transfer, lowering the required ammonia concentration, increasing cyclic capacity, and reducing regeneration energy consumption.

## Student Leadership Experience

- **Member of the Presidium, 49th Student Union, College of Energy Engineering, Zhejiang University** | April 2025 - April 2026
- **Head of the Academic Development Department, Academic Guidance and Advancement Center, Zhejiang University** | September 2024 - June 2025

<img src="https://zzh123-1325455460.cos.ap-nanjing.myqcloud.com/202606041720079.png"/>

<img src="https://zzh123-1325455460.cos.ap-nanjing.myqcloud.com/202606041726019.png"/>
