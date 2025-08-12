# Literature Review – Sustainable AI

## 1. Introduction
The rapid proliferation of large-scale AI models has raised significant concerns about their environmental impact. Landmark studies like Strubell et al. (2019) revealed that training a single NLP model could emit as much carbon as five cars over their lifetimes. With the growing use of transformer-based models such as GPT and BERT, reducing their energy footprint is becoming critical for sustainable development and responsible AI.

This project aims to explore the carbon and energy consumption of prompt-level inference using large language models (LLMs) and proposes an optimization pipeline for energy-efficient prompting.


## 2. Environmental Impact of AI Workloads

### 2.1 Carbon Emissions from Training & Inference
Patterson et al. (2021) compared carbon emissions across hardware types, cloud regions, and model sizes. Training models like GPT-3 consumed **hundreds of MWh**, making them environmentally unsustainable for repeated fine-tuning.

While training impact has been well studied, **inference** (especially in production-scale applications) also contributes significantly to carbon emissions and is often overlooked. Optimizing inference can therefore yield measurable sustainability gains.


## 3. Tools for Tracking Energy and Carbon Emissions

Several tools and libraries have emerged to monitor AI's carbon footprint:

- **[CodeCarbon](https://mlco2.github.io/codecarbon/)** (Lacoste et al., 2021): A lightweight Python package to estimate carbon emissions based on hardware, region, and runtime.
- **CarbonTracker**: Logs energy consumption over time and estimates carbon emissions during training.
- **[MLCO2 Impact Estimator](https://mlco2.github.io/impact/#compute)**: An interactive tool to approximate emissions from compute hours, GPU type, and cloud region.

However, these tools primarily track **training-time** energy use, with limited support for **real-time inference-level prompt assessment**.


## 4. Prompt Optimization for Efficient Inference

Recent studies suggest that prompt formulation can significantly impact token count and inference time. For instance, Li et al. (2023) showed that rephrased prompts with fewer tokens led to **10–15% faster inference** with minimal accuracy loss.

Hugging Face’s `tokenizers` library and tools like `textstat` can be used to quantify prompt complexity via:

- **Token Count** – A proxy for compute cost.
- **Readability Score** – Simplified prompts can reduce model perplexity and speed up response.

Embedding-based tools like **Sentence Transformers** and **BARTScore** (Yuan et al., 2021) can rank paraphrased prompts by semantic similarity, allowing selection of efficient alternatives without compromising intent.



## 5. Research Gap and Project Relevance

### 5.1 Identified Gap
While there are tools for measuring energy use during training, **very few frameworks offer real-time, prompt-level energy transparency during inference**. Additionally, prompt optimization is still largely **manual and heuristic**.

### 5.2 Project Relevance
This project fills the gap by integrating:

- Prompt complexity analysis using NLP tools
- Carbon estimation using lightweight tracking libraries
- A GUI-based prompt optimizer for real-time feedback
- A simulation-based module to optimize workload distribution in cloud environments


## 6. References

- Strubell, E., Ganesh, A., & McCallum, A. (2019). *Energy and Policy Considerations for Deep Learning in NLP*. arXiv:1906.02243  
- Patterson, D., Gonzalez, J., Le, Q., Liang, C., Munguia, L. M., Rothchild, D., ... & Dean, J. (2021). *Carbon Emissions and Large Neural Network Training*. arXiv:2104.10350  
- Lacoste, A., Luccioni, A., Schmidt, V., & Dandres, T. (2021). *Quantifying the Carbon Emissions of Machine Learning*. Journal of Machine Learning Research, 21  
- Yuan, W., Neubig, G., & Liu, P. (2021). *BARTScore: Evaluating Generated Text as Text Generation*. arXiv:2106.11520  
- Li, T., Qin, Z., & Chen, Y. (2023). *Prompt Compression for Efficient Inference*. NeurIPS Workshop