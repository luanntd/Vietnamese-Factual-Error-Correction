# **Vietnamese-Factual-Error-Correction**
**Factual Error Correction (FEC)** is a critical task in natural language processing, aimed at detecting and correcting factual inconsistencies in textual content. Although significant progress has been made in **English**, **Vietnamese FEC** remains under-explored due to the lack of annotated datasets and tailored approaches. This project presents a novel approach to **Vietnamese FEC** based on the baseline paper [**Zero-shot Faithful Factual Error Correction**](https://aclanthology.org/2023.acl-long.311/). This implementation is one of the first steps for this task in **Vietnamese**. Through extensive experiments, we evaluate our approach with human judgments, analyze its strengths and weaknesses, and discuss its applicability in real-world scenarios. Our findings contribute to the development of robust FEC systems for low-resource languages like **Vietnamese**.
![output_example](./output_example.png?size=50)

### **Modules**
This method follows a zero-shot approach, requiring no training in this implementation. However, its multi-component architecture results in substantial inference time and high memory consumption. It is important to consider the associated costs when running the system. The tools and LLMs used in this implementation are listed below:
- **Claim Answer Generator**: [Stanza](https://aclanthology.org/2020.acl-demos.14/)
- **Question Generator**: Model available at [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
- **Question Answering**: Model available at [ViT5-VQA](https://huggingface.co/PhucDanh/vit5-fine-tuning-for-question-answering)
- **QA-to-Claim**: Model available at [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
- **Correction Scoring**: Model available at [PhoBERT](https://huggingface.co/vinai/phobert-base-v2), [ROUGE](https://aclanthology.org/W04-1013/)

### **Dependencies**
All the required packages are listed in requirements.txt. To install all the dependencies, run
```bash
pip install -r requirements.txt
```

## **Dataset**
We conduct experiments on a Vietnamese dataset. [ViFactCheck](https://arxiv.org/abs/2412.15308), the first publicly available benchmark dataset designed specifically for Vietnamese fact-checking across multiple online news domains. This dataset contains 7,232 human-annotated pairs of claim-evidence combinations sourced from reputable Vietnamese online news, covering 12 diverse topics. We re-purpose it for the factual error correction task by removing claims that not enough information compared to evidence and taking all claims supported or refuted. Dataset available at HuggingFace: [ViFactCheck-Dataset](https://huggingface.co/datasets/tranthaihoa/vifactcheck_gold_evidence).

| **Features**      | **Description** |
|---------------|-----------------|
| Statement | Input claim |
| Context | Full evidence |
| Topic | Topic of context |
| Author | Newspaper source |
| Url | Online newspaper url |
| Evidence | Retrieval evidence |
| Label | label of input claim |

## **How to use**
First, you need to load the model and dataset.
```python
from utils.dataset import ViFactCheck
from model.vi_zerofec import Vi_ZeroFEC

# Load model
corrector = Vi_ZeroFEC()

# Load all dataset
vifactcheck = ViFactCheck("dataset_url")
dataset = vifactcheck.get_all()
```
You can process one sample by:
```python
sample = {
    'evidence': 'Giải trình sau đó, về quy định sở hữu nhà chung cư như dự thảo, Bộ trưởng Bộ Xây dựng ...', 
    'input_claim': 'Khi chung cư bị tiêu hủy thì các giấy tờ sở hữu chung cư vẫn còn hiệu lực.', 
    'label': 'Refuted'
}
output = corrector.correct(sample)
```
Or process on batch via:
```python
samples = dataset[0:500]
outputs = corrector.batch_correct(samples, "save_dir")
```
All of these are sourced from `main.py`.

## **Human Evaluation**
We split annotation process into **three round**, after each round we calculate **Cohen’s Kappa** and revised the guideline if necessary. We maintain frequent communication with each other, including answering any possible questions and resolving mismatch issues, to facilitate the evaluation process.

## **Contributors**

<a href="https://github.com/luanntd">
  <img src="https://github.com/luanntd.png?size=50" width="50" style="border-radius: 50%;" />
</a>
<a href="https://github.com/Khoa-Nguyen-Truong">
  <img src="https://github.com/Khoa-Nguyen-Truong.png?size=50" width="50" style="border-radius: 50%;" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>
