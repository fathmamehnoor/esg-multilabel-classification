# ESG Multi-Label Classification with BERT

This project fine-tunes a BERT transformer model to classify textual content from corporate sustainability reports into multiple ESG (Environmental, Social, and Governance) categories. It demonstrates the use of multi-label NLP classification for real-world sustainability analysis and responsible AI applications.

Model Deployed on Hugging Face: [me-r/esg-multilabel-classification](https://huggingface.co/me-r/esg-multilabel-classification)

---

## Overview

Sustainability reports often contain complex narratives addressing environmental policies, governance ethics, and social practices. This project applies Natural Language Processing (NLP) techniques to automatically categorize such text into 27 ESG dimensions, enabling scalable, data-driven sustainability analysis.

The model achieves an F1 Score of 0.9501 on the validation set, demonstrating strong generalization for nuanced ESG-related language.

## Key Features

- **Transformer-based NLP:** Fine-tuned [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) from [Hugging Face Transformers](https://huggingface.co/transformers).

- **Multi-Label Classification:** Handles texts that span multiple ESG categories simultaneously.

- **Custom PyTorch Pipeline:** Built from scratch with `Dataset`, `DataLoader`, and training/evaluation loops.

- **Automated Label Encoding:** Uses `MultiLabelBinarizer` to encode and decode ESG labels.

- **Deployed Model:** Accessible via [Hugging Face](https://huggingface.co) for real-time inference.

---
## Technical Implementation

**Tech Stack:**

- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **Scikit-learn**
- **Pandas / NumPy**


**Dataset and Preprocessing:**

The dataset was curated from corporate sustainability and annual reports, with each text segment annotated across multiple **ESG dimensions** (e.g., *Energy Management*, *GHG Emissions*, *Labor Practices*).

A custom preprocessing pipeline was built for **multi-label NLP classification**, including:

- **Label encoding** using `MultiLabelBinarizer`
- **Train/validation stratified splitting**
- **Tokenization** with the **BERT tokenizer** (dynamic padding, truncation to 256 tokens, and attention mask generation)

This ensures consistent, high-quality input representation for fine-tuning the transformer model.


**Model Architecture:**

- **Base model:** `BertForSequenceClassification`
- **Problem type:** `multi_label_classification`
- **Optimizer:** `AdamW`
- **Learning rate scheduler:** Linear warmup
- **Evaluation metric:** F1-score (micro-average)
  
**Training Configuration:**

| **Parameter**           | **Value** |
|--------------------------|-----------|
| Max sequence length      | 256       |
| Batch size               | 4         |
| Epochs                   | 4         |
| Learning rate            | 2e-5      |
| Random seed              | 42        |


---
## Research Significance

This work demonstrates how language models can be used for automated sustainability analytics, aligning machine learning with social impact research.  
It bridges NLP and environmental data science, offering a foundation for:

- **ESG risk assessment automation**
- **Sustainable investment analysis**
- **Responsible AI applications** in finance and policy research

## Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

## Inference

You can run predictions on new text like this:

```bash
python predict.py
```

## Inference Example

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch, json

# Load model and tokenizer
model_name = "me-r/esg-multilabel-classification"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Load label encoder
with open("esg_label_encoder.json", 'r') as f:
    labels = json.load(f)['classes']

# Example text
text = "Our board has introduced stricter ethical guidelines, improved transparency
        in business operations, and enhanced diversity across executive roles."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)

# Predict probabilities
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).numpy()[0]

# Apply threshold to get predicted labels
predicted_labels = [labels[i] for i, p in enumerate(probs) if p > 0.5]
print(predicted_labels)
```

**Output:**
```
Predicted ESG Categories: ['Employee_Engagement_Inclusion_And_Diversity']
```

You can modify example_text in predict.py with your own input.