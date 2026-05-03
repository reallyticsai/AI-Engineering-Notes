---
tags:
  - AI
  - Foundation Models
  - Modular AI
  - LangChain
  - LoRA
---

# From Monolith to Modular: Building Industry-Specific AI with Composable Foundation Models
![Modular Foundation Models for Customizable AI](../images/modular-foundation-models-for-customizab.jpg)

## TL;DR
* Fine-tune large language models efficiently using LoRA (Low-Rank Adaptation) for industry-specific applications.
* Leverage LangChain to compose multiple models and components into a single production pipeline.
* Build modular foundation models to overcome the limitations of monolithic AI architectures.

## Introduction
The rise of large language models (LLMs) and foundation models has transformed the AI landscape. However, their monolithic architecture can be a significant barrier to adoption in industry-specific applications. The need for customization, efficiency, and scalability has led to the development of modular foundation models. In this article, we'll explore the current state of the art in modular foundation models, their production architecture patterns, and code patterns that practitioners should know.

## Technical Deep Dive
To build industry-specific AI with composable foundation models, we need to fine-tune large models efficiently. LoRA (Low-Rank Adaptation of Large Language Models) is a recent breakthrough that achieves this by updating only a small subset of the model's parameters.

### Fine-Tuning with LoRA
Here's an example code block that demonstrates how to fine-tune a pre-trained BERT model using LoRA with the Hugging Face Transformers library:
```python
import torch
from transformers import BertTokenizer, BertModel
from lora import LoRA

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Initialize LoRA with the pre-trained model
lora = LoRA(model, rank=16)

# Define a custom dataset class for industry-specific data
class IndustryDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        labels = self.data[idx]['labels']

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels)
        }

# Create a dataset instance and data loader
dataset = IndustryDataset(data, tokenizer)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Fine-tune the model using LoRA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lora.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lora.parameters(), lr=1e-5)

for epoch in range(5):
    lora.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = lora(input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')
```
### Composing Models with LangChain
LangChain provides a modular architecture for building AI applications. We can compose multiple models and components into a single pipeline using LangChain's `Chain` API.

Here's an example code block that demonstrates how to compose a LoRA-fine-tuned BERT model with a downstream task-specific model using LangChain:
```python
from langchain import Chain, LLMChain
from langchain.llms import HuggingFacePipeline

# Create a Hugging Face pipeline with the LoRA-fine-tuned BERT model
pipeline = HuggingFacePipeline(
    model=lora,
    tokenizer=tokenizer,
    task='text-classification'
)

# Define a downstream task-specific model
class TaskModel(torch.nn.Module):
    def __init__(self):
        super(TaskModel, self).__init__()
        self.fc = torch.nn.Linear(768, 8)

    def forward(self, x):
        return self.fc(x)

# Create a LangChain instance with the composed model
chain = Chain(
    llm_chain=LLMChain(llm=pipeline),
    task_model=TaskModel()
)

# Use the composed model for inference
input_text = 'This is an example input text.'
output = chain.run(input_text)
print(output)
```
## Architecture Diagram
Our production architecture consists of the following components:

```
+---------------+
|  Model Hub   |
+---------------+
       |
       |
       v
+---------------+
|  LoRA Fine-   |
|  Tuning       |
+---------------+
       |
       |
       v
+---------------+
|  LangChain    |
|  Composition  |
+---------------+
       |
       |
       v
+---------------+
|  Downstream   |
|  Task Model   |
+---------------+
       |
       |
       v
+---------------+
|  Deployment   |
|  (e.g., REST  |
|   API, gRPC)  |
+---------------+
```
## Production Lessons Learned
From our experience, we've learned that:

* Modular foundation models require careful management of model versions and updates.
* LoRA fine-tuning can be sensitive to hyperparameter tuning, such as the rank and learning rate.
* LangChain's modular architecture allows for flexible composition of models and components, but requires careful consideration of data flow and dependencies.

## Key Takeaways
* Modular foundation models offer a promising approach to building industry-specific AI applications.
* LoRA fine-tuning enables efficient adaptation of large language models to specific tasks and datasets.
* LangChain provides a modular architecture for composing multiple models and components into a single production pipeline.

## Further Reading
* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
* [LangChain Documentation](https://langchain.readthedocs.io/en/latest/)
* [Hugging Face Transformers Library](https://huggingface.co/docs/transformers/index)

By Reallytics AI