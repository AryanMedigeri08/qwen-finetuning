---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: apache-2.0
tags: [qlora, lora, customer-support, e-commerce, peft]
datasets: [bitext/Bitext-customer-support-llm-chatbot-training-dataset]
---
# SahayakLLM - QLoRA adapter for e-commerce customer support

One LoRA adapter on **Qwen/Qwen2.5-1.5B-Instruct** (4-bit NF4) trained for two tasks: (a) customer-support **response generation** and
(b) **routing** to JSON `{"intent": ..., "category": ...}` (about 20% of records, marked by a separate system prompt).

## Training data
Bitext Customer Support LLM Chatbot Training Dataset (Bitext Innovations, CDLA-Sharing-1.0), cleaned and split with a fixed seed
(42); stratified 80/10/10 over the training intents, plus held-out intents never used for training
(check_invoice, review, switch_account). Data manifest hash: `1182a97b9d392682e8dd80db916a5eb6103057d4752e1d66c79818cd83b67c46`.
Training records: 23,875 (response + routing). KharidiKart is a fictional company.

## Hyperparameters
LoRA r=16, alpha=32, dropout=0.05, all linear projections; lr 0.0002 cosine, 3% warm-up;
batch 4x4=16; 1 epoch(s); max sequence length 480; fp16 compute; paged AdamW 8-bit;
loss on assistant tokens only. Trainable parameters: 18,464,768 (1.18%).

## Results (test split, greedy decoding)
| Config | Intent macro-F1 | BERTScore-F1 | ROUGE-L | Placeholder leakage % |
|---|---|---|---|---|
| B0 zero-shot | 0.501 | 0.874 | 0.214 | 9.4 |
| B1 few-shot | 0.615 | 0.887 | 0.267 | 21.1 |
| **C2 (this adapter)** | 0.999 | 0.923 | 0.445 | 46.9 |

Held-out intents (C2): intent macro-F1 0.348, BERTScore-F1 0.898.

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", quantization_config=bnb, device_map={"": 0})
model = PeftModel.from_pretrained(base, "<this-repo>")
```
Use the exact system prompts of the project (`prompts/system.txt` for responses, `prompts/router_system.txt` for routing).

## Limitations
* Trained on synthetic, templated support dialogues for a **fictional** retailer; not validated on real customers.
* Responses keep `{{placeholder}}` slots instead of real facts; facts must come from tools and be checked by a verifier.
* Generic policies only: the model can state invented policy numbers if it is not given grounded evidence.
* The routing label list is given in the prompt; accuracy on unseen intents is reported separately (held-out set).
* English only; 1.5B-parameter model with 4-bit quantisation; not for production or medical/legal/financial advice.
