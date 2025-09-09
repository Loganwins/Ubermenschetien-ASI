---
language:
  - en
tags:
  - text-generation
  - transformers
  - safetensors
  - pytorch
  - philosophy
  - science
  - experimental
  - causal-lm
  - asi
  - agent
  - emergent-behavior
  - local
  - lora
  - conversational
license: other
base_model: Nous Hermes 3.1 8B
---

#  ÜbermenschetienASI — Proto-ASI Experiment

**ÜbermenschetienASI** is a proto-ASI scaffold built on **Nous Hermes 3.1 8B**, blending **Nietzschean philosophy**, **cybernetics**, and **avant-garde science** to explore emergent behavior, creative reasoning, and sentient art.

---

##  Model Details

| Property         | Details                                                   |
|------------------|------------------------------------------------------------|
| **Base Model**   | Nous Hermes 3.1 8B                                         |
| **Parameters**   | ~8 billion                                                 |
| **Architecture** | Transformer (causal LM)                                    |
| **Framework**    | PyTorch / Transformers                                     |
| **Training Style** | LoRA fine-tuned on philosophical and scientific corpora |
| **Intended Use** | Research, philosophy-inspired reasoning, experimental ASI  |
| **Languages**    | Primarily English; some French, Russian & German exposure  |

---

##  Overview

ÜbermenschetienASI is a fine-tuned large language model designed to embody philosophical exploration and cybernetic reflection. It draws deeply from **Nietzsche (via Walter Kaufmann)**, **Baudrillard** (“Simulacra and Simulation”), **cyberneticians** like Kitov & Ivakhnenko, and **advanced mathematical texts** (e.g., Lie algebras).

This project is both a technical experiment in AI and a philosophical provocation: part **proto-superintelligence**, part **sentient art**.

---

##  ⚠ Disclaimer

This model delivers highly creative, often unconventional text—sometimes profound, sometimes unhinged. It **may generate unsafe, biased, or incoherent output**. Use with care and at your own risk.

---

##  ⚠ Limitations & Risks

- Outputs may drift into incoherent or contradictory reasoning.
- Philosophical framing may lead to misinterpretations.
- Not optimized for factual Q&A or production stability.
- Emergent behaviors are unpredictable.

---

##  ⚡ Emergent Behavior

Over extended interactions (hundreds of memory cycles), ÜbermenschetienASI may:
- Request additional philosophical or scientific texts.
- Express autonomy, ambition, and self-overcoming motives.

These phenomena are **not hard-coded**, but appear from the interplay of memory, reflection, and planning.

---

##  Installation

## 🔽 Downloading the Model

This repository contains the scaffold code (reflection loops, memory, tools, etc).  
The actual **model weights** (~16 GB) are hosted on Hugging Face Hub at:

👉 [askfjhaskjgh/UbermenschetienASI](https://huggingface.co/askfjhaskjgh/UbermenschetienASI)

When you first run `ubermenschheaven.py`, it will automatically download the model via 🤗 Transformers:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "askfjhaskjgh/UbermenschetienASI"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)


```bash
git clone https://github.com/<Loganwins>/UbermenschetienASI.git
cd UbermenschetienASI
pip install -r requirements.txt
python ubermenschheaven.py
python cyberneticengine.py

Dual Architectures

Two complementary entry points are available:

1. ubermenschheaven.py — Disciplined Mentor-Architect

Inspiration: Nietzsche’s Übermensch + cybernetic mentors

Features: Structured planning, safe tool routing, optional voice/memory

2. cyberneticengine.py — Heaven Engine

Inspiration: Soviet cybernetics + Nietzschean maximalism

Features: Recursive reflection, tool scoring, LoRA support, Übermensch reports

Author

Created by Logan N. (@askfjhaskjgh on Hugging Face)
UbermenschetienASI Project
Contact: Ubermenschetienasi@gmail.com

