# GSoC 2026 — ML4SCI SYMBA

## Foundation Models for Squared Amplitude Calculation

**Sairam Chennaka** | Junior Research Fellow, MNR University x IHub-Data, IIIT Hyderabad
**Project:** [SYMBA3 — Foundation Models for Squared Amplitude Calculation](https://ml4sci.org/gsoc/2026/proposal_SYMBA3.html)
**Organization:** ML4SCI / University of Alabama

---

## Overview

This repository contains the evaluation task solutions for the GSoC 2026 ML4SCI SYMBA project. The goal is to build physics-informed foundation models for automated **squared amplitude calculation** in particle physics — predicting symbolic squared amplitudes from Feynman diagram amplitudes using seq2seq transformers.

Two novel physics-informed techniques are proposed and implemented:

1. **Factored Amplitude Encoding** — Labels the propagator and amplitude components with physics type tokens (`[PROP]`, `[AMP]`), giving the encoder structured access to the propagator before processing the full amplitude. Applied to QED inputs.

2. **Denominator Pre-filling** — Injects the known propagator denominator as a structured prefix (`[DENOM] ... [NUMER] ...`) in the target, allowing the model to focus generative capacity on the harder numerator polynomial. Applied to both QED and QCD targets, this structural hint significantly reduces denominator prediction errors.

---

## Task Solutions

| Notebook | Task | Description |
|---|---|---|
| `task1_2_preprocessing.ipynb` | Task 1.2 | Physics-informed preprocessing, Mandelstam-safe index normalization, greedy atomic tokenization, 80-10-10 dataset splits |
| `task2_3_foundation_model.ipynb` | Task 2.3 | T5-style encoder-decoder trained from scratch, QED + QCD training, full evaluation suite |

---

## Results

| Model | Exact Match | Token Accuracy | BLEU Score | Num. Error <1% | Train Samples |
|---|---|---|---|---|---|
| QED 2-to-2 | 19.44% (7/36) | 0.6031 | 0.8689 | 22.2% | 288 |
| QCD 2-to-2 | 0.00% (0/19) | 0.1145 | 0.5495 | 0.0% | 148 |

> **Context:** The published SYMBA baseline (Alnuqaydan et al., 2023/2024) used 251K QED and 205K QCD training samples — approximately 900× more than this evaluation dataset. The performance gap is consistent with the authors' own finding that results are data-limited.

---

## Physics-Informed Design Decisions

### Tokenization

All Mandelstam invariants (`s_12`, `s_13`, `s_14`, `s_23`, `s_24`, `s_34`), mass symbols (`m_e`, `m_mu`, `reg_prop`, etc.), and couplings (`g²`, `g⁴`, `e²`, `e⁴`) are treated as **atomic tokens** via greedy longest-match tokenization. Standard BPE would split `s_12` → `['s','_','1','2']` (4 tokens), destroying the Lorentz invariant identity. Vocabulary size: **83 tokens**.

### Mandelstam-Safe Index Normalization

FeynCalc bookkeeping indices (`%gam_115`, `i_31`) are normalized to sequential integers, with Mandelstam variables protected via placeholder substitution before renaming and restored afterwards.

### Encoding (QED)

```
Input: [PROP] s_12 + 1/2*reg_prop  [AMP]  -1/2*i*e^2*gamma_{...}/(s_12+...)
```

### Decoding (QED + QCD)

```
Target: [DENOM] (s_12 + 1/2*reg_prop)^(-2)  [NUMER]  32/81*e^4*s_14*s_34 + ...
```

---

## Model Architecture

| Hyperparameter | Value |
|---|---|
| Architecture | T5-style encoder-decoder (from scratch) |
| d_model | 256 |
| d_ff | 1024 |
| num_heads | 8 |
| num_layers | 4 (encoder + decoder each) |
| d_kv | 32 |
| vocab_size | 83 |
| Total parameters | ~7.37M (7,367,424) |
| max_input (QED/QCD) | 512 / 1024 tokens |
| max_target (QED/QCD) | 256 / 512 tokens |

Training hardware: NVIDIA GeForce RTX 4050 Laptop GPU (6.4 GB VRAM)
Optimizations: fp16 mixed precision, gradient checkpointing, gradient accumulation, early stopping

---

## Setup & Reproduction

### Requirements

```bash
pip install transformers torch scikit-learn pandas numpy matplotlib nltk accelerate
```

### Dataset

Place the 17 SYMBA `.txt` files (QED and QCD 2-to-2 tree-level) in a `./data/` folder:

```
data/
  QED-2-to-2-diag-TreeLevel-0.txt
  QED-2-to-2-diag-TreeLevel-1.txt
  ...
  QCD-2-to-2-diag-TreeLevel-0.txt
  ...
```

### Run Order

```bash
# Step 1 — Preprocessing (generates ./task1_preprocessing/ CSVs)
jupyter notebook task1_2_preprocessing.ipynb

# Step 2 — Foundation model training + evaluation
jupyter notebook task2_3_foundation_model.ipynb
```

Outputs saved: `task1_preprocessing/{qed,qcd}_{train,val,test}.csv`, `results_summary.csv`, `vocab.json`, training loss curves.

---

## References

1. Alnuqaydan, A. et al. (2023). *Symbolic Machine Learning for High Energy Physics Calculations.* NeurIPS ML4PS Workshop 2023. [Paper](https://ml4physicalsciences.github.io/2023/files/NeurIPS_ML4PS_2023_183.pdf)
2. Alnuqaydan, A. et al. (2024). *SYMBA: Symbolic Computation of Squared Amplitudes in High Energy Physics with Machine Learning.* Machine Learning: Science and Technology, IOP. [Paper](https://iopscience.iop.org/article/10.1088/2632-2153/acb2b2)
3. Raffel, C. et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR 21(140).

---

## Links

- [SYMBA3 Project Page](https://ml4sci.org/gsoc/2026/proposal_SYMBA3.html)
- [ML4SCI SYMBA GitHub](https://github.com/ML4SCI/SYMBA)
- [GSoC 2026 ML4SCI](https://ml4sci.org/activities/gsoc2026.html)
