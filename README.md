# Entropy Preservation in Distilled Reinforcement Learning

**An Empirical Characterization for Small Language Models**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Code, training configurations, and evaluation pipeline for a Master's thesis on **Distilled Reinforcement Learning (DistilledRL)** ([Wang et al., 2026](https://arxiv.org/abs/2607.17247)), using a Llama-3.2-1B-Instruct student and a DeepSeek-R1-Distill-Llama-8B teacher.

**Contents:** [Overview](#overview) · [Key findings](#key-findings) · [Confidence-gated variant](#the-confidence-gated-variant) · [Results](#results) · [Setup](#setup) · [Training](#training) · [Gate calibration](#gate-calibration) · [Reproducibility](#reproducibility) · [Checkpoints](#checkpoints) · [Citation](#citation)

---

## Overview

The thesis asks one empirical question:

> **When a teacher signal reweights an RL learning signal instead of acting as an imitation target, what happens to the shape of the student's policy distribution, and how does that shape affect downstream coverage?**

It compares three training objectives under matched hyperparameters (DistilledRL, KL-regularized GRPO, KL-free GRPO), evaluates them on four benchmarks (GSM8K test, GSM-Symbolic, SVAMP, MATH-500), and uses a **confidence-gated variant** of the teacher weight as an independent probe of the proposed mechanism.

---

## Key findings

| Objective | Entropy behaviour | pass@1 | pass@32 |
|---|---|---|---|
| **KL-free GRPO** | Rapid collapse; rewarded format lost | Collapses | Collapses |
| **GRPO + KL** | Moderated, but still converges to low entropy | **Best** | Does not improve coverage |
| **DistilledRL** | Stays in a mid-range band throughout training | Improves | **Best** |

*Qualitative summary based on GSM8K test; numbers are in [Results](#results).*

- **DistilledRL's coverage advantage is clear on SVAMP, GSM8K test, and GSM-Symbolic, and is not measurable on MATH-500.** Over the base model, pass@32 improves by +9.6, +7.9, and +8.4 points on the three benchmarks respectively (DistilledRL, step 60).
- **Entropy preservation is the proposed mechanism.** DistilledRL keeps the policy in a mid-range entropy band (~1.1–1.3 nats), which we argue preserves access to lower-probability correct trajectories under repeated sampling and drives the coverage gain.
- **LoRA rank is not the limiting factor.** Increasing adapter rank from 16 to 64 does not change the MATH-500 outcome, ruling out adapter capacity as the primary constraint.

---

## The confidence-gated variant

The central theoretical contribution of this thesis is a **confidence-gated variant of the DistilledRL weight**, designed as an *independent probe* of the entropy-preservation account rather than as a proposed method.

The gate attenuates teacher influence at tokens where the teacher's next-token distribution is uncertain:

$$
w_{i,t} = 1 + c_{i,t}\left(\widetilde{\rho}_{i,t} - 1\right), \qquad
c_{i,t} = \operatorname{sigmoid}\!\left(\frac{H^\star - H_{\text{teacher},i,t}}{\tau}\right)
$$

where $H_{\text{teacher},i,t}$ is the teacher's per-token entropy, $H^\star$ is a calibrated entropy threshold, and $\tau$ is a transition scale. When the teacher is confident, $c \to 1$ and the variant recovers the original DistilledRL weight. When the teacher is uncertain, $c \to 0$ and the teacher's influence collapses toward the ordinary RL update.

### Calibration

$H^\star = 1.574$ (75th percentile of teacher entropy on base-student rollouts) and $\tau = 0.348$ (one quarter of the standard deviation).

The gate was validated before training:

- The Pearson correlation between $c$ and the teacher-ratio deviation is **+0.5217**.
- The risk/benefit ratio (fraction of tokens where the gate blocks a strong teacher signal vs. amplifies one) is **0.062**, so the gate preferentially amplifies rather than suppresses the teacher's strong preferences.

### Result

The gated run produced the co-movement predicted by the entropy-preservation account:

- Mean policy entropy fell by ~0.3–0.4 nats relative to ungated DistilledRL.
- GSM8K pass@32 fell by 3.8 percentage points (82.3% → 78.5%, $p = 0.0005$).
- GSM-Symbolic pass@1 improved significantly, making the gated variant the only one in the study to significantly exceed the base model on this benchmark (42.4% vs. 37.3%, $p = 0.0097$).

This is the strongest evidence in the thesis that entropy preservation contributes to the coverage benefit: an intervention built to reduce teacher influence at uncertain tokens lowered entropy, and coverage fell with it.

---

## Results

### Training dynamics

KL-free GRPO undergoes entropy collapse within the first 15–20 optimization steps and stops producing the rewarded `\boxed{}` format. GRPO + KL moderates the collapse but settles at a low entropy level. DistilledRL (V12) stays in a mid-range band (~1.1–1.3 nats) throughout training, with a transient excursion around step 33 that coincides with a temporary increase in response length.

### GSM8K test (1319 questions)

| Model | Step | pass@1 | pass@32 |
|---|---|---|---|
| Base | — | 46.1% | 74.4% |
| GRPO (no KL) | 40 | 29.7% | 48.4% |
| GRPO + KL | 60 | 53.0% | 72.7% |
| DistilledRL (V12) | 60 | 49.4% | 82.3% |
| Gated DistilledRL | 60 | 50.9% | 78.5% |

### SVAMP (300 questions)

| Model | Step | pass@1 | pass@32 |
|---|---|---|---|
| Base | — | 66.0% | 82.7% |
| DistilledRL (V12) | 60 | 64.1% | 92.3% |
| Gated DistilledRL | 60 | 62.7% | 86.7% |

### GSM-Symbolic (1000 questions)

| Model | Step | pass@1 | pass@32 |
|---|---|---|---|
| Base | — | 37.3% | 63.8% |
| DistilledRL (V12) | 60 | 39.8% | 72.2% |
| Gated DistilledRL | 60 | 42.4% | 70.8% |

The gated variant is the only checkpoint in the study to significantly improve pass@1 over base on GSM-Symbolic ($p = 0.0097$).

---

## Setup

**Requirements**

- Python 3.10+
- PyTorch 2.1+
- Two GPUs: student on `cuda:0`, teacher on `cuda:1`. Both fit in 16 GB with 4-bit NF4 quantization.
- vLLM (evaluation)

**Install**

```bash
git clone https://github.com/Moncefan/distilled-rl-thesis.git
cd distilled-rl-thesis
pip install -r requirements.txt
```

---

## Training

All training scripts share the same configuration. Only the training objective differs.

| Setting | Value |
|---|---|
| Student | `meta-llama/Llama-3.2-1B-Instruct` |
| Teacher | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| LoRA rank / alpha | 64 / 128 |
| Learning rate | 1e-4 |
| Group size | 10 |
| Sampling temperature / top-p | 1.2 / 0.99 |
| Max new tokens | 512 |
| Prompts per step | 2 |
| Accumulation steps | 4 |

### Confidence-gated variant

The gated script expects `gate_calibration.json` in the working directory. Generate it first with the calibration script, then train:

```bash
python calibrate_gate.py --num-prompts 30 --samples-per-prompt 2   # writes gate_calibration.json
python Gated_DistilledRL.py
```

---

## Gate calibration

The calibration script runs in two stages:

1. **Threshold and scale.** Sets $H^\star$ and $\tau$ from the teacher's per-token entropy on base-student rollouts.
2. **Validation.** Evaluates the gate on positive-advantage tokens and reports the Pearson correlation between $c$ and the teacher-ratio deviation, the 2×2 contingency table, and the risk/benefit ratio.

The diagnostic must be positive before training is justified. A negative or near-zero correlation would indicate that the gate suppresses the teacher signal where it is strongest.

---

## Reproducibility

The training and evaluation pipeline is fully reproducible:

- **Fixed tokenization path** (`add_special_tokens=False`) so inputs never contain a duplicate BOS token.
- **Two-phase training loop** (frozen rollout collection, then gradient accumulation) for memory efficiency on two T4 GPUs.
- **Shared judge** across all evaluations, with symbolic verification and a restricted regex fallback.

## Contributions

1. **A controlled three-way comparison** of DistilledRL, KL-regularized GRPO, and KL-free GRPO that isolates the effect of the objective from the surrounding RL recipe.
2. **A boundary-condition study** across four benchmarks (GSM8K test, GSM-Symbolic, SVAMP, MATH-500), showing where the DistilledRL advantage holds and where it does not.
3. **A confidence-gated variant** used as an independent probe of the entropy-preservation mechanism, with pre-training calibration and alignment diagnostics reported.
4. **A LoRA rank ablation** (16 vs. 64) ruling out adapter capacity as the primary constraint on the observed benefits.

---

## Checkpoints

LoRA adapter weights are not committed to this repository due to size. They are available on Hugging Face at [`MoncefAn`](https://huggingface.co/MoncefAn). The `checkpoints/` directory in this repo contains only training logs and validation predictions.

---

## Citation

The DistilledRL objective was introduced by:

```bibtex
@article{wang2026distilledrl,
  title   = {Distilled Reinforcement Learning for {LLM} Post-Training},
  author  = {Wang, Chen and Li, Zhaochun and Bai, Jionghao and Zhang, Yining
             and Deng, Hexuan and Lan, Ge and Wang, Yue},
  journal = {arXiv preprint arXiv:2607.17247},
  year    = {2026}
}
```

## Acknowledgements

Student and teacher models are [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and [`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B). Evaluation uses vLLM. The code was developed on two T4 GPUs with 4-bit NF4 quantization for both student and teacher.

## License

MIT. See [LICENSE](LICENSE).

## Contact

- **Author:** Anani Moncef
- **Email:** ananimoncef2@gmail.com
- **Thesis supervisor:** Abdelkader Amrane
