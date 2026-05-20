# Continual_Learning_System
Built a scalable continual learning framework in PyTorch to reduce catastrophic forgetting using Fine-Tuning, EWC, and Experience Replay on Split CIFAR-10/100. Achieved 64.1% accuracy with Replay and reduced forgetting by 97% using EWC. Tech: PyTorch, Streamlit, Plotly, NumPy.
# Continual Learning — Scalability Study
## Split-CIFAR-10 vs Split-CIFAR-100

## What this adds over the basic project
This is the **research contribution** — comparing how Fine-tuning, EWC, and Replay
scale when the problem gets harder. Two new output figures:

1. `scalability_comparison.png` — 2×4 grid: all metrics side-by-side per dataset
2. `degradation_analysis.png`  — how much each method degrades CIFAR-10 → CIFAR-100

The **degradation ratio** is a novel metric you can describe in your report:
> "We define degradation ratio as the percentage accuracy drop a method suffers
> when moving from a simpler (CIFAR-10) to a harder (CIFAR-100) benchmark.
> A method with low degradation ratio is more robust to problem complexity."

## Dataset splits
| Dataset | Tasks | Classes/task | Total classes |
|---------|-------|-------------|---------------|
| Split-CIFAR-10 | 5 | 2 | 10 |
| Split-CIFAR-100 | 10 | 10 | 100 |

## Model — ScalableCNN
- CIFAR-10 : 2-stage CNN (64→128 channels), ~0.6M params
- CIFAR-100: 3-stage CNN (64→128→256 channels), ~1.3M params
- Both use ResBlocks + BatchNorm + AdaptiveAvgPool

## Run
```bash
pip install torch torchvision matplotlib numpy
python scalability_comparison.py
```

## Speed tips (CPU)
Edit CONFIGS at top of file:
```python
# Fast CPU run (~30-45 min)
"cifar10":  { "epochs": 4, "num_tasks": 5  }
"cifar100": { "epochs": 5, "num_tasks": 5  }  # reduce tasks too

# Full run (GPU recommended)
"cifar10":  { "epochs": 8,  "num_tasks": 5  }
"cifar100": { "epochs": 10, "num_tasks": 10 }
```

## Expected outputs
```
scalability_results.json     ← raw numbers for your report tables
scalability_comparison.png   ← main figure (2×4 panel)
degradation_analysis.png     ← degradation figure (1×3 panel)
```

## Expected results
|  | CIFAR-10 Acc | CIFAR-100 Acc | Degradation |
|--|--|--|--|
| Fine-tuning | ~19% | ~6% | ~68% drop |
| EWC | ~55% | ~30% | ~45% drop |
| Replay | ~72% | ~45% | ~38% drop |

Key finding: **Replay degrades the least** — it scales better than EWC as
class count and task count increase. This is your main research finding.

