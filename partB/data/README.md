# Data Directory — Toy Dataset

## Overview

This directory contains supporting documentation for the **toy dataset** used in the Part B reproduction of *"Discovery of Significant Emerging Trends"* (Goorha & Ungar, 2010).

## Why Toy Data?

The original paper operates on approximately **100,000 news and blog articles per day** from the Spinn3r blog corpus and Lydia news analysis system. Reproducing this scale is infeasible within the assignment constraints. Instead, we use a small, manually constructed dataset that captures the **mathematical properties** needed to demonstrate the scoring formula's behaviour.

## Dataset Description

The toy dataset consists of **20 phrase entries** across three categories, simulating iPod mentions within an IT Industry corpus:

| Category | Phrases | Count Range | Purpose |
|:---------|--------:|:------------|:--------|
| **iPod (Emerging Trend)** | 5 | 0–18 per week | Signal — phrases that spike in weeks 5–6 |
| **Generic IT (Noise)** | 10 | 9–50 per week | Background noise — stable high-frequency phrases |
| **Rare Junk** | 5 | 0–1 per week | Sparsity test — ghost phrase vulnerability |

**Total unique phrases:** 20
**Simulated time span:** 6 weeks (4 baseline + 2 current)

## Design Rationale

The dataset demonstrates three key behaviours:
1. **α = 0.0** (pure volume) → noise phrases ("technology industry") dominate
2. **α = 1.0** (pure specificity) → junk phrases tie with real trends
3. **α = 0.95** → iPod emerging trends correctly surface above noise and junk

## Limitations

- **No temporal realism:** Simulated weeks, not calendar data
- **No entity extraction:** Phrases are pre-defined
- **No proximity filtering:** Co-occurrences are assumed
- **Scale:** 20 phrases vs. millions in the original corpus

## Usage

The dataset is defined **inline in the notebooks** (`task_2_1.ipynb`, `task_2_2.ipynb`, `task_2_3.ipynb`, `task_3_1.ipynb`) rather than as separate CSV files, since the dataset is small enough to be self-contained.
