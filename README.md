<h1 align="center">A GLANCE into the Future: Grade-Guided Liquid AMD Network for Continuous-Time Evolution of AMD Progression from Fundus Images</h1>

<!--
<p align="center">
  <a href="https://arxiv.org/abs/"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-CC5500.svg?style=for-the-badge"></a>
</p>
-->

<p align="center">
  <b><i>Submitted for IEEE ISBI 2027</i></b>
</p>

<p align="center">
  <i>Jovana Panic</i> &nbsp;&middot;&nbsp;
  <i>Niccolò Fineschi</i> &nbsp;&middot;&nbsp;
  <i>Yuki Maeno</i> &nbsp;&middot;&nbsp;
  <i>Hideaki Hayashi</i> &nbsp;&middot;&nbsp;
  <i>Kazuya Nishimura</i> &nbsp;&middot;&nbsp;
  <i>Atsushi Watanabe</i> &nbsp;&middot;&nbsp;
  <i>Samanta Rosati</i> &nbsp;&middot;&nbsp;
  <i>Gabriella Balestra</i> &nbsp;&middot;&nbsp;
  <i>Kohji Nishida</i> &nbsp;&middot;&nbsp;
  <i>Hajime Nagahara</i>
</p>

![Framework](./figures/glance.gif)

## Abstract

Age-related macular degeneration (AMD) progression varies across patients, motivating forecasts of continuous disease trajectories beyond fixed-horizon risk prediction.

We propose **GLANCE**, *Grade-guided Liquid AMD Network for Continuous-time Evolution*, a continuous-time Liquid Neural Network trained with clinically grounded objectives that penalize predicted severity decreases, balance forecasting horizons, and encourage agreement with observed progression trends.

![Framework](./figures/framework_dark.png#gh-dark-mode-only)
![Framework](./figures/framework_light.png#gh-light-mode-only)

 
Our contributions are:
- We formulate AMD **severity-trajectory forecasting** from a single baseline visit using bilateral retinal fundus images and clinician-assigned severity scores.
- We introduce a **progression penalty** against decreasing forecasts, with **horizon weighting** and a **robust slope loss** for distant follow-up and progression speed.
- We introduce an **evaluation framework** assessing forecasting error across time horizons, trajectory plausibility, and trend agreement.

<!-- ![Trajectories](./figures/trajectories_dark.png#gh-dark-mode-only)
![Trajectories](./figures/trajectories_light.png#gh-light-mode-only) -->

On the AREDS dataset, GLANCE achieves Harrell's C-index of $0.96$ while reducing implausible severity drops from $70.1\%$ to $1.3\%$, preserving accuracy while ensuring monotonically plausible 10-year trajectories.
Comparisons with discrete and continuous time models contextualize these findings, highlighting trade-offs between predictive performance and trajectory plausibility.
This work establishes a foundation for clinically oriented AMD trajectory forecasting to support risk-informed screening and longitudinal monitoring.

## Cloning repository

To **clone** this repository, including the necessary submodules, use:

```bash
git clone --recursive https://github.com/niccofine/GLANCE
```

In case the repository is cloned without the ``` --recursive ```  flag, to include correctly all the necessary submodules run:

```bash
git submodule update --init --recursive
``` 

It is recommended to define a **Python Virtual Environment** for this repository, installing the required packages and the ```src``` functions running:

```bash
pip install -r requirements.txt

pip install -e .
```

## Repository tree

The repository is **organized** into Python scripts and source code contained into *topic-specific* folders:
- **Python** source code contain the necessary functions to run the experiments;
- **Python** scripts call source functions to run the experiments.

```
├── figures/
├── scripts/
│   ├── dataset.py                # cohort split
│   ├── crossvalidation.py        # cross-validation on construction set
│   ├── lossselection.py          # loss variant selection
│   ├── finaltraining.py          # final training
│   ├── evaluation.py             # cross-validation and test metrics, statistical tests
│   ├── explainability.py         # feature attribution, concept projection, pixel attribution
│   ├── figures.py                # paper figures and figure-backing data tables
│   ├── posthoc.py                # post-hoc analyses
│   ├── allresults.py             # consolidated per-patient scoring across models
│   ├── extrametrics.py           # additional post-hoc metrics
│   ├── tests.py                  # unit tests
│   └── driver.sh                 # runs every step above in order
├── src/glance/
│   ├── config.py                 # paths and constants
│   ├── data.py                   # cohort, sample, split loading, cross-validation, dataset
│   ├── models.py                 # architectures, loss, training loop, GPU queue
│   ├── irregular.py              # irregular-visit (posthoc)
│   ├── annual.py                 # annual-grid (posthoc)
│   ├── continuous.py             # LNN variants (posthoc)
│   ├── encoder.py                # RET-CLIP visual encoder and image preprocessing
│   └── evaluate.py               # metrics, statistical tests, evaluation
├── results/
│   └── supplementary.md          # supplementary material
├── .gitignore
├── LICENSE.md
├── README.md
├── requirements.txt
└── setup.py                      # src setup script
```

## Data prerequisites

Before running, GLANCE expects precomputed RET-CLIP embeddings and their source image paths.
These are produced by [AMD-Latent](https://github.com/niccofine/AMD-Latent)'s encoding notebook (`notebooks/dataset/01_encoding.ipynb`), which saves them as `latentvec.npy` and `imagepaths.npy`.
Place those two files into `data/raw/retclip_features/` as `latentvec_RETCLIP.npy` and `imagepaths_RETCLIP.npy` before building the dataset.