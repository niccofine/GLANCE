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

![Trajectories](./figures/trajectories_dark.png#gh-dark-mode-only)
![Trajectories](./figures/trajectories_light.png#gh-light-mode-only)

## Abstract

Age-related macular degeneration (AMD) progresses at variable rates, and forecasting late AMD onset is critical for patient management.
Rather than predicting binary risk at fixed horizons, we forecast the full grade trajectory from a single fundus visit using **GLANCE**, *Grade-guided Liquid AMD Network for Continuous-time Evolution*.

![Framework](./figures/framework_dark.png#gh-dark-mode-only)
![Framework](./figures/framework_light.png#gh-light-mode-only)

 
Our contributions are:
- We formulate AMD progression as a **grade-guided continuous-time trajectory forecasting**: frozen RET-CLIP features of both eyes, $\mathbf{z_\mathrm{LE}}$ and $\mathbf{z_\mathrm{RE}}$, and the current grade, $s$, condition a closed-form Liquid Neural Network queried at any time.
- We introduce a **progression penalty** against decreasing forecasts, with **horizon weighting** and a **robust slope loss** for distant follow-up and progression speed.
- On AREDS dataset, with a pre-specified protocol, we show that the penalty removes implausible forecasts at comparable error, and that trajectory forecasters match fixed-horizon baselines on late AMD prediction while estimating **onset time**.

Moreover, attribution analysis confirms the prognostic signal for disease worsening is concentrated around the macula and the optic disc.

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
│   ├── figures.py                # data for figure building
│   ├── posthoc.py                # posthoc evaluations
│   ├── tests.py                  # unit tests
│   └── driver.sh                 # runs every step above in order
├── src/glance/
│   ├── config.py                 # paths and constants
│   ├── data.py                   # cohort, sample, split loading, cross-validation, dataset
│   ├── models.py                 # architectures, loss, training loop, GPU queue
│   ├── irregular.py              # irregular-visit training experiment (posthoc)
│   ├── encoder.py                # RET-CLIP visual encoder and image preprocessing
│   └── evaluate.py               # metrics, statistical tests, evaluation
├── .gitignore
├── LICENSE.md
├── README.md
├── requirements.txt
├── setup.py                      # src setup script
└── supplementary.md              # supplementary material
```

## Data prerequisites

Before running, GLANCE expects precomputed RET-CLIP embeddings and their source image paths.
These are produced by [AMD-Latent](https://github.com/niccofine/AMD-Latent)'s encoding notebook (`notebooks/dataset/01_encoding.ipynb`), which saves them as `latentvec.npy` and `imagepaths.npy`.
Place those two files into `data/raw/retclip_features/` as `latentvec_RETCLIP.npy` and `imagepaths_RETCLIP.npy` before building the dataset.