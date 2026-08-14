<h1 align="center">Cross-Domain Attribute Alignment with CLIP: A Rehearsal-Free Approach for Class-Incremental Unsupervised Domain Adaptation</h1>

<p align="center">
  <strong>Official implementation of VisTA framework.</strong><br>
  ACM International Conference on Multimedia (ACM MM), 2025
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2509.11264"><img src="https://img.shields.io/badge/arXiv-2509.11264-b31b1b.svg" alt="arXiv"></a>
  &nbsp;
  <a href="https://dl.acm.org/doi/10.1145/3746027.3755184">
    <img src="https://img.shields.io/badge/ACM%20Multimedia-2025-blue?logo=acm&logoColor=white" alt="ACM Multimedia">
  </a>
</p>

---

## Overview

This repository introduces **VisTA**, a rehearsal-free Class-Incremental Unsupervised Domain Adaptation (**CI-UDA**) method built on the large-scale vision-language model CLIP. VisTA respectively constructs an attribute dictionary for source domain and target domain, and encourages Visual Attention Consistency and Prediction Consistency to learn domain-invariant attributes. Experiments show that VisTA effectively reduces catastrophic forgetting and mitigates domain shift.

<div align="center">
  <img src="assets/VisTA.png" width="900px" />
</div>

The full paper with **Supplementary Materials** is available [here](https://arxiv.org/abs/2509.11264).

## Installation

Our code is implemented in Python (version >= 3.8) with PyTorch (version >= 1.11.0). Please follow the steps below to install the dependencies:

```bash
# Install CLIP
git clone https://github.com/openai/CLIP.git
cd CLIP
pip install .

# Install Dassl
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch
pip install -r requirements.txt
pip install .

# Install other dependent packages of VisTA.
git clone https://github.com/RyunMi/VisTA.git
cd VisTA
pip install -r requirements.txt
```

## Datasets

Please follow the [instructions](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md) to prepare three datasets for CI-UDA: Office-31, Office-Home, and Mini-DomainNet. After preparing the datasets, please update the `DATA` variable in `scripts/{dataset}.sh` accordingly.

## Training

For CI-UDA, we provide scripts to run experiments:

```bash
# Training on Office-31
sh scripts/office31.sh

# Training on Office-Home
sh scripts/officehome.sh

# Training on Mini-DomainNet
sh scripts/minidomainnet.sh
```

> Note: Set `MODEL.BACKBONE.NAME` in `configs/trainers/VisTA/*.yaml` to switch visual encoders. The `.pt` weights will auto-download to `assets/`.

## Citation

If you find the code useful in your research, please consider citing:

```bibtex
@inproceedings{mi2025vista,
author = {Mi, Kerun and Kang, Guoliang and Li, Guangyu and Zhao, Lin and Zhou, Tao and Gong, Chen},
title = {Cross-Domain Attribute Alignment with CLIP: A Rehearsal-Free Approach for Class-Incremental Unsupervised Domain Adaptation},
year = {2025},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {7883–7892},
numpages = {10},
series = {MM '25}
}
```

## Acknowledgments

We would like to acknowledge the following projects:

[CoOp](https://github.com/KaiyangZhou/CoOp)

[DAPrompt](https://github.com/LeapLabTHU/DAPrompt)

[AttriCLIP](https://github.com/vanity1129/AttriCLIP)
