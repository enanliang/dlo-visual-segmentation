# DLO Visual Segmentation

Source code and dataset links for our Pattern Recognition Letters paper:

> **【Layer-wise Dynamic Precision Scheduling for Efficient DLO Segmentation: A Simulation-based Evaluation】**
> YongYinan Liang. Guanghua School of Management, Peking University.

---

## Overview

This repository provides the implementation of a visual recognition and segmentation method for **Deformable Linear Objects (DLO)** — i.e. flexible cables/wires — which is a core perception module for robotic cable manipulation, untangling and assembly tasks.

Key features:
- Shape feature extraction for thin, occluded, tangled flexible cables
- Robust segmentation under clutter and self-occlusion
- Lightweight pipeline suitable for real-time robotic perception

## Repository Structure

dlo-visual-segmentation/
├── README.md
├── src/                # source code
├── configs/            # experiment configurations
├── demo/               # qualitative result figures
└── requirements.txt    # python dependencies

##Citation

@article{liang2026dlo,
  title   = {【Layer-wise Dynamic Precision Scheduling for Efficient DLO Segmentation: A Simulation-based Evaluation】},
  author  = {Liang, Yinan},
  journal = {Pattern Recognition Letters},
  year    = {2026}
}

## License

This project is released under the [MIT License](LICENSE).


