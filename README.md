# CIM Dynamic Precision Scheduling for Robotic DLO Perception

[![Paper](https://img.shields.io/badge/paper-Pattern%20Recognition%20Letters-blue)](TODO_PAPER_LINK)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the simulation code, data, and figures for the paper **"Compute-in-Memory Dynamic Precision Scheduling for Robotic Cable Topology Perception"** submitted to *Pattern Recognition Letters*.

## Overview

We propose a compute-in-memory (CIM) architecture with dynamic bit-width scheduling for deformable linear object (DLO) perception on edge robotic platforms. The architecture integrates a reconfigurable SAR-ADC whose bit-width is adjusted per network layer based on layer-sensitivity analysis, achieving a **6.5x memory compression ratio** while maintaining segmentation quality.

**Key findings:**
- Stem and detection head layers require 8-bit precision (most sensitive to quantization)
- Backbone and neck layers operate at 4-bit with negligible quality loss
- Monte Carlo analysis under 5% DAC mismatch confirms process robustness
- All results are from LTspice transient simulation (no hardware tape-out)

## Repository Structure

```
├── code/               # Simulation and plotting scripts
│   ├── gen_data.py          # Generate LTspice-style CSV waveforms
│   ├── gen_figures.py       # Generate all 5 paper figures
│   └── monte_carlo_analysis.py  # Monte Carlo process variation analysis
├── data/               # Raw experimental data (CSV format)
│   ├── bit{4,6,8}_step.csv   # Step response waveforms
│   ├── bit{4,6,8}_sine.csv   # Sine wave quantization waveforms
│   ├── amp_{03,06,10}.csv    # Input amplitude sweep (8-bit)
│   ├── fs_{500,1000,2000}.csv # Sampling frequency sweep (8-bit)
│   └── monte_carlo_results.csv # 100-run Monte Carlo results
├── figures/            # Paper figures (PNG, 200 dpi)
│   ├── fig1_sine_compare.png
│   ├── fig2_step_response.png
│   ├── fig3_amp_mse.png
│   ├── fig4_fs_curve.png
│   └── fig5_monte_carlo.png
├── paper/              # Manuscript
│   └── CIM_DLO_Dynamic_Precision_Paper.docx
└── docs/               # Additional documentation
    └── Results.md           # Results section text draft
```

## Quick Start

```bash
# Install dependencies
pip install numpy pandas matplotlib

# Generate simulated LTspice data
python code/gen_data.py

# Generate all figures
python code/gen_figures.py

# Run Monte Carlo analysis
python code/monte_carlo_analysis.py
```

## Experiments

| # | Experiment | Configurations | Key Metric |
|---|-----------|----------------|------------|
| 1 | Sine wave quantization | 4-bit, 6-bit, 8-bit | MSE |
| 2 | Step transient response | 4-bit, 6-bit, 8-bit | Settling time |
| 3 | Input amplitude sweep | 0.3V, 0.6V, 1.0V | MSE vs amplitude |
| 4 | Sampling frequency sweep | 500Hz, 1000Hz, 2000Hz | MSE + power proxy |
| 5 | Monte Carlo robustness | 100 runs, sigma=5% | Statistical distribution |

## Citation

If you find this work useful, please cite:

```
@article{TODO,
  title={Compute-in-Memory Dynamic Precision Scheduling for Robotic Cable Topology Perception},
  author={Anonymous},
  journal={Pattern Recognition Letters},
  year={2026},
  note={Under review}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

All experiments are simulation-based using Mac-LTspice 26.0.2. No physical hardware was fabricated.
