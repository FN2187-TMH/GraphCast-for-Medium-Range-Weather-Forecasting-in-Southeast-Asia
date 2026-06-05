# GraphCast SEA: Regional Medium-Range Weather Forecasting using Graph Neural Networks

[![Project Status](https://img.shields.io/badge/Project-Deep%20Learning-blue.svg)](https://github.com/FN2187-TMH/GraphCast-for-Medium-Range-Weather-Forecasting-in-Southeast-Asia)
[![Framework](https://img.shields.io/badge/Backend-PyTorch-ee4c2c.svg)](https://pytorch.org/)

**GraphCast SEA** is a specialized, regional deep learning weather forecasting system designed for Southeast Asia (SEA). It predicts key atmospheric and surface variables up to 72 hours ahead. Inspired by Google DeepMind's original GraphCast, this repository implements a highly optimized, compact variant (~0.54M parameters) custom-engineered to execute efficiently on accessible hardware (such as a single NVIDIA Tesla T4 GPU) without sacrificing structural fidelity.
<p align="center">
  <img src="Slides\southeast-asia-region-map-of-countries-in-southeastern-asia-vector-illustration-2BYTAAB.jpg" alt="Southeast Asia Spatial Scope" width="60%">
</p>
---

## 🎯 Project Objectives
* **Regional Fine-Grained Forecasting:** Deliver high-resolution regional forecasts over the complex geographical domain of Southeast Asia ($0.25^\circ \times 0.25^\circ$ spatial grid).
* **Dual-Stream Data Fusion:** Blend multi-source climate inputs by coupling **ERA5 Reanalysis** (the "Brain" branch) with real-time **NASA GPM MERGIR** satellite precipitation data (the "Eye" branch).
* **Hardware-Optimized Parallelism:** Utilize fully vectorized GNN operations to eliminate heavy Python `for-loops` across batch dimensions, keeping VRAM footprints lightweight while allowing long-range autoregressive rollouts.

---

## 🏗️ Core Pipeline & Architecture
The model relies on a non-Euclidean **Grid-Mesh-Grid** framework structured into an **Encoder ➔ Processor ➔ Decoder** topology. This formulation bypasses the geometric distortions common to standard convolutional grids over spherical domains.

1. **Encoder (Grid-to-Mesh - G2M):** Maps flat 2D geographic grid arrays onto an internal, optimized multi-scale spherical mesh. Spatial edges are established via $k$-Nearest Neighbors ($k=4$) to smoothly interpolate regional climate states into $128$-dimensional latent node features.
2. **Processor (Mesh-to-Mesh - M2M):** Comprises **4 stacked homogeneous GNN Layers**. Nodes interact via vectorized message passing across the mesh topology, effectively modeling large-scale advection, fluid dynamics, and thermodynamic atmospheric interactions.
3. **Decoder (Mesh-to-Grid - M2G):** Project the updated hidden states from the abstract mesh nodes back down to the target 2D coordinate grid. It outputs a latitude-weighted local residual delta ($\Delta X$).

$$\hat{X}_{t+6\text{h}} = X_{t} + \Delta X$$

This **Residual Mapping** prevents the neural network from wasting capacity on identity transformations, accelerating gradient convergence and stabilizing multi-step autoregressive inference loops.

---

## 📊 Key Atmospheric Parameters
* **Pressure-Level Dynamics (@850hPa and @500hPa):** Geopotential (`z`), Temperature (`t`), Specific humidity (`q`), and Wind vector components (`u`, `v`).
* **Surface-Level Variables:** 2-meter temperature (`t2m`), Mean sea-level pressure (`msl`), 10-meter wind components (`u10`, `v10`), and Total accumulated precipitation (`tp`).

---

## 📈 Experimental Validation
Evaluated across lead times ranging from $+24\text{h}$ out to $+72\text{h}$ using Latitude-Weighted Root Mean Squared Error (RMSE) and Anomaly Correlation Coefficient (ACC):
* **GraphCast SEA (GNN)** demonstrates substantial skill improvements over baseline methods, consistently outperforming both **Persistence** (inherent inertia) and **Climatology** (historical mean values).
* When compared to a deep convolutional **U-Net** baseline, the GNN variant proves significantly more robust against errors compounding over extended rollouts ($+72\text{h}$ lead time), confirming its ability to emulate long-range fluid dynamics rather than simply memorizing training sequences.

---
