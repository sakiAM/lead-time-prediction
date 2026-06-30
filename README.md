# Dynamic Lead-Time Prediction Engine: A Classical-to-Quantum Architecture Study

## 🎯 Executive Product Summary
In global supply chains, static delivery schedules create severe inventory inefficiencies—either driving up capital holding costs due to "safety stock" bloat, or triggering expensive SLA breaches from stockouts. 

This project designs and implements an adaptive, data-driven machine learning engine to predict actual shipment transit days (`Days for shipping (real)`). The product vision balances immediate operational scalability with long-term technology horizon mapping, evolving a classical statistical baseline into a production-grade ensemble pipeline, and benchmarking a futuristic hybrid Quantum Machine Learning (QML) prototype.

---

## 📊 Core Performance & Benchmark Matrix

To evaluate the feasibility and trade-offs of different computational eras, the engine was built across three distinct architectural phases:

| Model Phase | Technical Paradigm | Test RMSE (Days) | Test MAE (Days) | Operational Status & Product Feasibility |
| :--- | :--- | :---: | :---: | :--- |
| **Phase 1** | Linear Regression | 1.3876 | 1.1297 | **Legacy Baseline.** Fast but assumes linear scaling; completely misses operational bottlenecks. |
| **Phase 2** | XGBoost Regressor | **1.2655** | **0.9848** | **Production Champion.** Sub-day accuracy (~23.5 hrs). Optimally handles high-cardinality categories. |
| **Phase 3 (A)**| Variational Quantum (VQR)| 3.2547 | 2.8632 | **R&D Track.** Heavily resource-constrained; highly susceptible to optimization barren plateaus. |
| **Phase 3 (B)**| Quantum Kernel (QSVR) | **1.9777** | **1.8056** | **R&D Track.** Peak Quantum expressiveness via feature entanglement, but limited by $O(N^2)$ simulation bottlenecks. |

---

## 💡 Strategic Technical & Product Insights

### 1. Product Thinking: Turning Metrics into Capital Efficiency
* **Upstream Optimization:** Dropping the forecasting variance below 24 hours (Phase 2 MAE: 0.98 days) directly optimizes warehouse labor allocation and dynamically downsizes safety stock requirements, freeing up millions in trapped holding capital.
* **Client-Facing Value:** Achieving high-confidence, sub-day accuracy allows front-end engineering teams to surface precise delivery guarantees at user checkout, directly accelerating e-commerce conversion rates.

### 2. Technical Thinking: Data Leakage & Validation Guardrails
* Traditional randomized sampling introduces severe chronological data leakage (using future data to predict past events). This architecture enforces clean validation splits, ensuring data sampling and scaling (mapping continuous features into bounded $[0, \pi]$ quantum dimensions) occur strictly *after* independent dataset partitioning.

### 3. Innovative Thinking: Navigating the Quantum Scaling Wall
* **The Constraint:** Shifting to Quantum Support Vector Regression (QSVR) with a `ZZFeatureMap` successfully captured deeper multi-variable correlations (dropping quantum error to 1.97 RMSE). However, calculating a massive, multi-dimensional similarity matrix triggered an $O(N^2)$ computational complexity wall, timing out the classical CPU simulator.
* **The PM Recommendation:** Maintain the classical **XGBoost engine for immediate production path workloads** to protect real-time API latency SLAs (<200ms). Isolate the **Quantum Kernel track as an offline, hybrid batch-processing optimization loop** for high-value routing lanes.

---

## 📂 Project Architecture & PM Artifacts

The repository is structured like an enterprise product feature launch, separating source code execution from product governance:

```text
├── artifacts/                # 📂 Product Management Portfolio
│   ├── product_requirements_document.md  # Core PRD, Personas & SLAs
│   └── business_case_roi_model.xlsx      # Financial Safety Stock Impact Model
├── src/                      # 📂 Production Source Code
│   ├── baseline_regression.py            # Phase 1: Linear Model
│   ├── xgboost_regression.py             # Phase 2: Production Pipeline
│   └── quantum_regression.py             # Phase 3: Qiskit QSVR/VQR Simulator
└── README.md                 # 📄 Executive Brief
