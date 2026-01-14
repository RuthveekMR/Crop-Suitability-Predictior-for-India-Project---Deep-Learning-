# 🌱 Crop Suitability Prediction for India

*A Spatially-Aware, Ranking-Based Agricultural Decision Support System*

A **state-of-the-art, end-to-end Machine Learning & Deep Learning system** that **ranks the top-5 most suitable crops** for any geographic location in India using climate, soil, and crop-requirement data.

This project combines **scientific datasets**, **spatially leakage-safe modeling**, and **modern deep learning (Tabular ResNet)** with a **live interactive deployment**.

🔗 **Live Application:**
[https://ruthveekmr-crop-suitability-predictior-for-india-pro-app-pdchk6.streamlit.app/](https://ruthveekmr-crop-suitability-predictior-for-india-pro-app-pdchk6.streamlit.app/)

---

## 📌 Project Highlights

* ✅ Formulated as a **learning-to-rank problem** (not classification)
* ✅ **Spatially aware train–test splitting** to prevent geographic leakage
* ✅ Integrates **IMD climate data, SoilGrids soil data, and FAO EcoCrop**
* ✅ Achieves **near-perfect ranking accuracy** (Spearman ≈ **0.98**)
* ✅ Fully deployed as an **interactive Streamlit application**
* ✅ Designed as a **decision-support framework**, not a black-box predictor

---

## 🧠 Problem Statement

> **Given a latitude and longitude in India, which crops are most suitable for cultivation under local environmental conditions?**

Instead of predicting a single crop or a binary outcome, the system **ranks all candidate crops** and returns the **top-5 recommendations**, aligning with real agricultural decision-making where choices are comparative.

---

## 🗂️ Data Sources

| Dataset         | Description                      | Source      |
| --------------- | -------------------------------- | ----------- |
| 🌧️ Rainfall    | Daily gridded rainfall (NetCDF)  | IMD Pune    |
| 🌡️ Temperature | Daily maximum temperature (.GRD) | IMD Pune    |
| 🌱 Soil         | Sand–silt–clay composition       | SoilGrids   |
| 🌾 Crop Traits  | Climate & soil requirements      | FAO EcoCrop |

**Official Links**

* IMD Rainfall: [https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_1_NetCDF.html](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_1_NetCDF.html)
* IMD Temperature: [https://www.imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html](https://www.imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html)
* SoilGrids: [https://soilgrids.org/](https://soilgrids.org/)
* EcoCrop: [https://github.com/OpenCLIM/ecocrop](https://github.com/OpenCLIM/ecocrop)

---

## 🔄 Methodology Overview (High-Level)

1. **Climate Data Processing**

   * Scientific NetCDF / GRD formats converted to structured grids
   * Median, 25th & 75th percentiles computed per grid

2. **Soil Feature Engineering**

   * Sand–silt–clay → USDA texture classes
   * Multi-hot encoded soil constraints

3. **Land–Ocean Filtering**

   * Ocean grid points removed using geospatial joins

4. **Grid–Crop Pair Construction**

   * Each land grid evaluated against all crops
   * Enables learning-to-rank formulation

5. **Spatially Safe Data Splitting**

   * India divided into 3°×3° tiles
   * Tiles held out to ensure **zero spatial leakage**

6. **Normalization & Feature Engineering**

   * RobustScaler + MinMaxScaler
   * Climate variability & interaction features

7. **Model Benchmarking**

   * Distance baseline
   * LightGBM
   * MLP, Two-Tower MLP
   * **Tabular ResNet (best performing)**

8. **Final Training & Evaluation**

   * Early stopping, LR scheduling
   * Tested on fully unseen spatial regions

9. **Deployment**

   * Real-time inference pipeline
   * Interactive map-based interface

---

## 🏆 Model Performance (Unseen Spatial Test Set)

| Metric                        | Value      |
| ----------------------------- | ---------- |
| **Spearman Rank Correlation** | **0.9817** |
| RMSE                          | 0.0297     |
| MAE                           | 0.0053     |

➡️ Confirms **exceptional ranking accuracy** and strong spatial generalization.

---

## 🏗️ Project Structure (Public)

```
crop_suitability_app/
│
├── model/
│   ├── best_final_model.keras
│   ├── final_scalers.pkl
│   ├── grid_lookup.csv
│   └── Ecocrop_cleaned_final_v5.csv
│
├── utils/
│   ├── __init__.py
│   └── predictor.py
│
├── app.py
├── requirements.txt
├── LICENSE
└── README.md
```

> 🔒 **Note:** Internal research notebooks and intermediate artifacts are intentionally excluded from the public repository.

---

## 🚀 Running the Application Locally

```bash
git clone https://github.com/RuthveekMR/Crop-Suitability-Predictior-for-India-Project---Deep-Learning-
cd Crop-Suitability-Predictior-for-India-Project---Deep-Learning-
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚠️ Scope & Limitations

This system predicts **environmental suitability**, not yield or profitability.

Currently **not included**:

* Solar radiation & sunlight duration
* Frost days & growing season length
* Pest and disease prevalence
* Irrigation availability
* Market economics & pricing

➡️ The system serves as a **high-quality decision-support baseline**, not a standalone farming advisory.

---

## 🔮 Future Work

* 🌤️ Solar radiation, frost & humidity integration
* 💧 Irrigation and water-availability modeling
* 📈 Economic & policy-aware recommendations
* 🌍 Expansion beyond India
* 🧠 Uncertainty estimation & explainability

---

## 🔐 Intellectual Property Notice

This repository represents an **academic research and development project**.

The **core methodology, learning framework, and system architecture** are under
**intellectual property review through Manipal Academy of Higher Education (MAHE), India**.

Use of this code does **not confer inventorship, ownership, or patent rights**
over the underlying system, methods, or ideas.

---

## 👤 Author

**Ruthveek M R**
B.E. Data Science & Engineering
MIT Manipal

---

## ⭐ Acknowledgements

* India Meteorological Department (IMD)
* SoilGrids
* FAO EcoCrop Database

---

If you find this project insightful, consider ⭐ starring the repository.

---
