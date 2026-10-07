# Hidden Markov Anomaly Detection
### From OC-SVM to Structured Sequence Models

[![Course](https://img.shields.io/badge/Course-Introduction%20to%20Machine%20Learning-blue)](https://github.com/AmirAli-jb/Hidden-Markov-Anomaly-Detection)
[![University](https://img.shields.io/badge/University-Sharif%20University%20of%20Technology-red)](https://www.sharif.edu/)
[![Python](https://img.shields.io/badge/Python-3.x-yellow?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)

## Overview

This project investigates **anomaly detection in sequential data**, comparing a classical **One-Class Support Vector Machine (OC-SVM)** with a sequence-aware **Hidden Markov Anomaly Detection (HMAD)** model.

The main question is:

> **When does modeling temporal and latent sequential structure improve anomaly detection compared with a fixed-vector method such as OC-SVM?**

The project starts with OC-SVM using different sequence representations and then extends the analysis using:

- Markov Chains
- Hidden Markov Models (HMMs)
- Log-space Viterbi decoding
- Hidden-state transition features
- State-dependent emission features
- A simplified iterative HMAD training procedure

The models are evaluated on two real-world time-series datasets:

- **ECG5000**
- **Daphnet Freezing of Gait**

---

## Project Structure

```text
Hidden-Markov-Anomaly-Detection/
│
├── data/
│   ├── ECG5000/
│   └── Daphnet_FOG/
│
├── notebooks/
│   ├── ECG5000.ipynb
│   └── Daphnet_FOG.ipynb
│
├── plots/
│   └── Generated figures and experimental results
│
├── report/
│   ├── Project_Description.pdf
│   └── Final_Report.pdf
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Datasets

### ECG5000

ECG5000 contains electrocardiogram signals where each sample represents one complete heartbeat with **140 time steps**.

For the anomaly-detection formulation:

- Class `1` is considered **normal**.
- Classes `2`, `3`, `4`, and `5` are considered **anomalous**.
- Only normal samples are used during one-class training.

| Split | Normal | Anomalous | Total |
|---|---:|---:|---:|
| Training | 292 | 0 | 292 |
| Test | 2627 | 1873 | 4500 |

---

### Daphnet Freezing of Gait

The Daphnet dataset contains accelerometer measurements collected for detecting **Freezing of Gait (FoG)** events.

In this project:

- the thigh accelerometer is used;
- the three acceleration axes are converted to acceleration magnitude;
- signals are divided into sequences of length **256**;
- a sliding window with **50% overlap** is used;
- subjects are separated between training and testing to avoid data leakage.

| Split | Normal | Anomalous | Total |
|---|---:|---:|---:|
| Training | 600 | 0 | 600 |
| Test | 540 | 60 | 600 |

---

## OC-SVM Baseline

The first model is an **RBF One-Class SVM** trained only on normal sequences.

The main configuration is:

```text
Kernel = RBF
gamma  = scale
nu     = 0.1
```

All features are standardized using `StandardScaler`, fitted only on the training set.

Three sequence representations are compared:

1. **Flattened Sequence**
2. **Global Summary Statistics**
3. **Window-Based Summary Statistics**

### Feature Representation Results

| Dataset | Representation | Dimension | ROC-AUC |
|---|---|---:|---:|
| ECG5000 | Flattened | 140 | **0.984025** |
| ECG5000 | Global Summary | 5 | 0.951839 |
| ECG5000 | Window Summary | 8 | 0.982451 |
| Daphnet FOG | Flattened | 256 | **0.375309** |
| Daphnet FOG | Global Summary | 5 | 0.372130 |
| Daphnet FOG | Window Summary | 32 | 0.317037 |

The results show that fixed-vector representations work particularly well for ECG5000 but struggle to capture the more complex temporal structure of the Daphnet dataset.

---

## Hidden Markov Anomaly Detection

To explicitly model sequential structure, the project implements a simplified **HMAD** framework.

The model uses:

- `K = 2` hidden states;
- Gaussian emission distributions;
- K-Means initialization;
- log-space Viterbi decoding;
- iterative HMM parameter updates;
- OC-SVM on structural sequence features.

For every sequence, Viterbi decoding estimates the most likely hidden-state path:

```text
x → Viterbi → z*
```

A joint feature vector is then constructed from:

- four normalized hidden-state transition features;
- two state-dependent emission features.

Therefore,

```text
Ψ(x, z) ∈ R⁶
```

The resulting representation captures not only the observed values but also **how the hidden states evolve through time**.

---

## HMAD Training Pipeline

```text
Normal Training Sequences
          ↓
K-Means Initialization
          ↓
Gaussian HMM
          ↓
Viterbi Decoding
          ↓
Joint Feature Extraction
          ↓
Standardization
          ↓
One-Class SVM
          ↓
Update HMM Parameters
          ↓
Repeat
```

Training is performed for up to **10 iterations**, while monitoring changes in hidden-state assignments and OC-SVM decision scores.

---

## Results

### OC-SVM vs. HMAD

| Dataset | OC-SVM Baseline | HMAD + Viterbi |
|---|---:|---:|
| ECG5000 | **0.984025** | 0.933161 |
| Daphnet FOG | 0.375309 | **0.647469** |

The experiments show two different behaviors:

- On **ECG5000**, the flattened signal itself already provides a highly informative representation, and OC-SVM achieves the best result.
- On **Daphnet FOG**, HMAD significantly improves performance, indicating that temporal structure and hidden-state transitions contain important anomaly information.

---

## Effect of Viterbi Decoding

To evaluate the importance of hidden-state inference, HMAD was also tested without meaningful Viterbi decoding.

| Dataset | Without Viterbi | With Viterbi |
|---|---:|---:|
| ECG5000 | 0.496717 | **0.933161** |
| Daphnet FOG | 0.602222 | **0.647469** |

The decrease in performance without Viterbi demonstrates the importance of correctly estimating hidden-state sequences when constructing HMAD features.

---

## Running the Project

Clone the repository:

```bash
git clone https://github.com/AmirAli-jb/Hidden-Markov-Anomaly-Detection.git
cd Hidden-Markov-Anomaly-Detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter lab
```

Then open:

```text
notebooks/ECG5000.ipynb
```

or

```text
notebooks/Daphnet_FOG.ipynb
```

---

## Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Key Takeaway

A more complex sequence model is not necessarily better for every dataset.

For ECG5000, the observed signal shape already provides enough information for OC-SVM to perform extremely well.

For Daphnet Freezing of Gait, however, explicitly modeling hidden temporal structure substantially improves anomaly detection.

> **The effectiveness of a machine-learning model depends strongly on how well its assumptions match the structure of the data.**

---

## Reports

More detailed theoretical explanations, implementations, experiments, plots, and discussions are available in:

- [`Project_Description.pdf`](report/Project_Description.pdf)
- [`Final_Report.pdf`](report/Final_Report.pdf)

---

## Dataset References

**Daphnet Freezing of Gait**

D. Roggen, M. Plotnik, and J. Hausdorff,  
*Daphnet Freezing of Gait*, UCI Machine Learning Repository, 2010.

**ECG5000**

ECG5000 is distributed through the UCR/UEA Time Series Classification Archive and originates from ECG recordings related to the BIDMC Congestive Heart Failure Database.

A. L. Goldberger et al.,  
*PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals*, 2000.

---

## Authors

**Helia Tajabadi** — [@HeliaTJB](https://github.com/HeliaTJB)  
**AmirAli Jahanbakhshi** — [@AmirAli-jb](https://github.com/AmirAli-jb)

**Introduction to Machine Learning**  
Sharif University of Technology

Instructor: **Dr. Sajjad Amini**


