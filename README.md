# 🧠 Diabetes Neural Network Classifier

> A simple feedforward neural network built with TensorFlow/Keras to predict the onset of diabetes using the Pima Indians Diabetes Dataset. Built as part of the **GoMyCode Deep Learning Checkpoint**.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Training Pipeline](#training-pipeline)
- [Results](#results)
- [Key Concepts Demonstrated](#key-concepts-demonstrated)
- [References](#references)
- [Author](#author)

---

## Project Overview

This project implements a **binary classification neural network** from scratch using TensorFlow/Keras to predict whether a patient is likely to develop diabetes based on diagnostic measurements.

The notebook follows the same structured deep learning pipeline introduced in `audiobooks_dl.ipynb`, adapted for a medical classification problem:

1. Load and explore the dataset
2. Balance classes to eliminate bias
3. Standardise features
4. Split into train / validation / test sets
5. Save data in `.npz` format
6. Build, compile, and train the neural network
7. Evaluate performance and make predictions
8. Manually trace the forward pass using NumPy

---

## Dataset

| Property | Detail |
|---|---|
| **Name** | Pima Indians Diabetes Dataset |
| **Source** | `diabetes (1).csv` via Google Drive |
| **Samples** | 768 patients |
| **Features** | 8 numeric diagnostic measurements |
| **Target** | `Outcome` — `0` (No Diabetes) / `1` (Diabetes) |

### Features

| # | Feature | Description |
|---|---|---|
| 1 | `Pregnancies` | Number of times pregnant |
| 2 | `Glucose` | Plasma glucose concentration (2h OGTT) |
| 3 | `BloodPressure` | Diastolic blood pressure (mm Hg) |
| 4 | `SkinThickness` | Triceps skin fold thickness (mm) |
| 5 | `Insulin` | 2-Hour serum insulin (mu U/ml) |
| 6 | `BMI` | Body mass index (kg/m²) |
| 7 | `DiabetesPedigreeFunction` | Diabetes pedigree function score |
| 8 | `Age` | Age in years |

---

## Model Architecture

```
Input Layer        →   8 features (standardised)
        ↓
Hidden Layer       →   4 neurons, ReLU activation
        ↓
Output Layer       →   1 neuron,  Sigmoid activation
        ↓
Output             →   Probability ∈ [0, 1]
                       ≥ 0.5 → Diabetic (1)
                       < 0.5 → Not Diabetic (0)
```

| Layer | Type | Units | Activation | Parameters |
|---|---|---|---|---|
| Input | — | 8 | — | — |
| Hidden | Dense | 4 | ReLU | 36 |
| Output | Dense | 1 | Sigmoid | 5 |
| **Total** | | | | **41** |

**Training Configuration:**

| Setting | Value |
|---|---|
| Loss function | Binary Cross-Entropy |
| Optimiser | Adam |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping | patience = 5 (monitors `val_loss`) |

---

## Project Structure

```
📁 diabetes-neural-network/
│
├── 📓 diabetes_neural_network.ipynb   # Main notebook (all code + explanations)
├── 📄 README.md                       # Project documentation (this file)
├── 📊 diabetes (1).csv                # Raw dataset (loaded from Google Drive)
│
├── 📁 saved_data/                     # Auto-generated during runtime
│   ├── diabetes_train_data.npz        # Training split
│   ├── diabetes_validation_data.npz   # Validation split
│   └── diabetes_test_data.npz         # Test split
│
└── 📓 audiobooks_dl.ipynb             # Reference notebook (GoMyCode)
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- Google Colab (recommended) **or** a local environment with the packages below

### Required Libraries

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

### Google Drive Setup

The notebook loads the dataset directly from your Google Drive. Ensure the file is placed at:

```
MyDrive/GoMyCode /diabetes (1).csv
```

> ⚠️ Note the space in `GoMyCode ` — this must match exactly.

---

## How to Run

### Option A — Google Colab (Recommended)

1. Upload `diabetes_neural_network.ipynb` to [Google Colab](https://colab.research.google.com)
2. Upload `diabetes (1).csv` to `MyDrive/GoMyCode /` in your Google Drive
3. Click **Runtime → Run All**
4. Authorise Google Drive access when prompted

### Option B — Local Jupyter

```bash
# Clone or download the project files
jupyter notebook diabetes_neural_network.ipynb
```

> Update the `path` variable in Section 2 to point to your local CSV file path.

---

## Training Pipeline

The notebook follows the same 7-step pipeline as `audiobooks_dl.ipynb`:

```
Step 1  Load raw CSV from Google Drive
   ↓
Step 2  Separate features (X) and target (y)
   ↓
Step 3  Balance dataset — match class-0 count to class-1 count
   ↓
Step 4  Standardise features with StandardScaler
   ↓
Step 5  Split → Train (80%) / Validation (10%) / Test (10%)
   ↓
Step 6  Save splits to .npz → Reload (tensor-ready format)
   ↓
Step 7  Build → Compile → Train → Evaluate → Predict
```

---

## Results

> Results are generated when the notebook is run. Expected ranges based on this architecture and dataset:

| Metric | Expected Range |
|---|---|
| Test Accuracy | 72% – 80% |
| Test Loss | 0.45 – 0.55 |
| Epochs to converge | 20 – 50 |

The loss curve (train vs. validation) is plotted after training to visually confirm learning and detect any overfitting.

---

## Key Concepts Demonstrated

### Forward Pass
Each input sample travels through the network layer by layer:

```
z₁ = W₁ᵀ · x + b₁        # Hidden pre-activation
a₁ = ReLU(z₁)             # Hidden activation
z₂ = W₂ᵀ · a₁ + b₂       # Output pre-activation
ŷ  = Sigmoid(z₂)          # Final prediction (probability)
```

This is reproduced manually in NumPy in **Section 9** of the notebook to verify the math against `model.predict()`.

### Loss Calculation (Binary Cross-Entropy)
```
L = -[y · log(ŷ) + (1 - y) · log(1 - ŷ)]
```
Measures the distance between predicted probabilities and true labels.

### Backpropagation & Weight Update
The Adam optimiser computes gradients of the loss with respect to every weight using the chain rule, then updates weights to minimise the loss:

```
W ← W - α · ∂L/∂W
```

TensorFlow handles this automatically during `model.fit()`.

---

## References

- [Pima Indians Diabetes Dataset — UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/diabetes)
- [TensorFlow / Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)

---

## Author

**
Sobowale Ahmed
Deep Learning Module — Simple Neural Network

---

*Built with TensorFlow 2.x · Python 3.x · Google Colab*
