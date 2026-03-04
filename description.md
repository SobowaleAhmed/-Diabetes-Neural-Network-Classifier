# Project Description

## Title
**Diabetes Binary Classification using a Simple Neural Network**

## Short Description
A feedforward neural network built with TensorFlow/Keras that predicts diabetes onset from 8 clinical features. Demonstrates the full deep learning pipeline: data balancing, preprocessing, forward pass, loss calculation, backpropagation, and evaluation.

## Long Description

This project was developed as part of the **GoMyCode Deep Learning Checkpoint — Module: Simple Neural Networks**.

The goal is to implement and understand the core mechanics of a basic neural network applied to a real-world medical dataset — the **Pima Indians Diabetes Dataset** — to predict whether a patient is likely to develop diabetes based on diagnostic measurements.

### What It Does
- Loads the diabetes dataset from Google Drive
- Balances the binary classes to prevent model bias (same technique used in the `audiobooks_dl.ipynb` reference notebook)
- Standardises all 8 input features using `StandardScaler`
- Splits data into training (80%), validation (10%), and test (10%) sets
- Saves and reloads data in NumPy `.npz` tensor format
- Builds a neural network: **Input(8) → Dense(4, ReLU) → Dense(1, Sigmoid)**
- Trains using **Binary Cross-Entropy loss** and the **Adam optimiser** with early stopping
- Plots loss and accuracy curves across training epochs
- Evaluates final performance on the held-out test set
- Demonstrates the forward pass step-by-step in NumPy, extracting actual trained weight matrices to verify predictions manually

### Technologies Used
- **Python 3** — Core language
- **TensorFlow / Keras** — Neural network construction and training
- **NumPy** — Numerical operations and manual forward pass
- **Pandas** — Data loading and exploration
- **Scikit-learn** — Feature scaling and data splitting
- **Matplotlib / Seaborn** — Training visualisation
- **Google Colab** — Runtime environment

### Key Learning Outcomes
1. Understanding how a neural network maps inputs to outputs (forward pass)
2. How binary cross-entropy loss quantifies prediction error
3. How backpropagation and the Adam optimiser update weights each epoch
4. Why class balancing matters for fair model training
5. How to split, scale, and save data in a reproducible ML pipeline

## Tags
`deep-learning` `neural-network` `binary-classification` `tensorflow` `keras` `diabetes` `healthcare-ai` `gomycode` `python` `numpy`

## Dataset
Pima Indians Diabetes Dataset — 768 samples, 8 features, binary target (`Outcome`)

## Checkpoint Criteria Addressed
| Criterion | How It's Met |
|---|---|
| Tech Skills | Full TF/Keras model with proper layer configuration |
| Quality of Work | Clean, commented code; visualisations; manual forward pass verification |
| Problem-Solving | Class balancing, early stopping, standardisation |
| Submission Format | Structured notebook + README documentation |
