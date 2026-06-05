# Rice Stress Predictive Gene Panels

## Deep Learning and Feature Selection Reveal Compact Gene Panels Characterizing Drought and Heat Stress Responses in Rice

This repository contains the source code used in the study:

**Deep Learning and Feature Selection Reveal Compact Gene Panels Characterizing Drought and Heat Stress Responses in Rice**

### Overview

Rice (*Oryza sativa* L.) is one of the world's most important staple crops, yet its productivity is increasingly threatened by drought and heat stress. This repository provides the complete computational workflow used to identify compact predictive gene panels associated with drought and heat stress responses in rice.

The framework integrates RNA-seq data, feature selection algorithms, and deep learning models to discover biologically meaningful and highly predictive gene signatures. Feature subsets were evaluated using stratified 5-fold cross-validation and F1-score as the primary performance metric.

The final predictive gene panels consisted of:

* **10 genes** for drought stress classification
* **4 genes** for heat stress classification

Among all evaluated approaches, the combination of **Minimum-Redundancy Maximum-Relevance (mRMR)** feature selection and **Multi-Layer Perceptron Neural Network (MLPNN)** achieved the highest predictive performance.

---

## Workflow

RNA-seq Data → Feature Selection → Predictive Gene Panels → Deep Learning Evaluation → Performance Assessment

---

## Feature Selection Methods

The repository includes implementations of the following feature selection approaches:

### 1. Stability Selection (SS)

A subsampling-based feature selection framework that combines repeated random sampling with sparse logistic regression to identify features consistently selected across data partitions.

### 2. Boruta

An all-relevant feature selection method based on Random Forests and shadow features, designed to identify all biologically relevant genes associated with stress responses.

### 3. Recursive Feature Elimination (RFE) and RFECV

Wrapper-based methods that iteratively remove less informative features and identify optimal gene subsets through cross-validation.

### 4. Mutual Information (MI)

An information-theoretic method that ranks genes according to their statistical dependence with the stress phenotype.

### 5. Minimum-Redundancy Maximum-Relevance (mRMR)

A feature selection strategy that simultaneously maximizes relevance to the target phenotype while minimizing redundancy among selected genes.

### 6. Embedded Gradient Boosting Feature Selection (EGBFS)

An embedded approach that ranks genes according to feature importance scores derived from gradient boosting classifiers.

### 7. Genetic Algorithm (GA)

An evolutionary optimization approach that searches for optimal gene subsets using selection, crossover, and mutation operators.

---

## Deep Learning Methods

### 1. Multi-Layer Perceptron Neural Network (MLPNN)

Fully connected neural networks with extensive hyperparameter optimization including:

* Hidden layer architecture
* Learning rate
* Activation function
* L2 regularization

### 2. Convolutional Neural Network (CNN)

One-dimensional convolutional neural networks designed to learn local relationships among selected gene features.

### 3. Generative Adversarial Network (GAN)

GAN-based data augmentation was used to generate synthetic minority-class samples and mitigate class imbalance.

### 4. Denoising Autoencoder (DAE)

Autoencoder-based representation learning for generating compact latent feature embeddings prior to classification.

---

## Repository Structure

```text
rice-stress-predictive-gene-panels/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── feature_selection/
│   ├── stability_selection.py
│   ├── boruta.py
│   ├── rfe_rfecv.py
│   ├── mi_mrmr.py
│   ├── egbfs.py
│   └── genetic_algorithm.py
│
├── deep_learning/
│   ├── mlpnn.py
│   ├── cnn.py
│   ├── gan.py
│   └── dae.py
│
├── data/
│   ├── drought/
│   └── heat/
│
└── results/
```

---

## Software Requirements

The analyses were performed using Python and the following libraries:

```text
numpy
pandas
scikit-learn
tensorflow
keras
matplotlib
seaborn
xgboost
boruta
joblib
scipy
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Usage

### Feature Selection

Example:

```bash
python feature_selection/mi_mrmr.py
```

### Deep Learning Evaluation

Example:

```bash
python deep_learning/mlpnn.py
```

Input files, output directories, and hyperparameters can be adjusted within the corresponding scripts.

---

## Reproducibility

All feature selection methods and deep learning models were evaluated using:

* Stratified 5-fold cross-validation
* F1-score as the primary evaluation metric
* Independent training and testing datasets
* Hyperparameter optimization for each feature selection pipeline

---

## Citation

If you use this repository in your research, please cite:

Shahsavari M. *Deep Learning and Feature Selection Reveal Compact Gene Panels Characterizing Drought and Heat Stress Responses in Rice.*

(Journal information will be added upon publication.)

---

## Author

**Masoud Shahsavari**

Department of Agronomy and Plant Breeding
College of Agriculture and Natural Resources
University of Tehran
Karaj, Iran

Email: [mshahsavari@ut.ac.ir](mailto:mshahsavari@ut.ac.ir)

---

## License

This project is released under the MIT License.
