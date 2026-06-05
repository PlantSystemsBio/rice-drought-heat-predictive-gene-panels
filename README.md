Rice Stress Predictive Gene Panels

This repository contains the code used for the study:

"Deep Learning and Feature Selection Reveal Compact Gene Panels Characterizing Drought and Heat Stress Responses in Rice"

Masoud Shahsavari — University of Tehran

Overview

This project integrates RNA-seq data, multiple feature selection (FS) methods, and deep learning (DL) algorithms to identify compact predictive gene panels associated with drought and heat stress responses in rice (Oryza sativa L.).

The workflow evaluates candidate gene subsets using machine learning and deep learning classifiers, with performance assessed using stratified 5-fold cross-validation and F1-score.

Feature Selection Methods

The repository includes implementations of the following feature selection approaches:

Stability Selection (SS)

Subsampling-based sparse feature selection using logistic regression.

Boruta

All-relevant feature selection using Random Forest and shadow features.

RFE / RFECV

Recursive feature elimination with optional cross-validation for automatic subset-size determination.

Mutual Information (MI)

Feature ranking based on information-theoretic relevance.

Minimum-Redundancy Maximum-Relevance (mRMR)

Greedy selection balancing feature relevance and redundancy.

Embedded Gradient Boosting Feature Selection (EGBFS)

Tree-based embedded feature ranking using gradient boosting.

Genetic Algorithm (GA)

Evolutionary optimization of fixed-size gene subsets.

Deep Learning Algorithms

The selected gene panels are evaluated using:

Multi-layer Perceptron Neural Network (MLPNN)

Grid-searched architectures with L2 regularization and stratified 5-fold CV.

Convolutional Neural Network (CNN)

1D convolutional architectures with automated hyperparameter tuning.

Generative Adversarial Network (GAN)

Minority-class data augmentation for stress-condition samples.

Denoising Autoencoder (DAE)

Representation learning and dimensionality reduction prior to classification.

Repository Structure

rice-stress-predictive-gene-panels/

├── README.md

├── LICENSE

├── requirements.txt

│

├── feature_selection/

│ ├── stability_selection.py

│ ├── boruta.py

│ ├── rfe_rfecv.py

│ ├── mi_mrmr.py

│ ├── egbfs.py

│ └── ga_selection.py

│

├── deep_learning/

│ ├── mlpnn.py

│ ├── cnn.py

│ ├── gan_augmentation.py

│ └── dae.py

│

├── data/

│ ├── drought/

│ └── heat/

│

└── results/

Requirements

The code was developed in Python and depends on common scientific computing and machine learning libraries, including:

numpy

pandas

scikit-learn

tensorflow / keras

xgboost

boruta

matplotlib

seaborn

A complete environment can be recreated using:

pip install -r requirements.txt

Running the Workflow

Example execution order:

# 1. Perform feature selection

python feature_selection/mi_mrmr.py

# 2. Train and evaluate the best-performing deep learning model

python deep_learning/mlpnn.py

Adjust input paths and parameters according to the dataset structure.

Citation

If you use this code in your research, please cite the associated publication:

Shahsavari M. Deep Learning and Feature Selection Reveal Compact Gene Panels Characterizing Drought and Heat Stress Responses in Rice.

[Journal details to be added upon publication.]

License

This repository is released under the MIT License unless otherwise specified.
