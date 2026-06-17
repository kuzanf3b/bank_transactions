# Bank Transactions Analysis

A comprehensive machine learning project for analyzing and classifying bank transaction data using clustering and classification techniques.

## 📋 Project Overview

This project implements data science workflows for analyzing bank transactions, including:
- **Clustering Analysis**: Grouping transactions into distinct patterns using unsupervised learning
- **Classification Modeling**: Predicting transaction categories using supervised learning approaches
- **Model Optimization**: Hyperparameter tuning and model selection

## 📁 Repository Structure

```
bank_transactions/
├── classification.ipynb                    # Classification model development and evaluation
├── clustering.ipynb                        # Clustering analysis and visualization
├── data_clustering.csv                     # Processed data for clustering tasks
├── data_clustering_inverse.csv             # Inverse-transformed clustering data
├── decision_tree_model.h5                  # Trained Decision Tree model
├── explore_random_forest_classification.h5 # Random Forest exploration model
├── tuning_classification.h5                # Tuned classification model
├── PCA_model_clustering.h5                 # Principal Component Analysis model
└── model_clustering.h5                     # Trained clustering model
```

## 📊 Key Components

### Notebooks

- **`classification.ipynb`**: Develops and evaluates classification models for transaction prediction
  - Multiple machine learning algorithms tested
  - Model evaluation and performance metrics
  - Hyperparameter optimization

- **`clustering.ipynb`**: Performs clustering analysis on transaction data
  - Unsupervised learning approach to identify transaction patterns
  - PCA dimensionality reduction
  - Cluster visualization and interpretation

### Data Files

- **`data_clustering.csv`**: Primary dataset for clustering analysis
- **`data_clustering_inverse.csv`**: Inverse-transformed data for interpretability

### Model Files

Pre-trained models saved in HDF5 format:
- `decision_tree_model.h5`: Decision Tree classifier
- `explore_random_forest_classification.h5`: Random Forest model exploration
- `tuning_classification.h5`: Optimized classification model
- `PCA_model_clustering.h5`: PCA transformation model
- `model_clustering.h5`: Clustering model

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Libraries: pandas, scikit-learn, numpy, matplotlib, seaborn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kuzanf3b/bank_transactions.git
cd bank_transactions
```

2. Install required packages:
```bash
pip install jupyter pandas scikit-learn numpy matplotlib seaborn
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

### Usage

1. Open `clustering.ipynb` to explore transaction clustering patterns
2. Open `classification.ipynb` to see classification model development
3. Examine the pre-trained models loaded from `.h5` files

## 📈 Methodology

### Clustering Approach
- Dimensionality reduction using PCA
- Unsupervised clustering to identify transaction groups
- Pattern analysis and visualization

### Classification Approach
- Multiple supervised learning algorithms
- Model evaluation using standard metrics
- Hyperparameter tuning for optimal performance
- Model comparison and selection

## 🔧 Technologies Used

- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Model Serialization**: HDF5 format
- **Development**: Jupyter Notebook

## 📝 Notes

- All models are pre-trained and saved as `.h5` files
- Data has been processed and normalized for analysis
- The project includes both original and inverse-transformed datasets for interpretability

## 📄 License

This project is open source and available on GitHub.

## 👤 Author

**kuzanf3b**

---

**Last Updated**: June 2026
