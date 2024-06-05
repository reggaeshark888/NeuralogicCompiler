# Project Title

This repository contains various files and notebooks related to the development, training, and evaluation of MLP models with ternary weights, including an ILP algorithm for determining an integer set of weights, hyperparameter optimization, and logic expression weight generation.

## Repository Structure

- **minimum_integer_set_algorithm.ipynb**: This notebook contains an Integer Linear Programming (ILP) algorithm for determining an optimal integer set of weights.
- **hyperparameter_experiments.ipynb**: This notebook performs a grid search for finding the optimal hyperparameters for the MLP models.
- **ternary_weight_mlp.py**: This script contains the code for creating MLP models, datasets, and training algorithms with ternary weights.
- **experiments.ipynb**: This notebook contains experiments related to the interpretability of the MLP models.
- **Logic*.py**: These scripts contain code for generating the corresponding MLP weights for logic expressions.
- **MyVisitor.py**: This script contains additional code for generating the corresponding MLP weights for logic expressions.

## File Descriptions

### minimum_integer_set_algorithm.ipynb

This notebook implements an ILP algorithm to determine an optimal set of integer weights. It explores different approaches and methodologies to solve the problem efficiently.

### hyperparameter_experiments.ipynb

This notebook conducts a comprehensive grid search to find the best hyperparameters for the MLP models. It includes detailed experiments and results analysis to identify the optimal settings for model training.

### ternary_weight_mlp.py

This script defines the architecture for MLP models with ternary weights. It includes the following components:
- **Model Architecture**: Code for creating MLP models with ternary weights.
- **Dataset Creation**: Functions to generate and preprocess datasets for training.
- **Training Algorithms**: Implementations of training algorithms for the MLP models.

### experiments.ipynb

This notebook focuses on the interpretability experiments of the MLP models. It includes various experiments to analyze and interpret the model behavior and decision-making processes.

### Logic*.py and MyVisitor.py

These scripts are responsible for generating the corresponding MLP weights for logic expressions. They include:
- **Logic Expression Parsing**: Code to parse and process logic expressions.
- **Weight Generation**: Functions to generate MLP weights based on the parsed logic expressions.

## Usage

1. Clone the repository to your local machine.
2. Ensure you have the required dependencies installed (listed in `requirements.txt`).
3. Explore the notebooks and scripts to understand and run the experiments and algorithms.

## Installation

```bash
git clone <repository_url>
cd <repository_directory>
pip install -r requirements.txt
