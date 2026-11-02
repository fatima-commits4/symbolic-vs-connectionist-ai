# AI Paradigms Comparison: Symbolic vs. Connectionist Models

A comparative Machine Learning study implementing and evaluating two fundamental artificial intelligence paradigms—Symbolic (Rule-Based) AI and Connectionist (Neural Network) AI—on tabular classification tasks.

## Overview & Architecture
This project contrasts the decision-making mechanics and performance of traditional logic-driven models against sub-symbolic artificial neural networks using Python and `scikit-learn`:

- **Symbolic AI:** Utilizes a `DecisionTreeClassifier` to model explicit, tree-based logical decisions.
- **Connectionist AI:** Utilizes a Multi-Layer Perceptron (`MLPClassifier`) with hidden layers to evaluate sub-symbolic pattern learning.

## Technical Workflow
1. **Dataset Loading & Preprocessing:** Loads the Iris feature dataset and performs a standardized train/test split ($70/30$).
2. **Model Training:** Trains both classifiers on identical training feature sets.
3. **Evaluation:** Measures and compares performance accuracy metrics across both paradigm implementations.

## Tech Stack
- **Language:** Python 3
- **Libraries:** `scikit-learn` (`datasets`, `model_selection`, `tree`, `neural_network`, `metrics`)

## Project Documentation
*Full experimental write-up and analytical reports are available upon request.*
