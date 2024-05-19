# Heart Disease Classification Using Optimization Techniques

This project aims to classify a heart disease dataset using neural networks optimized with various techniques, including Gradient Descent, Genetic Algorithm, Simulated Annealing, and Randomized Hill Climbing. The performance of these techniques is compared based on metrics such as accuracy, sensitivity, specificity, and AUC.

## Dataset

The dataset used for this project is the [Heart Disease Dataset](https://www.kaggle.com/datasets/mexwell/heart-disease-dataset) from Kaggle. The dataset contains the following attributes:

1. Age
2. Sex
3. Chest pain type
4. Resting blood pressure
5. Serum cholesterol
6. Fasting blood sugar
7. Resting electrocardiogram results
8. Maximum heart rate achieved
9. Exercise induced angina
10. Oldpeak = ST depression induced by exercise
11. The slope of the peak exercise ST segment
12. Target (0: No heart disease, 1: Heart disease)

## Project Structure

The project consists of the following files:

- `code.ipynb`: Jupyter Notebook containing the implementation of neural network training with various optimization techniques.
- `comparison_table.csv`: CSV file containing the comparison table of performance metrics for different optimization techniques.
- `README.md`: Project documentation file.
- `REPORT.pdf`: Project report file.

## Installation

To run this project, you need to have the following libraries installed:

- pandas
- numpy
- scikit-learn
- matplotlib
- keras
- tensorflow
- deap
- tqdm
- simanneal

You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib keras tensorflow deap tqdm simanneal
```
# Usage
Clone the repository:
```bash
git clone https://github.com/khaireddine-arbouch/Heart-Disease-Classification-Using-Optimization-Techniques.git
cd heart-disease-classification
```
Open the Jupyter Notebook:
```bash
jupyter notebook Code.ipynb
```
Run the notebook to train the neural network using different optimization techniques and generate the comparison table.
Optimization Techniques
1. Gradient Descent
Implemented using Keras with the Adam optimizer.

2. Genetic Algorithm
Implemented using the DEAP library to optimize the hidden layer sizes of the neural network.

3. Simulated Annealing
Implemented using the Simanneal library to optimize the hidden layer sizes of the neural network.

4. Randomized Hill Climbing
Implemented using a custom function to optimize the hidden layer sizes of the neural network.

# Results
The results of the different optimization techniques are compared based on the following performance metrics:

- Accuracy: The proportion of correctly classified instances.
- Sensitivity: The proportion of actual positives correctly identified.
- Specificity: The proportion of actual negatives correctly identified.
- AUC: The Area Under the ROC Curve.

The comparison table is saved as comparison_table.csv.

# Acknowledgments
The Heart Disease Dataset is provided by Kaggle.
The DEAP library for Genetic Algorithm implementation.
The Simanneal library for Simulated Annealing implementation.
