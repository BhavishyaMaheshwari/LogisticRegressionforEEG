# EEG Emotion Classification using Logistic Regression

This project implements a classical machine learning pipeline for emotion recognition from EEG features stored in MATLAB (`.mat`) files, using the **SEED-IV dataset**.  
The model classifies **Fear**, **Sadness**, and **Disgust** using **Logistic Regression** after feature aggregation and normalization.

---

## Overview

- **Dataset**: SEED-IV (emotion recognition from EEG)
- **Input**: EEG differential entropy (DE) features stored in `.mat` files  
- **Feature Processing**:
  - Trial-wise aggregation
  - Time dimension reduction using mean pooling
  - Feature vector flattening
- **Model**: Multiclass Logistic Regression
- **Evaluation**:
  - Accuracy
  - Classification report
  - Confusion matrix visualization

---

## Dataset: SEED-IV

The experiments are conducted on the **SEED-IV** dataset, a benchmark EEG emotion recognition dataset collected using film stimuli.

### Dataset Characteristics

- **Subjects**: 15
- **Sessions**: 3 per subject
- **Emotions**:
  - Fear
  - Sadness
  - Disgust
  - Happiness
- **EEG Channels**: 62
- **Sampling Rate**: 200 Hz
- **Features Used**: Differential Entropy (DE)

Each subject file contains **24 trials**, corresponding to emotional stimuli.

---

## Dataset Structure

All EEG feature files must be placed inside the following directory:

```

./mat/

```

- Each `.mat` file corresponds to one subject/session
- Each file contains **24 trials**, named:

```

de_LDS1, de_LDS2, ..., de_LDS24

````

### Emotion Labels

| Emotion    | Label | Trials  |
|------------|-------|---------|
| Fear       | 0     | 1 – 6   |
| Sadness    | 1     | 7 – 12  |
| Disgust    | 2     | 13 – 18 |
| Happiness  | 3     | 19 – 24 |

> **Note:** The happiness class is removed in this experiment to focus on negative emotional states.

---

## Feature Processing Pipeline

1. Load SEED-IV `.mat` files for each subject
2. For each trial:
   - Compute the mean across the time dimension
   - Flatten channel–frequency features into a 1D vector
3. Ensure uniform feature length across trials
4. Concatenate all trials from all subjects
5. Remove happiness samples
6. Standardize features using `StandardScaler`

---

## Model Details

- **Classifier**: Logistic Regression
- **Solver**: `lbfgs`
- **Max Iterations**: 3000
- **Multiclass Handling**: Softmax (internal)

```python
LogisticRegression(max_iter=3000, solver="lbfgs")
````

---

## Train–Test Split

* **Train/Test Ratio**: 80% / 20%
* **Stratified split** to preserve class distribution
* **Random State**: 42

---

## Model Diagram
<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/94e89fc9-db42-4dee-9901-4483bdf3266d" />




## Results (Baseline Analysis)

The trained Logistic Regression model demonstrates effective discrimination between the three emotional states using aggregated SEED-IV EEG features.

### Quantitative Performance


* Precision and recall values indicate balanced performance, suggesting that no single class dominates the predictions.
* The F1-scores show consistent separability between **Fear**, **Sadness**, and **Disgust**.

### Confusion Matrix Analysis

* Most predictions lie along the diagonal, indicating correct classification.
* Minor confusion is observed between **Fear** and **Sadness**, which is expected due to overlapping neurophysiological patterns.
* **Disgust** shows comparatively stronger separability.

### Interpretation

These results confirm that:

* Mean-pooled DE features from SEED-IV still retain emotional information.
* Linear classifiers can serve as strong baselines for EEG emotion recognition.
* Further improvements are likely with temporal modeling, nonlinear classifiers, and quantumization of logistic regression

---

## Results 


### Classification Performance

| Class   | Precision | Recall | F1-score |
| ------- | --------- | ------ | -------- |
| Fear    |    0.74   |  0.94  |   0.83   |
| Sadness |   0.80    |  0.89  |   0.84   |
| Disgust |   0.91    |  0.56  |   0.69   |

**Overall Accuracy:** `79.6%`

---

### Confusion Matrix

<img width="633" height="476" alt="image" src="https://github.com/user-attachments/assets/ad73c3de-da51-4cb0-8649-0d6f0aa93f17" />



## Evaluation Metrics

* Overall **Accuracy**
* **Precision, Recall, F1-score** for each class
* **Confusion Matrix** visualization

Evaluated classes:

```
Fear, Sadness, Disgust
```

---

## Dependencies

Install required packages using:

```bash
pip install numpy scipy scikit-learn matplotlib tqdm
```

---

## How to Run

1. Place all SEED-IV `.mat` files in the `./mat/` directory
2. Run the script:

```bash
python LRfE.py
```

3. Outputs:

   * Accuracy score
   * Confusion matrix plot



