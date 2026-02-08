import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data_path = r"./mat"

X = []
y = []

for fname in tqdm(os.listdir(data_path)):
    if not fname.endswith(".mat"):
        continue

    mat = loadmat(os.path.join(data_path, fname))

    subject_trials = []

    for trial in range(1, 25):
        data = mat[f"de_LDS{trial}"]
        data = np.mean(data, axis=-1)   # remove time dimension
        subject_trials.append(data.flatten())

    # make all trials same length (safety)
    min_len = min(len(v) for v in subject_trials)
    subject_trials = [v[:min_len] for v in subject_trials]

    X.extend(subject_trials)

    # labels: 6 fear, 6 sadness, 6 disgust, 6 happiness
    y.extend([0]*6 + [1]*6 + [2]*6 + [3]*6)


X = np.array(X)
y = np.array(y)

# drop happiness
keep = y != 3
X = X[keep]
y = y[keep]

print("Data shape:", X.shape)
print("Classes:", np.unique(y))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


clf = LogisticRegression(max_iter=3000, solver="lbfgs")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(
    y_test, y_pred,
    target_names=["Fear", "Sadness", "Disgust"]
))


cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm, cmap="Blues")
plt.colorbar()

labels = ["Fear", "Sadness", "Disgust"]
plt.xticks(range(3), labels)
plt.yticks(range(3), labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()
