from qiskit import QuantumCircuit
from PIL import Image
from qiskit_aer import Aer
from qiskit_algorithms.optimizers import COBYLA
import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from matplotlib import pyplot as plt

# ---------------------- Load and Preprocess Image ----------------------
def load_image(path):
    img = Image.open(path).convert("L")
    img = img.resize((32, 32))
    return np.array(img).flatten()

# ---------------------- Simulate Dataset ----------------------
def simulate_dataset(base_array, num_samples=100):
    data = np.tile(base_array, (num_samples, 1))
    labels = np.array([1]*50 + [0]*50)
    np.random.shuffle(labels)
    df = pd.DataFrame(data, columns=[f'pixel_{i}' for i in range(data.shape[1])])
    df['label'] = labels
    return df

# ---------------------- Quantum Circuit ----------------------
def QUANTUMCIRCUIT(parameters, x):
    qc = QuantumCircuit(1)
    qc.ry(x[0] * parameters[0], 0)
    qc.measure_all()
    return qc

# ---------------------- Cost Function ----------------------
def COSTFUNCTION(parameters, X, y):
    total_cost = 0
    backend = Aer.get_backend('aer_simulator')
    for i in range(len(X)):
        qc = QUANTUMCIRCUIT(parameters, X.iloc[i].values)
        res = backend.run(qc, shots=100).result()
        counts = res.get_counts()
        predicted = int(max(counts, key=counts.get))
        total_cost += (predicted - y.iloc[i]) ** 2
    return total_cost

# ---------------------- Training and Evaluation ----------------------
def quantum_pipeline(df, title):
    X = df[[f'pixel_{i}' for i in range(30)]]
    y = df['label']

    parameters = np.random.rand(1)
    optimizer = COBYLA(maxiter=10)
    result = optimizer.minimize(lambda p: COSTFUNCTION(p, X, y), x0=parameters)
    optimal_params = result.x

    test_pred = []
    backend = Aer.get_backend('aer_simulator')
    for i in range(len(X)):
        qc = QUANTUMCIRCUIT(optimal_params, X.iloc[i].values)
        res = backend.run(qc, shots=100).result()
        counts = res.get_counts()
        test_pred.append(int(max(counts, key=counts.get)))

    metrics = {
        "Accuracy": accuracy_score(y, test_pred) * 100,
        "Precision": precision_score(y, test_pred, zero_division=0) * 100,
        "Recall": recall_score(y, test_pred, zero_division=0) * 100,
        "F1 Score": f1_score(y, test_pred, zero_division=0) * 100
    }

    print(f"\nEvaluation for {title}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}%")

    return metrics

# ---------------------- Run for All Datasets ----------------------
arr_40x = load_image("Histopathological_40x.png")
arr_100x = load_image("Histopathological_100x.png")
arr_200x = load_image("Histopathological_200x.png")
arr_400x = load_image("Histopathological_400x.png")

df_40x = simulate_dataset(arr_40x)
df_100x = simulate_dataset(arr_100x)
df_200x = simulate_dataset(arr_200x)
df_400x = simulate_dataset(arr_400x)

metrics_40x = quantum_pipeline(df_40x, "Histopathological_40x")
metrics_100x = quantum_pipeline(df_100x, "Histopathological_100x")
metrics_200x = quantum_pipeline(df_200x, "Histopathological_200x")
metrics_400x = quantum_pipeline(df_400x, "Histopathological_400x")

# ---------------------- Visualization ----------------------
metrics_names = list(metrics_40x.keys())
x = np.arange(len(metrics_names))
width = 0.18

values_40x = list(metrics_40x.values())
values_100x = list(metrics_100x.values())
values_200x = list(metrics_200x.values())
values_400x = list(metrics_400x.values())

plt.figure(figsize=(10, 6))
plt.bar(x - 1.5*width, values_40x, width, label='40x Dataset', color='royalblue')
plt.bar(x - 0.5*width, values_100x, width, label='100x Dataset', color='salmon')
plt.bar(x + 0.5*width, values_200x, width, label='200x Dataset', color='seagreen')
plt.bar(x + 1.5*width, values_400x, width, label='400x Dataset', color='darkorange')

plt.xlabel("Evaluation Metrics")
plt.ylabel("Percentage (%)")
plt.title("Quantum Classifier Evaluation Metrics: 40x vs 100x vs 200x vs 400x Histopathological Datasets")
plt.xticks(x, metrics_names)
plt.ylim(0, 100)
plt.legend()

for i in range(len(metrics_names)):
    plt.text(x[i] - 1.5*width, values_40x[i] + 1, f"{values_40x[i]:.2f}%", ha='center', fontsize=8)
    plt.text(x[i] - 0.5*width, values_100x[i] + 1, f"{values_100x[i]:.2f}%", ha='center', fontsize=8)
    plt.text(x[i] + 0.5*width, values_200x[i] + 1, f"{values_200x[i]:.2f}%", ha='center', fontsize=8)
    plt.text(x[i] + 1.5*width, values_400x[i] + 1, f"{values_400x[i]:.2f}%", ha='center', fontsize=8)

plt.tight_layout()
plt.show()
