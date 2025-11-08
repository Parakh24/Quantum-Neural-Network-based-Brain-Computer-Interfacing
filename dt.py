# ---------------------- Importing Libraries ----------------------
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

# ---------------------- Load and Preprocess Images ----------------------
def load_image(path):
    img = Image.open(path).convert("L")
    img = img.resize((32, 32))
    return np.array(img).flatten()

arr_40x = load_image("Histopathological_40x.png")
arr_100x = load_image("Histopathological_100x.png")
arr_200x = load_image("Histopathological_200x.png")
arr_400x = load_image("Histopathological_400x.png")

# ---------------------- Simulate Dataset ----------------------
def simulate_dataset(base_array, num_samples=100):
    data = []
    for _ in range(num_samples):
        noise = np.random.normal(0, 5, base_array.shape)
        sample = np.clip(base_array + noise, 0, 255)
        data.append(sample)
    df = pd.DataFrame(data, columns=[f'pixel_{i}' for i in range(base_array.shape[0])])
    labels = np.array([1]*50 + [0]*50)
    np.random.shuffle(labels)
    df['label'] = labels
    return df

df_40x = simulate_dataset(arr_40x)
df_100x = simulate_dataset(arr_100x)
df_200x = simulate_dataset(arr_200x)
df_400x = simulate_dataset(arr_400x)
    
# ---------------------- Model Pipeline ----------------------
def process_decision_tree(df, title):
    X = df.drop(columns=['label'])
    y = df['label']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    param_grid = {
        'max_depth': [3, 5],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred) * 100,
        "Precision": precision_score(y_test, y_pred) * 100,
        "Recall": recall_score(y_test, y_pred) * 100,
        "F1 Score": f1_score(y_test, y_pred) * 100
    }

    print(f"\nEvaluation for {title}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}%")

    return metrics

# ---------------------- Run for All Datasets ----------------------
metrics_40x = process_decision_tree(df_40x, "Histopathological_40x")
metrics_100x = process_decision_tree(df_100x, "Histopathological_100x")
metrics_200x = process_decision_tree(df_200x, "Histopathological_200x")
metrics_400x = process_decision_tree(df_400x, "Histopathological_400x")

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
plt.title("Decision Tree Evaluation Metrics: 40x vs 100x vs 200x vs 400x Histopathological Datasets")
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
