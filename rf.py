# ---------------------- Importing Libraries ----------------------
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

# ---------------------- Load and Preprocess Images ----------------------
img_40x = Image.open('Histopathological_40x.png').convert("L")
img_100x = Image.open('Histopathological_100x.png').convert("L")
img_200x = Image.open('Histopathological_200x.png').convert("L")
img_400x = Image.open('Histopathological_400x.png').convert("L")

img_40x = img_40x.resize((32, 32))
img_100x = img_100x.resize((32, 32))
img_200x = img_200x.resize((32, 32))
img_400x = img_400x.resize((32, 32))

arr_40x = np.array(img_40x).flatten()
arr_100x = np.array(img_100x).flatten()
arr_200x = np.array(img_200x).flatten()
arr_400x = np.array(img_400x).flatten()

# ---------------------- Simulate Dataset ----------------------
num_samples = 100
def simulate_data(base_array):
    data = []
    for _ in range(num_samples):
        noise = np.random.normal(0, 5, base_array.shape)
        sample = np.clip(base_array + noise, 0, 255)
        data.append(sample)
    return pd.DataFrame(data, columns=[f'pixel_{i}' for i in range(base_array.shape[0])])

df_40x = simulate_data(arr_40x)
df_100x = simulate_data(arr_100x)
df_200x = simulate_data(arr_200x)
df_400x = simulate_data(arr_400x)

# ---------------------- Create Labels ----------------------
def generate_labels():
    labels = np.array([1]*50 + [0]*50)
    np.random.shuffle(labels)
    return labels

df_40x['label'] = generate_labels()
df_100x['label'] = generate_labels()
df_200x['label'] = generate_labels()
df_400x['label'] = generate_labels()

# ---------------------- Feature and Label Separation ----------------------
def split_features_labels(df):
    X = df.drop(columns=['label'])
    y = df['label']
    return X, y

X_40x, y_40x = split_features_labels(df_40x)
X_100x, y_100x = split_features_labels(df_100x)
X_200x, y_200x = split_features_labels(df_200x)
X_400x, y_400x = split_features_labels(df_400x)

# ---------------------- Data Preprocessing ----------------------
scaler = MinMaxScaler()
X_40x_scaled = scaler.fit_transform(X_40x)
X_100x_scaled = scaler.fit_transform(X_100x)
X_200x_scaled = scaler.fit_transform(X_200x)
X_400x_scaled = scaler.fit_transform(X_400x)

# ---------------------- Train-Test Split ----------------------
def train_test(X_scaled, y):
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_40x_train_raw, X_40x_test_raw, y_40x_train, y_40x_test = train_test(X_40x_scaled, y_40x)
X_100x_train_raw, X_100x_test_raw, y_100x_train, y_100x_test = train_test(X_100x_scaled, y_100x)
X_200x_train_raw, X_200x_test_raw, y_200x_train, y_200x_test = train_test(X_200x_scaled, y_200x)
X_400x_train_raw, X_400x_test_raw, y_400x_train, y_400x_test = train_test(X_400x_scaled, y_400x)

# ---------------------- Dimensionality Reduction ----------------------
def apply_pca(X_train_raw, X_test_raw):
    pca = PCA(n_components=2)
    return pca.fit_transform(X_train_raw), pca.transform(X_test_raw)

X_40x_train, X_40x_test = apply_pca(X_40x_train_raw, X_40x_test_raw)
X_100x_train, X_100x_test = apply_pca(X_100x_train_raw, X_100x_test_raw)
X_200x_train, X_200x_test = apply_pca(X_200x_train_raw, X_200x_test_raw)
X_400x_train, X_400x_test = apply_pca(X_400x_train_raw, X_400x_test_raw)

# ---------------------- Model Training ----------------------
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

model_40x = train_model(X_40x_train, y_40x_train)
model_100x = train_model(X_100x_train, y_100x_train)
model_200x = train_model(X_200x_train, y_200x_train)
model_400x = train_model(X_400x_train, y_400x_train)

y_40x_pred = model_40x.predict(X_40x_test)
y_100x_pred = model_100x.predict(X_100x_test)
y_200x_pred = model_200x.predict(X_200x_test)
y_400x_pred = model_400x.predict(X_400x_test)

# ---------------------- Hyperparameter Tuning ----------------------
param_grid = {
    'n_estimators': [50],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5]
}

def tune_model(model, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=2, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search

grid_search_40x = tune_model(model_40x, X_40x_train, y_40x_train)
grid_search_100x = tune_model(model_100x, X_100x_train, y_100x_train)
grid_search_200x = tune_model(model_200x, X_200x_train, y_200x_train)
grid_search_400x = tune_model(model_400x, X_400x_train, y_400x_train)

# ---------------------- Model Evaluation ----------------------
def evaluate_model(y_test, y_pred):
    return [
        accuracy_score(y_test, y_pred) * 100,
        precision_score(y_test, y_pred) * 100,
        recall_score(y_test, y_pred) * 100,
        f1_score(y_test, y_pred) * 100
    ]

values_40x = evaluate_model(y_40x_test, y_40x_pred)
values_100x = evaluate_model(y_100x_test, y_100x_pred)
values_200x = evaluate_model(y_200x_test, y_200x_pred)
values_400x = evaluate_model(y_400x_test, y_400x_pred)

# ---------------------- Visualization ----------------------
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
x = np.arange(len(metrics))
width = 0.18

plt.figure(figsize=(10, 6))
plt.bar(x - 1.5*width, values_40x, width, label='40x Dataset', color='royalblue')
plt.bar(x - 0.5*width, values_100x, width, label='100x Dataset', color='salmon')
plt.bar(x + 0.5*width, values_200x, width, label='200x Dataset', color='seagreen')
plt.bar(x + 1.5*width, values_400x, width, label='400x Dataset', color='darkorange')

plt.xlabel("Evaluation Metrics")
plt.ylabel("Percentage (%)")
plt.title("Random Forest Evaluation Metrics: 40x vs 100x vs 200x vs 400x Histopathological Datasets")
plt.xticks(x, metrics)
plt.ylim(0, 100)
plt.legend()

for i in range(len(metrics)):
    plt.text(x[i] - 1.5*width, values_40x[i] + 1, f"{values_40x[i]:.2f}%", ha='center', fontsize=8)
    plt.text(x[i] - 0.5*width, values_100x[i] + 1, f"{values_100x[i]:.2f}%", ha='center', fontsize=8)
    plt.text(x[i] + 0.5*width, values_200x[i] + 1, f"{values_200x[i]:.2f}%", ha='center' , fontsize = 8) 
    plt.text(x[i] + 1.5*width, values_400x[i] + 1, f"{values_400x[i]:.2f}%", ha='center' , fontsize = 8) 

plt.tight_layout() 
plt.show() 