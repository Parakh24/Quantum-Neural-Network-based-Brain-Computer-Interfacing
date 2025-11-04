"""This script performs:
1. Data preprocessing and scaling
2. Dimensionality reduction (PCA)
3. Model training using Decision Tree, Random Forest, and SVR
4. Hyperparameter tuning using GridSearchCV
5. Model evaluation using accuracy_score , precision_score , recall_score , f1_score
6. Visualization (Actual vs Predicted scatter plots)
"""

# ---------------------- Importing Libraries ----------------------
import pandas as pd 
import numpy as np 
from PIL import Image 
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

# ---------------------- Load Image ----------------------
img = Image.open('Histopathological_100x.png').convert("L")  # converts it to grayscale 
img = img.resize((32, 32)) 

# ---------------------- Load and Preprocess the Image ----------------------
arr = np.array(img) 
df = pd.DataFrame(np.random.randint(0, 256, (100, 1024)), columns=[f'pixel_{i}' for i in range(1024)])

# Create 50 ones and 50 zeros
labels = np.array([1]*50 + [0]*50)
np.random.shuffle(labels)
df['label'] = labels

X = df[['pixel_0', 'pixel_1', 'pixel_2', 'pixel_3', 'pixel_4', 'pixel_5', 'pixel_6', 'pixel_7',
        'pixel_8', 'pixel_9', 'pixel_10', 'pixel_11', 'pixel_12', 'pixel_13', 'pixel_14', 'pixel_15',
        'pixel_16', 'pixel_17', 'pixel_18', 'pixel_19', 'pixel_20', 'pixel_21', 'pixel_22', 'pixel_23',
        'pixel_24', 'pixel_25', 'pixel_26', 'pixel_27', 'pixel_28', 'pixel_29']]
y = df['label']

# ---------------------- Data Preprocessing ----------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(X)  

# ---------------------- Dimensionality Reduction ----------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled_data) 

# ---------------------- Train-Test Split ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

# =====================================================================
#  Naive Bayes
# =====================================================================
model = GaussianNB() 
model.fit(X_train, y_train)
y_pred_2 = model.predict(X_test)

# ---------------------- Hyperparameter Tuning ----------------------
# Fixed: GaussianNB supports only var_smoothing
param_grid_2 = {
    'var_smoothing': np.logspace(-9, -1, 5)
}

grid_search_2 = GridSearchCV(
    estimator=model,
    param_grid=param_grid_2,
    cv=3, 
    scoring='accuracy',
    verbose=1
)
grid_search_2.fit(X_train, y_train)

# ---------------------- Model Evaluation ----------------------
print("Accuracy Score for Histopathological_40x (NB):", accuracy_score(y_test, y_pred_2)*100)
print("Precision Score for Histopathological_40x (NB):", precision_score(y_test, y_pred_2)*100)
print("Recall Score for Histopathological_40x (NB):", recall_score(y_test, y_pred_2)*100)
print("F1 Score for Histopathological_40x (NB):", f1_score(y_test, y_pred_2)*100)
