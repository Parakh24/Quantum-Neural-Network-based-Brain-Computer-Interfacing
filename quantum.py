from qiskit import QuantumCircuit
from PIL import Image
from qiskit_aer import Aer
from qiskit_algorithms.optimizers import COBYLA
import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
                                                                                      
# Sample data


#load Image 
img = Image.open('Histopathological.png').convert("L")  #converts it to grayscale 
img = img.resize((32, 32)) 

print(img) 

#load and preprocess the image 

arr = np.array(img) 
df = pd.DataFrame(np.random.randint(0, 256, (100, 1024)), columns=[f'pixel_{i}' for i in range(1024)])

# Create 50 ones and 50 zeros
labels = np.array([1]*50 + [0]*50)

# Shuffle the labels randomly
np.random.shuffle(labels)

# Assign to DataFrame
df['label'] = labels

print(df['label'].value_counts())
print(df.head(3))


X = df[['pixel_0' , 'pixel_1' , 'pixel_2' , 'pixel_3' , 'pixel_4' , 'pixel_5' , 'pixel_6' , 'pixel_7' ,'pixel_8', 'pixel_9', 'pixel_10', 'pixel_11', 'pixel_12', 'pixel_13', 'pixel_14', 'pixel_15', 'pixel_16', 'pixel_17', 'pixel_18', 'pixel_19', 'pixel_20', 'pixel_21', 'pixel_22', 'pixel_23', 'pixel_24', 'pixel_25', 'pixel_26', 'pixel_27', 'pixel_28', 'pixel_29']]
y = df['label']

# Define quantum circuit
def QUANTUMCIRCUIT(parameters, x):
    qc = QuantumCircuit(1)
    qc.ry(x[0] * parameters[0], 0)
    qc.measure_all()
    return qc

# Define cost function
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

# Training
epochs = 10
parameters = np.random.rand(1)
optimizer = COBYLA(maxiter=epochs)
result = optimizer.minimize(lambda p: COSTFUNCTION(p, X, y), x0=parameters)
optimal_params = result.x

# Evaluation
test_pred = []
backend = Aer.get_backend('aer_simulator')

for i in range(len(X)):
    qc = QUANTUMCIRCUIT(optimal_params, X.iloc[i].values)
    res = backend.run(qc, shots=100).result()
    counts = res.get_counts()
    test_pred.append(int(max(counts, key=counts.get)))

accuracy = accuracy_score(y, test_pred)
precision = precision_score(y, test_pred, zero_division=0)
recall = recall_score(y, test_pred, zero_division=0)
f1 = f1_score(y, test_pred, zero_division=0)

print(f"\nAccuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-Score:  {f1:.2f}")
