# 🧠 Model Training Steps

---

## 1. Define the Output

### Logistic Regression
z = w * x + b  
f(x) = 1 / (1 + np.exp(-z))

### Neural Network
model = Sequential([
    Dense(...),
    Dense(...)
])

---

## 2. Specify the Loss and Cost

### Logistic Loss
loss = -y * log(y_hat) - (1 - y) * log(1 - y_hat)

### Neural Network Compilation
model.compile(loss=BinaryCrossentropy())  
# or  
model.compile(loss=MeanSquaredError())

---

## 3. Training on Data to Minimize the Cost

### Logistic Regression
w = w - alpha * dj_dw  
b = b - alpha * dj_db

### Neural Network
model.fit(X, y, epochs=100)

---

## ⚙️ Different Activation Functions

| Activation Function | Formula | Notes |
|----------------------|----------|--------|
| **ReLU** | g(z) = max(0, z) | Most common in hidden layers, faster convergence |
| **Linear** | g(z) = w * x + b | Used for regression problems |
| **Sigmoid** | g(z) = 1 / (1 + np.exp(-z)) | Used for binary classification |

> **Note:** Without activation functions, a neural network behaves like simple linear regression.  
> Activation functions introduce non-linearity, allowing the model to handle complex patterns.

---

## 🧩 Output Layer Activations

- Choose **ReLU**, **Sigmoid**, or **Linear** based on the problem type.  
- **Hidden Layers:** ReLU is preferred because it only flattens on one side (faster and easier optimization).  
- **Sigmoid:** Flattens at both ends (0 and 1), leading to slower convergence.

---

## 🔢 Softmax — Generalization of Logistic Regression

- **Logistic Regression** → Binary Classification  
- **Softmax** → Multi-class Classification  

### Logistic Regression
z = w * x + b  
a1 = g(z) = 1 / (1 + np.exp(-z)) = P(y=1 | x)  
a2 = 1 - a1 = P(y=0 | x)

### Softmax for Multiclass Classification
For classes y = 1, 2, 3, 4:

z1 = w1 * x + b1  
z2 = w2 * x + b2  
z3 = w3 * x + b3  
z4 = w4 * x + b4  

a1 = e^(z1) / (e^(z1) + e^(z2) + e^(z3) + e^(z4))  
a2 = e^(z2) / (e^(z1) + e^(z2) + e^(z3) + e^(z4))  
a3 = e^(z3) / (e^(z1) + e^(z2) + e^(z3) + e^(z4))  
a4 = e^(z4) / (e^(z1) + e^(z2) + e^(z3) + e^(z4))

### General Softmax Formula
For y = 1, 2, ..., N:

z_j = w_j * x + b_j    (for j = 1, 2, ..., N)  
a_j = exp(z_j) / Σ(exp(z_k))   (for k = 1 to N)  
P(y = j | x) = a_j

---

✅ **Summary**
- Logistic Regression → Single layer (binary)  
- Neural Network → Multi-layer (nonlinear)  
- ReLU → Hidden layers  
- Sigmoid / Softmax → Output layers
