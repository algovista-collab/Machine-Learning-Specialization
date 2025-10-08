# ✍️ Machine Learning Notes

## 1. Introduction

1. **Machine Learning** is a field where machines are trained to predict outputs without being explicitly programmed.  
2. **Supervised Learning**: Both inputs (feature variables) and outputs (target variables) are given to train the model. The machine predicts the output for a new input.  

   - Two types:  
     - **Linear Regression** – Output can be any continuous value (e.g., Housing Price Prediction)  
     - **Classification** – Output can be any finite categories/classes (e.g., Spam Filtering)  

3. **Unsupervised Learning**: Only input is given to the machine. It recognizes patterns in the input data and clusters the output with common characteristics (e.g., Clustering Algorithms).  

---

## 2. Linear Regression Model (Univariate)

**Model Equation:**  

$$
f(x) = wx + b
$$

- **Parameters:** $w$ (weight) and $b$ (bias)  
- **Predicted Variable:** $f(x)$ or $\hat{y}$  

**Cost Function (Mean Squared Error):**  

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
$$

where $m$ is the number of training examples.  

**Explanation:**  

- The model predicts output based on the weight $w$ and bias $b$.  
- The **weight** represents the importance of the feature variable. A higher weight indicates a more significant feature.  
- The **bias** represents the base output when input is 0.  
- We square the error in the cost function to avoid cancellation of positive and negative errors.  
- Dividing by $2m$ averages the total error and makes the derivative cleaner for Gradient Descent.  

**Example:** In multivariate regression, the size of a house may have a positive weight, and the age may have a negative weight, indicating that older houses tend to have lower prices.  

---

## 3. Gradient Descent

To find optimal values of $w$ and $b$, we use **Gradient Descent**, which updates parameters iteratively to minimize the cost function.  

- If the cost function increases when $w$ increases, we decrease $w$.  
- If the cost function decreases when $w$ increases, we keep increasing $w$.  
- This process continues until the cost function reaches a minimum or the parameters converge.  

**Learning Rate ($\alpha$):** Determines how fast or slow we update the parameters.  

**Gradient Descent Update Equations:**  

$$
\theta := \theta - \alpha \frac{\partial J(w, b)}{\partial \theta}
$$

For linear regression:  

$$
w := w - \alpha \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x^{(i)}
$$

$$
b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
$$
