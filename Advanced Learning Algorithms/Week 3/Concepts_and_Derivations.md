# Diagnosing and Selecting a Machine Learning Model

When we have a trained model but it has unexpectedly huge errors, we need to decide what to do next. **Diagnostics** is a test we run to gain insight into what is and is not working with a learning algorithm, helping guide improvements in performance.

---

## Step 1: Split the Data
Divide the data into:

- **Training set**
- **Test set**

Calculate the cost (or error) for both sets:

- If **training cost is low** but **test cost is high**, the model is **not generalizing well** to new data (overfitting).

> For classification problems, we can measure performance using the **fraction of misclassified data** instead of computing logistic loss.

---

## Step 2: Model Selection
We can try models of different complexity, e.g., polynomials of varying degrees.

- Calculate the **test cost \(J_{test}\)** to see if it is reducing.
- **Warning:** Selecting the model with the lowest test cost on the same test set can lead to overfitting the test data.  
  ✅ **Do not finalize the model by repeatedly testing on the same test set.**

---

## Step 3: Use Cross-Validation
Introduce a **cross-validation set** to properly evaluate models:

- Track:
  - **Training error**
  - **Cross-validation error**
  - **Test error**

- Evaluate different polynomial degrees on the **cross-validation set** and pick the best one.
- Test the final model on the **test set**, which the model has never seen.  
  ✅ This gives an **unbiased estimate** of the model's performance.

---
