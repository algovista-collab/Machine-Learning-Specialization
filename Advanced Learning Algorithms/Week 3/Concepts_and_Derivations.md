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

# Using Training, Cross-Validation, and Test Sets

## Roles of Each Set
- **Training set** → Learn model parameters.
- **Cross-validation (CV) set** → Evaluate different models/hyperparameters and pick the best.
- **Test set** → Estimate final model performance (unbiased).

---

## Workflow for Polynomial Model Selection

1. **Choose a polynomial degree** (e.g., 1, 2, 3, …).  
2. **Train model** on the **training set** and get the model parameters.  
3. **Compute CV error** on the **cross-validation set**.  
4. Repeat for all candidate degrees.  
5. **Select model** with lowest CV error.  
6. (Optional) Retrain best model on **training + CV set**.  
7. **Evaluate final model** on **test set** → unbiased performance.

---

### Key Points
- Training set is **always used** to learn model parameters.  
- Cross-validation set is **never used for training**; only for evaluation.  
- Test set is **never touched** until the final evaluation.

---

# Bias, Variance, and Regularization

## Bias and Variance Cases
- **Case 1: High Bias**  
  - `J_train` and `J_cv` are both high  
  - Model underfits the data → poor fit on both training and CV sets

- **Case 2: High Variance**  
  - `J_train` is low, `J_cv` is high  
  - Model overfits the training data → does not generalize well

- **Case 3: Low Bias / Low Variance**  
  - `J_train` is low, `J_cv` is low  
  - Model fits well and generalizes

---

## Effect of Polynomial Degree on Error

- As **polynomial degree `d` increases**:
  - `J_train` **decreases** (model fits training data better)
  - `J_cv` **decreases initially** and then **increases** after a point (overfitting occurs)

- **Error curve behavior**:
  - Very low degree → `J_train` and `J_cv` both high → underfitting  
  - Moderate degree → `J_train` low, `J_cv` low → good fit  
  - Very high degree → `J_train` very low, `J_cv` increases → overfitting  

<img width="595" height="381" alt="image" src="https://github.com/user-attachments/assets/46c8a979-e57e-4595-8944-f0c16933343e" />

> Sometimes underfitting and overfitting can occur simultaneously depending on the data and model.

---

## Effect of Regularization Parameter `λ` on Error

Consider a model with 4th order polynomial:  

- **Large λ**  
  - `J_train` is high  
  - Parameters shrink toward 0 → model underfits  
  - Regularization term dominates the cost

- **Small λ**  
  - `J_train` is low, `J_cv` is high  
  - Model overfits → focuses too much on training error

- **Intermediate λ**  
  - `J_train` and `J_cv` both reasonably low → good tradeoff

---

## Establishing a Baseline Performance
1. Determine a **reasonable error level** based on:
   - Human-level performance
   - Competing algorithm performance
   - Experience or intuition

2. Observing `J_train` and `J_cv` as `λ` changes:
   - Low λ → `J_train` low, `J_cv` high → overfitting  
   - High λ → `J_train` high, `J_cv` high → underfitting  
   - Choose **intermediate λ** for best tradeoff
     
---

<img width="619" height="351" alt="image" src="https://github.com/user-attachments/assets/29bdad24-16f4-44e4-83a2-584f1baf6964" />
