# Machine Learning Notes

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

In practice, the regression line is the estimate that minimizes the sum of squared residual values (sum of square of the difference between actual and predicted), also called the residual sum of squares or RSS: The method of minimizing the sum of the squared residuals is termed least squares regression, or ordinary least squares (OLS) regression. It is often attributed to Carl Friedrich Gauss, the German mathematician. Regression is used both for prediction and explanation. A regression can show that X and Y are related, but it cannot prove that X causes Y. To claim causation, you need additional evidence like theory, timing, or experimental design—not just the regression results. Like we cannot say either X causes Y or Y causes X.

<img width="883" height="386" alt="image" src="https://github.com/user-attachments/assets/f8cda86b-f201-4193-ae08-22fb829c2528" />

## Model Selection Methods — Summary

### Stepwise Methods
- **Forward Selection**:  
  Start with no predictors and add them one at a time, choosing the variable that most improves \(R^2\). Stop when added variables are no longer statistically significant.
- **Backward Selection (Elimination)**:  
  Start with all predictors and iteratively remove those that are not statistically significant until all remaining predictors are significant.

### Penalized Regression
- Instead of searching across discrete models, penalized regression adds a **penalty term** to the fitting equation to discourage overly complex models.
- Coefficients are **shrunk toward zero** rather than eliminated outright.
- Closely related in spirit to **AIC**.
- Common methods:
  - **Ridge Regression**
  - **Lasso Regression**

### Overfitting and Validation
- Stepwise and all-subsets regression are **in-sample** methods and can lead to overfitting.
- **Cross-validation** is commonly used to assess and tune models on unseen data.
- In **linear regression**, overfitting is usually less severe due to the model’s simplicity.
- For **more complex or iterative models**, cross-validation is essential to ensure good generalization.

## Weighted Regression — Summary

Weighted regression assigns different importance (weights) to observations during model fitting. It is commonly used in statistics and can be useful for data science in the following cases:

- **Inverse-Variance Weighting**  
  When observations have different measurement precision, weights are set inversely proportional to their variance.  
  - Higher variance → lower weight  
  - Lower variance → higher weight

- **Aggregated Data Analysis**  
  When each row represents multiple original observations, a weight variable encodes how many cases each row stands for.

- Weighted regression is especially important in the analysis of **complex survey data**.
- Stepwise regression is a way to automatically determine which variables should be included in the model.
- Weighted regression is used to give certain records more or less weight in fitting the equation.
- A prediction interval pertains to uncertainty around a single value, while a confidence interval pertains to a mean or other statistic calculated from multiple values. Thus, a prediction interval will typically be much wider than a confidence interval for the same value.

## Key Terms for Factor Variables

Factor variables, also termed categorical variables, take on a limited number of discrete values. The keyword argument drop_first will return P – 1 columns. Use this to avoid the problem of multicollinearity.

- **Dummy Variables**  
  Binary (0–1) variables created by recoding categorical factors for use in regression and other models.

- **Reference Coding**  
  Uses one factor level as a baseline; other levels are compared against it.  
  *Also called:* **treatment coding**

- **One-Hot Encoding**  
  Retains all factor levels as separate binary variables. Common in machine learning, but not suitable for multiple linear regression.

- **Deviation Coding**  
  Compares each factor level to the overall mean rather than a reference level.  
  *Also called:* **sum contrasts**

## Key Terms for Interpreting the Regression Equation

- **Correlated Variables**  
  Predictor variables that are highly correlated make individual coefficient interpretation difficult.

- **Multicollinearity**  
  Perfect or near-perfect correlation among predictors, leading to unstable or non-estimable regression coefficients. Perfect multicollinearity occurs when one predictor variable can be expressed as a linear combination of others. Multicollinearity in regression must be addressed—variables should be removed until the multicollinearity is gone. A regression does not have a well-defined solution in the presence of perfect multicollinearity.
  *Also called:* **collinearity**

- **Confounding Variables**  
  Important predictors that, when omitted, create misleading or spurious relationships in the model.

- **Main Effects**  
  The direct relationship between a predictor and the outcome, holding other variables constant.

## Main Effects vs. Interactions

- **Main Effects** (independent variables) are the predictors in a regression model.
- Using only main effects assumes that each predictor’s relationship with the response is **independent of all other predictors**.
- In practice, this assumption often fails: the effect of one predictor can **depend on** the value of another predictor.
- **Interactions** capture these dependent relationships and allow predictors to influence the response jointly.
- **Interactions**  Occur when the effect of one predictor on the response depends on the value of another predictor.

  <img width="897" height="437" alt="image" src="https://github.com/user-attachments/assets/ac545853-40ed-403f-a0fe-d5c8ea5325f6" />

## Key Terms for Regression Diagnostics

- **Standardized Residuals**  
  Residuals scaled by their standard error, used to identify unusual observations.

- **Outliers**  
  Observations with outcome values far from the rest of the data or from model predictions.

- **Influential Values**  
  Records whose inclusion or removal substantially changes the regression results. This is useful to identify only in smaller datasets.

- **Leverage**  
  Measures how much influence a single observation has on the fitted model.  
  *Also called:* **hat-value**

- **Non-Normal Residuals**  
  Residuals that are not normally distributed; often violate technical assumptions but usually are not critical in data science.

- **Heteroskedasticity**  
  Non-constant variance of residuals across outcome ranges, often signaling a missing predictor. A key assumption of linear regression is homoskedasticity — residuals have roughly the same variance everywhere. Heteroskedasticity violates this assumption. The distribution of the residuals is relevant mainly for the validity of formal statistical inference (hypothesis tests and p-values), which is of minimal importance to data scientists concerned mainly with predictive accuracy.

- **Partial Residual Plots**  
  Diagnostic plots that show the relationship between the outcome and one predictor while accounting for others.  
  *Also called:* **added variable plots**

- **Spline Regression**  
  Fits a smooth curve using multiple connected polynomial segments.

- **Knots**  
  Points in the predictor space where spline segments join.

- **Generalized Additive Models (GAMs)**  
  Spline-based models that automatically select and smooth knots.  
  *Also called:* **GAM**
