# Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is an **unsupervised learning algorithm** primarily used for **dimensionality reduction**. It transforms a large set of correlated variables into a smaller set of uncorrelated variables, called **Principal Components (PCs)**, while retaining most of the variability (information) in the original dataset.

## Core Idea: Dimensionality Reduction for Visualization

The main application discussed here is **visualization**, where PCA reduces the huge number of features ($n$) down to 2 or 3 features (or dimensions, $k$) to allow the data to be easily plotted.

* **Goal:** Find new axes and coordinates that capture the size and information contained in the original features using a fewer number of dimensions.

---

## PCA Mechanics: Finding the Optimal Axes

### 1. Preprocessing Steps (Crucial)
Before applying PCA, the data must be prepared:
* **Normalization:** Each feature should be **normalized to have zero mean**. ($\mu_j = 0$).
* **Feature Scaling:** Features should be scaled (e.g., divided by the standard deviation or range) to put them in a **similar range**, preventing features with large scales from dominating the calculation of variance.

### 2. Identifying Principal Components (New Axes)
PCA essentially rotates the coordinate system to align with the directions of maximum variance in the data.

* **Principal Component:** The new axis (a straight line, often called the $z$-axis) that maximizes the **variance** of the projected data points. This axis **captures the most information** about the original data.
* **Projection:** The original data examples (e.g., 5 examples with features $x_1$ and $x_2$) are **projected** onto this new $z$-axis.
* **Maximum Variance:** The goal is to choose a $z$-axis such that the variance of the projected points (their spread or distance from the new origin) is the largest possible. A large variance means the principal component retains most of the useful information from the original two dimensions.

---

## PCA in Practice (scikit-learn Workflow)

When using a library like `scikit-learn`, PCA is typically executed in three steps:

1.  **Fit:** Train the PCA model on the normalized data to obtain the new axes (Principal Components).
    * Example: `pca.fit(X)` to obtain $k=2$ new axes.
2.  **Examine Variance:** Optionally, examine how much of the original data's variability is explained by each principal component.
    * **Metric:** `explained_variance_ratio_`
    * *Rationale:* This helps determine if the chosen $k$ components are sufficient.
3.  **Transform:** Project the original data onto the new principal component axes.
    * Example: `X_new = pca.transform(X)`

---

## Applications of PCA

### 1. Data Compression
* Reducing data size (e.g., from 100 features down to 10 features).
* Saves memory/storage space.

### 2. Speed Up Training of Supervised Learning Models
* By reducing the number of input features ($n$), the computational cost of training algorithms (e.g., Neural Networks, Logistic Regression) is significantly reduced. This is a form of **preprocessing** for supervised learning.

---

## Reconstruction (Inverse Transform)

The process of going from the compressed/projected data back to an approximation of the original space is called **Reconstruction**.

* **Process:** Given a compressed data point $z$ (e.g., $z = 3.55$), the **reconstruction step** attempts to find the corresponding original point $(\hat{x}_1, \hat{x}_2)$ in the original feature space.
* **Note:** Because information is lost during compression (dimensionality reduction), the reconstructed point $(\hat{x}_1, \hat{x}_2)$ is only an **approximation** of the original point $(x_1, x_2)$, not an exact replica.
