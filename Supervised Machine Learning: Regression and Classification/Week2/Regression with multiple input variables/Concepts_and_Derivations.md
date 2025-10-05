In the Linear Regression, we used a model with only single variable: f(x) = w*x + b

In the Multiple Linear Regression: we use a model with multiple variables: $$f_{\vec{w}, b}(\vec{x}) = w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + \dots + w_n \cdot x_n + b$$ ---- (1)

We can simply denote this using vector which is nothing but a list of numbers, we usually use row vector: $$\vec{x} = [x_1, x_2, x_3, \dots, x_n] \quad (x_j \text{ is the } j\text{th feature})$$
$$\vec{w}$$ is a vector with multiple weights for the corresponding features. These numbers indicate how important that particular feature is while predicting the output.

Vector Representation of the Model: $$f_{\vec{w}, b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$
The dot product from Linear Algebra, denotes each value of $$\vec{w}$$ is multipled with the values in $$\vec{x}$$ and the size of the $$\vec{w}$$ and $$\vec{x}$$ should be the same.

The benefits of using vector representation is to speed up the calculations by using a popular numerical linear algebra library called NumPy (Numerical Python). It has a function called dot which is used to get the product of 2 vectors. The dot function is able to use parallel hardware whether it is a normal computer CPU or a GPU (Graphics Processing Unit) to accelerate the ML jobs especially when the dataset is very large.

Vectorization - The equation 1 in the concise code: $$f = np.dot(\vec{w}, \vec{x}) + b$$

The number of elements in the array is referred to as the dimension or rank of the vector. The arrays in NumPy is used to represent 1-D or more of the same data type called dtype.

Gradient Descent for multiple variables are calculated for different feature parameters. 

$$\text{Update for } w_j: \quad w_j := w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

$$\text{Update for } b: \quad b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right)$$

Feature Scaling: If the range of any feature varies significantly like size of the house (x1) from 200 to 3000 sq ft and other feature like number of bedrooms (x2) varies only from 0 to 5, we have to scale to bring them to the same range.
Without scaling, any small change in the parameter w1 for x1 will cause the cost function to change significantly because w1 is multiplied by a big number and to make significant change in the cost function, we need to multiply with big number of w2. This results in thin ellipse when we take the contour plot of the cost function w.r.t w1 and w2. To minimize the cost function, we calculate gradient descent which can bounce back nd forth for a long time to reach the minimum. Hence, we need scaling to get circles in the contour plot.

1. Min-Max Scaling: Dividing by maximum value will scale between 0 and 1

$$X_{\text{scaled}} := \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$

2. Mean Normalization: Taking mean of the feature values and subtract the mean from the value and dividing by (maximum-minimum)

$$X_{\text{scaled}} := \frac{X - \mu}{X_{\text{max}} - X_{\text{min}}}$$
   
3. Z-Score Normalization: Subtracting the mean from the value and dividing by the standard deviation

$$X_{\text{scaled}} := \frac{X - \mu}{\sigma}$$

The formulas for calculating the mean ($\mu_j$) and variance ($\sigma^2_j$) for feature $j$ are:

$$\mu_j = \frac{1}{m} \sum_{i=0}^{m-1} x^{(i)}_j$$

$$\sigma^2_j = \frac{1}{m} \sum_{i=0}^{m-1} (x^{(i)}_j - \mu_j)^2 $$

Note that $\sigma_j$ (standard deviation) is the square root of $\sigma^2_j$ (variance).

When generating the plot, the normalized features were used. Any predictions using the parameters learned from a normalized training set must also be normalized.

In order to know after how many iterations, we should stop - we can either check the learning curve and stop when cost function flattens and does not change much when iterations increases. This is called convergence. 
Another method is called Automatic convergence test where we set a threshold called epsilon (ε) which is a small number and we check whether the change in the cost function between iterations is smaller than the epsilon. If it is, then gradient descent has converged. But if the chosen epsilon is too small, then it might never converge or if its big then it might stop too early before even converging.

How to pick a learning rate? 
1. Typical starting values: 0.001, 0.01, 0.1 (depending on the problem and feature scaling).
2. Observing the cost function J over iterations.
3. Increasing or decreasing based on the plot.

Feature Engineering is using the intuition to design new features by transforming or combining original features. This can include squaring some of the features which might result in non-linear curve. In such cases, Polynomial Regression will fit the curves to the data. The quadratic function is a parabola which decreases eventually, hence it might be better to pick cubic function.
