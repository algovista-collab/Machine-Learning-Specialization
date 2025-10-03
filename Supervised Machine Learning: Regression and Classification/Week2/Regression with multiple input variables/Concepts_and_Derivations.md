In the Linear Regression, we used a model with only single variable: f(x) = w*x + b

In the Multiple Linear Regression: we use a model with multiple variables: $$f(\vec{x}) = w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + \dots + w_n \cdot x_n + b$$

We can simply denote this using vector which is nothing but a list of numbers, we usually use row vector: $$\vec{x} = [x_1, x_2, x_3, \dots, x_n] \quad (x_j \text{ is the } j\text{th feature})$$
$$\vec{w}$$ is a vector with multiple weights for the corresponding features. These numbers indicate how important that particular feature is while predicting the output.

Vector Representation of the Model: $$f_{\vec{w}, b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$
The dot product from Linear Algebra, denotes each value of w is multipled with the values in x and the size of the vector w and vector x should be same.
