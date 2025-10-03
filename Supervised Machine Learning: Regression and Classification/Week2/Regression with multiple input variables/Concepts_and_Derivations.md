In the Linear Regression, we used a model with only single variable: f(x) = w*x + b
In the Multiple Linear Regression: we use a model with multiple variables: $$f(\vec{x}) = w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + \dots + w_n \cdot x_n + b$$

We can simply denote this using vector which is nothing but a list of numbers, we usually use row vector: x arrow = [x1, x2, x3, ..., xn] (xj is the jth feature)
w arrow is a vector with multiple weights for the corresponding features. These numbers indicate how important that particular feature is while predicting the output.

Vector Representation of the Model: $$f_{\vec{w}, b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$
The dot product from Linear Algebra, denotes each value of w is multipled with the values in x and the size of the vector w and vector x should be same. We can use row vector with one row and multiple columns denoting different features.
