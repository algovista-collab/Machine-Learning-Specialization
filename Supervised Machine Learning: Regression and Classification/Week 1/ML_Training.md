# Machine Learning Training Paradigms

## 1. Data-Based Paradigms

These methods categorize training based on how the data is labeled.

| Paradigm | Description |
| :--- | :--- |
| **Self-supervised Learning** | A method where the system generates its own supervisory signals (labels) from the input data. The data itself provides the guidance. *Example: BERT model predicting masked words.* |
| **Semi-supervised Learning** | Training involves a mix of a **small amount of labeled data** and a **large amount of unlabeled data**. The labeled data is used to anchor the learning, and the unlabeled data helps the model learn the underlying structure and patterns. |

---

## 2. Temporal/Training Flow Paradigms

These methods categorize training based on how the data is presented to the system over time.

### Batch Learning (Offline Learning)

* **Definition:** The system is **incapable of learning incrementally**. It must be trained using all the available data at once. Training is performed **offline**.
* **Drawback (Model Rot / Data Drift):** Since the real world continually changes, the model's performance slowly degrades over time. This decay is called **model rot** or **data drift**.
* **Solution:** The model must be periodically **re-trained** from scratch using a fresh, complete dataset.

### Online Learning (Incremental Learning)

* **Definition:** The system is trained **incrementally** by feeding it data instances sequentially, either individually or in small groups called **mini-batches**.
* **Usage:**
    * Useful for systems that need to **adapt quickly to change** (e.g., stock trading).
    * Used to train **huge datasets** that cannot fit into a single machine's memory (**out-of-core learning**). The system loads a part of the data, runs a training step, and repeats.
* **Adaptation Rate (Learning Rate $\alpha$):**
    | Learning Rate ($\alpha$) | Effect | Agent Behavior |
    | :--- | :--- | :--- |
    | **High** | Adapts **rapidly** to new data. | Quickly **forgets old data**. |
    | **Small** | Has more **inertia**. Learns more **slowly**. | Less sensitive to new, potentially noisy, data. |
* **Process:** The model is launched into production and continues to learn and update as new data streams in.

## Main Approaches to Generalization in Machine Learning

Generalization is a model's ability to perform well on new, unseen data after being trained on a specific dataset.

---

## 1. Instance-Based Learning (Lazy Learning)

This approach does not build an explicit, abstract model during the training phase.

* **Learning Process:** The system **learns the examples by heart** by simply storing the entire training dataset.
* **Generalization:** When presented with a new, unseen instance, the system generalizes by using a **similarity measure** (or distance measure, like Euclidean distance) to compare the new instance to the learned examples. It then predicts the label based on the most similar (closest) stored instances.
* **Example Algorithms:** K-Nearest Neighbors (KNN).

---

## 2. Model-Based Learning (Eager Learning)

This approach uses the training data to build an explicit, generalizable model.

* **Learning Process:** The goal is to **form a model** that represents the underlying patterns and relationships in the data.
* **Generalization:** Once the model is trained, it can make predictions on new data based on the learned mathematical function or structure.
* **Model Evaluation:** The quality of the model is measured using specific functions:
    * **Utility/Fitness Function:** Measures **how good** the model is (e.g., accuracy, precision, F1-score). The goal is to **maximize** this function.
    * **Cost/Loss Function:** Measures **how bad** the model is (e.g., Mean Squared Error, cross-entropy). The goal is to **minimize** this function.
* **Example Algorithms:** Linear Regression, Support Vector Machines, Neural Networks.

## Data and Model Preparation Concepts ⚙️

## 1. Sampling Issues

When collecting data, issues can arise that introduce bias and affect the representativeness of the dataset.

* **Sampling Bias:** Occurs when the data collection method is flawed, leading to certain members of the population being under-represented or over-represented in the sample. This prevents the model from generalizing accurately.
* **Nonresponse Bias:** A specific type of sampling bias that occurs when a significant fraction of people fail to respond to a survey or request for data. The non-respondents may differ systematically from those who do respond, skewing the results.

---

## 2. Feature Engineering

Feature Engineering is the process of using domain knowledge to create new input variables that help the machine learning model learn better.

* **Feature Selection:** The process of choosing the **most relevant and useful existing features** from the dataset to be used for model training. This helps reduce dimensionality, speeds up training, and often improves generalization by eliminating noise.
* **Feature Extraction:** The process of **combining existing features to produce useful features** and **create new features** through transformation (e.g., Principal Component Analysis, or creating a `total_rooms` feature by multiplying `houses` and `rooms_per_house`).

---

## 3. Hyperparameters

Hyperparameters are critical settings that control the learning process itself, rather than being learned from the data.

* **Definition:** A hyperparameter is a **parameter of the learning algorithm** and **not of the model**. Its value is set prior to the training process.
* **Function:** Hyperparameters often control the **amount of regularization** to apply during learning (e.g., $L_2$ regularization strength) or define the structure of the model (e.g, number of layers in a neural network, depth of a decision tree).
* **Tuning:** The process of finding the optimal set of hyperparameters is called hyperparameter tuning (or optimization).
