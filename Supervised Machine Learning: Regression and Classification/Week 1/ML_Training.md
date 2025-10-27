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
