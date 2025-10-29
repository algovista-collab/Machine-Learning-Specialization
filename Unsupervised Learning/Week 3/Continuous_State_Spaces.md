# Continuous State Spaces and Q-Function Approximation

## 1. Defining Continuous State Spaces

In many real-world Reinforcement Learning problems, the environment cannot be described by a finite number of states.

* **Discrete State:** The system can be in any of a fixed, countable number of states (e.g., $n$ possible positions on a chessboard).
* **Continuous State:** The system's parameters can take on any real-valued number within a specific range.

| Example System | State Vector ($S$) Components | Total Parameters |
| :--- | :--- | :--- |
| **Simple Rover** | Position $x$, $y$ (where $x, y \in [0, 6]$ km) | 2 |
| **Complex Vehicle** | Position ($x, y$), Orientation ($\theta$), Velocity ($\dot{x}, \dot{y}$), Angular Velocity ($\dot{\theta}$) | 6 |
| **Helicopter** | $x, y, z$ (position), $\phi$ (roll), $\omega$ (pitch), $\dots$ | $\approx 12$ |

For a **Continuous State Markov Decision Process (MDP)**, the state $S$ is a **vector of numbers**, where any component can contain a large (or infinite) range of values.

---

## 2. The Lunar Lander Example

The Lunar Lander problem is a classic example of a continuous control task.

* **State Vector ($S$):** $S = [x, y, \dot{x}, \dot{y}, \theta, \dot{\theta}, l_r, r_r]$
    * $x, y$: Position.
    * $\dot{x}, \dot{y}$: Velocity.
    * $\theta, \dot{\theta}$: Angle and angular velocity.
    * $l_r, r_r$: Binary indicators (0 or 1) showing whether the left leg ($l_r$) or right leg ($r_r$) is touching the ground.

* **Action Vector ($A$):** The actions are discrete (e.g., main engine, left thruster, right thruster, do nothing). They are represented as one-hot feature vectors:
    * Action 1 (Main Engine): $[1, 0, 0, 0]$
    * Action 2 (Left Thruster): $[0, 1, 0, 0]$
    * Action 3 (Right Thruster): $[0, 0, 1, 0]$
    * Action 4 (Do Nothing): $[0, 0, 0, 1]$

* **Input to NN:** The combined **State-Action vector** $[S \ A]$ serves as the input $\mathbf{x}$.
    * Total Input Size: $8 \text{ (State)} + 4 \text{ (Action)} = \mathbf{12}$ inputs.

---

## 3. Q-Function Approximation via Neural Network

In continuous state spaces, it's impossible to store every $Q(s, a)$ pair in a table (Q-table). Instead, we use a **Neural Network (NN)** as a function approximator.

### NN Goal

The job of the Neural Network $f_{w,b}(\mathbf{x})$ is to approximate the State-Action Value Function $Q(s, a)$.

$$\text{NN Output } \mathbf{y} = Q(s, a)$$

$$\text{NN Function: } f_{w,b}(\mathbf{x}) \approx Q(s, a)$$

### Training the NN using the Bellman Equation

The key to training the NN is turning the complex RL problem into a standard **supervised learning** problem.

1.  **Experience Collection:** The agent interacts with the environment, taking random actions and saving the experience as a tuple: $(S, a, R(S), S')$.
2.  **Target Creation (Bellman Equation):** For each experience tuple, we calculate the desired target output $\mathbf{y}$, which is the **expected return** according to the Bellman Equation:

    $$ \text{Target } \mathbf{y} = \underbrace{R(S)}_{\text{Immediate Reward}} + \gamma \cdot \underbrace{\max_{a'} Q(S', a')}_{\text{Discounted Max Future Value}}$$

3.  **Supervised Training:** The NN is trained to minimize the difference between its prediction $f_{w,b}(\mathbf{x})$ and the calculated Bellman target $\mathbf{y}$.

    * **Input ($\mathbf{x}$):** The combined state-action vector $[S, a]$.
    * **Desired Output ($\mathbf{y}$):** The calculated Bellman target (the optimal return).

By minimizing the difference between the NN's prediction and the Bellman target, the network learns a robust mapping from any state-action input to its expected optimal return.
