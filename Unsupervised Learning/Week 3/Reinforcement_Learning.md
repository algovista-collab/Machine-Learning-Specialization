# Reinforcement Learning (RL) Fundamentals

## Core Components

| Component | Symbol | Description |
| :--- | :--- | :--- |
| **State** | $S$ (or $s$) | A complete description of the environment at a given time. |
| **Action** | $a$ | A move or decision made by the agent within the state $s$. |
| **Reward** | $R(s')$ (or $R_{t+1}$) | The immediate scalar feedback received after transitioning to the new state $s'$. |
| **New State** | $S'$ (or $s'$) | The resulting state after the action $a$ is taken from state $s$. |

---

## Return and Discount Factor

The goal of an RL agent is to maximize the **expected Return** $G_t$, which is the total discounted reward from the current time step $t$ until the end of the episode.

### Return Formula

The return $G_t$ is calculated as the sum of future rewards, discounted by a factor $\gamma$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### Discount Factor ($\gamma$)

| Symbol | Range | Description |
| :--- | :--- | :--- |
| $\gamma$ | $0 \le \gamma \le 1$ | The **Discount Factor**. It determines the present value of future rewards. |

* **$\gamma \approx 1$ (Farsighted):** Future rewards are nearly as important as immediate rewards.
* **$\gamma \approx 0$ (Shortsighted/Impatient):** Immediate rewards are much more important. Every step "costs" the agent something in terms of future return.

## RL Return Comparison: Impatient Agent ($\gamma=0.1$)

**Scenario:** Starting at State 4.
**Reward Convention:** $R_{t+1}$ is the reward received upon leaving state $S_t$. Assume $R_{t+1}$ (for leaving $S_4$) is $\mathbf{10}$ for both paths.

---

### Path 1: Move Left (Longer Path to $R=100$)

**Sequence:** $S_4 \xrightarrow{R=10} S_3 \xrightarrow{R=0} S_2 \xrightarrow{R=0} S_1(R=100, \text{Terminal})$

**Formula:** $G_4 = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4}$

**Calculation:**
$$G_{\text{Left}} = 10 + (0.1)(0) + (0.1)^2 (0) + (0.1)^3 (100)$$
$$G_{\text{Left}} = 10 + 0 + 0 + 0.1$$
$$G_{\text{Left}} = \mathbf{10.1}$$

---

### Path 2: Move Right (Shorter Path to $R=40$)

**Sequence:** $S_4 \xrightarrow{R=10} S_5 \xrightarrow{R=40} S_6(\text{Terminal})$

**Formula:** $G_4 = R_{t+1} + \gamma R_{t+2}$

**Calculation:**
$$G_{\text{Right}} = 10 + (0.1)(40)$$
$$G_{\text{Right}} = 10 + 4$$
$$G_{\text{Right}} = \mathbf{14.0}$$

---

## Conclusion

| Path | Return ($G_4$) |
| :--- | :--- |
| **Left** (to $R=100$) | $10.1$ |
| **Right** (to $R=40$) | $14.0$ |

Since $14.0 > 10.1$, the $\gamma=0.1$ (impatient) agent chooses the **Right** path. The high discount factor heavily penalizes the extra steps required for the higher $R=100$ reward, making the quicker $R=40$ reward path more valuable.

### Example $\gamma = 0.5$ Calculation

Given a path moving left towards a terminal State 1 ($R=100$) from State 4:

| Starting State | Path | Return ($G_t$) Calculation | Return Value |
| :--- | :--- | :--- | :--- |
| **State 4** | $\rightarrow R_{t+1}=0, R_{t+2}=0, R_{t+3}=0, R_{t+4}=100$ | $0 + 0.5 \cdot 0 + 0.5^2 \cdot 0 + 0.5^3 \cdot 100$ | $12.5$ |
| **State 3** | $\rightarrow R_{t+1}=0, R_{t+2}=0, R_{t+3}=100$ | $0 + 0.5 \cdot 0 + 0.5^2 \cdot 100$ | $25$ |
| **State 2** | $\rightarrow R_{t+1}=0, R_{t+2}=100$ | $0 + 0.5 \cdot 100$ | $50$ |
| **State 1** | $R_{t+1}=100$ | $100$ | $100$ |

**Comparison (State 3):**
* **Move Left** (to $R=100$): Return is $25$.
* **Move Right** (to $R=40$ at State 6, 3 steps away): $0 + 0.5 \cdot 0 + 0.5^2 \cdot 0 + 0.5^3 \cdot 40 = 5$.
* **Decision:** Move Left ($25 > 5$).

**Comparison (State 5):**
* **Move Left** (to $R=100$ at State 1, 4 steps away): $0 + 0.5^4 \cdot 100 = 6.25$.
* **Move Right** (to $R=40$ at State 6, 1 step away): $0 + 0.5 \cdot 40 = 20$.
* **Decision:** Move Right ($20 > 6.25$). **It is better to move to lower reward terminal state only if it's quick.**

---

## Policy ($\pi$)

The **Policy** is the agent's strategy. It is the function $\pi$ that the RL algorithm aims to learn.

| Component | Symbol | Description |
| :--- | :--- | :--- |
| **Policy** | $\pi(s)$ | A function that takes a **state** $s$ as input and maps it to an **action** $a$. |
| **Goal** | $\pi: S \rightarrow A$ | To select the action $a$ in state $s$ that **maximizes the expected Return** $G_t$. |

Based on the example calculations above, the optimal policy $\pi(s)$ would be:

* $\pi(2)$ says **go left**
* $\pi(3)$ says **go left**
* $\pi(4)$ says **go left**
* $\pi(5)$ says **go right**

---

<img width="1169" height="557" alt="Screenshot 2025-10-27 091534" src="https://github.com/user-attachments/assets/555d294a-f653-4133-891f-589a2427267a" />

## Markov Decision Process (MDP)

The foundation of most Reinforcement Learning problems is the Markov Decision Process.

**Key Principle: The Markov Property**

* **Definition:** The future depends only on the **current state** ($S_t$) and not on the sequence of events (states and actions) that led to it.
* **Formal Statement:** The state $S_{t+1}$ and the reward $R_{t+1}$ are conditionally independent of all past states and actions, given the current state $S_t$ and action $A_t$.
    $$\mathbb{P}(S_{t+1} | S_t, A_t) = \mathbb{P}(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, \dots)$$

---

## State-Action Value Function ($$Q(s, a)$$)

The $Q$-function is central to many RL algorithms (like Q-Learning). It estimates the "quality" of taking a specific action in a specific state.

**Definition: $$Q(s, a)$$**

The **State-Action Value Function** $Q(s, a)$ is the expected **Return** ($G_t$) achieved if the agent:
1.  **Starts** in state $s$.
2.  **Takes action $a$** immediately.
3.  **Behaves optimally** (following the optimal policy $\pi^*$) for all subsequent steps.

* **Goal:** $Q(s, a)$ represents the expected maximum total discounted reward that can be obtained from state $s$ by starting with action $a$.

---

## Optimal Decision Making

The $Q$-function allows the agent to make the best possible decision in any given state.

**Optimal Value Function ($$V^*(s)$$):**
The best possible return achievable from state $s$ is the maximum $Q$-value over all possible actions $a$ from that state.
$$V^*(s) = \max_{a} Q(s, a)$$

**Optimal Policy ($$\pi^*(s)$$):**
The best possible action to take in state $s$ (the optimal policy) is the action $a$ that yields the maximum $Q$-value.
$$\pi^*(s) = \arg \max_{a} Q(s, a)$$

---

## Example (Using $$\gamma=0.5$$ from prior context)

| State | Action | $$Q(s, a)$$ Calculation | $$Q(s, a)$$ Value |
| :---: | :---: | :--- | :--- |
| **2** | Left ($\leftarrow$) | $R_{t+1}(100)$ | $\mathbf{50}$ |
| **2** | Right ($\rightarrow$) | $R_{t+1}(0) + \gamma \cdot V^*(3) = 0 + 0.5 \cdot 25$ | $12.5$ |

**Decision for State 2:**
* **$$V^*(2)$$** = $$\max (50, 12.5) = \mathbf{50}$$
* **$$\pi^*(2)$$** = The action that gives 50, which is **Left ($$\leftarrow$$)**

<img width="798" height="285" alt="Screenshot 2025-10-27 095047" src="https://github.com/user-attachments/assets/c0def647-6170-42f8-a45e-a9596c860078" />

## The Bellman Equation

The Bellman Equation is a recursive relationship that describes the optimal value functions ($$V^*$$ and $$Q^*$$). It states that the optimal value of a state (or state-action pair) is equal to the immediate reward plus the discounted optimal value of the next state (or state-action pair).

---

## 1. Bellman Optimality Equation for State Value ($$V^*$$)

This equation defines the optimal value of a state $$s$$, $$V^*(s)$$, as the reward received for the best action, $$a$$, plus the discounted value of the resulting state, $$s'$$.

$$V^*(s) = \max_{a} \left( R(s, a) + \gamma \sum_{s'} P(s'|s, a) V^*(s') \right)$$

Where:
* $V^*(s)$: The optimal value of state $s$.
* $\max_{a}$: The choice of action $a$ that maximizes the entire expression.
* $R(s, a)$: The immediate reward received for taking action $a$ in state $s$.
* $\gamma$: The discount factor.
* $P(s'|s, a)$: The probability of transitioning to state $s'$ given state $s$ and action $a$.
* $V^*(s')$: The optimal value of the resulting next state $s'$.

---

## 2. Bellman Optimality Equation for State-Action Value ($Q^*$)

This is arguably the more common form used in Q-Learning, defining the optimal value of a state-action pair, $Q^*(s, a)$.

$$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s, a) \max_{a'} Q^*(s', a')$$

Where:
* $Q^*(s, a)$: The optimal expected return for starting in state $s$ and taking action $a$.
* $R(s, a)$: The immediate reward received.
* $\max_{a'} Q^*(s', a')$: The highest possible future value (the optimal action $a'$) from the next state $s'$.

---

## 3. Calculation Example (Deterministic Environment)

In a **deterministic environment** (where $P(s'|s, a)=1$), the summation $\sum_{s'} P(s'|s, a)$ disappears, simplifying the $Q$-function to:

$$Q^*(s, a) = R(s, a) + \gamma \cdot \max_{a'} Q^*(s', a')$$

**Scenario:**
* **Current State:** $S_A$
* **Action:** $a_1$ (leads deterministically to $S_B$)
* **Reward:** $R(S_A, a_1) = 5$
* **Discount Factor:** $\gamma = 0.9$
* **Known Optimal Future Values in $S_B$:**
    * $Q^*(S_B, a_x) = 15$
    * $Q^*(S_B, a_y) = 25$
    * $Q^*(S_B, a_z) = 10$

**Step 1: Find the Optimal Future Value from $S_B$**
The optimal future value is the maximum Q-value in the successor state $S_B$:
$$\max_{a'} Q^*(S_B, a') = \max(15, 25, 10) = \mathbf{25}$$

**Step 2: Apply the Bellman Equation to find $$Q^*(S_A, a_1)$$**

$$Q^*(S_A, a_1) = $$

$$R(S_A, a_1) + \gamma \max_{a'} Q^*(S_B, a')$$

$$Q^*(S_A, a_1) = 5 + 0.9 \cdot 25$$

$$Q^*(S_A, a_1) = 5 + 22.5$$

$$Q^*(S_A, a_1) = \mathbf{27.5}$$

**Interpretation:**
The total optimal expected return for taking action $a_1$ from state $S_A$ is 27.5. This value is composed of the immediate reward (5) and the discounted value of behaving optimally in the next state (22.5).
