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
