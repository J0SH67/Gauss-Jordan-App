#  Concrete Compressive Strength Predictor & Linear Systems Solver
### *Supervised Machine Learning via Custom Gauss-Jordan Elimination*

[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Language-Python%203.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic%20Use-blue?style=for-the-badge)](#)

---

## 📌 Project Overview

In civil and structural engineering, determining the 28-day compressive strength of concrete traditionally requires preparing cylinder specimens and waiting weeks for standard water-tank curing.

This application demonstrates how **Machine Learning** can predict compressive strength from mix design parameters from first principles. Rather than relying on commercial "black box" machine learning libraries (such as Scikit-Learn), this system implements an **algorithmic numerical engine from scratch**:
1. A **Gauss-Jordan Elimination solver with Partial Pivoting** capable of solving general $n \times n$ linear systems with full step-by-step row operation auditing.
2. A **Multiple Linear Regression (Ordinary Least Squares)** training layer that computes the **Normal Equations** and uses the custom Gauss-Jordan solver to derive the optimal regression coefficients ($\beta$).

---

##  Proponents & Contributors

* **John Joshua D. Ilisan**
* **Darius Lape**
* **Briel Jan M. Lacia**

---

##  Mathematical Formulation

### 1. The Normal Equations Bridge
In concrete mix evaluation, the number of experimental cylinder batches ($N$) far exceeds the number of unknown weights ($k$). This yields an **overdetermined rectangular system** ($X\beta \approx y$) with no direct matrix inverse.

To minimize the sum of squared prediction errors ($\sum (y - \hat{y})^2$), the Ordinary Least Squares method formulates the **Normal Equations**:

$$(X^T X)\beta = X^T y \quad \iff \quad A\beta = b$$

Where:
* $X$ is the $(N \times 4)$ feature design matrix augmented with a leading column of $1$s (intercept).
* $X^T X$ is a compressed, square, and symmetric $(4 \times 4)$ coefficient matrix ($A$).
* $X^T y$ is a $(4 \times 1)$ column vector of constants ($b$).
* $\beta = [\beta_0, \beta_1, \beta_2, \beta_3]^T$ is the vector of model weights.

### 2. Physical Engineering Variables
$$\text{Strength (MPa)} = \beta_0 + \beta_1(\text{Cement}) + \beta_2(w/c) + \beta_3(\text{Age})$$

| Parameter | Type | Unit | Engineering Interpretation |
| :--- | :--- | :--- | :--- |
| **$\beta_0$** | Unknown | $\text{MPa}$ | **Baseline Intercept**: Mathematical datum of the regression plane. |
| **$x_1$ / $\beta_1$** | Independent | $\text{kg/m}^3$ | **Cement Content**: Binder density. Positive coefficient ($\beta_1 > 0$). |
| **$x_2$ / $\beta_2$** | Independent | Decimal | **Water-Cement Ratio ($w/c$)**: Governed by Abrams' Law; excess water creates capillary voids. Negative coefficient ($\beta_2 < 0$). |
| **$x_3$ / $\beta_3$** | Independent | Days | **Curing Age**: Hydration progression over time. Positive coefficient ($\beta_3 > 0$). |
| **$y$** | Dependent | $\text{MPa}$ | **Compressive Strength**: Ultimate failure load under Universal Testing Machine (UTM) compression. |

---

##  Features

- **Custom Gauss-Jordan Engine (`gauss_jordan.py`):**
  - Implements **Partial Pivoting** (row swapping to select the maximum absolute pivot entry) to eliminate division-by-zero risks and minimize floating-point roundoff drift.
  - Generates an auditable, step-by-step transformation log from augmented matrix $[A \mid b]$ to Reduced Row Echelon Form (RREF) $[I \mid x]$.
- **Machine Learning Integration (`linear_regression.py`):**
  - Translates multi-variable regression into an $Ax = b$ problem via the Normal Equations.
  - Does not rely on pre-built machine learning solvers.
- **Interactive Web Interface (`app.py`):**
  - **Tab 1:** General $n \times n$ linear equation solver with editable matrix cells and subscript rendering ($x_1, x_2, \dots$).
  - **Tab 2:** Concrete mix design training interface, dynamic summation breakdown tables, model validation charts ($R^2$, $\text{MAE}$, $\text{RMSE}$), and real-time interactive mix prediction sliders.
  - Fully functional in offline environments.

---

## 📂 Project Structure

```text
├── app.py                 # Streamlit web application (UI, charts, sliders)
├── gauss_jordan.py        # Algorithmic solver with partial pivoting & step logging
├── linear_regression.py   # OLS Normal Equations engine powered by Gauss-Jordan
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
