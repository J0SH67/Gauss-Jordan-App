import numpy as np
from gauss_jordan import solve_gauss_jordan


class MultipleLinearRegression:

  def __init__(self):
    self.coefficients = None
    self.XT_X = None
    self.XT_y = None
    self.steps = None

  def fit(self, X, y, return_steps=False):
    """Solves (X^T * X) * beta = (X^T * y) using the custom Gauss-Jordan solver."""
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # 1. Add bias column (vector of 1s for the intercept beta_0)
    ones = np.ones((X.shape[0], 1))
    X_design = np.hstack([ones, X])

    # 2. Compute Normal Equations matrices
    XT_X = np.dot(X_design.T, X_design)
    XT_y = np.dot(X_design.T, y)

    self.XT_X = XT_X
    self.XT_y = XT_y

    # 3. Solve system using our custom Gauss-Jordan engine
    if return_steps:
      self.coefficients, self.steps = solve_gauss_jordan(
          XT_X.tolist(), XT_y.tolist(), return_steps=True
      )
      return self.coefficients, self.steps
    else:
      self.coefficients = solve_gauss_jordan(
          XT_X.tolist(), XT_y.tolist(), return_steps=False
      )
      return self.coefficients

  def predict(self, X):
    """Computes y_pred = X_design * beta."""
    if self.coefficients is None:
      raise ValueError("Model has not been trained yet. Call fit() first.")

    X = np.array(X, dtype=float)
    if X.ndim == 1:
      X = X.reshape(1, -1)
    ones = np.ones((X.shape[0], 1))
    X_design = np.hstack([ones, X])

    return np.dot(X_design, self.coefficients)