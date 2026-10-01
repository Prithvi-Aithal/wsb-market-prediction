"""Models from the report: Newton's-method logistic regression (from scratch)
and a 3x256 ReLU feed-forward network with a sigmoid output."""
import copy

import numpy as np
from sklearn.neural_network import MLPClassifier


class NewtonLogisticRegression:
    """Binary logistic regression fit by Newton's method (solves dF/dtheta = 0).

    The feature matrix already carries a bias column (feature 7), so no
    separate intercept is fitted. A tiny L2 term keeps the Hessian invertible
    when classes are (nearly) separable - without it the weights diverge, which
    is the non-convergence the report observed for AMC.
    Iteration history of ||delta w|| is stored in `deltas_` (Figure 3).
    """

    def __init__(self, max_iter=50, tol=1e-6, l2=1e-4):
        self.max_iter, self.tol, self.l2 = max_iter, tol, l2

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        n, d = X.shape
        w = np.zeros(d)
        self.deltas_ = []
        for _ in range(self.max_iter):
            p = self._sigmoid(X @ w)
            grad = X.T @ (p - y) / n + self.l2 * w
            H = (X * (p * (1 - p))[:, None]).T @ X / n + self.l2 * np.eye(d)
            step = np.linalg.solve(H, grad)
            w = w - step
            self.deltas_.append(float(np.linalg.norm(step)))
            if self.deltas_[-1] < self.tol:
                break
        self.coef_ = w
        self.converged_ = self.deltas_[-1] < self.tol
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.coef_)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def fit_neural_net(Xtr, ytr, Xva, yva, batch_size=16, epochs=30, lr=1e-3, seed=0,
                   hidden=(256, 256, 256), alpha=1e-4):
    """Default = report's net: 3 x 256 ReLU layers + sigmoid head, Adam(lr=1e-3),
    30 epochs. The epoch with the best validation accuracy is kept (model
    selection on the validation split; the test split is never touched here)."""
    net = MLPClassifier(hidden_layer_sizes=tuple(hidden), activation="relu",
                        solver="adam", learning_rate_init=lr, batch_size=batch_size,
                        alpha=alpha, random_state=seed)
    classes = np.array([0, 1])
    best, best_acc, hist = None, -1.0, []
    for _ in range(epochs):
        net.partial_fit(Xtr, ytr, classes=classes)
        acc = float((net.predict(Xva) == yva).mean())
        hist.append(acc)
        if acc > best_acc:
            best_acc, best = acc, copy.deepcopy(net)
    return best, hist
