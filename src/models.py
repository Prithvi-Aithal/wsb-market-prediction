"""Models from the report: Newton's-method logistic regression (from scratch)
and a 3x256 ReLU feed-forward network with a sigmoid output; plus our new model,
a soft-voting ensemble shrunk toward the class prior and tuned by forward-chaining CV."""
import copy

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


class NewtonLogisticRegression:
    """Binary logistic regression fit by Newton's method (solves dF/dtheta = 0).

    The feature matrix already carries a bias column (feature 7), so no
    separate intercept is fitted. A small L2 term keeps the Hessian invertible
    when classes are (nearly) separable (its strength is tuned on validation data).
    On this data the solver converges in 4-8 iterations for every stock, with or
    without the L2 term.
    Iteration history of ||delta w|| is stored in `deltas_` (cf. Figure 3 of the CS229 report).
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


class ShrinkageEnsemble:
    """Soft-voting ensemble shrunk toward the class prior.

    Base learners, all on the same features: the Newton logistic regression above,
    a random forest (bagged trees), gradient boosting (boosted trees) and Gaussian
    naive Bayes. Their probabilities are averaged (soft vote) and blended with the
    training base rate:

        p(up | x) = lam * p_vote(x) + (1 - lam) * prior

    lam = 0 is the majority-class rule and lam = 1 the plain ensemble. lam is chosen
    by forward-chaining cross-validation (tune_ensemble), so if the post features do
    not carry forward in time the model falls back to the prior instead of fitting
    noise (bias-variance trade-off).
    """

    def __init__(self, lam=1.0, l2=1e-2, rf_leaf=20, gb_trees=100, gb_depth=3, gb_lr=0.05, seed=0):
        self.lam, self.l2, self.rf_leaf, self.seed = lam, l2, rf_leaf, seed
        self.gb_trees, self.gb_depth, self.gb_lr = gb_trees, gb_depth, gb_lr

    def fit(self, X, y):
        Z = X[:, :-1]   # trees and naive Bayes do not need the bias column
        self.prior_ = float(np.mean(y))
        self.lr_ = NewtonLogisticRegression(l2=self.l2).fit(X, y)
        self.rf_ = RandomForestClassifier(200, min_samples_leaf=self.rf_leaf, random_state=self.seed).fit(Z, y)
        self.gb_ = GradientBoostingClassifier(n_estimators=self.gb_trees, max_depth=self.gb_depth,
                                              learning_rate=self.gb_lr, subsample=0.8,
                                              random_state=self.seed).fit(Z, y)
        self.nb_ = GaussianNB().fit(Z, y)
        return self

    def vote_proba(self, X):
        Z = X[:, :-1]
        return np.mean([self.lr_.predict_proba(X)] +
                       [m.predict_proba(Z)[:, 1] for m in (self.rf_, self.gb_, self.nb_)], axis=0)

    def predict_proba(self, X):
        return self.lam * self.vote_proba(X) + (1 - self.lam) * self.prior_

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


ENS_LAMS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def ensemble_configs(rng, n=8):
    """Random search space for the ensemble; the first draw is the default configuration."""
    yield {"l2": 1e-2, "rf_leaf": 20, "gb_trees": 100, "gb_depth": 3, "gb_lr": 0.05}
    for _ in range(n - 1):
        yield {"l2": float(rng.choice([1e-4, 1e-2, 1.0, 10.0])),
               "rf_leaf": int(rng.choice([5, 10, 20, 50, 100])),
               "gb_trees": int(rng.choice([50, 100, 200])),
               "gb_depth": int(rng.choice([1, 2, 3])),
               "gb_lr": float(rng.choice([0.01, 0.05, 0.1]))}


def tune_ensemble(X, y, n_train, n_configs=8, seed=42):
    """Random search over the base learners' hyper-parameters and a grid over lam.

    X, y are the training rows followed by the validation rows (time order). Each
    candidate is scored by mean log-loss over forward-chaining folds: fit on the first
    40 / 60 / 80 % of the training split and score the next 20 %, then fit on the whole
    training split and score the validation split. The test split is never used.
    Ties go to the smaller lam (the simpler model). Returns (cfg, lam, cv_logloss).
    """
    cuts = [int(n_train * f) for f in (0.4, 0.6, 0.8)] + [n_train]
    folds = [(a, b) for a, b in zip(cuts, cuts[1:] + [len(y)]) if len(np.unique(y[:a])) == 2 and b > a]
    best = (None, None, np.inf)
    for cfg in ensemble_configs(np.random.default_rng(seed), n_configs):
        losses = np.zeros(len(ENS_LAMS))
        for a, b in folds:
            m = ShrinkageEnsemble(**cfg).fit(X[:a], y[:a])
            pv = m.vote_proba(X[a:b])
            losses += [log_loss(y[a:b], lam * pv + (1 - lam) * m.prior_, labels=[0, 1]) for lam in ENS_LAMS]
        losses /= len(folds)
        i = int(np.argmin(losses))   # argmin returns the first (smallest) lam on ties
        if losses[i] < best[2]:
            best = (cfg, ENS_LAMS[i], float(losses[i]))
    return best
