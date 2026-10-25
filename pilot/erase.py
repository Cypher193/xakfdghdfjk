"""LEACE (Belrose et al. 2023) for a scalar concept, closed form, Ledoit-Wolf-shrunk covariance.

    python -m pilot.erase        # self-check on synthetic data

fit(X, z, cols) -> Eraser(mu, V, U); r(x) = x - ((x - mu) V) U, affine, rank k, so mean-pooling commutes with it.
Concept columns Z: z, or with cols (a label per row, e.g. the denoising step) one column per label, z * 1{label}, so
cov(r(x), z) = 0 within every label. With Sig = cov(x, Z) and S_a = (1-a) S + a tr(S)/D I:
    r(x) = x - Sig (Sig^T B)^-1 B^T (x - mu),  B = S_a^-1 Sig   (= W+ P W with W = S_a^-1/2; a = 0 is exact LEACE).
a by Ledoit-Wolf: with D > n the exact S^-1 interpolates the fit data, the shrunk one is a ridge direction.
random_like(X, z, seed, cols): fit to a permutation of z: same construction and size, erases nothing about z.
crossfit(...): the out-of-sample check (fit on train scenes, score held-out scenes).
"""
from dataclasses import dataclass

import numpy as np
from sklearn.covariance import ledoit_wolf_shrinkage


@dataclass
class Eraser:
    mu: np.ndarray      # (D,)
    V: np.ndarray       # (D, k)
    U: np.ndarray       # (k, D)
    shrink: float

    def apply(self, h):
        """h: (..., D) torch tensor (any dtype/device, result in h's dtype) or numpy array."""
        if isinstance(h, np.ndarray):
            return h - ((h - self.mu) @ self.V) @ self.U
        import torch
        mu, V, U = (torch.as_tensor(a, dtype=torch.float32, device=h.device) for a in (self.mu, self.V, self.U))
        x = h.float()
        return (x - ((x - mu) @ V) @ U).to(h.dtype)


def fit(X, z, cols=None, shrink=None):
    X, z = np.asarray(X, np.float64), np.asarray(z, np.float64)
    n, D = X.shape
    cols = np.zeros(n, int) if cols is None else np.asarray(cols)
    # z centred within each label, so X^T Z is the within-label covariance (no between-label mean offsets)
    Z = np.stack([np.where(cols == c, z - z[cols == c].mean(), 0) for c in np.unique(cols)], 1)
    mu = X.mean(0)
    Xc = X - mu
    a = ledoit_wolf_shrinkage(X) if shrink is None else shrink
    t = (Xc ** 2).sum() / (n - 1) / D
    Sig = Xc.T @ Z / (n - 1)
    B = Xc.T @ np.linalg.solve((1 - a) * Xc @ Xc.T / (n - 1) + a * t * np.eye(n), Z / (n - 1))    # S_a^-1 Sig, n x n solve
    return Eraser(mu, B, np.linalg.solve(Sig.T @ B, Sig.T), float(a))


def random_like(X, z, seed, cols=None):
    return fit(X, np.random.default_rng(seed).permutation(z), cols)


def cov_rel(X, z, er=None):
    """|cov(r(x), z)| / |cov(x, z)|."""
    c = lambda A: np.linalg.norm((A - A.mean(0)).T @ (z - z.mean()))
    return c(X if er is None else er.apply(X)) / c(X)


def crossfit(Xs, z, groups, k=5):
    """Xs (T, n, D): T captured steps of n clips. GroupKFold by scene; per fold fit leace and rand on all T steps of the
    train clips (cols = step), then per step on the held-out clips. -> res (2 conds [leace, rand], T, 4):
    R^2 of a ridge probe clean->clean, (a) clean-trained -> erased, (b) erased-trained -> erased, and the median
    perturbation |r(x) - x| / |x - train mean|. R^2 is pooled over folds (0 = predicting the mean)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from pilot.probe import ALPHAS
    probe = lambda A, y, g: GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": ALPHAS},
                                         cv=GroupKFold(3), scoring="neg_mean_squared_error").fit(A, y, groups=g)
    T, n, _ = Xs.shape
    pred, pert = np.zeros((2, T, 3, n)), np.zeros((2, T, n))
    for f, (tr, te) in enumerate(GroupKFold(k).split(z, z, groups)):
        A, za, cols = Xs[:, tr].reshape(T * len(tr), -1), np.tile(z[tr], T), np.repeat(np.arange(T), len(tr))
        ers = fit(A, za, cols), random_like(A, za, f, cols)
        for s in range(T):
            X = Xs[s]
            clean = probe(X[tr], z[tr], groups[tr])
            for c, er in enumerate(ers):
                Etr, Ete = er.apply(X[tr]), er.apply(X[te])
                pred[c, s, :, te] = np.stack([clean.predict(X[te]), clean.predict(Ete),
                                              probe(Etr, z[tr], groups[tr]).predict(Ete)], 1)
                pert[c, s, te] = np.linalg.norm(Ete - X[te], axis=1) / np.linalg.norm(X[te] - X[tr].mean(0), axis=1)
    r2 = 1 - ((pred - z) ** 2).sum(-1) / ((z - z.mean()) ** 2).sum()
    return np.concatenate([r2, np.median(pert, -1)[..., None]], -1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, D = 600, 40
    A = rng.normal(size=(D, D)) / np.sqrt(D) * 2
    z = rng.normal(size=n) + 2
    noise = lambda: rng.normal(size=(n, D)) @ A + 3
    X = noise() + np.outer(z, rng.normal(size=D))      # correlated features + a z direction
    er = fit(X, z)
    print("cov rel:", cov_rel(X, z, er), "| rand", cov_rel(X, z, random_like(X, z, 0)))
    assert cov_rel(X, z, er) < 1e-6
    res = crossfit(X[None], z, np.arange(n) // 3)
    print("cross-fitted [clean, (a), (b), pert]: leace", res[0, 0].round(3), "| rand", res[1, 0].round(3))
    assert res[0, 0, 0] > 0.5 and (res[0, 0, 1:3] <= 0.05).all() and (res[1, 0, 1:3] > 0.6 * res[0, 0, 0]).all()
    # two steps, z along different directions: cols erases it within each
    X2 = np.concatenate([X, noise() + np.outer(z, rng.normal(size=D))])
    er2 = fit(X2, np.tile(z, 2), np.repeat([0, 1], n))
    assert max(cov_rel(X2[i * n:(i + 1) * n], z, er2) for i in (0, 1)) < 1e-6
    assert np.allclose(er.apply(X.astype(np.float32)), er.apply(X), atol=1e-3)
    import torch
    assert np.allclose(er.apply(torch.tensor(X[:, None])).numpy()[:, 0], er.apply(X), atol=1e-4)      # (..., D) torch, computed in float32
    print("erase selftest ok")
