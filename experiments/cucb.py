import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import shap


# ============================================================
# Friedman DGP with p total features
# First 5 are signal, remaining p-5 are pure noise
# ============================================================

def gen_friedman(
    n: int,
    p: int,
    rng: np.random.Generator,
    noise_std: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Friedman-1 style data with p total features.
    X_j ~ Uniform(0,1), j=1,...,p
    y = 10 sin(pi x1 x2) + 20 (x3 - 0.5)^2 + 10 x4 + 5 x5 + eps

    Returns
    -------
    X : shape (n, p)
    y : shape (n,)
    """
    if p < 5:
        raise ValueError("p must be at least 5 for Friedman DGP.")

    X = rng.uniform(0.0, 1.0, size=(n, p))
    eps = rng.normal(0.0, noise_std, size=n)

    y = (
        10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20.0 * (X[:, 2] - 0.5) ** 2
        + 10.0 * X[:, 3]
        + 5.0 * X[:, 4]
        + eps
    )
    return X, y


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class CUCBConfig:
    n_train: int = 200
    n_val: int = 100
    p: int = 500
    k_select: int = 5
    n_rounds: int = 300
    noise_std: float = 1.0
    seed: int = 123

    # Random forest
    rf_n_estimators: int = 300
    rf_max_depth: int | None = None
    rf_min_samples_leaf: int = 5
    rf_max_features: str | int | float = "sqrt"

    # UCB bonus
    ucb_alpha: float = 2.0

    # Bootstrap
    bootstrap_size: int | None = None  # if None, use n_train


# ============================================================
# Utilities
# ============================================================

def bootstrap_indices(n: int, rng: np.random.Generator, size: int | None = None) -> np.ndarray:
    """
    Sample bootstrap indices with replacement.
    """
    if size is None:
        size = n
    return rng.integers(0, n, size=size)


def compute_shap_weights(
    model: RandomForestRegressor,
    X_val_sub: np.ndarray,
) -> np.ndarray:
    """
    Compute normalized absolute SHAP weights for selected features.

    Returns
    -------
    weights : shape (k_select,)
        Nonnegative weights summing to 1.
    """
    explainer = shap.TreeExplainer(model)

    # For regression, TreeExplainer returns array of shape (n_val, k_select)
    shap_values = explainer.shap_values(X_val_sub)

    # Robust handling in case SHAP returns a list-like object
    shap_values = np.asarray(shap_values)

    # Mean absolute SHAP over validation set
    abs_mean = np.mean(np.abs(shap_values), axis=0)

    total = abs_mean.sum()
    if total <= 0:
        # fallback: uniform split if all SHAP are exactly zero
        weights = np.ones_like(abs_mean) / len(abs_mean)
    else:
        weights = abs_mean / total

    return weights


def select_top_k_ucb(
    mu_hat: np.ndarray,
    counts: np.ndarray,
    t: int,
    k: int,
    alpha: float,
) -> np.ndarray:
    """
    Standard top-k UCB oracle.
    """
    bonus = alpha * np.sqrt(np.log(max(t, 2)) / np.maximum(counts, 1e-12))
    ucb = mu_hat + bonus
    chosen = np.argsort(ucb)[-k:][::-1]
    return chosen


def select_initial_covering_subset(
    counts: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Force initial exploration so every feature is sampled at least once.
    Among unsampled features, choose up to k at random.
    If fewer than k remain, fill the rest randomly from all features.
    """
    unsampled = np.where(counts == 0)[0]
    if len(unsampled) >= k:
        return rng.choice(unsampled, size=k, replace=False)

    chosen = list(unsampled)
    need = k - len(chosen)

    remaining_pool = np.setdiff1d(np.arange(len(counts)), np.array(chosen, dtype=int), assume_unique=False)
    fill = rng.choice(remaining_pool, size=need, replace=False)
    chosen.extend(fill.tolist())
    return np.array(chosen, dtype=int)


# ============================================================
# Main CUCB routine
# ============================================================

def run_cucb_friedman(cfg: CUCBConfig) -> Dict[str, object]:
    rng = np.random.default_rng(cfg.seed)

    # Generate one fixed training set and one fixed validation set
    X_train, y_train = gen_friedman(
        n=cfg.n_train,
        p=cfg.p,
        rng=rng,
        noise_std=cfg.noise_std,
    )
    X_val, y_val = gen_friedman(
        n=cfg.n_val,
        p=cfg.p,
        rng=rng,
        noise_std=cfg.noise_std,
    )

    # Per-feature CUCB state
    counts = np.zeros(cfg.p, dtype=float)
    reward_sums = np.zeros(cfg.p, dtype=float)
    mu_hat = np.zeros(cfg.p, dtype=float)

    # History
    selected_history: List[np.ndarray] = []
    mse_history: List[float] = []
    global_reward_history: List[float] = []
    shap_weight_history: List[np.ndarray] = []
    per_round_feature_rewards: List[np.ndarray] = []

    # Track how often true features are selected
    true_support = set(range(5))
    true_count_history: List[int] = []

    for t in range(1, cfg.n_rounds + 1):
        # ----------------------------------------------------
        # Oracle: top-k by UCB, but first force full coverage
        # ----------------------------------------------------
        if np.any(counts == 0):
            selected = select_initial_covering_subset(
                counts=counts,
                k=cfg.k_select,
                rng=rng,
            )
        else:
            selected = select_top_k_ucb(
                mu_hat=mu_hat,
                counts=counts,
                t=t,
                k=cfg.k_select,
                alpha=cfg.ucb_alpha,
            )

        # ----------------------------------------------------
        # Bootstrap training data
        # ----------------------------------------------------
        boot_idx = bootstrap_indices(
            n=cfg.n_train,
            rng=rng,
            size=cfg.bootstrap_size,
        )

        X_boot = X_train[boot_idx][:, selected]
        y_boot = y_train[boot_idx]

        X_val_sub = X_val[:, selected]

        # ----------------------------------------------------
        # Fit downstream learner
        # ----------------------------------------------------
        model = RandomForestRegressor(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            max_features=cfg.rf_max_features,
            random_state=cfg.seed + t,
            n_jobs=-1,
        )
        model.fit(X_boot, y_boot)

        # ----------------------------------------------------
        # Global reward = negative validation MSE
        # ----------------------------------------------------
        y_pred_val = model.predict(X_val_sub)
        mse_val = mean_squared_error(y_val, y_pred_val)
        global_reward = -mse_val

        # ----------------------------------------------------
        # SHAP-based credit assignment
        # ----------------------------------------------------
        shap_weights = compute_shap_weights(model=model, X_val_sub=X_val_sub)

        # Per-feature rewards for selected features only
        feature_rewards = global_reward * shap_weights

        # ----------------------------------------------------
        # Update CUCB statistics
        # ----------------------------------------------------
        for j_local, j_global in enumerate(selected):
            counts[j_global] += 1.0
            reward_sums[j_global] += feature_rewards[j_local]
            mu_hat[j_global] = reward_sums[j_global] / counts[j_global]

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------
        selected_history.append(selected.copy())
        mse_history.append(mse_val)
        global_reward_history.append(global_reward)
        shap_weight_history.append(shap_weights.copy())

        round_rewards = np.zeros(cfg.p)
        round_rewards[selected] = feature_rewards
        per_round_feature_rewards.append(round_rewards)

        n_true = sum(int(j in true_support) for j in selected)
        true_count_history.append(n_true)

        if t % 25 == 0 or t == 1:
            print(
                f"Round {t:3d} | "
                f"MSE={mse_val:8.4f} | "
                f"selected={selected.tolist()} | "
                f"n_true_in_subset={n_true}"
            )

    # Final ranking by empirical mean reward
    final_rank = np.argsort(mu_hat)[::-1]

    results = {
        "config": cfg,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "counts": counts,
        "reward_sums": reward_sums,
        "mu_hat": mu_hat,
        "final_rank": final_rank,
        "top_20_features": final_rank[:20],
        "selected_history": selected_history,
        "mse_history": np.array(mse_history),
        "global_reward_history": np.array(global_reward_history),
        "shap_weight_history": shap_weight_history,
        "per_round_feature_rewards": per_round_feature_rewards,
        "true_count_history": np.array(true_count_history),
        "true_support": np.array(sorted(true_support)),
    }
    return results


# ============================================================
# Simple summary
# ============================================================

def summarize_results(results: Dict[str, object], top_m: int = 10) -> None:
    mu_hat = results["mu_hat"]
    counts = results["counts"]
    top_features = results["top_20_features"]
    mse_history = results["mse_history"]
    true_count_history = results["true_count_history"]

    print("\n===== Summary =====")
    print(f"Average validation MSE over rounds: {mse_history.mean():.4f}")
    print(f"Best (lowest) validation MSE over rounds: {mse_history.min():.4f}")
    print(f"Average # true features in selected subset: {true_count_history.mean():.3f}")
    print(f"Top-{top_m} features by empirical mean reward:")
    for rank, j in enumerate(top_features[:top_m], start=1):
        print(
            f"  {rank:2d}. feature {j:3d} | "
            f"mu_hat={mu_hat[j]: .6f} | "
            f"count={int(counts[j])}"
        )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    cfg = CUCBConfig(
        n_train=200,
        n_val=100,
        p=200,
        k_select=10,
        n_rounds=100,
        noise_std=1.0,
        seed=123,
        rf_n_estimators=300,
        rf_max_depth=None,
        rf_min_samples_leaf=5,
        rf_max_features="sqrt",
        ucb_alpha=2.0,
        bootstrap_size=200,
    )

    results = run_cucb_friedman(cfg)
    summarize_results(results, top_m=10)


