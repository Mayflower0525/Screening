"""
REE Prediction Pipeline (clean) — Augmentation × Model × repeated-CV study
=========================================================================
Train-on-Synthetic+Real, Test-on-Real (TSTR) evaluation of generative data
augmentation for resting-energy-expenditure (REE) prediction.

Models       : SVR, Random Forest, XGBoost (ML) · MLP, FT-Transformer (DL)
Augmenters   : TVAE_default, TVAE_reweighing, GaussianCopula
Features     : a single feature set (ALL_COLS predictors + BMI dummies)
Ratios       : synthetic:real ratio r in DEFAULT_RATIOS (0 = no augmentation)
Evaluation   : repeated stratified k-fold CV (TSTR; augmenters never see test)

Each model carries its own no-augmentation baseline (ratio 0), so the effect
of every augmenter is measured against the same model (within-model design).

Run:  python -c "import ree_pipeline_clean as p; p.main()"
Outputs: repeated_kfold_results.csv / _summary.csv / _paired_tests.csv /
         _within_model_tests.csv / _subgroup_summary.csv
"""

# =============================================================================
# Imports
# =============================================================================
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from xgboost import XGBRegressor

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer

# SHAP / parsimony analysis (added module)
import matplotlib
matplotlib.use("Agg")            # safe for headless / notebook file output
import matplotlib.pyplot as plt
try:
    import shap
except ImportError as _shap_err:  # pragma: no cover
    shap = None                   # only needed for run_shap_parsimony()

warnings.filterwarnings("ignore")

# =============================================================================
# Config
# =============================================================================
RANDOM_STATE = 42
TARGET = "REE"

CONT_COLS = ["Age", "Weight", "Height", "Calf", "HandGrip", "MUAC"]
CAT_COLS  = ["Gender", "BMI_Category_New"]      # BMI is back
ALL_COLS  = CONT_COLS + CAT_COLS + [TARGET]

# Tail columns used by the reweighing TVAE (now includes Calf)
TAIL_COLS = ["Weight", "MUAC", "REE", "Age", "Height", "HandGrip", "Calf"]

# BMI handling
BMI_BASELINE = "Normal Weight"   # category dropped from dummies (= reference)

# Sweep config
DEFAULT_RATIOS    = [0.0, 1.0]
DEFAULT_N_SEEDS   = 3
DEFAULT_TEST_SIZE = 0.2
DEFAULT_TUNE_ITER = 30

# CV config
DEFAULT_N_REPEATS = 5
DEFAULT_N_FOLDS   = 5

INPUT_CSV = "REPOSE_240326_amended.csv"

# Models and augmentation methods to sweep
ML_MODELS  = ["SVR", "RF", "XGBoost"]
DL_MODELS  = {"MLP", "FTTransformer"}            # SAINT dropped
MODELS     = ML_MODELS + ["MLP", "FTTransformer"]
SWEEP_AUGMENTERS = ["TVAE_default", "TVAE_reweighing", "GaussianCopula"]
REWEIGH_PARAMS   = {"low_q": 0.05, "high_q": 0.95, "max_w": 15.0}

REWEIGH_PARAMS_DEFAULT = REWEIGH_PARAMS  # alias

# =============================================================================
# Reproducibility & metrics
# =============================================================================

def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ccc(y, p):
    mu_y, mu_p = y.mean(), p.mean()
    return (2 * np.mean((y - mu_y) * (p - mu_p))) / (
        y.var() + p.var() + (mu_y - mu_p) ** 2
    )

def acc_cats(y, p, thr=0.10):
    e = (p - y) / y
    a = np.abs(e) <= thr
    return tuple(round(100 * m.sum() / len(y), 1)
                 for m in (a, (~a) & (e < 0), (~a) & (e > 0)))

def icc21(y, p):
    n = len(y)
    d = np.column_stack([y, p])
    gm = d.mean()
    SSR = 2 * np.sum((d.mean(1) - gm) ** 2)
    SSC = n * np.sum((d.mean(0) - gm) ** 2)
    MSR = SSR / (n - 1)
    MSE = (d.var() * 2 * n - SSR - SSC) / (n - 1)
    MSC = SSC
    icc = (MSR - MSE) / (MSR + MSE + (2 / n) * (MSC - MSE))
    F = MSR / MSE
    ci = lambda q: round(
        (F / stats.f.ppf(q, n - 1, n - 1) - 1)
        / (F / stats.f.ppf(q, n - 1, n - 1) + 1),
        3,
    )
    return round(icc, 3), ci(0.975), ci(0.025)

def calibration_slope(y, p):
    slope, intercept, *_ = stats.linregress(p, y)
    return round(slope, 4), round(intercept, 4)

def bootstrap_ci(y_true, y_pred, metric_fn,
                 n_boot=1000, ci=0.95, seed=RANDOM_STATE):
    """Percentile bootstrap CI for `metric_fn(y_true, y_pred)`.

    Resamples the (y, p) pairs with replacement `n_boot` times, computes
    the metric on each resample, and returns (lower, upper) bounds at
    the specified confidence level.  This captures sampling variability
    of the test set itself — *complementary* to the across-seed std,
    which captures variability from the train/test split.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = metric_fn(y_true[idx], y_pred[idx])
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.percentile(vals, 100 * alpha)),
        float(np.percentile(vals, 100 * (1 - alpha))),
    )

# =============================================================================
# Data & single feature set
# =============================================================================

def load_data(path=INPUT_CSV):
    """Load CSV, normalise column names, drop NA, return ALL_COLS."""
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: c.strip()
                   .replace(" ", "_").replace("-", "_")
                   .replace("(", "").replace(")", ""))
    df = df.dropna()
    real = df[ALL_COLS].dropna().reset_index(drop=True)
    # Make BMI a clean categorical (strip stray whitespace like 'Overweight ')
    real["BMI_Category_New"] = (
        real["BMI_Category_New"].astype(str).str.strip()
    )
    return real

def expand_features(df, baseline=BMI_BASELINE):
    """Add BMI-category dummies (drop `baseline` as the reference level).
    No interaction terms — a single feature set is used."""
    out = df.copy()
    dummies = pd.get_dummies(out["BMI_Category_New"], prefix="BMI_Category")
    drop_col = f"BMI_Category_{baseline}"
    if drop_col in dummies.columns:
        dummies = dummies.drop(columns=[drop_col])
    return pd.concat([out, dummies.astype(int)], axis=1)


def get_features(expanded_df):
    """The single feature set: continuous clinical vars + Gender + BMI dummies."""
    bmi_cols = [c for c in expanded_df.columns
                if c.startswith("BMI_Category_") and c != "BMI_Category_New"]
    return [c for c in CONT_COLS if c in expanded_df.columns] + ["Gender"] + bmi_cols

# =============================================================================
# Augmenters
# =============================================================================

def fit_tvae_default(train_data, seed=RANDOM_STATE):
    seed_everything(seed)
    md = SingleTableMetadata()
    md.detect_from_dataframe(train_data)
    tvae = TVAESynthesizer(
        md, epochs=300, batch_size=50, embedding_dim=32,
        compress_dims=(64, 64), decompress_dims=(64, 64),
        l2scale=1e-4, cuda=False,
    )
    tvae.fit(train_data)
    return tvae

def get_bilateral_weights(df, cols, low_q=0.05, high_q=0.95, max_w=10.0):
    weights = np.ones(len(df))
    for col in cols:
        val = df[col].values
        q_lo, q_hi = df[col].quantile([low_q, high_q])
        mask_hi = val > q_hi
        if mask_hi.any() and (val.max() > q_hi):
            w_hi = 1 + (val[mask_hi] - q_hi) / (val.max() - q_hi) * (max_w - 1)
            weights[mask_hi] = np.maximum(weights[mask_hi], w_hi)
        mask_lo = val < q_lo
        if mask_lo.any() and (q_lo > val.min()):
            w_lo = 1 + (q_lo - val[mask_lo]) / (q_lo - val.min()) * (max_w - 1)
            weights[mask_lo] = np.maximum(weights[mask_lo], w_lo)
    return weights

def fit_tvae_reweighing(train_data, seed=RANDOM_STATE,
                        low_q=0.05, high_q=0.95, max_w=10.0,
                        bootstrap_factor=2.0):
    """Reweighing TVAE.  All reweighing knobs are exposed so they can be
    swept by `sweep_reweighing_params`.  Defaults match the original
    implementation, so existing callers keep their previous behaviour."""
    seed_everything(seed)
    weights = get_bilateral_weights(
        train_data, TAIL_COLS, low_q=low_q, high_q=high_q, max_w=max_w
    )
    boot_idx = np.random.choice(
        len(train_data),
        size=int(len(train_data) * bootstrap_factor),
        replace=True, p=weights / weights.sum(),
    )
    boot_df = train_data.iloc[boot_idx].reset_index(drop=True)

    is_tail = pd.DataFrame({
        c: (train_data[c] < train_data[c].quantile(low_q))
        | (train_data[c] > train_data[c].quantile(high_q))
        for c in TAIL_COLS
    }).any(axis=1)
    tail_rows = train_data[is_tail]

    rebalanced = (
        pd.concat([boot_df, tail_rows], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    seed_everything(seed)
    md = SingleTableMetadata()
    md.detect_from_dataframe(rebalanced)
    tvae_b = TVAESynthesizer(
        md, epochs=300, batch_size=50, embedding_dim=16, cuda=False,
    )
    tvae_b.fit(rebalanced)
    return tvae_b

def fit_gaussian_copula(train_data, seed=RANDOM_STATE):
    seed_everything(seed)
    md = SingleTableMetadata()
    md.detect_from_dataframe(train_data)
    gc = GaussianCopulaSynthesizer(md, default_distribution="norm")
    gc.fit(train_data)
    return gc

AUGMENTERS = {
    "TVAE_default":    fit_tvae_default,
    "TVAE_reweighing": fit_tvae_reweighing,
    "GaussianCopula":  fit_gaussian_copula,
}

def sample_synthetic(augmentor, n_synth, real_for_clip):
    """Sample n_synth rows; clip continuous fields to real-data support."""
    if n_synth <= 0:
        return pd.DataFrame(columns=real_for_clip.columns)
    synth = augmentor.sample(num_rows=n_synth, batch_size=n_synth)
    for col in CONT_COLS + [TARGET]:
        synth[col] = synth[col].clip(real_for_clip[col].min(),
                                     real_for_clip[col].max())
    # Ensure BMI category is a string (needed for expand_features later)
    if "BMI_Category_New" in synth.columns:
        synth["BMI_Category_New"] = synth["BMI_Category_New"].astype(str)
    return synth

# =============================================================================
# Hyperparameter grids
# =============================================================================
PARAM_GRIDS = {
    "SVR": {
        "svr__C":       [0.1, 1, 10, 100],
        "svr__epsilon": [0.01, 0.05, 0.1, 0.5],
        "svr__kernel":  ["rbf", "linear"],
        "svr__gamma":   ["scale", "auto", 0.001, 0.01],
    },
    "RF": {
        "n_estimators":      [100, 300, 500, 1000],
        "max_depth":         [None, 4, 6, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
    },
    "XGBoost": {
        "n_estimators":     [100, 300, 600, 1000],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "max_depth":        [3, 4, 6, 8],
        "subsample":        [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    },
}

# =============================================================================
# Deep-learning models (sklearn-compatible PyTorch regressors)
# =============================================================================

class _BaseTabularDL(BaseEstimator, RegressorMixin):
    """Common training loop for tabular DL regressors.

    Standardises X and y, splits off 15 % of train as an internal
    validation set for early stopping, restores best-validation weights.
    Subclasses implement `_build_model(n_features)`.
    """

    def __init__(self, lr=1e-3, weight_decay=1e-4, batch_size=64,
                 max_epochs=150, patience=20, device="cpu",
                 random_state=42, verbose=False):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # subclasses must implement this
    def _build_model(self, n_features):
        raise NotImplementedError

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Standardise X and y for stable training
        self._x_mean = X.mean(axis=0)
        self._x_std  = X.std(axis=0) + 1e-8
        X_norm = (X - self._x_mean) / self._x_std

        self._y_mean = float(y.mean())
        self._y_std  = float(y.std() + 1e-8)
        y_norm = (y - self._y_mean) / self._y_std

        # 85/15 train / internal-val split for early stopping
        n = len(X)
        n_val = max(int(0.15 * n), 8)
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n)
        v_idx, t_idx = perm[:n_val], perm[n_val:]

        Xt = torch.from_numpy(X_norm[t_idx]).to(self.device)
        yt = torch.from_numpy(y_norm[t_idx]).to(self.device).unsqueeze(1)
        Xv = torch.from_numpy(X_norm[v_idx]).to(self.device)
        yv = torch.from_numpy(y_norm[v_idx]).to(self.device).unsqueeze(1)

        self.model_ = self._build_model(X.shape[1]).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.lr, weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(Xt, yt), batch_size=self.batch_size,
            shuffle=True, drop_last=False,
        )

        best_val = float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(self.max_epochs):
            self.model_.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                vloss = float(loss_fn(self.model_(Xv), yv))

            if vloss < best_val - 1e-6:
                best_val = vloss
                best_state = {
                    k: v.detach().clone()
                    for k, v in self.model_.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self._n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_norm = (X - self._x_mean) / self._x_std
        Xt = torch.from_numpy(X_norm).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(Xt).cpu().numpy().flatten()
        return pred * self._y_std + self._y_mean

class TabularMLP(_BaseTabularDL):
    """Plain MLP with batch-norm + dropout.

    Simple, fast, often competitive on small tabular data.  Defaults:
    3 hidden layers × 128 units, dropout 0.3.
    """

    def __init__(self, hidden_dim=128, n_layers=3, dropout=0.3, **kw):
        super().__init__(**kw)
        self.hidden_dim = hidden_dim
        self.n_layers  = n_layers
        self.dropout   = dropout

    def _build_model(self, n_features):
        layers, prev = [], n_features
        for _ in range(self.n_layers):
            layers += [
                nn.Linear(prev, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ]
            prev = self.hidden_dim
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

class _TransformerBlock(nn.Module):
    """Pre-LN transformer block used by FT-Transformer."""

    def __init__(self, d_model, n_heads=4, ff_dim=64, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # x: (B, T, D)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h)
        x = x + a
        h = self.ln2(x)
        x = x + self.ff(h)
        return x

class _FTTransformerNet(nn.Module):
    """FT-Transformer network: feature tokenizer → CLS + features →
    transformer blocks → linear head on CLS."""

    def __init__(self, n_features, d_model=32, n_blocks=2,
                 n_heads=4, ff_dim=64, dropout=0.1):
        super().__init__()
        # Per-feature linear tokenizer (each feature value -> d_model vector)
        self.weight = nn.Parameter(
            torch.randn(n_features, d_model) / (d_model ** 0.5)
        )
        self.bias   = nn.Parameter(torch.zeros(n_features, d_model))
        # CLS token
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) / (d_model ** 0.5))
        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):                       # x: (B, F)
        B = x.shape[0]
        tokens = x.unsqueeze(-1) * self.weight + self.bias   # (B, F, D)
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)             # (B, F+1, D)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.head(tokens[:, 0])

class FTTransformer(_BaseTabularDL):
    """Feature Tokenizer + Transformer (Gorishniy et al., 2021).

    Scaled down for small-N CPU training: 2 blocks, 32-dim embeddings,
    4 heads, 64-dim FF.  Current SOTA family on tabular benchmarks.
    """

    def __init__(self, d_model=32, n_blocks=2, n_heads=4,
                 ff_dim=64, dropout=0.1, **kw):
        super().__init__(**kw)
        self.d_model  = d_model
        self.n_blocks = n_blocks
        self.n_heads  = n_heads
        self.ff_dim   = ff_dim
        self.dropout  = dropout

    def _build_model(self, n_features):
        return _FTTransformerNet(
            n_features=n_features, d_model=self.d_model,
            n_blocks=self.n_blocks, n_heads=self.n_heads,
            ff_dim=self.ff_dim, dropout=self.dropout,
        )

# =============================================================================
# Build / tune / train-eval
# =============================================================================
def build_model(model_name, params=None, seed=RANDOM_STATE):
    """Construct an estimator. ML: SVR / RF / XGBoost.  DL: MLP / FTTransformer."""
    if model_name == "SVR":
        if params is None:
            return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0))
        kw = {k.replace("svr__", ""): v for k, v in params.items()}
        return make_pipeline(StandardScaler(), SVR(**kw))
    if model_name == "RF":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **(params or {}))
    if model_name == "XGBoost":
        return XGBRegressor(random_state=seed, verbosity=0, n_jobs=-1, **(params or {}))
    if model_name == "MLP":
        return TabularMLP(random_state=seed, **(params or {}))
    if model_name == "FTTransformer":
        return FTTransformer(random_state=seed, **(params or {}))
    raise ValueError(f"Unknown model: {model_name}")

def tune_model(train_df, model_name, feat_cols,
               n_iter=DEFAULT_TUNE_ITER, seed=RANDOM_STATE,
               param_grid=None):
    """Tune model on real-only data; return best params dict.

    `param_grid=None`  → use the default `PARAM_GRIDS[model_name]`.
    Pass an explicit dict to override (e.g. `SVR_PARAM_GRID_RBF`).

    DL models (MLP / FT-Transformer) skip RandomizedSearchCV
    and use their built-in sensible defaults — repeated DL re-tuning at
    every fold would dominate runtime and provide little benefit.
    """
    if model_name in DL_MODELS:
        return {}  # use the class's default hyperparameters

    X = train_df[feat_cols].astype(float).values
    y = train_df[TARGET].astype(float).values
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    base = build_model(model_name, params=None, seed=seed)
    grid = param_grid if param_grid is not None else PARAM_GRIDS[model_name]
    search = RandomizedSearchCV(
        base, grid, n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=kf, random_state=seed, n_jobs=-1,
    ).fit(X, y)
    return search.best_params_

def train_eval_model(train_df, test_df, model_name, params,
                     feat_cols, seed=RANDOM_STATE, n_boot=1000,
                     slice_col=None):
    """Fit on train_df, evaluate on test_df (real-only).  Return metrics.


    `n_boot` : int, default 1000
        Number of bootstrap resamples for 95 % CIs on RMSE / MAE / MAPE /
        R² / CCC / accuracy_5pct / accuracy_pct / accuracy_15pct.  Set
        to 0 to disable bootstrap (faster — useful for the wide
        full-grid sweep).

    `slice_col` : str or None, default None
        If a column name (e.g. "BMI_Category_New") is given, also compute
        per-slice descriptive metrics (n, RMSE, MAE, bias, R², CCC, ICC,
        cal_slope) on each unique value of that column in test_df.
        Columns are named `slice_<value>_<metric>` and flow through to the
        sweep aggregation.  Correlation metrics (R², CCC, ICC, cal_slope)
        require n ≥ 8 per slice per fold; smaller slices skip them.
    """
    X_tr = train_df[feat_cols].astype(float).values
    y_tr = train_df[TARGET].astype(float).values
    X_te = test_df[feat_cols].astype(float).values
    y_te = test_df[TARGET].astype(float).values

    model = build_model(model_name, params=params, seed=seed)
    model.fit(X_tr, y_tr)
    p = model.predict(X_te)

    rmse = float(np.sqrt(mean_squared_error(y_te, p)))
    mae  = float(mean_absolute_error(y_te, p))
    r2   = float(r2_score(y_te, p))
    cc   = float(ccc(y_te, p))
    bias = float(np.mean(p - y_te))
    icc, ilo, ihi = icc21(y_te, p)
    ac, un, ov    = acc_cats(y_te, p)
    cs, ci_       = calibration_slope(y_te, p)

    # Additional clinical metrics
    mape = float(np.mean(np.abs((p - y_te) / y_te)) * 100)
    rel = (p - y_te) / y_te
    ac_5  = round(100 * (np.abs(rel) <= 0.05).mean(), 1)
    ac_15 = round(100 * (np.abs(rel) <= 0.15).mean(), 1)

    # Bland-Altman 95 % limits of agreement
    diffs = p - y_te
    sd_diff = float(diffs.std())
    loa_lower = float(diffs.mean() - 1.96 * sd_diff)
    loa_upper = float(diffs.mean() + 1.96 * sd_diff)

    # Bootstrap 95 % CIs on the key metrics (test-set sampling variability)
    boot_results = {}
    if n_boot and n_boot > 0:
        boot_specs = {
            "rmse":          lambda y, q: float(np.sqrt(mean_squared_error(y, q))),
            "mae":           lambda y, q: float(mean_absolute_error(y, q)),
            "mape":          lambda y, q: float(np.mean(np.abs((q - y) / y)) * 100),
            "r2":            lambda y, q: float(r2_score(y, q)),
            "ccc":           lambda y, q: float(ccc(y, q)),
            "accurate_5pct":  lambda y, q: float(100 * (np.abs((q - y) / y) <= 0.05).mean()),
            "accurate_pct":   lambda y, q: float(100 * (np.abs((q - y) / y) <= 0.10).mean()),
            "accurate_15pct": lambda y, q: float(100 * (np.abs((q - y) / y) <= 0.15).mean()),
        }
        for name, fn in boot_specs.items():
            lo, hi = bootstrap_ci(
                y_te, p, fn, n_boot=n_boot, ci=0.95, seed=seed,
            )
            boot_results[f"{name}_boot_lo"] = round(lo, 4)
            boot_results[f"{name}_boot_hi"] = round(hi, 4)

    # ---- Per-slice descriptive metrics (e.g. BMI category subgroups) ----
    slice_metrics = {}
    if slice_col is not None and slice_col in test_df.columns:
        slice_vals = test_df[slice_col].astype(str).values
        for sv in pd.unique(slice_vals):
            mask = (slice_vals == sv)
            n_s = int(mask.sum())
            if n_s < 3:
                continue
            y_s = y_te[mask]
            p_s = p[mask]
            sv_safe = (str(sv).strip()
                       .replace(" ", "_").replace("|", "_")
                       .replace("/", "_"))
            pref = f"slice_{sv_safe}"
            slice_metrics[f"{pref}_n"]    = n_s
            slice_metrics[f"{pref}_rmse"] = round(
                float(np.sqrt(mean_squared_error(y_s, p_s))), 4)
            slice_metrics[f"{pref}_mae"]  = round(
                float(mean_absolute_error(y_s, p_s)), 4)
            slice_metrics[f"{pref}_bias"] = round(
                float(np.mean(p_s - y_s)), 4)
            # Correlation metrics need a minimum n to be meaningful
            if n_s >= 8:
                try:
                    slice_metrics[f"{pref}_r2"]  = round(
                        float(r2_score(y_s, p_s)), 4)
                    slice_metrics[f"{pref}_ccc"] = round(
                        float(ccc(y_s, p_s)), 4)
                    icc_v, _, _ = icc21(y_s, p_s)
                    slice_metrics[f"{pref}_icc"] = icc_v
                    cs_v, _ = calibration_slope(y_s, p_s)
                    slice_metrics[f"{pref}_cal_slope"] = cs_v
                except (ValueError, ZeroDivisionError):
                    pass  # metric undefined for this fold; aggregate skips NaN

    return {
        "n_train":       len(train_df),
        "n_test":        len(test_df),
        "rmse":          round(rmse, 4),
        "mae":           round(mae, 4),
        "mape":          round(mape, 4),
        "r2":            round(r2, 4),
        "ccc":           round(cc, 4),
        "bias":          round(bias, 4),
        "icc":           icc,
        "icc_lower_95":  ilo,
        "icc_upper_95":  ihi,
        "accurate_5pct":  ac_5,
        "accurate_pct":   ac,        # ±10 % (kept for back-compat)
        "accurate_15pct": ac_15,
        "under_pct":     un,
        "over_pct":      ov,
        "loa_lower":     round(loa_lower, 4),
        "loa_upper":     round(loa_upper, 4),
        "cal_slope":     cs,
        "cal_intercept": ci_,
        **boot_results,
        **slice_metrics,
    }

# =============================================================================
# Config sweep (within-model design)
# =============================================================================
def _config_label(cfg):
    """Compact label, e.g. 'TVAE_reweighing|w=15.0 | RF | r=0.5'."""
    rew = cfg.get("reweighing_params")
    rew_str = f"|q={rew['low_q']}-{rew['high_q']},w={rew['max_w']}" if rew else ""
    return f"{cfg['augmenter']}{rew_str} | {cfg['model']} | r={cfg['ratio']}"


def make_configs(models=MODELS, augmenters=SWEEP_AUGMENTERS, ratios=DEFAULT_RATIOS):
    """Per model: one no-aug baseline (ratio 0) + every augmenter at every
    ratio > 0.  reweighing_params are attached to TVAE_reweighing configs."""
    cfgs = []
    for m in models:
        cfgs.append({"augmenter": "NoAug", "model": m, "ratio": 0.0})
        for aug in augmenters:
            for r in ratios:
                if float(r) == 0.0:
                    continue
                c = {"augmenter": aug, "model": m, "ratio": float(r)}
                if aug == "TVAE_reweighing":
                    c["reweighing_params"] = dict(REWEIGH_PARAMS)
                cfgs.append(c)
    return cfgs


ALL_CONFIGS = make_configs()


def within_model_comparison(results_df, configs, metric="rmse",
                            lower_is_better=True, index_col="iteration"):
    """Paired tests: each augmented config vs its OWN model's no-aug baseline
    (ratio 0), matched by `index_col`.  Isolates the augmentation effect
    within each model."""
    def _is_baseline(c):
        return float(c["ratio"]) == 0.0
    base_by_model = {c["model"]: _config_label(c) for c in configs if _is_baseline(c)}
    available = set(results_df["config"].unique())
    out = []
    for c in configs:
        if _is_baseline(c):
            continue
        base_label = base_by_model.get(c["model"])
        cfg_label = _config_label(c)
        if not base_label or cfg_label not in available or base_label not in available:
            continue
        sub = results_df[results_df["config"].isin([cfg_label, base_label])]
        pc = paired_comparison(sub, base_label, metric=metric,
                               lower_is_better=lower_is_better, index_col=index_col)
        pc = pc[pc["config"] == cfg_label].copy()
        pc.insert(0, "model", c["model"])
        out.append(pc)
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    return res.rename(columns={"vs_baseline": "vs_own_baseline",
                               "better_than_baseline": "better_than_own_baseline"})

def paired_comparison(focused_df, baseline_config_label,
                      metric="rmse", lower_is_better=True,
                      index_col="seed"):
    """Paired t-test (and Wilcoxon signed-rank) comparing every config
    against `baseline_config_label`, matched by the column given in
    `index_col` (default 'seed').  For repeated k-fold results, pass
    `index_col='iteration'` to match by (repeat, fold) pair."""
    pivot = focused_df.pivot(index=index_col, columns="config", values=metric)
    if baseline_config_label not in pivot.columns:
        raise ValueError(
            f"Baseline '{baseline_config_label}' not in configs.  "
            f"Available: {list(pivot.columns)}"
        )
    base = pivot[baseline_config_label]

    rows = []
    for cfg in pivot.columns:
        if cfg == baseline_config_label:
            continue
        diff = pivot[cfg] - base
        # Paired t-test
        t_stat, p_t = stats.ttest_rel(pivot[cfg], base)
        # Wilcoxon signed-rank (non-parametric)
        try:
            w_stat, p_w = stats.wilcoxon(pivot[cfg], base)
        except ValueError:
            w_stat, p_w = np.nan, np.nan
        rows.append({
            "config":          cfg,
            "vs_baseline":     baseline_config_label,
            "metric":          metric,
            "mean_diff":       round(float(diff.mean()), 4),
            "ci95_lo":         round(float(diff.mean() - 1.96 * diff.std() / np.sqrt(len(diff))), 4),
            "ci95_hi":         round(float(diff.mean() + 1.96 * diff.std() / np.sqrt(len(diff))), 4),
            "paired_t":        round(float(t_stat), 4),
            "paired_t_pval":   round(float(p_t), 4),
            "wilcoxon_pval":   round(float(p_w), 4) if not np.isnan(p_w) else np.nan,
            "better_than_baseline": (
                (diff.mean() < 0) if lower_is_better else (diff.mean() > 0)
            ),
        })
    return pd.DataFrame(rows)

# =============================================================================
# Repeated stratified k-fold CV (TSTR)
# =============================================================================
def run_repeated_kfold(real_data, configs=None,
                       n_repeats=DEFAULT_N_REPEATS, n_folds=DEFAULT_N_FOLDS,
                       tune_iter=DEFAULT_TUNE_ITER, n_boot=1000,
                       slice_col=None, verbose=True):
    """For each (repeat, fold): stratified split on REE quartiles; tune each
    model once on real-only train; fit each augmenter (ratio>0) once on
    real-only train; for each config sample synthetic at its ratio, train on
    (real + synthetic), evaluate on the real-only test fold.  Single feature set."""
    if configs is None:
        configs = ALL_CONFIGS

    unique_models  = {cfg["model"] for cfg in configs}
    unique_aug_keys = set()
    for cfg in configs:
        rew = cfg.get("reweighing_params")
        rew_key = frozenset(rew.items()) if rew else None
        if cfg["ratio"] > 0:
            unique_aug_keys.add((cfg["augmenter"], rew_key))

    rows = []
    ree_q = pd.qcut(real_data[TARGET], q=4, labels=False, duplicates="drop")

    for repeat in range(n_repeats):
        if verbose:
            print(f"\n>>> Repeat {repeat + 1}/{n_repeats}")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=repeat)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(real_data, ree_q)):
            iter_seed = repeat * n_folds + fold_idx
            train_real_raw = real_data.iloc[train_idx].reset_index(drop=True)
            test_real_raw  = real_data.iloc[test_idx].reset_index(drop=True)

            train_real = expand_features(train_real_raw)
            test_real  = expand_features(test_real_raw)
            feat_cols  = get_features(train_real)

            best_params = {
                m: tune_model(train_real, m, feat_cols,
                              n_iter=tune_iter, seed=iter_seed)
                for m in unique_models
            }

            fitted = {}
            for aug_key in unique_aug_keys:
                aug_name, rew_key = aug_key
                if aug_name == "TVAE_reweighing" and rew_key is not None:
                    fitted[aug_key] = fit_tvae_reweighing(
                        train_real_raw, seed=iter_seed, **dict(rew_key))
                else:
                    fitted[aug_key] = AUGMENTERS[aug_name](train_real_raw, iter_seed)

            for cfg in configs:
                aug_name = cfg["augmenter"]
                rew = cfg.get("reweighing_params")
                rew_key = frozenset(rew.items()) if rew else None
                model_name = cfg["model"]
                ratio = cfg["ratio"]

                n_synth = int(round(ratio * len(train_real_raw)))
                if n_synth == 0:
                    train_combined_raw = train_real_raw.copy()
                else:
                    synth_raw = sample_synthetic(
                        fitted[(aug_name, rew_key)], n_synth, train_real_raw)
                    train_combined_raw = pd.concat(
                        [train_real_raw, synth_raw], ignore_index=True)

                train_combined = expand_features(train_combined_raw)
                metrics = train_eval_model(
                    train_combined, test_real, model_name,
                    best_params[model_name], feat_cols,
                    seed=iter_seed, n_boot=n_boot, slice_col=slice_col)

                rows.append({
                    "config":            _config_label(cfg),
                    "augmenter":         aug_name,
                    "reweighing_params": str(rew) if rew else "",
                    "model":             model_name,
                    "ratio":             ratio,
                    "repeat":            repeat,
                    "fold":              fold_idx,
                    "iteration":         iter_seed,
                    **metrics,
                })

            if verbose:
                last = rows[-len(configs):]
                print(f"    Fold {fold_idx + 1}/{n_folds}  "
                      f"n_train={len(train_real_raw):>4}  n_test={len(test_real_raw):>3}")

    return pd.DataFrame(rows)

def aggregate_repeated_kfold(rkf_df, metric_cols=None, ci=0.95):
    """Aggregate repeated k-fold results across all (repeat, fold)
    iterations.  Reports n_evaluations, n_repeats, n_folds alongside the
    standard mean ± std + t-distribution CI + average per-iteration
    bootstrap CI for every metric."""
    if metric_cols is None:
        metric_cols = [
            "rmse", "mae", "mape", "r2", "ccc", "bias",
            "icc", "accurate_5pct", "accurate_pct", "accurate_15pct",
            "loa_lower", "loa_upper", "cal_slope",
        ]
        # Auto-detect slice metric columns produced by train_eval_model
        # when slice_col is set (e.g. slice_Underweight_rmse,
        # slice_Obese_icc, etc.) and include them in the aggregation.
        slice_suffixes = ("_n", "_rmse", "_mae", "_bias",
                          "_r2", "_ccc", "_icc", "_cal_slope")
        slice_cols = sorted(
            c for c in rkf_df.columns
            if c.startswith("slice_")
            and any(c.endswith(s) for s in slice_suffixes)
        )
        metric_cols = metric_cols + slice_cols

    out_rows = []
    for cfg_label, grp in rkf_df.groupby("config", sort=False):
        n = len(grp)
        t_crit = stats.t.ppf(0.5 + ci / 2, df=n - 1) if n > 1 else 0
        row = {
            "config":        cfg_label,
            "n_evaluations": n,
            "n_repeats":     grp["repeat"].nunique(),
            "n_folds":       grp["fold"].nunique(),
            "augmenter":     grp["augmenter"].iloc[0],
            "model":         grp["model"].iloc[0],
            "ratio":         grp["ratio"].iloc[0],
        }
        for m in metric_cols:
            mean = grp[m].mean()
            std  = grp[m].std()
            sem  = std / np.sqrt(n) if n > 1 else 0
            row[f"{m}_mean"]  = round(float(mean), 4)
            row[f"{m}_std"]   = round(float(std), 4)
            row[f"{m}_ci_lo"] = round(float(mean - t_crit * sem), 4)
            row[f"{m}_ci_hi"] = round(float(mean + t_crit * sem), 4)
            boot_lo_col = f"{m}_boot_lo"
            boot_hi_col = f"{m}_boot_hi"
            if boot_lo_col in grp.columns and boot_hi_col in grp.columns:
                row[f"{m}_boot_lo_avg"] = round(float(grp[boot_lo_col].mean()), 4)
                row[f"{m}_boot_hi_avg"] = round(float(grp[boot_hi_col].mean()), 4)
        out_rows.append(row)
    return pd.DataFrame(out_rows)

# =============================================================================
# Main entrypoint
# =============================================================================
def main(real_data=None, input_csv=INPUT_CSV,
         n_repeats=DEFAULT_N_REPEATS, n_folds=DEFAULT_N_FOLDS,
         tune_iter=DEFAULT_TUNE_ITER, n_boot=500,
         configs=None, slice_col="BMI_Category_New"):
    """Run the full repeated-CV augmentation study and save all CSVs."""
    seed_everything(RANDOM_STATE)
    if real_data is None:
        print(">>> Loading data ...")
        real_data = load_data(input_csv)
        print(f"    rows: {len(real_data)}")
    if configs is None:
        configs = make_configs()

    n_iter_total = n_repeats * n_folds
    print(f"\n{'=' * 68}")
    print(f"  REPEATED STRATIFIED {n_folds}-FOLD CV  |  {n_repeats} repeats "
          f"= {n_iter_total} evals/config")
    print(f"  models: {MODELS}")
    print(f"  augmenters: {SWEEP_AUGMENTERS}  |  ratios: {DEFAULT_RATIOS}")
    print(f"  configs: {len(configs)}  |  n_boot: {n_boot}  |  subgroups: {slice_col}")
    print(f"{'=' * 68}")

    rkf_df = run_repeated_kfold(
        real_data, configs=configs, n_repeats=n_repeats, n_folds=n_folds,
        tune_iter=tune_iter, n_boot=n_boot, slice_col=slice_col, verbose=True)
    rkf_df.to_csv("repeated_kfold_results.csv", index=False)
    print("\n✓ Saved per-iteration results → repeated_kfold_results.csv")

    agg = aggregate_repeated_kfold(rkf_df)
    agg.to_csv("repeated_kfold_summary.csv", index=False)
    print("✓ Saved aggregated summary → repeated_kfold_summary.csv")

    show = [c for c in ["config", "n_evaluations", "rmse_mean", "rmse_std",
            "mae_mean", "mape_mean", "r2_mean", "ccc_mean", "icc_mean",
            "accurate_pct_mean", "cal_slope_mean"] if c in agg.columns]
    print("\n  ── Summary (sorted by RMSE) ──")
    print(agg[show].sort_values("rmse_mean").to_string(index=False))

    baseline_label = _config_label(configs[0])
    if baseline_label in rkf_df["config"].unique():
        paired = paired_comparison(rkf_df, baseline_label, metric="rmse",
                                   lower_is_better=True, index_col="iteration")
        paired.to_csv("repeated_kfold_paired_tests.csv", index=False)
        print("✓ Saved → repeated_kfold_paired_tests.csv")

    within = within_model_comparison(rkf_df, configs, metric="rmse",
                                     lower_is_better=True, index_col="iteration")
    if not within.empty:
        within.to_csv("repeated_kfold_within_model_tests.csv", index=False)
        print("\n  ── Within-model paired tests (vs each model's own no-aug baseline) ──")
        print(within.to_string(index=False))
        print("✓ Saved → repeated_kfold_within_model_tests.csv")

    if slice_col:
        slice_names = sorted(set(
            c.replace("slice_", "").replace("_rmse_mean", "")
            for c in agg.columns if c.startswith("slice_") and c.endswith("_rmse_mean")))
        if slice_names:
            sub_rows = []
            for _, r in agg.sort_values("rmse_mean").iterrows():
                for s in slice_names:
                    pref = f"slice_{s}"
                    if f"{pref}_rmse_mean" not in r.index:
                        continue
                    sub_rows.append({
                        "config": r["config"], "subgroup": s.replace("_", " "),
                        "n_mean": r.get(f"{pref}_n_mean", float("nan")),
                        "rmse": r.get(f"{pref}_rmse_mean", float("nan")),
                        "mae":  r.get(f"{pref}_mae_mean", float("nan")),
                        "bias": r.get(f"{pref}_bias_mean", float("nan")),
                        "r2":   r.get(f"{pref}_r2_mean", float("nan")),
                        "ccc":  r.get(f"{pref}_ccc_mean", float("nan")),
                        "icc":  r.get(f"{pref}_icc_mean", float("nan")),
                        "cal_slope": r.get(f"{pref}_cal_slope_mean", float("nan")),
                    })
            pd.DataFrame(sub_rows).to_csv("repeated_kfold_subgroup_summary.csv", index=False)
            print("✓ Saved → repeated_kfold_subgroup_summary.csv")

    print(f"\n{'=' * 68}\n  DONE. {len(configs)} configs × {n_iter_total} evals.\n{'=' * 68}")
    return rkf_df, agg


# =============================================================================
# SHAP feature-importance & RMSE parsimony analysis  (added module)
# =============================================================================
# Part 2 of the study.  On top of the augmentation sweep this adds:
#   1. Selection of the TOP-N configs by OVERALL mean RMSE (a config may be
#      with OR without an augmenter, and the same model may appear more than
#      once — e.g. RF/no-aug and RF/GaussianCopula can both make the cut).
#   2. For each selected config: fit the final model on ALL data and produce
#      a SHAP bar plot (mean|SHAP| importance, annotated) and a beeswarm,
#      computed OVERALL and within the Underweight / Obese subgroups.
#   3. A SHAP-ranked forward-selection RMSE "parsimony" curve for the same
#      configs (overall + subgroups), to find the smallest feature set that
#      reaches near-minimal RMSE.
#
# Subgroup handling: within a single-BMI subgroup the BMI dummies are constant,
# so they carry no within-subgroup information; they are dropped from that
# scope's SHAP ranking, bar plot, beeswarm, and parsimony sweep.
#
# Outputs (per selected config, tagged by a short config id):
#   shap_importance_<scope>.csv            mean|SHAP| per feature × config
#   shap_bar_<cfgid>_<scope>.png           annotated importance bar plot
#   shap_summary_<cfgid>_<scope>.png       beeswarm
#   parsimony_rmse.csv                     RMSE vs n-features, every config×scope
#   optimal_feature_sets.csv               parsimonious set per config × scope
#   parsimony_rmse_<scope>.png             top-N parsimony plot per scope
#
# Depends on names already defined earlier in this file:
#   TARGET, CONT_COLS, MODELS, DL_MODELS, RANDOM_STATE, BMI_BASELINE,
#   REWEIGH_PARAMS, load_data, expand_features, get_features, build_model,
#   tune_model, fit_tvae_default, fit_tvae_reweighing, fit_gaussian_copula,
#   sample_synthetic, seed_everything, make_configs, _config_label


# ---- BMI dummies are ranked/selected as one clinical unit ------------------
def _feature_groups(feat_cols):
    """Group the flat feature list into clinical units.

    The BMI one-hot dummies are collapsed into a single 'BMI_Category' group
    so the parsimony curve adds/removes BMI as one decision, not half a
    dummy set.  Every other feature is its own group.
    """
    bmi = [c for c in feat_cols if c.startswith("BMI_Category_")
           and c != "BMI_Category_New"]
    groups = {}
    for c in feat_cols:
        if c not in bmi:
            groups[c] = [c]
    if bmi:
        groups["BMI_Category"] = bmi
    return groups


def _cfg_id(cfg):
    """Short filesystem-safe id for a config, e.g. 'RF_GaussianCopula_r1.0'."""
    aug = cfg.get("augmenter", "NoAug")
    return f"{cfg['model']}_{aug}_r{cfg.get('ratio', 0.0)}".replace(".", "p")


def _cfg_from_summary_row(row):
    """Rebuild a config dict (for train_at_config) from a summary CSV row."""
    cfg = {
        "augmenter": row["augmenter"],
        "model": row["model"],
        "ratio": float(row["ratio"]),
    }
    if row["augmenter"] == "TVAE_reweighing":
        cfg["reweighing_params"] = dict(REWEIGH_PARAMS)  # noqa: F821
    return cfg


# ---- pick the TOP-N configs by overall RMSE (with or without augmentation) --
def choose_top_configs(summary_csv="repeated_kfold_summary.csv",
                       metric="rmse", top_n=5, verbose=True):
    """Return the top-N configs by lowest overall mean `metric`.

    Unlike a per-model rule, this simply ranks every config in the summary and
    keeps the best N — so a single model can appear multiple times and some
    models may not appear at all.  Returns a list of config dicts (ordered
    best -> worst) ready for `train_at_config`.
    """
    summ = pd.read_csv(summary_csv)
    mcol = f"{metric}_mean"
    ranked = summ.sort_values(mcol).head(top_n).reset_index(drop=True)
    configs = [_cfg_from_summary_row(r) for _, r in ranked.iterrows()]
    if verbose:
        print(f">>> Top {top_n} configs by overall {metric.upper()}:")
        for i, (_, r) in enumerate(ranked.iterrows(), 1):
            print(f"  {i}. {r['config']:<45} {mcol}={r[mcol]:.3f}")
    return configs


# ---- train one model at a given config on a train split --------------------
def train_at_config(cfg, train_raw, feat_cols, seed=RANDOM_STATE,
                    tune_iter=30):
    """Fit `cfg`'s model on (real + synthetic) built per its augmenter/ratio.

    Returns a fitted estimator plus the expanded training frame used, so the
    same feature columns can be reused for SHAP.  Mirrors the sweep's
    train side exactly (tune on real-only, augment real-only, then fit).
    """
    model_name = cfg["model"]
    ratio = cfg.get("ratio", 0.0)
    aug_name = cfg.get("augmenter", "NoAug")
    rew = cfg.get("reweighing_params")

    train_real = expand_features(train_raw)              # noqa: F821
    best_params = tune_model(train_real, model_name, feat_cols,   # noqa: F821
                             n_iter=tune_iter, seed=seed)

    if ratio and aug_name != "NoAug":
        if aug_name == "TVAE_default":
            aug = fit_tvae_default(train_raw, seed)                # noqa: F821
        elif aug_name == "TVAE_reweighing":
            aug = fit_tvae_reweighing(train_raw, seed=seed,        # noqa: F821
                                      **(rew or {}))
        elif aug_name == "GaussianCopula":
            aug = fit_gaussian_copula(train_raw, seed)            # noqa: F821
        else:
            raise ValueError(f"Unknown augmenter {aug_name}")
        n_synth = int(round(ratio * len(train_raw)))
        synth_raw = sample_synthetic(aug, n_synth, train_raw)     # noqa: F821
        train_combined_raw = pd.concat([train_raw, synth_raw],
                                       ignore_index=True)
    else:
        train_combined_raw = train_raw.copy()

    train_combined = expand_features(train_combined_raw)          # noqa: F821
    model = build_model(model_name, params=best_params, seed=seed)  # noqa: F821
    X = train_combined[feat_cols].astype(float).values
    y = train_combined[TARGET].astype(float).values              # noqa: F821
    model.fit(X, y)
    return model, train_combined


# ---- variance filter: which features actually vary within a set of rows ----
def _varying_features(X, feat_cols, tol=1e-9):
    """Return the subset of feat_cols with non-zero variance in X.

    Within a single-BMI subgroup (e.g. all-Obese rows) the BMI dummies are
    constant, so they carry no within-subgroup information and are removed
    from that scope's SHAP ranking / plots.  Also catches any other
    accidentally-constant column (e.g. a subgroup that is single-sex).
    """
    X = np.asarray(X, dtype=float)
    var = X.var(axis=0)
    return [f for f, v in zip(feat_cols, var) if v > tol]


# ---- SHAP values (model-agnostic, works for ML + DL) -----------------------
def compute_shap(model, X_background, X_explain, feat_cols,
                 model_name, max_bg=100, max_explain=300, seed=RANDOM_STATE):
    """Return (shap_values, X_explain_used) tuple, or (None, None) on failure.

    Tree models use the fast exact TreeExplainer.  Everything else (SVR, MLP,
    FT-Transformer) uses a Permutation explainer on a sampled background,
    which is more robust than KernelExplainer on small/constant-column data
    and does not hang on the DL models.  Any explainer failure is caught and
    returns (None, None) so one config can't abort the whole run.

    Returns the ACTUAL X_explain used (may be downsampled if > max_explain), so
    shap_values and X always have matching row counts for plotting.

    `X_background` should be the SAME scope being explained (e.g. the Obese
    rows when explaining Obese), so constant features get ~zero attribution.
    """
    if shap is None:
        raise ImportError(
            "SHAP is required for run_shap_parsimony(): pip install shap"
        )
    X_background = np.asarray(X_background, dtype=float)
    X_explain = np.asarray(X_explain, dtype=float)

    X_explain_used = X_explain
    if len(X_explain) > max_explain:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_explain), max_explain, replace=False)
        X_explain_used = X_explain[idx]

    try:
        if model_name in ("RF", "XGBoost"):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_explain_used)
            return np.asarray(sv), X_explain_used

        # generic robust path (SVR, MLP, FTTransformer)
        n_bg = max(min(max_bg, len(X_background)), 1)
        bg = shap.sample(X_background, n_bg, random_state=seed) \
            if len(X_background) > n_bg else X_background
        predict = (lambda d: np.asarray(model.predict(d)).ravel())
        try:
            explainer = shap.PermutationExplainer(predict, bg, seed=seed)
            sv = explainer(X_explain_used, silent=True).values
        except Exception:
            bg_k = shap.kmeans(X_background, min(max_bg, len(X_background)))
            explainer = shap.KernelExplainer(predict, bg_k, seed=seed)
            sv = explainer.shap_values(X_explain_used, nsamples="auto",
                                       silent=True)
        return np.asarray(sv), X_explain_used
    except Exception as e:                                    # pragma: no cover
        print(f"    [SHAP] {model_name} explainer failed ({type(e).__name__}: "
              f"{e}) — SHAP skipped for this scope")
        return None, None


def _grouped_importance(shap_matrix, feat_cols, groups, keep_feats=None):
    """Mean|SHAP| aggregated to clinical groups (BMI dummies summed).

    If `keep_feats` is given, only those features contribute — constant
    (dropped) features are excluded, and any group with no surviving member
    is omitted entirely.
    """
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    per_feat = dict(zip(feat_cols, mean_abs))
    out = {}
    for gname, members in groups.items():
        live = [m for m in members
                if keep_feats is None or m in keep_feats]
        if not live:
            continue
        out[gname] = float(sum(per_feat.get(m, 0.0) for m in live))
    return out


def _plot_shap_bar(imp_dict, title, out_png):
    """Annotated horizontal bar plot of mean|SHAP| importance (grouped)."""
    items = sorted(imp_dict.items(), key=lambda kv: kv[1])   # low -> high
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    plt.figure(figsize=(7, max(3, 0.45 * len(names) + 1)))
    bars = plt.barh(names, vals, color="#4C72B0")
    vmax = max(vals) if vals else 1.0
    for b, v in zip(bars, vals):
        plt.text(b.get_width() + vmax * 0.01,
                 b.get_y() + b.get_height() / 2,
                 f"{v:.2f}", va="center", ha="left", fontsize=9)
    plt.xlabel("mean(|SHAP value|)  —  average impact on REE prediction")
    plt.title(title)
    plt.xlim(0, vmax * 1.15)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ---- RMSE parsimony via SHAP-ranked forward selection ----------------------
def parsimony_curve(cfg, real_data, ranked_groups, groups,
                    scope_mask_fn=None, n_repeats=5, n_folds=5,
                    tune_iter=15, seed=RANDOM_STATE):
    """Forward-select feature GROUPS in SHAP-importance order and record
    test RMSE at each step, via repeated stratified k-fold CV.

    `ranked_groups`  : group names ordered most->least important.
    `groups`         : {group: [member columns]} mapping.
    `scope_mask_fn`  : optional fn(test_expanded_df)->bool mask, to score RMSE
                       only on a subgroup (e.g. Underweight) of each test fold.
    Returns a DataFrame: n_features, RMSE_mean, RMSE_std for this scope.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import mean_squared_error

    ree_q = pd.qcut(real_data[TARGET], q=4, labels=False,   # noqa: F821
                    duplicates="drop")
    rows = []

    for k in range(1, len(ranked_groups) + 1):
        active_groups = ranked_groups[:k]
        active_cols = [c for g in active_groups for c in groups[g]]
        fold_rmses = []
        for repeat in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                  random_state=repeat)
            for f_i, (tr, te) in enumerate(skf.split(real_data, ree_q)):
                it_seed = repeat * n_folds + f_i
                tr_raw = real_data.iloc[tr].reset_index(drop=True)
                te_raw = real_data.iloc[te].reset_index(drop=True)
                model, _ = train_at_config(
                    cfg, tr_raw, active_cols, seed=it_seed, tune_iter=tune_iter)
                te_x = expand_features(te_raw)               # noqa: F821
                for col in active_cols:
                    if col not in te_x.columns:
                        te_x[col] = 0
                Xte = te_x[active_cols].astype(float).values
                yte = te_x[TARGET].astype(float).values      # noqa: F821
                pred = model.predict(Xte)
                if scope_mask_fn is not None:
                    m = scope_mask_fn(te_x)
                    if m.sum() < 3:
                        continue
                    yte, pred = yte[m], pred[m]
                fold_rmses.append(float(np.sqrt(mean_squared_error(yte, pred))))
        if fold_rmses:
            rmse_mean = round(float(np.mean(fold_rmses)), 4)
            rmse_std = round(float(np.std(fold_rmses)), 4)
        else:
            rmse_mean, rmse_std = float("nan"), float("nan")
        rows.append({
            "n_features": k,
            "added_group": ranked_groups[k - 1],
            "features": "|".join(active_groups),
            "rmse_mean": rmse_mean,
            "rmse_std":  rmse_std,
        })
    return pd.DataFrame(rows)


def _elbow(parsimony_df, tol_frac=0.02):
    """Smallest n_features whose RMSE is within `tol_frac` of the best RMSE
    on the curve (default within 2 %).  NaN-safe."""
    df = parsimony_df.dropna(subset=["rmse_mean"])
    if df.empty:
        return int(parsimony_df["n_features"].iloc[-1])
    best = df["rmse_mean"].min()
    thresh = best * (1 + tol_frac)
    ok = df[df["rmse_mean"] <= thresh]
    return int(ok["n_features"].min()) if len(ok) else \
        int(df["n_features"].iloc[-1])


# ---- top-level driver ------------------------------------------------------
def run_shap_parsimony(real_data=None, input_csv=None,
                       summary_csv="repeated_kfold_summary.csv",
                       within_csv="repeated_kfold_within_model_tests.csv",
                       slice_col="BMI_Category_New",
                       subgroups=("Underweight", "Obese"),
                       top_n=5, n_repeats=5, n_folds=5, tune_iter=15,
                       shap_seed=RANDOM_STATE):
    """Part 2: SHAP (bar + beeswarm) and RMSE-parsimony for the TOP-N configs
    by overall RMSE, computed overall + within Underweight / Obese.

    `within_csv` is accepted for backward compatibility but is not required
    for the top-N selection (kept so older call sites don't break).
    """
    seed_everything(shap_seed)                               # noqa: F821
    if real_data is None:
        real_data = load_data(input_csv) if input_csv else load_data()  # noqa: F821
    print(f">>> SHAP/parsimony on {len(real_data)} rows")

    # top-N configs by overall RMSE (with or without augmenter)
    configs = choose_top_configs(summary_csv, metric="rmse", top_n=top_n)

    full_expanded = expand_features(real_data)               # noqa: F821
    feat_cols = get_features(full_expanded)                  # noqa: F821
    groups = _feature_groups(feat_cols)
    print(f"    feature groups ({len(groups)}): {list(groups.keys())}")

    scopes = {"overall": None}
    for sg in subgroups:
        scopes[sg] = (lambda df, v=sg: (df[slice_col].astype(str)
                                        .str.strip().values == v))

    # importance tables keyed by config label (a model may repeat)
    importance_tables = {s: {} for s in scopes}
    parsimony_frames = {s: {} for s in scopes}

    for cfg in configs:
        label = _config_label(cfg)                           # noqa: F821
        cid = _cfg_id(cfg)
        model_name = cfg["model"]
        print(f"\n>>> {label}")

        model, _ = train_at_config(cfg, real_data, feat_cols,
                                   seed=shap_seed, tune_iter=tune_iter)
        X_all = full_expanded[feat_cols].astype(float).values

        for scope, mask_fn in scopes.items():
            if mask_fn is None:
                X_exp = X_all
                X_bg = X_all
            else:
                m = mask_fn(full_expanded)
                if m.sum() < 5:
                    print(f"    [{scope}] too few rows ({m.sum()}) — skipped")
                    continue
                X_exp = X_all[m]
                X_bg = X_all[m]                  # subgroup: explain vs itself

            keep = _varying_features(X_exp, feat_cols)
            keep_idx = [feat_cols.index(f) for f in keep]

            sv, X_used = compute_shap(model, X_bg, X_exp, feat_cols,
                                      model_name, seed=shap_seed)
            if sv is None:
                continue
            imp = _grouped_importance(sv, feat_cols, groups, keep_feats=keep)
            importance_tables[scope][label] = imp

            # (1) annotated bar plot of grouped importance
            try:
                _plot_shap_bar(
                    imp, f"SHAP importance — {label} ({scope})",
                    f"shap_bar_{cid}_{scope}.png")
            except Exception as e:                            # pragma: no cover
                plt.close("all")
                print(f"    [{scope}] bar plot skipped: {e}")

            # (2) beeswarm — only varying features (no constant dummies)
            try:
                plt.figure()
                shap.summary_plot(sv[:, keep_idx], X_used[:, keep_idx],
                                  feature_names=keep, show=False,
                                  plot_size=(7, 4))
                plt.title(f"SHAP beeswarm — {label} ({scope})")
                plt.tight_layout()
                plt.savefig(f"shap_summary_{cid}_{scope}.png", dpi=150)
                plt.close()
            except Exception as e:                            # pragma: no cover
                plt.close("all")
                print(f"    [{scope}] beeswarm skipped: {e}")

            # parsimony curve ranked by this scope's SHAP importance
            ranked = sorted(imp, key=imp.get, reverse=True)
            if not ranked:
                print(f"    [{scope}] no varying features — parsimony skipped")
                continue
            pc = parsimony_curve(
                cfg, real_data, ranked, groups,
                scope_mask_fn=mask_fn, n_repeats=n_repeats,
                n_folds=n_folds, tune_iter=tune_iter, seed=shap_seed)
            pc.insert(0, "config", label)
            pc.insert(1, "scope", scope)
            pc["elbow_n"] = _elbow(pc)
            parsimony_frames[scope][label] = pc

    # ---- save importance CSVs (one row per config, cols = features) ----
    for scope in scopes:
        if not importance_tables[scope]:
            continue
        imp_df = pd.DataFrame(importance_tables[scope]).T
        imp_df.index.name = "config"
        imp_df = imp_df[sorted(imp_df.columns,
                               key=lambda c: -imp_df[c].mean())]
        imp_df.round(4).to_csv(f"shap_importance_{scope}.csv")
        print(f"✓ Saved → shap_importance_{scope}.csv")

    # ---- save parsimony CSV + parsimonious sets ----
    all_pc, optimal_sets = [], []
    for scope in scopes:
        for label, pc in parsimony_frames[scope].items():
            all_pc.append(pc)
            elbow = int(pc["elbow_n"].iloc[0])
            chosen = pc[pc["n_features"] == elbow].iloc[0]
            optimal_sets.append({
                "config": label, "scope": scope,
                "n_features": elbow,
                "features": chosen["features"],
                "rmse_at_elbow": chosen["rmse_mean"],
                "rmse_full": pc["rmse_mean"].iloc[-1],
            })
    if all_pc:
        pd.concat(all_pc, ignore_index=True).to_csv(
            "parsimony_rmse.csv", index=False)
        print("✓ Saved → parsimony_rmse.csv")
        pd.DataFrame(optimal_sets).to_csv(
            "optimal_feature_sets.csv", index=False)
        print("✓ Saved → optimal_feature_sets.csv")

    # ---- parsimony figures: one per scope, one line per top-N config ----
    for scope in scopes:
        frames = parsimony_frames[scope]
        if not frames:
            continue
        plt.figure(figsize=(8.5, 5))
        for label, pc in frames.items():
            plt.plot(pc["n_features"], pc["rmse_mean"], marker="o", label=label)
            plt.fill_between(pc["n_features"],
                             pc["rmse_mean"] - pc["rmse_std"],
                             pc["rmse_mean"] + pc["rmse_std"], alpha=0.10)
            e = int(pc["elbow_n"].iloc[0])
            sel = pc[pc["n_features"] == e]
            if len(sel) and pd.notna(sel["rmse_mean"].iloc[0]):
                plt.scatter([e], [float(sel["rmse_mean"].iloc[0])], s=140,
                            facecolors="none", edgecolors="black",
                            linewidths=1.5, zorder=5)
        plt.xlabel("Number of features (added in SHAP-importance order)")
        plt.ylabel("Test RMSE (repeated 5×5-fold CV)")
        plt.title(f"RMSE parsimony — top {top_n} configs — {scope}  "
                  f"(○ = parsimonious set, within 2% of best)")
        plt.legend(title="Config", frameon=False, fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"parsimony_rmse_{scope}.png", dpi=150)
        plt.close()
        print(f"✓ Saved → parsimony_rmse_{scope}.png")

    print("\n=== SHAP/parsimony complete ===")
    return importance_tables, parsimony_frames, configs


if __name__ == "__main__":
    import sys
    if "--shap" in sys.argv:
        run_shap_parsimony()
    elif "--all" in sys.argv:
        main()
        run_shap_parsimony()
    else:
        main()
