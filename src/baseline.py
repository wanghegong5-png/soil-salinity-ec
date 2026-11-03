"""Stage 2: baseline EC regression with spatial validation.

Validation schemes
  A. Spatial block CV: ~0.1 deg grid blocks -> GroupKFold (5 folds)
  B. Leave-one-municipality-out: train Baicheng -> test Songyuan, and reverse
  (Random KFold is also reported, only to show how optimistic it is.)

Target: log1p(EC); metrics reported back on original EC scale.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "samples_clean.csv"
REP = ROOT / "reports"
FIG = REP / "figures"
FEATURES = ["TIR", "SIT", "PDI"]
BLOCK_DEG = 0.1
SEED = 42


def models():
    return {
        "Mean (baseline)": DummyRegressor(strategy="mean"),
        "Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                              random_state=SEED, n_jobs=-1),
        "GradBoost": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                   max_iter=300, random_state=SEED),
    }


def metrics(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
            "R2": r2_score(y_true, y_pred)}


def cv_predict(model, X, y, splits):
    pred = np.full(len(y), np.nan)
    for tr, te in splits:
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    return pred


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    X = df[FEATURES].to_numpy()
    y_log = df["log_EC"].to_numpy()
    ec = df["EC"].to_numpy()

    # spatial blocks; identical coordinates always fall in the same block, so no leakage
    bx = np.floor(df["Lon"] / BLOCK_DEG).astype(int)
    by = np.floor(df["Lat"] / BLOCK_DEG).astype(int)
    block = pd.factorize(bx.astype(str) + "_" + by.astype(str))[0]
    n_blocks = len(np.unique(block))

    schemes = {
        "Random KFold (optimistic)": list(KFold(5, shuffle=True, random_state=SEED).split(X)),
        f"Spatial block CV ({BLOCK_DEG} deg, {n_blocks} blocks)": list(GroupKFold(5).split(X, groups=block)),
    }

    rows, preds = [], {}
    for sname, splits in schemes.items():
        for mname, m in models().items():
            p = np.expm1(cv_predict(m, X, y_log, splits))
            preds[(sname, mname)] = p
            rows.append({"Scheme": sname, "Model": mname, **metrics(ec, p)})

    # leave-one-municipality-out
    for test_city in df["Municipal"].unique():
        tr = (df["Municipal"] != test_city).to_numpy(); te = ~tr
        for mname, m in models().items():
            m.fit(X[tr], y_log[tr])
            p = np.expm1(m.predict(X[te]))
            rows.append({"Scheme": f"Train others -> test {test_city} (n={te.sum()})",
                         "Model": mname, **metrics(ec[te], p)})

    res = pd.DataFrame(rows)
    res[["MAE", "RMSE", "R2"]] = res[["MAE", "RMSE", "R2"]].round(3)

    # error by EC range (spatial CV, best non-dummy model by R2)
    sp = [k for k in schemes if k.startswith("Spatial")][0]
    sp_res = res[(res["Scheme"] == sp) & (res["Model"] != "Mean (baseline)")]
    best = sp_res.sort_values("R2", ascending=False).iloc[0]["Model"]
    p_best = preds[(sp, best)]
    bins = pd.qcut(ec, 4)
    seg = (pd.DataFrame({"bin": bins, "err": np.abs(p_best - ec), "bias": p_best - ec})
           .groupby("bin", observed=True).agg(n=("err", "size"), MAE=("err", "mean"), Bias=("bias", "mean"))
           .round(3))

    # plots
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ec, p_best, s=6, alpha=0.5)
    lim = [0, max(ec.max(), np.nanmax(p_best))]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Observed EC"); ax.set_ylabel("Predicted EC")
    ax.set_title(f"{best} - spatial block CV")
    fig.tight_layout(); fig.savefig(FIG / "pred_vs_obs.png", dpi=120); plt.close(fig)

    piv = res.pivot_table(index="Scheme", columns="Model", values="R2")
    fig, ax = plt.subplots(figsize=(9, 4))
    piv.plot.barh(ax=ax); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("R2 (original EC scale)"); ax.set_title("Model x validation scheme")
    fig.tight_layout(); fig.savefig(FIG / "r2_by_scheme.png", dpi=120); plt.close(fig)

    lines = ["# 基线模型结果", "",
             f"特征：{', '.join(FEATURES)}；目标：log1p(EC) 训练，指标在 EC 原尺度计算；样本 {len(df)}。", "",
             "## 各验证方案", "", res.to_markdown(index=False), "",
             f"## 分段误差（{sp}，{best}，按 EC 四分位）", "", seg.to_markdown(), "",
             "## 图", "", "![](figures/r2_by_scheme.png)", "", "![](figures/pred_vs_obs.png)", ""]
    (REP / "baseline_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
