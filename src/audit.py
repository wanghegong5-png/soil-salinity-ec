"""Stage 1: label audit for Soil_EC_sampling_points.csv.

Outputs:
  reports/audit_summary.md
  reports/figures/*.png
  data/processed/samples_clean.csv  (adds log_EC and location_group columns)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Soil_EC_sampling_points.csv"
OUT = ROOT / "data" / "processed"
REP = ROOT / "reports"
FIG = REP / "figures"
FEATURES = ["TIR", "SIT", "PDI"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW)
    lines = ["# 数据审计报告", "", f"源文件：`{RAW.relative_to(ROOT)}`，{len(df)} 行", ""]

    # 1. schema / missing
    lines += ["## 1. 字段与缺失", "", "| 列 | 类型 | 缺失 |", "|---|---|---|"]
    for c in df.columns:
        lines.append(f"| {c} | {df[c].dtype} | {df[c].isna().sum()} |")
    lines.append("")
    lines.append(f"ID 唯一：{df['ID'].is_unique}；完全重复行：{df.duplicated().sum()}")
    lines.append("")

    # 2. by municipality
    lines += ["## 2. 地区分布", "", df["Municipal"].value_counts().to_frame("n").to_markdown(), ""]

    # 3. coordinates
    lines += ["## 3. 坐标范围", "",
              df[["Lon", "Lat"]].describe().round(4).to_markdown(), ""]

    # 4. duplicate locations -> group id
    df["location_group"] = df.groupby(["Lon", "Lat"]).ngroup()
    dup = df[df.duplicated(["Lon", "Lat"], keep=False)].sort_values(["Lon", "Lat"])
    lines += ["## 4. 重复坐标", "",
              f"唯一坐标数：{df['location_group'].nunique()}；涉及重复的记录：{len(dup)}", ""]
    if len(dup):
        lines += [dup[["ID", "Lon", "Lat", "Municipal", "EC"] + FEATURES].to_markdown(index=False), ""]
    lines += ["处理：保留全部记录，但以 `location_group` 分组，保证同一坐标不会同时进入训练和测试集。", ""]

    # 5. target distribution
    ec = df["EC"]
    lines += ["## 5. 目标变量 EC", "",
              df.groupby("Municipal")["EC"].describe().round(3).to_markdown(), "",
              f"整体偏度：{ec.skew():.2f}；log1p 后偏度：{np.log1p(ec).skew():.2f}；"
              f"非正值数：{(ec <= 0).sum()}", ""]
    df["log_EC"] = np.log1p(ec)

    # 6. features
    lines += ["## 6. 特征", "", df[FEATURES].describe().round(3).to_markdown(), "",
              "### 与 EC 的相关系数（Spearman）", "",
              df[FEATURES + ["EC"]].corr(method="spearman")["EC"].drop("EC").round(3).to_frame("rho").to_markdown(),
              "", "### 特征间相关（Pearson）", "",
              df[FEATURES].corr().round(3).to_markdown(), ""]

    # 7. outliers (IQR rule, report only)
    lines += ["## 7. 离群值（IQR×3，仅报告不删除）", "", "| 列 | 数量 |", "|---|---|"]
    for c in FEATURES + ["EC"]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        n = ((df[c] < q1 - 3 * (q3 - q1)) | (df[c] > q3 + 3 * (q3 - q1))).sum()
        lines.append(f"| {c} | {n} |")
    lines += ["", "## 8. 已知缺口", "",
              "- 无采样日期/年份、采样深度列 → 暂不能自行重新匹配卫星影像。",
              "- TIR / SIT / PDI 的具体定义与单位需对照原论文确认。",
              "- 仅覆盖吉林白城、松原，不代表黑龙江/内蒙古。", ""]

    # figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(ec, bins=50); ax[0].set_title("EC"); ax[0].set_xlabel("EC")
    ax[1].hist(df["log_EC"], bins=50); ax[1].set_title("log1p(EC)")
    fig.tight_layout(); fig.savefig(FIG / "ec_distribution.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, g in df.groupby("Municipal"):
        sc = ax.scatter(g["Lon"], g["Lat"], c=g["log_EC"], s=8, cmap="viridis",
                        vmin=df["log_EC"].min(), vmax=df["log_EC"].max(), label=name)
    fig.colorbar(sc, ax=ax, label="log1p(EC)")
    ax.set_xlabel("Lon"); ax.set_ylabel("Lat"); ax.set_title("Sample locations")
    fig.tight_layout(); fig.savefig(FIG / "sample_map.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, c in zip(ax, FEATURES):
        a.scatter(df[c], df["log_EC"], s=5, alpha=0.5); a.set_xlabel(c); a.set_ylabel("log1p(EC)")
    fig.tight_layout(); fig.savefig(FIG / "feature_vs_ec.png", dpi=120); plt.close(fig)

    lines += ["## 图", "", "![](figures/ec_distribution.png)", "", "![](figures/sample_map.png)", "",
              "![](figures/feature_vs_ec.png)", ""]

    df.to_csv(OUT / "samples_clean.csv", index=False)
    (REP / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
