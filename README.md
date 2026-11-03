# 松嫩平原西部土壤电导率（EC）遥感估计：空间验证基线

用公开实测样本（吉林白城、松原，942 点，EC1:5）检验遥感衍生特征预测土壤盐分的能力，**重点比较随机划分与空间划分下的泛化差异**。

数据来源：Wang et al., 2025, *Scientific Data*，[论文](https://www.nature.com/articles/s41597-025-06057-7) / [Zenodo](https://zenodo.org/records/17044261)（MD5 `9c2e663b52c0631ecdb35c5c46d33d2e`）。

## 运行

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python src\audit.py      # -> reports/audit_summary.md
.\.venv\Scripts\python src\baseline.py   # -> reports/baseline_results.md
```

## 目前结论（阶段 2，仅用作者提供的 TIR / SIT / PDI 三个特征）

| 验证方式 | 最佳模型 R² | 均值基线 R² |
|---|---|---|
| 随机 5 折 | 0.23（随机森林） | −0.03 |
| 0.1° 空间分块 5 折 | 0.05（线性） | −0.04 |
| 跨市留出（白城 ↔ 松原） | < 0 | < 0 |

- 随机划分明显高估性能；空间独立验证下，三个特征几乎不优于均值预测。
- 高盐样本（EC > 1.1）被系统性低估，误差集中在高值段。
- 结论仅适用于这三个特征；论文报告的 R²=0.467 使用了更完整的建模设置，下一步需补充特征再比较。

## 结构

```
data/raw/            原始 CSV（未修改）
src/audit.py         数据审计：缺失、重复坐标、分布、相关性、离群值
src/baseline.py      基线模型 × 三种验证方案、分段误差
reports/             自动生成的报告与图表
```
