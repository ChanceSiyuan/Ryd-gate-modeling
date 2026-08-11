# cz_grape_e2e — `qoc` 离散伴随 GRAPE 接缝的端到端可信度审计

## 问题与模型

本仓库所有脉冲优化研究都依赖 `qoc.grape` 提供的解析梯度。这份 artifact 回答的是一个
前置问题：**它给出的数字能不能信**。

被审计的对象是一次双原子 `01r` CZ 优化（$T = 2$ µs，间距 4.0 µm，70S，
$C_6 = 5.4207\times10^{12}$ rad/s·µm⁶，$V_{rr} = 1.3234\times10^{9}$ rad/s），
8+8+1 系数的 B 样条脉冲，目标函数

$$J = \overline{L} + w_\phi\,(\text{相位偏差})^2 + w_F\,\text{fluence},
\qquad w_\phi = 0.6,\;\; w_F = 10^{-4}$$

验收 $L_{\max}\le 10^{-4}$、相位误差 $\le 0.05$ rad。搜索用 2048 个传播片，
L-BFGS-B（maxiter 200 / maxfun 400），8 个随机起点，`rng_seed = 7`。

一个 $1-F\approx 8\times10^{-8}$ 的声称，只有在**每一条误差来源都比它小一个数量级以上**时
才成立。于是设四道闸，全部记录在 `acceptance` 里：

| 闸 | 问的是什么 | 判据 |
|---|---|---|
| 梯度 | 解析伴随梯度是否等于有限差分 | 相对误差 $\ll 1$ |
| 多起点 | 解是否只在精心挑的起点才找得到 | 至少一个随机起点可行 |
| 网格收敛 | 离散化误差是否小于声称值 | 最后一次加倍的位移 $<$ 声称值的 10% |
| 模型一致 | 搜索模型是否与公开 `exact_ode` 一致 | 差距 $\ll$ 声称值 |

## 核心结果

**四道闸全过**（`acceptance.all_passed = true`），声称 $1-F = 8.214\times10^{-8}$，
10% 容差带 $8.214\times10^{-9}$：

| 闸 | 实测 | 相对声称值 |
|---|---|---|
| 梯度最坏相对误差 | $1.126\times10^{-9}$（梯度范数 17.96，2048 片） | 1.4% |
| 网格收敛（最后一次加倍） | $\Delta J_{2N} = 1.135\times10^{-10}$ | **0.14%** |
| 搜索模型 vs `exact_ode` | $1.270\times10^{-9}$ | 1.5% |
| 多起点 | 8 个随机起点中最优为 `random_06` | 可行 |

网格收敛的完整阶梯（`grid_convergence.rows`）显示目标值在 $N\ge 256$ 后就稳定在
$J\approx 1.9448\times10^{-5}$：

| $N$ | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| $\Delta J_{2N}$ | 2.65e-5 | 5.31e-7 | 2.78e-8 | 3.20e-10 | 3.56e-10 | **1.14e-10** |

也就是说：**从 $N=64$ 到 $N=256$ 就已经把离散化误差压到声称值以下**，生产用的 2048 片
有两个数量级的余量。三道数值闸各自只占声称值的 1.4% / 0.14% / 1.5%，加起来仍远小于 10%
的容差带 —— 这才是"8e-8 可信"的实际含义。

总耗时 `wall_time_s = 3316.9`（约 55 min）。

## 图

本目录**无图文件**。图是 `scripts/notebooks/07_cz_grape_e2e_validation.ipynb` 的渲染输出
（已随 notebook 提交）。

## 数据与复现

| 路径 | 内容 |
|---|---|
| `validation.json` | `schema_version` 2、`kind`、`config`、`gradient_check`、`multistart`、`best`、`grid_convergence`、`acceptance`、`wall_time_s` |

`config` 钉死了全部设定：`duration_s` 2e-6、`ryd_level` 70、`spacing_um` 4.0、
`C6_rad_s_um6`、`V_RR_rad_s`、样条 `basis_n_coeffs` 8 / `basis_degree` 3、
`w_phase` / `w_fluence`、`n_search_slices` 2048、`acceptance`、
`exact_options`（dense / rtol 1e-10 / atol 1e-12）、`optimizer`、以及假定的斜率上限
`slew_caps_assumed_rad_s2`。

```bash
uv run python scripts/cz_grape_e2e_validation.py          # ~55 min
# 只出图（回放 validation.json）
uv run jupyter nbconvert --to notebook --execute --inplace \
    scripts/notebooks/07_cz_grape_e2e_validation.ipynb
```

**产出脚本** `scripts/cz_grape_e2e_validation.py`（末次提交 `059fb1c`，2026-07-24，
比 artifact 晚一天）；notebook 07 只回放绘图。设计
`docs/designs/2026-07-28-direct-qoc-zxz-design.md` 中引用本审计作为 qoc 接缝的依据。
