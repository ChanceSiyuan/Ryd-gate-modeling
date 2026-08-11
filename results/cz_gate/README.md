# cz_gate — 两族 CZ 脉冲各自能做到的最好保真度

## 问题与模型

同一个双原子 CZ 目标，两条互不相同的脉冲路线，各自优化到底能到哪：

- **AR（绝热回归，adiabatic return）** —— 缓变包络把两原子绝热地带过 Rydberg 再带回，
  纠缠相位由阻塞下的能级排斥累积。对 `our`（本项目原子参数）与 `lukin`（文献对照参数）
  两套参数各跑一遍，回答的是"参数集本身值不值"。
- **TO（时间最优，time-optimal）** —— 固定形状族内直接搜索最短路径。
  对 `mp` 与 `pm` **两个超精细流形**分别标定（间距 3.0 µm），因为两者的中间态失谐结构不同，
  补偿差分 Stark 的参数**不能互相搬用**。

TO 的搜索坐标是 6 维 `x`，其中 `theta` 为条件相位，`amps` 为 4×4 的分段振幅表；
产物同时给两种容差下的相干不保真度，这一点在下面很关键。

## 核心结果

### AR：`our` 参数集完胜，且 `lukin` 那条根本没跑完

| | `our` | `lukin` |
|---|---|---|
| `best_infidelity` | **$9.46\times10^{-6}$** | $2.14\times10^{-4}$ |
| 评估次数 `n_eval` | 3674（3 个起点） | 303 |
| `done` | **true** | **false** |
| 耗时 `elapsed_s` | 44 028 | **124 432** |

两件事要分开说：`lukin` 差 23 倍是**表面**结论；真正的结论是它在**三倍于 `our` 的算力**下
只完成了 303 次评估就停在 `done: false`。所以 $2.14\times10^{-4}$ 只是**该参数集的上界**
（即"至少能做到这么好"），不是它的最优值 —— 引用时必须带上这一条。

`ar_opt_our_search_best.json` 不是结果：它记录的 $0.309$ 来自粗搜阶段的
`ar_opt_our_w24.json`，只用于指向源文件。9 个 `ar_opt_our_w{N}.json` 是逐窗口的粗搜记录。

### TO：两个流形相差 5.5 倍，且积分器偏差对 mp 不可忽略

间距均为 3.0 µm：

| 流形 | `theta`（条件相位, rad） | `coherent_infidelity`（生产容差） | `tight_tol_infidelity`（严容差） | 差值 | `elapsed_s` |
|---|---|---|---|---|---|
| `mp` | $-3.0527$ | $2.122\times10^{-5}$ | **$1.398\times10^{-5}$** | $7.2\times10^{-6}$ | 3 225 |
| `pm` | $-2.5959$ | $8.534\times10^{-5}$ | **$7.674\times10^{-5}$** | $8.6\times10^{-6}$ | 1 202 |

- **要引用的是 `tight_tol_infidelity`**。两者之差（$\sim10^{-5}$）是积分器自身的偏差，
  对 pm 只占 11%，但对 mp 占了 **34%** —— 在 $10^{-5}$ 量级上，用生产容差的数字会
  系统性高估不保真度三成。
- **mp 比 pm 好 5.5 倍**，且条件相位差了 0.46 rad，两个流形不是彼此的重标度。
- `traces_{mp,pm}.npz` 存了对应的布居轨迹（`t(200,)`、`pops(4, 5, 200)` = 4 逻辑输入 ×
  5 能级 × 200 时刻），可直接查"泄漏发生在哪一段"。

## 图

本目录**无图文件**。图是 `scripts/notebooks/01_cz_gate.ipynb` 的渲染输出（已随 notebook 提交）。

## 数据与溯源

| 路径 | 内容 |
|---|---|
| `ar_optimization/ar_opt_{our,lukin}.json` | 优化器状态：`param_set`、`n_eval`、`best_infidelity`、`done`、`elapsed_s` |
| `ar_optimization/ar_opt_our_w{N}.json` | 9 份逐窗口粗搜记录 |
| `ar_optimization/ar_opt_our_search_best.json` | 指向最优粗搜窗口的指针，非结果 |
| `to_calibration/to_{mp,pm}.json` | 11 键：`manifold`、`spacing_um`、`x(6)`、`theta`、`amps`、两种容差的不保真度、`elapsed_s`、`done` |
| `to_calibration/traces_{mp,pm}.npz` | 7 键：`t(200,)`、`pops(4,5,200)`、`amps(4,4)`、`x(6,)`、`fixed(4,)`、`spacing_um`、`rtol` |
| `error_characterization/` | 见下方告示 |

### 告示：`error_characterization/` 是重写前的旧架构产物

仓库里**没有任何代码引用**这个子目录，且 6 个 `mc_*.txt` 首行自述
`# MonteCarloResult saved 2026-02-11` / `2026-02-12`。它们是在 2026-07-15
由 `eebce3e` 提交进来的 —— 正是 `src/ryd_gate` 重写当天。也就是说这是**二月份、
旧架构衰减通道契约下**的结果被顺手带了进来。

不要把这里的数字当作当前结论引用，除非重新导出。本目录其余部分
（`ar_optimization/`、`to_calibration/`）都在重写之后，不受影响。

## 复现

```bash
# 出图与表格（读缓存的 JSON/NPZ，不跑优化器）
uv run jupyter nbconvert --to notebook --execute --inplace \
    scripts/notebooks/01_cz_gate.ipynb
```

重新导出 `ar_optimization/` 或 `to_calibration/` 需要在 notebook 内重跑优化 cell，
按上表 `elapsed_s`，各是**数万 CPU 秒**量级。

**产出** `scripts/notebooks/01_cz_gate.ipynb`（末次提交 `6e43959`，2026-07-20，
比数据 2026-07-16…18 晚两天）。
