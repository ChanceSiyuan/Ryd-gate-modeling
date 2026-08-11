# max_leakage_297 — 297 nm 单光子 CZ 的相干泄漏图族（五个晶格间距）

## 问题与模型

[`../max_leakage_ode`](../max_leakage_ode/) 的单光子对照组。用 297 nm 直接驱动
$|1\rangle\to|r\rangle$，**没有中间态**，于是双光子路线里占主导的那条误差通道
（中间态散射 $\propto (\Omega/2\Delta_e)^2$）根本不存在。问题因此变成：
剩下的是什么，以及要多大激光功率。

对每个晶格间距扫一张 **8 × 9 的 $n_{\rm ryd}\times t_{\rm gate}$ 图**（注意：
两族的横轴不同 —— 双光子那族扫的是 $\Delta_e$，这里没有中间态可扫，改扫主量子数）。
每个格点内部再扫两条锚点轴（$\log_{10}$ 插值）：

- 297 nm Rabi `omega297_anchors_mhz` = 9, 12, 15, 18 MHz
- 失谐扫描幅度 `dsweep_anchors_mhz` = 2, 10, 20, 30 MHz，硬件上限
  `dsweep_hw_limit_mhz = 20.0`

固定物理：磁场 **20.0 G**、`n_eval_trajectory` 301、`atol_production` $10^{-12}$、
`atol_audit` $10^{-13}$。求解器同为分块最大 DOP853（逐 (格点, 逻辑输入) 块强制容差）。

## 核心结果

### 这是 `results/` 里 schema 最整齐的存储

五个子库 `a3.0 / a4.0 / a5.0 / a7.0 / a10.0` **逐类文件键数完全一致**
（`a3.0` 另有一个 `filter/` 序列，只在该子库存在，见下）：

| 文件族 | 键数 | 双光子族对照 |
|---|---|---|
| `manifest.json` | 13 | 16 |
| `chunks/chunk_*.npz` | 38 | 40 |
| `scatter/scatter_*.npz` | 28 | 30 |
| `trajectories/traj_*.npz` | 11 | 11 |
| `reports/verification.json` | 8 | 10 |

比双光子族少的那几个键正是中间态相关量 —— **schema 的差异本身就是"没有中间态"这件事的
直接体现**，不是疏漏。

### 求解器接缝逐位一致，且无任何失败

| 检验 | 全部五个子库 |
|---|---|
| `hamiltonian_equivalence_rel_dev` | **0.0** |
| `error_norm_max_dev` | $4.89\times10^{-16}$ |
| `error_norm_verified` / `swap_symmetric` | true / true |
| `failures` / `deferred_points` | 0 / 0 |
| 环境 | scipy 1.17.1 / numpy 2.4.6 |

### 完成度

| 子库 | `phase` | 本轮完成点数 | 耗时 | workers | `inflation_p90` |
|---|---|---|---|---|---|
| **a3.0** | run-done | **32 832** | 4 042 s | 20 | **1.43** |
| a4.0 | scatter-25-done | 94 | 22 s | 16 | 15.9 |
| a5.0 | scatter-25-done | 94 | 8 s | 16 | 17.6 |
| a7.0 | scatter-25-done | 94 | 6 s | 16 | 14.4 |
| a10.0 | scatter-25-done | 94 | 5 s | 16 | 14.0 |

只有 a3.0 是完整网格（32 832 点，约 1.1 h）。`inflation_p90` 1.43 vs 散射档的 14–18，
同样说明档跑落在昂贵区域。

## 图

每个间距**五张**图（比双光子族少一张 `p_mid`，因为没有中间态）。以完整的 a3.0 为例：

![最大泄漏 a = 3 µm](a3.0/plots/max_leakage_8x9.png)
主图：最坏情况相干泄漏。

![总误差 a = 3 µm](a3.0/plots/total_error_8x9.png)
![总损耗 a = 3 µm](a3.0/plots/p_loss_total_8x9.png)
![Rydberg 布居 a = 3 µm](a3.0/plots/p_ryd_8x9.png)
![旁观 Rydberg 泄漏 a = 3 µm](a3.0/plots/p_r_garb_8x9.png)

`a4.0/plots/`、`a5.0/`、`a7.0/`、`a10.0/` 下有同样五张（png + pdf）。图不进 git。

## 数据与溯源

```
a{3,4,5,7,10}.0/
  manifest.json      13 键：axes, physics, *_hash, git, created_at, policies, run_meta
  chunks/            相干序列，每块 38 键
  scatter/           散射序列，每块 28 键
  filter/            相位噪声滤波核序列（**仅 a3.0**）：kernel, f_bins；118 MB / 12168 点，未跟踪
  trajectories/      96 条采样轨迹，11 键
  reports/           status, pilot, candidates, verification, audit_summary, failures, scatter_gate
  exports/           latest_merged.npz（可再生，已 gitignore）
  plots/             五张图，png + pdf
  logs/store.lock    fcntl.flock 文件；进程退出即释放，PID 记录刻意保留
```

git 里追踪了 **43 个文件，全部在 `a3.0/` 下**（18 chunks + 16 scatter + 6 reports +
2 trajectories + manifest）—— 这是刻意留的小基线夹具，不是意外。
`reports/candidates.json` 自带说明 *"per-panel minima over exact ODE nodes only"*。

### 告示：`a3.0` 正在被另一条研究线扩写

相位噪声研究（`results/297_laser_noise/README.md`）正在给这个图族
加上相位噪声度量与"功率↔Rabi"换算表。它已经写入了
`a3.0/reports/phase_noise_mc.json` —— 一份**另外四个间距都没有**的报告 ——
并在 2026-07-30 改动了 `audit_summary` / `candidates` / `pilot` / `verification`。
CLI 里也已多出 `filter` 子命令。

**任何关于 `a3.0` 的文件计数或报告清单都应当现场重新测量，不要照抄本文。**

## 复现

```bash
# 续算/扩展某个间距
uv run python scripts/max_leakage_297_sweep.py run --spacing-um 3.0 --help
# 只看状态，不碰数据
uv run python scripts/max_leakage_297_sweep.py status \
    --output results/max_leakage_297/a3.0
# 从既有存储重渲图
uv run python scripts/max_leakage_297_sweep.py plot \
    --output results/max_leakage_297/a3.0
```

子命令全集：`status, pilot, run, audit, scatter, filter, export, plot`
（`filter` 是相位噪声滤波核，由上述研究线新增）。完整 a3.0 网格约 4 042 s / 20 workers。
在 DGX 上跑，不要隔着挂载。

**产出脚本** `scripts/max_leakage_297_sweep.py`（末次提交 `e3dfcb9`，2026-07-25），
底座 `scripts/sweeplib/`。设计
`docs/designs/2026-07-24-max-leakage-297-sweep-design.md`。
