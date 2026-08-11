# max_leakage_ode — 双光子 `rb87_7_mp` CZ 的相干泄漏图族（五个晶格间距）

## 问题与模型

双原子双光子 CZ 的**原始系（original-frame）**最大泄漏扫描 —— 不做旋转波近似之外的化简，
直接在七能级 `rb87_7_mp` 模型上积分。对每个晶格间距扫一张
**8 × 9 的 $\Delta_e\times t_{\rm gate}$ 图**：

$$\Delta_e/2\pi \in \{9,\,12,\,15,\,20,\,25,\,30,\,40,\,50\}\ \text{GHz}
\;\times\;
t_{\rm gate}:\ 9\ \text{个取值}$$

每个格点内部再扫两条**锚点轴**（在 $\log_{10}$ 空间插值）：

- Rabi 频率 `omega_anchors_mhz`（4 个锚点）
- 失谐扫描幅度 `dsweep_anchors_mhz` = 2, 10, 20, 30 MHz，
  硬件上限 `dsweep_hw_limit_mhz = 20.0` —— 这是**硬件限制**而非建模选择。

固定物理：`beam_factor` 140、`detuning_sign` +1、`n_eval_trajectory` 301、
生产容差 `atol_production` $10^{-12}$、审计容差 `atol_audit` $10^{-13}$。

求解器是分块最大 DOP853：误差范数**按 (格点, 逻辑输入) 分块**独立估计并取最大，
所以容差是逐块强制的，而不是被大分量稀释掉。

存储是可续算的：`chunks/` 相干序列、`scatter/` 散射序列、`trajectories/` 采样态历史、
`reports/` 审计链、`plots/` 渲染图。

## 核心结果

### 驱动强度随间距显著下降 —— 不同间距的图不在同一驱动下

`verification.json` 记录了每个子库实际拿到的 1013 nm Rabi 及其相对参考值的偏离：

| 间距 | $\Omega_{1013}$ (rad/s) | `omega_1013_rel_dev_from_reference` |
|---|---|---|
| a3.0 | $3.0764\times10^{9}$ | $6.9\times10^{-13}$（参考） |
| a4.0 | $2.6642\times10^{9}$ | 0.134 |
| a5.0 | $2.3830\times10^{9}$ | 0.225 |
| a10.0 | $1.6850\times10^{9}$ | **0.452** |

200 原子光束铺在更大的阵列上，10 µm 处可用 $\Omega_{1013}$ 掉了 45%。
**这是被测的物理，不是不一致** —— 但它意味着跨间距比较"同一 $(\Delta_e,t)$ 格点"时，
比的是不同驱动强度下的门，解读时必须带上这一条。

### 求解器接缝在每个间距上都逐位一致

| 检验 | 全部五个子库 |
|---|---|
| `hamiltonian_equivalence_rel_dev` | **0.0** |
| `error_norm_max_dev`（自定义分块范数 vs 安装版 SciPy） | $4.89\times10^{-16}$ |
| `error_norm_verified` / `swap_symmetric` | true / true |
| 环境 | scipy 1.17.1 / numpy 2.4.6 |

### 完成度与代价

| 子库 | `phase` | 本轮完成点数 | 耗时 | `inflation_p90` |
|---|---|---|---|---|
| **a3.0** | scatter-13-done | **12 168** | 26 587 s (7.4 h, 40 workers) | **1.20** |
| a4.0 | scatter-13-done | 806 | 3 562 s | 7.35 |
| a5.0 | scatter-13-done | 94 | 1 830 s | 6.84 |
| a7.0 | scatter-13-done | 94 | 1 764 s | 6.96 |
| a10.0 | scatter-13-done | 94 | 1 918 s | 7.50 |

只有 **a3.0 是完整网格**，其余四个是散射档。`failures = 0`、`deferred_points = 0`
在五个子库全部成立。`inflation_p90`（代价膨胀的 90 分位）是网格质量的指示：
完整跑是 1.20，散射档 6.8–7.5 —— 档跑落在参数空间中昂贵的那一片。

## 图

每个间距六张图，同一 8×9 网格。以完整的 a3.0 为例：

![最大泄漏 a = 3 µm](a3.0/plots/max_leakage_8x9.png)
主图：最坏情况相干泄漏。

![总误差 a = 3 µm](a3.0/plots/total_error_8x9.png)
![总损耗 a = 3 µm](a3.0/plots/p_loss_total_8x9.png)
![中间态布居 a = 3 µm](a3.0/plots/p_mid_8x9.png)
![Rydberg 布居 a = 3 µm](a3.0/plots/p_ryd_8x9.png)
![旁观 Rydberg 泄漏 a = 3 µm](a3.0/plots/p_r_garb_8x9.png)

`a4.0/plots/`、`a5.0/`、`a7.0/`、`a10.0/` 下有同样六张（png + pdf）。图不进 git。

## 数据与溯源

```
a{3,4,5,7,10}.0/
  manifest.json      16 键：axes, physics, *_hash, git, created_at, policies, run_meta
  chunks/            相干序列，每块 40 键
  scatter/           散射序列，每块 30 键
  trajectories/      96 条采样轨迹，11 键
  reports/           status, pilot, candidates, verification, audit_summary, failures, scatter_gate
  exports/           latest_merged.npz（可再生，已 gitignore）
  plots/             六张图，png + pdf
  logs/store.lock    fcntl.flock 文件；进程退出即释放，PID 记录刻意保留
legacy_c6-874/       已归档的 pinned-C6-874 时代存储（见下）
```

`reports/candidates.json` 自带说明 *"per-panel minima over exact ODE nodes only"* ——
候选点是**精确 ODE 节点上的逐面板极小**，没有插值，这是正确的读法。

### 两处已知不规整

- **`a3.0/reports/pilot.json` 有 9 个键，a4.0–a10.0 只有 8 个**。`legacy_c6-874` 也是 9 个，
  所以 a3.0 与旧库同代，另外四个是后来按稍作精简的报告 schema 跑的。没有消费端用到那个多出的键。
- **`legacy_c6-874/` 已于 2026-07-30 撤出 git 追踪**，仅保留在磁盘。它有 508 MB ——
  在一个 361 MB 的 `.git` 里占了 363 MB 的 blob，几乎就是整个仓库的体积，而其数据已被
  ARC-C6 时代的存储取代。没有任何代码读它；唯一提及是
  `tests/test_max_leakage_ode_sweep.py` 里一个 argparse 字符串常量。现已被 `.gitignore` 覆盖。

## 复现

```bash
# 续算/扩展某个间距
uv run python scripts/max_leakage_ode_sweep.py run --spacing-um 3.0 --help
# 只看状态，不碰数据
uv run python scripts/max_leakage_ode_sweep.py status \
    --output results/max_leakage_ode/a3.0
# 从既有存储重渲六张图
uv run python scripts/max_leakage_ode_sweep.py plot \
    --output results/max_leakage_ode/a3.0
```

子命令全集：`status, pilot, run, audit, scatter, export, plot`。
完整网格约 7.4 h / 40 workers。在 DGX 上跑，不要隔着挂载。

**产出脚本** `scripts/max_leakage_ode_sweep.py`（末次提交 `e3dfcb9`，2026-07-25），
底座 `scripts/sweeplib/`；`scripts/notebooks/04_quench_and_state_prep.ipynb` 消费这些图。
设计 `docs/designs/2026-07-24-spacing-family-sweep-design.md`、
`docs/designs/2026-07-25-sweep-merge-sweeplib-design.md`。
