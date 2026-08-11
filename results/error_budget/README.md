# error_budget — CZ 误差预算图族：泄漏在参数空间的哪里最小

## 问题与模型

`1r` 两能级（$|1\rangle,|r\rangle$）往返失谐扫描：把构型驱向 Rydberg 再扫回来，
在 $L_x\times L_y$ 方阵上用精确后端演化（最近邻 vdW）。每个格点评分三条**逐原子**误差通道
（对格点取平均）及其和：

$$\varepsilon_{\rm coh} = \overline{n_{r,i}(T)}
\qquad\text{(残余非绝热激发)}$$

$$\varepsilon_{\rm SE} = \overline{\Gamma_r\!\int n_{r,i}\,dt}
\qquad\text{(Rydberg 自发辐射)}$$

$$\varepsilon_{\rm sc} = \overline{\Gamma_e\!\int\Big[\tfrac{4}{3}\Big(\tfrac{\Omega_{420}}{2\Delta_e}\Big)^{2} n_{1,i}
+\Big(\tfrac{\Omega_{1013}}{2\Delta_e}\Big)^{2} n_{r,i}\Big]dt}
\qquad\text{(中间态散射，两条激光腿)}$$

$\varepsilon_{\rm SE}/\varepsilon_{\rm sc}$ 是在**无损幺正**轨迹上的事后微扰估计（没有 Lindblad），
之所以成立是因为 1r 轨迹保范数（$n_1 = 1-n_r$）。

**Rabi 的关键约定**：420 nm（6.4 W）与 1013 nm（100 W）光束照亮的是**整个约 200 原子阵列**，
不是单个格点。间距 $a$ 时阵列足迹为 $\sqrt{N_{\rm beam}}\,a \times 6\ \mu$m，
顶帽强度 $I = P/(\sqrt{N_{\rm beam}}\,a\cdot 6\,\mu\mathrm{m})$，于是

$$\Omega \sim 1/a,\qquad \Omega_{\rm eff}\sim 1/a^{2}$$

被模拟的 $L_x\times L_y$ 只是一块代表性补丁：它决定多体 Hilbert 空间，**不决定驱动强度**。
另外失谐扫描半幅是**固定的**（默认 20 MHz，实验只能扫约 ±20 MHz），刻意不随 $\Omega_{\rm eff}$ 缩放。

每张图存同样六个通道，因而可以逐面板对照：`grid_phase_err`（条件相位偏离 π，rad）、
`grid_max_leakage`、`grid_p_mid_max`（中间态峰值布居）、`grid_p_ryd_max`、
`grid_p_r_garb_max`（旁观 Rydberg 能级）、`grid_p_loss_total_max`。

三个图族回答三个不同问题，各有 8×8（`g8`）与 20×20（`g20`）两档分辨率：

| 族 | x 轴 | y 轴 |
|---|---|---|
| **figA** | $P_{420}$ 0.25–6.41 W | $P_{1013}$ 5–100 W（$\Delta_e\in\{20,30,45\}$ GHz 各一张） |
| **figB** | $\Delta_e/2\pi$ 15–80 GHz | $K_{\rm eff}/2\pi$ 1–12 MHz |
| **figC** | $\Delta_e/2\pi$ 15–80 GHz | $D_{\rm sweep}/2\pi$ 2–30 MHz |

## 核心结果

### 最优工作点在 $\Delta_e$–$D_{\rm sweep}$ 面

各族 20×20 图上 `grid_max_leakage` 的最小值：

| 族 | 最小泄漏 | 相对 figC |
|---|---|---|
| **figC**（$\Delta_e$–$D_{\rm sweep}$） | **$2.06\times10^{-5}$** | 1 |
| figB（$\Delta_e$–$K_{\rm eff}$） | $2.25\times10^{-4}$ | 11× |
| figA / $\Delta_e = 45$ GHz（功率面） | $1.64\times10^{-3}$ | 80× |

### 中间态是损耗预算的全部，Rydberg 不是

figA 族沿 $\Delta_e$ 的变化（20 → 30 → 45 GHz）：

| 通道 | 20 GHz | 30 GHz | 45 GHz | 走势 |
|---|---|---|---|---|
| `grid_p_mid_max` | $7.25\times10^{-2}$ | $3.21\times10^{-2}$ | $1.41\times10^{-2}$ | 按 $1/\Delta_e$ 单调压制 |
| `grid_p_loss_total_max` | $7.56\times10^{-2}$ | $3.52\times10^{-2}$ | $1.72\times10^{-2}$ | 跟着中间态走 |
| `grid_p_ryd_max` | $3.49\times10^{-3}$ | $3.44\times10^{-3}$ | $3.41\times10^{-3}$ | **几乎不动** |

这就是"为什么最优候选都在大 $\Delta_e$"的直接证据：总损耗跟着中间腿走，
Rydberg 腿在整个族里纹丝不动。旁观 Rydberg 泄漏在所有图上都可忽略
（`grid_p_r_garb_max` 全族 $\le 2.3\times10^{-5}$，跨全部图不超过 $1.3\times10^{-4}$）。

### 8×8 对相位不够用，对泄漏够用

同一张图两档分辨率的最优 `grid_phase_err`：

| 图 | g8 | g20 | 比值 |
|---|---|---|---|
| figC | 0.194 | **0.0072** | 27× |
| figA / 30 GHz | 0.230 | **0.0075** | 31× |
| figB | 0.0486 | 0.0085 | 5.7× |

而同样两档的 `grid_max_leakage` 最小值只差几个百分点。结论很实用：
**相位最优点只能从 g20 读，泄漏最优点 g8 就够**。

## 图

![晶格扫描探索图](lattice_sweep/exploratory_plots/ebudget_sweep_plots.png)
![晶格扫描探索图（续）](lattice_sweep/exploratory_plots/ebudget_sweep_plots2.png)

这两张是晶格扫描的探索性图。主图族是 `scripts/notebooks/error_buget.ipynb` 的渲染输出
（已随 notebook 提交），它读缓存的 `*_ode_g{8,20}.npz`，不跑 ODE。图不进 git。

## 数据与溯源

| 路径 | 内容 |
|---|---|
| `cz_gate_maps/error_budget_figA_De{20,30,45}_ode_g{8,20}.npz` | 6 张功率面图，各 11 键 |
| `cz_gate_maps/error_budget_figB_ode_g{8,20}.npz` | 2 张 $\Delta_e$–$K_{\rm eff}$ |
| `cz_gate_maps/error_budget_figC_ode_g{8,20}.npz` | 2 张 $\Delta_e$–$D_{\rm sweep}$ |
| `cz_gate_maps/error_budget_scan_{effective,full7}.npz` | 各 22 键：有效模型 vs 完整七能级扫描 |
| `cz_gate_maps/error_budget_3param_scan.npz` | 19 键，三参数扫描 |
| `lattice_sweep/exploratory_plots/*.png` | 2 张探索图 |

每张图自带 `records`、`axis_x`、`axis_y`、`axis_x_name`、`axis_y_name` 与六个 `grid_*`，
所以是自描述的 —— 上文的轴含义是从文件里读出来的，不是假定的。

### 告示：驱动脚本的成本模型已作废

批量求解器改造之后，单个格点要 **125–199 s**，而原驱动假设的是约 0.1 s —— 相差约 1000 倍。
**这里的数据仍然有效**，但要重扫大网格必须先重新设计驱动，不要照旧预期启动
`error_budget_sweep.py`。

## 复现

```bash
# 出图（读缓存图族，不跑 ODE）
uv run jupyter nbconvert --to notebook --execute --inplace \
    scripts/notebooks/error_buget.ipynb
# 重生成某一族的 20x20（昂贵，先读上面的成本告示）
uv run python scripts/gen_error_budget_g20.py --help
# 4D 扫描 / 绘图两种模式
uv run python scripts/error_budget_sweep.py --mode plot
```

**产出脚本** `scripts/error_budget_sweep.py` 与 `scripts/gen_error_budget_g20.py`
（末次提交均为 `735a284`，2026-07-15 的 `src/ryd_gate` 重写；数据日期
2026-06-17…07-16，最新的图在重写之后）。
