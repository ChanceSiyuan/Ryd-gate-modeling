# 01r_adiabatic_optimization — 双原子 `01r` 平滑回计算态脉冲能压到多短

## 问题与模型

两原子 `01r` 三能级链（间距 4.0 µm，ARC 70S 有符号对相互作用
$C_6 = 5.420666\times10^{12}$ rad/s·µm⁶，即 $V_{rr}/2\pi = 210.6264$ MHz），单支路驱动：

$$H(t) = \frac{A(t)}{2}\sum_j\big(|r\rangle\langle 1|_j e^{-i\phi(t)} + \mathrm{h.c.}\big)
\;-\;\chi(t)\sum_j n_{r,j}\;+\;V_{rr}\,n_{r,1}n_{r,2},
\qquad \phi(t)=\int_0^t\chi\,dt'$$

门允许过程中占据 Rydberg 态，只要求四个逻辑输入 $b\in\{00,01,10,11\}$ **全部回到计算子空间**
并带上纠缠相位。以 $a_b=\langle b|U(T)|b\rangle$ 记末态振幅：

$$L_b = 1-|a_b|^2,\qquad
q = a_{00}a_{11}\overline{a_{01}}\,\overline{a_{10}},\qquad
\Phi_{ZZ}=\arg q,\qquad
C_\phi = \big|\sin(\Phi_{ZZ}/2)\big|$$

**验收判据**：$L_{\max}\le 10^{-4}$ 且 $C_\phi\ge 0.5$（`config.acceptance`）。

**参数化**：17 个坐标 —— 8 个幅度 B 样条系数 $p$、8 个啁啾系数 $d$、1 个端点啁啾标度 $\eta$，
$s=t/T$，包络为 15% 五次升降的平台：

$$A = A_{\max}\,\mathrm{env}(s)\,(Bp),\qquad
\chi = \chi_{\max}\frac{x+v}{1+xv},\quad x=-\eta\cos 2\pi s,\;\; v=\tanh\!\big(\mathrm{env}(s)\,(Bd)\big)$$

这个分式形式让 $|\chi|\le\chi_{\max}$ 自动成立而无需惩罚项。硬件盒约束：
$A_{\max}/2\pi = 17$ MHz、$\chi_{\max}/2\pi = 20$ MHz、端点啁啾 $\ge 2\pi\cdot 5$ MHz。

**目标函数**（内部搜索，非验收）：$J = \overline{L} + \mathrm{hinge}^2$，
$\mathrm{hinge}=\max(0,\,1-S_\phi/S_{\phi,\min})$，$S_\phi=(|q|-\mathrm{Re}\,q)/2$，
$S_{\phi,\min}=C_\phi^2=0.25$ —— 相位不足才罚，够了就只压泄漏。

搜索走 `qoc.grape` 离散伴随（分段常数中点片，1 ns 上限），梯度经样条链式法则回拉，
所以代价与坐标数无关。每级选中的脉冲再用**公开** `exact_ode` 后端
（dense，rtol 1e-10，atol 1e-12，301 个轨迹点）独立复验。

## 核心结果

3 个解析种子分支 × 16 个时长（2.0 → 0.5 µs，步长 0.1 µs）= 48 级，**46 级通过**：

| 分支 | 种子 $(A_0,\ \Delta_{\rm edge})$ MHz | 2.0 µs 基线 | 阶梯终点 |
|---|---|---|---|
| `phase_near_pi` | (13, 15) | 通过 | 走完 0.5 µs |
| `negative_phase` | (14, 12) | 通过 | 走完 0.5 µs |
| `positive_phase` | (10, 13) | 通过 | 走完 0.5 µs |

- **2 µs → 0.5 µs 的四倍压缩是可行的**，阶梯从不停摆：某级即使两个候选点都不可行，
  也取目标值较低者继续（`select_fallback`），该级标为 failed 但仍完整记录 —— 失败被保留在
  记录里而不是被丢掉，所以"哪里开始不行"是可查的。
- **每级都带离散化自检**：选中脉冲在**加倍细化**的网格上重算一次，差值记入
  `stages[*].grid_error` 的 `delta_L_max` / `delta_C_phi`。1 ns 步长下端点振幅与旧 CF4 引擎
  符合到 $\sim2\times10^{-5}$，远低于两个验收阈值。
- **成本由验证而非优化主导**：全跑 ~75 min，其中 48 次 `exact_ode` 约 57 min，
  L-BFGS-B 本身仅约 18 min（1015 条 history，967 次迭代）。
- **产物逐位可复现**。2026-07-31 完整重跑与已记录 artifact 比对：`config`、四个 metric 块、
  exposure、选中参数的**最大相对偏差 0.000e+00**，全部离散字段相同。

## 图

本目录**无图文件**。图是 `scripts/notebooks/06_01r_adiabatic_optimization.ipynb` 的渲染输出
（已随 notebook 提交）：种子池、GRAPE/延拓历史（目标值、泄漏、$\Phi_{ZZ}$、$C_\phi$、
Rydberg 曝光量，按分支分列）、以及物理脉冲的演化轨迹。notebook 只回放 `result.json`，
零优化器/ODE/ARC 调用，秒开。

## 数据与复现

| 路径 | 内容 |
|---|---|
| `result.json` | 全部：`schema_version` 4、`kind`、`complete`、`config`、`history`(1015)、`stages`(48) |

每级 `stages[i]` 同时存 `seed_metrics`、`optimized_metrics`、`search_metrics`（细网格）、
`exact_metrics`（公开 `exact_ode`，含四个逻辑输入各自的 `exposures_s`），
所以**搜索模型与精确模型的差距逐级可审**。

```bash
# 复现完整 artifact（48 级，~75 min）
uv run python scripts/adiabatic_01r_optimization.py --force
# 只跑三个 2 µs 基线
uv run python scripts/adiabatic_01r_optimization.py --force --no-continuation
# 只出图（回放，无计算）
uv run jupyter nbconvert --to notebook --execute --inplace \
    scripts/notebooks/06_01r_adiabatic_optimization.ipynb
```

不加 `--force` 且已有同 schema artifact 时为空操作，可安全重复执行。

**产出脚本** `scripts/adiabatic_01r_optimization.py` —— 本研究的唯一计算源；
notebook 06 导入该模块，只回放与绘图，两边不可能对不上。搜索侧的双线性控制模型由
`ryd_gate.bilinear_control_model` 导出（ADR-0024），与 `simulate` 共用同一个编译器。
