# 297_to_calibration — 297 nm 单光子 CZ 的 Time-Optimal 相位族标定

把双光子 TO 标定流程（notebook 01 + `results/cz_gate/to_calibration/`）移植到
`rb87_297_clock_4` 单光子模型，在 297 家族的规范工作点上标定 5 个 TO 形状参数，
并给出相干误差、门内自发辐射预算和逐能级布居轨迹。

## 问题与模型

两原子链（a = 3.0 µm），每原子 4 能级 $\{|0\rangle,|1\rangle,|r\rangle,|r_{\rm garb}\rangle\}$。
$|0\rangle$ 是暗旁观态；一束全局 σ⁻ 297 nm 光同时驱动目标腿
$|1\rangle\!\to\!|r\rangle$（53P₃/₂, $m_J=-3/2$）和 garbage 腿
$|1\rangle\!\to\!|r_{\rm garb}\rangle$（$m_J=-1/2$，偶极比 $\kappa=1/\sqrt3\approx0.5774$，
即 `fixed.omega_297_garb_rad_s / fixed.omega_297_max_rad_s`）：

$$
H(t)=\sum_{j=1,2}\Big[\frac{\Omega(t)e^{-i\phi(t)}}{2}
\big(|r\rangle\langle1|_j+\kappa\,|r_{\rm garb}\rangle\langle1|_j\big)+\mathrm{h.c.}
+\delta_Z\,n^{(j)}_{r_{\rm garb}}\Big]
+\sum_{cc'}V_{cc'}\,|cc'\rangle\langle cc'|,
$$

其中 $\delta_Z/2\pi=+37.323$ MHz 是 20 G 下 $r_{\rm garb}$ 相对 $r$ 的 Zeeman 分裂
（`to_297.json:garb_zeeman_rad_s`），$V_{cc'}$ 是预设内建的通道分辨 ARC C6；
rr 主通道在 3 µm 给 $V_{\rm nn}/2\pi=-131.873$ MHz（`to_297.json:blockade_V_nn_rad_s`）。

**TO 相位族**（`Direct297TOProtocol`，与双光子 `TOProtocol` 同构）：

$$
\Omega(t)=\Omega_{\max}\,B(t;t_{\rm rise}),\qquad
\phi(t)=A\cos(w\,\Omega_{\max}t+\phi_0)+\delta\,\Omega_{\max}t,\qquad
t_{\rm gate}=T\cdot\frac{2\pi}{\Omega_{\max}},
$$

$B$ 为 Blackman 包络（$t_{\rm rise}=20$ ns）。目标函数是对 CZ 的平均门保真度
（次归一化振幅自动惩罚 leakage），单比特相位 $\theta$ 对固定振幅解析可优：

$$
1-F,\qquad
F=\tfrac{1}{20}\Big(\big|a_{00}+2e^{-i\theta}a_{01}-e^{-2i\theta}a_{11}\big|^2
+|a_{00}|^2+2|a_{01}|^2+|a_{11}|^2\Big),
$$

$a_s=\langle s|U|s\rangle$ 由 `exact_ode` 求出（$|10\rangle\equiv|01\rangle$ 对称）。

**工作点**（`to_297.json:fixed`）：原子处 0.6 W、束斑 420 µm² → 目标腿
$\Omega_{\max}/2\pi=16.614$ MHz；n = 53、B = 20 G、a = 3.0 µm。

**标定流程**（移植自双光子标定的"批并行候选"方案）：32-worker differential
evolution（种群 32、40 代、mp/pm 双光子形状注入初始种群、搜索容差 rtol 1e-6）
→ Nelder-Mead 抛光（默认容差 rtol 1e-8）→ rtol 1e-10 / atol 1e-13 复核。
默认容差与紧容差终点差 6×10⁻⁶（9.232e-4 → 9.173e-4），与双光子标定观察到的
~7×10⁻⁶ 求解器偏差一致；下表全部取紧容差数字。

## 核心结果

标定终点 `x = [A, w, φ₀, δ, θ, T]`（`to_297.json:x`）：

```
[0.82458, 0.94157, 0.53710, -0.65717, 0.89609, 2.47357]
```

与双光子 TO 标定（notebook 01，2026-07-17，同为 rtol 1e-10）对比：

| | **297 单光子（本目录）** | `rb87_7_mp`（70S） | `rb87_7_pm`（53S） |
|---|---:|---:|---:|
| coherent 1−F（含 leakage） | **9.173e-4** | 1.398e-5 | 7.674e-5 |
| 门内自发辐射 | **3.561e-4** | 2.391e-3 | 1.803e-3 |
| **total incl. SE** | **1.273e-3** | 2.405e-3 | 1.880e-3 |
| t_gate | 148.9 ns | 268.7 ns | 287.1 ns |
| Ryd interaction | 131.9 MHz | 1183 MHz | 43 MHz |
| CZ 相位距 ±π | 2.6e-6 rad | <5e-5 rad | <5e-5 rad |

**在上述四能级、对角 $C_6$ 模型内部，单光子 TO 的总误差（1.27e-3）低于两个
双光子 manifold（2.41e-3 / 1.88e-3）：没有中间态散射、门更短，门内自发辐射低
5–7 倍；代价是 coherent 部分被 garbage 腿限制在 ~9e-4，比双光子高一个量级。**
附录的显式 pair-basis 审计表明该对角相互作用模型在 20 G 和 160 G 都不自洽，
所以这些门误差是**原模型内结果**，不能作为已验证的物理性能排序。相干误差几乎全部是泄漏
（`summary.json:per_state`：|01⟩/|11⟩ leakage 1.20e-3 / 1.22e-3，回归概率
0.9988），其中 Nielsen 加权的末态残留以 $r_{\rm garb}$ 为主（8.37e-4 原子，
`se_budget.r_garb.residual`）——σ⁻ 全局光的 garbage 腿又强（κ=0.577）又近
（$\delta_Z/\Omega_{\max}=2.25$），是这个工作点的硬瓶颈。

逐输入态指标（`summary.json:per_state`，rtol 1e-10）：

| 输入 | 回归概率 | 裸相位 (rad) | leakage |
|---|---:|---:|---:|
| `00` | 1.000000 | +1.65538 | 8.7e-08 |
| `01` = `10` | 0.998804 | +2.54328 | 1.20e-03 |
| `11` | 0.998776 | +0.30596 | 1.22e-03 |

门内自发辐射预算（Nielsen 权重 ¼/½/¼，$\Gamma_{\rm rad}=3184\ \mathrm{s^{-1}}$、
$\Gamma_{\rm BBR}=6787\ \mathrm{s^{-1}}$，`summary.json:se_budget`）：

| 误差源 | radiative | blackbody | 末态残留* |
|---|---:|---:|---:|
| 目标 $\vert r\rangle$ | 1.114e-4 | 2.374e-4 | 3.186e-4 |
| garbage $\vert r_{\rm garb}\rangle$ | 2.33e-6 | 4.96e-6 | 8.370e-4 |
| **SE 合计（不含残留）** | | | **3.561e-4** |

*残留列单位是"末态原子数期望"（双占据计 2），已包含在 coherent 1−F 里，
不重复计入 SE 合计。

## 图

图为 `.png`，按仓库规则不入 git；重生成命令见"复现"。

![各能级布居随时间演化，按输入态分列](populations_297.png)

上排：$|0\rangle,|1\rangle,|r\rangle$ 每原子布居（|00⟩ 全程冻结 = 暗态；|11⟩ 的
$|r\rangle$ 峰值 ~0.25/原子，blockade 抑制双激发）；下排：$|r_{\rm garb}\rangle$
每原子布居（|01⟩ 峰值 ~0.5%，|11⟩ ~1.2%，门末残留即 coherent 误差主体）。

![Nielsen 加权累计衰变概率与最终误差预算](error_budget_297.png)

左：门内累计衰变概率（目标 $|r\rangle$ 主导，3.5e-4）；右：各项对 1−F 的贡献
（对数轴）。

![标定出的 297 nm TO 波形](pulse_297.png)

Blackman 包络（20 ns 沿）+ 双周期余弦相位调制；瞬时扫频 $\dot\phi/2\pi$ 在
−24 至 +2 MHz 之间，均值 −10.9 MHz（$\delta=-0.657$）。

## 四能级对角模型内检查：增大 B 可压低 garbage 腿

在 B = 160 G（$\delta_Z\times8$）用**同一 TO 族、同样的优化算法、同一起点**重新优化得到的最优值
**2.369e-6**，比 20 G 低约 390 倍（低于双光子 mp 的 1.40e-5），参数几乎不动
（x5 = [0.840, 0.927, 0.531, −0.629, 2.515]）。
这只说明在原四能级对角模型内，garbage 单粒子腿确实是主要优化瓶颈；显式 pair-basis
结果在 160 G 给出 $|rr\rangle$ 的弱移权重 0.918，故该重抛光数字不能再作为高磁场
修复完整相互作用模型的证据。

## 告示

- **B 场口径**：`single_photon.ipynb` 开头声明 B = 100 G，但 cell 4 把 `B`
  覆盖成 8 G（notebook 实际运行值）。本标定取 **20 G** —— 预设默认值，也是
  `max_leakage_297` / `297_laser_noise` 扫描家族的口径；三处数值并存，引用时注意。
- **ARC C6 警告**：53P₃/₂ $|-3/2,-3/2\rangle$ 在 θ=π/2 不是主导本征通道
  （最大重叠 0.46），`arc_pair_c6_rad_s_um6` 返回最大重叠本征值——与
  `max_leakage_297` 扫描相同的已知行为，运行日志有对应 UserWarning。
  附录的 2276 维显式 pair-basis 对角化进一步表明：旧模型在 20 G 和 160 G
  都遗漏大量弱移谱权重，后者并不因 Zeeman 劈裂而自洽。
- **DE 搜索盒的 T 上界（2.2）在 DE 终点处于边界**，但后续 Nelder-Mead 无界，
  把 T 推到内点 2.474 后收敛（`search.de.x[4]=2.199` → `x[5]=2.474`）；
  终点不是搜索边界 artifact。未做超出 DE 盒的第二盆地宽域探索；B=160 G 重抛光
  只约束原四能级目标函数的优化景观，不约束显式 pair-basis 模型的最优解。
- **无 XYZ/AL/LG 分解**：该预设 `branching_ratios` 为空，无法按衰变去向拆分；
  预算按 radiative / blackbody / 残留报告，与 notebook 01 的双光子表不完全同构
  （SE 合计口径一致：门内衰变事件概率积分）。

## 数据与溯源

| 文件 | 内容 |
|---|---|
| `to_297.json` | 标定记录：`fixed`（工作点 7 项）、`x`（6 参数）、默认/紧容差 1−F、`search`（DE 种群 32 × 40 代 nfev 1312 → 2.893e-2；NM nfev 219 → 9.232e-4；5 个种子含 mp/pm 形状；rng_seed 1）、`elapsed_s` 5219 |
| `summary.json` | 逐输入态指标、θ、CZ 相位距、$\Gamma$、SE 预算、total 1.273e-3 |
| `traces_297.npz` | `t (301,)`、`pops (4,4,301)`（[输入态 00/01/10/11, 能级 0/1/r/r_garb, 时间]，两原子求和）、`amps (4,4) complex`（[输入态, 基态]）、`x (6,)`、`fixed (7,)`（`sorted(fixed)` 键序）、`spacing_um`、`rtol=1e-10` |
| `garb_leg_check.json` | 稳健性检查记录：`scan`（冻结 x\* 的 4 点 B 扫描，紧容差 1−F 与逐态泄漏）、`repolish`（B=160 G 同族 NM 重抛光，nfev 341 → 紧容差 2.369e-6）；由 `scripts/check_297_garb_leg.py` 生成，2026-08-05，全流程确定性（无 RNG） |
| `pair_channels.json` | schema 2 对相互作用审计：`full_pair.fields.{20,160}.channels.{rr,r_rgarb}` 是显式 pair-basis 局域谱、弱移权重与捕获权重；`effective_c6_comparison` 是旧二阶 $C_6$ + PP-Zeeman 对照；`radial_defect_ranking` 仅按径向矩阵元和 $|\delta|$ 排序；ARC 3.10.2，由 `scripts/check_297_pair_channels.py` 生成，2026-08-09，确定性复跑一致 |
| `*.png` | 三张图（未入 git） |

生成脚本 `scripts/calibrate_to_297.py`（分支 `laser-phase-noise`，本次运行时
尚未提交）；2026-08-04 在 DGX（40 核）上运行，标定段 5219 s
（`to_297.json:elapsed_s`；DE 976 s + NM 4243 s，见运行日志，日志在
DGX `/tmp/to297_run.log`，临时文件）。双光子对照数字取自
`results/cz_gate/to_calibration/to_{mp,pm}.json:tight_tol_infidelity` 与
notebook 01 cell 11 表（SE、t_gate、blockade）。

对相互作用审计由 `scripts/check_297_pair_channels.py` 在提交 `7634116` 的实现生成；
设计依据见 `docs/superpowers/specs/2026-08-09-full-pair-channel-diagonalization-design.md`
（提交 `b1aa67d`），执行步骤见
`docs/superpowers/plans/2026-08-09-full-pair-channel-diagonalization.md`。最终生产运行耗时
220.6 s；连续两次运行删除耗时字段后 JSON 逐字一致。

## 复现

```bash
# 回放：读取已有 to_297.json，重算紧容差轨迹 + 三张图 + summary.json（~1 分钟，已验证）
uv run python scripts/calibrate_to_297.py

# 全量重标定（覆盖记录；本次 32 worker 用时 ~87 分钟）
uv run python scripts/calibrate_to_297.py --force --workers 32

# 稳健性检查：冻结脉冲 B 扫描 + B=160 G 重抛光（确定性，~80 分钟）
uv run python scripts/check_297_garb_leg.py

# 显式 pair-basis 相互作用审计（确定性，pair_channels.json:params.elapsed_s = 220.6 s）
HOME=/tmp/arc297home MPLCONFIGDIR=/tmp/mpl297 uv run python scripts/check_297_pair_channels.py
```

## 附录：显式 pair-basis 对角化与二阶 C6 对照

### 最终审计对角化的矩阵

本次不再把中间 Förster 对态二阶消去。ARC 先在裸 pair states

$$
|\nu\rangle=|n_1l_1j_1m_{j1};n_2l_2j_2m_{j2}\rangle
$$

上建立有限维哈密顿量，再直接对角化

$$
H_{\rm pair}(B,R,\theta)
=\sum_\nu \epsilon_\nu(B)|\nu\rangle\langle\nu|
+V_{dd}(R,\theta).
$$

代码中它就是 ARC 的 `matDiagonal + matR[0] / (R*1e-6)^3`；前者已把线性
Zeeman 位移加到**所有保留的 pair states**，包括 S、P、D sectors，后者是完整角动量
矩阵的 dipole-dipole coupling。计算参数为 $n_i=48\ldots58$、$l_i\le2$、零场
pair defect 小于 30 GHz、`interactionsUpTo=1`、$R=3\ \mu$m、
$\theta=\pi/2$、$\phi=0$。20 G 与 160 G 的基维数均为 2276，矩阵均有
114932 个非零元（`pair_channels.json:full_pair.fields`）。因此这里的“全 pair”是指
**在上述截断内显式对角化**，不是无截断的严格原子哈密顿量。

对裸通道 $|c\rangle$，代码用其对角裸能 $\epsilon_c(B)$ 作参考，并定义频率位移与
可达权重

$$
\bar\Delta_k=\frac{E_k-\epsilon_c}{h},\qquad
p_k=|\langle\Psi_k|c\rangle|^2,qquad
W_{\rm weak}=\sum_{|\bar\Delta_k|<83.07\,{\rm MHz}}p_k.
$$

83.07 MHz 是 $5\times16.614$ MHz；所有表中能量均以 $E/h$ 的 MHz 表示。
程序用确定初始向量的 shift-invert sparse eigensolver 在每个裸能附近取局域谱，并自动
把本征对数从 32 加倍，直到弱移窗口两侧都被包围且捕获权重至少 0.99。
四个结果都满足 `window_bracketed=true`、`capture_converged=true`；捕获权重为
0.9969–0.9996，故表中的弱移权重不是任取若干谱线所得。

### 显式基结果

下表的“弱窗主谱线”写成“位移 MHz（裸态重叠）”；只列决定
$W_{\rm weak}$ 的谱线及紧邻阈值的主谱线。完整成分在
`pair_channels.json:full_pair.fields.{20.0,160.0}.channels`。

| B | 裸通道 | 弱窗主谱线 | $W_{\rm weak}$ | 捕获权重 |
|---:|---|---|---:|---:|
| 20 G | $|{-3/2},{-3/2}\rangle$ (`rr`) | −84.873 (0.733，刚在阈值外); +45.792 (0.131); +79.058 (0.009) | **0.139987** | 0.996906 |
| 160 G | `rr` | −54.543 (0.614); −8.893 (0.304) | **0.918024** | 0.998374 |
| 20 G | $|{-3/2},{-1/2}\rangle$ (`r_rgarb`) | −27.993 (0.416); −35.748 (0.374); +51.953 (0.100); +65.417 (0.054) | 0.944124 | 0.998528 |
| 160 G | `r_rgarb` | −13.127 (0.374); +2.928 (0.274); −22.061 (0.215); −0.315 (0.115) | 0.978549 | 0.999592 |

决定性结论是：**160 G 没有恢复单一 scalar blockade channel。** `rr` 的 91.8%
可达谱权重仍在 $\pm83.07$ MHz 内，其中 30.4% 位于 −8.89 MHz。对应本征矢也显示
被旧 16 维 $53P_{3/2}+53P_{3/2}$ 模型排除的 pair sectors：例如 −54.54 MHz
本征态含两种 $53P_{3/2}+53P_{1/2}$ 排列各 14.3%，−8.89 MHz 本征态各含
34.4%。这是显式 $V_{dd}$ 混合产生的成分；当前模型并未加入磁场算符自身在不同
$j$ manifolds 间的非对角混合。

### 旧二阶 C6 模型为何给出相反结论

`effective_c6_comparison` 保留了旧算法作为非权威对照。它先在 $B=0$ 将中间对态消去，
只在 16 维 $53P_{3/2}+53P_{3/2}$ 流形中形成

$$
H_{\rm eff}^{(2)}(0)=-\sum_\lambda
\frac{P V_{dd}|\lambda\rangle\langle\lambda|V_{dd}P}{\delta_\lambda},
$$

随后仅在这个 PP 流形内手工加线性 Zeeman 项。它既不让中间态 defect 随 $B$ 变化，
也不能保留对准共振 pair sectors 的非微扰杂化。

| B | 模型 | `rr` 主谱线：位移 MHz（重叠） | $W_{\rm weak}$ |
|---:|---|---|---:|
| 20 G | 二阶 C6 + PP Zeeman | −158.860 (0.803); +7.085 (0.150); +77.527 (0.026) | 0.175648 |
| 20 G | 显式 2276 维 | −84.873 (0.733); +45.792 (0.131); +79.058 (0.009) | 0.139987 |
| 160 G | 二阶 C6 + PP Zeeman | −129.555 (0.989) | **0.000000** |
| 160 G | 显式 2276 维 | −54.543 (0.614); −8.893 (0.304) | **0.918024** |

160 G 行的定性翻转说明“初始 PP manifold 的 Zeeman splitting 大于零场 C6 谱宽”
不是自洽性判据：磁场也移动其它保留 pair states，而显式对角化还能产生旧模型没有的
跨 sector 杂化。零场二阶对照本身也不是 scalar：`rr` 最大本征通道重叠仅 0.463，
另有 0.187 权重落在 +3.89 MHz 通道；裸态期望值为 −121.61 MHz。

### Förster channel 排名只用于筛查

`radial_defect_ranking` 另列

$$
w_\lambda\propto\frac{(R_1R_2)^2}{|\delta_\lambda|}
$$

的归一化排序。它省略角 CG/Wigner 因子，并丢掉分母符号，因而**不是**实际 $C_6$
分解，也不能证明某通道支配相互作用。

| 候选虚 pair channel | defect (GHz) | 径向/失谐权重 |
|---|---:|---:|
| $53S_{1/2}+54S_{1/2}$ | +0.3188 | 1.000 |
| $53S_{1/2}+52D_{5/2}$ / $52D_{3/2}$ | −10.50 / −10.58 | 0.0481 / 0.0478 |
| $54S_{1/2}+51D_{5/2}$ / $51D_{3/2}$ | −11.16 / −11.25 | 0.0131 / 0.0129 |
| $51D+52D$（四种 $j$ 组合） | −21.98 至 −22.14 | 0.0104–0.0105 |

### 仍然保留的近似与对门结果的含义

- pair basis 尚未对 `n_range`、$l_{\max}$ 和 30 GHz 能窗做收敛扫描；只计算了
  $R=3\ \mu$m，没有扫描 avoided crossings 随 $R$ 的演化。
- 只保留 dipole-dipole coupling。ARC 使用弱场线性 paramagnetic Zeeman；忽略
  diamagnetic、hyperfine，以及磁场算符自身在不同 $j$ manifolds 间的非对角混合。
- 求的是目标裸能附近的稀疏局域谱，不是全部 2276 个本征态；不过弱窗已完整包围，
  且四个裸通道捕获权重都超过 0.996。
- 门动力学仍是原四能级、对角 scalar-interaction 模型；本次只审计 pair spectrum，
  尚未把显式通道接入时间演化或重新优化脉冲。

因此主表 20 G 的 $9.173\times10^{-4}$ 与四能级 160 G 重抛光的
$2.369\times10^{-6}$ 都应标记为**相互作用模型未验证的优化输出**，不能解释为显式
pair physics 下的门误差。下一步必须先把这些可达 pair eigenchannels 接入动力学，再
重新评估或优化门；单纯继续增大 $B$ 不是本次结果支持的修复方案。
