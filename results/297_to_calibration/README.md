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
| `pair_potential_curves.json` | schema 1 的 Zeeman-resolved pair-potential sidecar：四个 $53P_{3/2}$ doorway（$m_j=-3/2,-1/2,+1/2,+3/2$）及 70S benchmark；`manifolds.<state>.fields.<B>.angles.<theta>.curves` 保存 41 个 $R$ 点的完整亮谱、$W_{\rm weak}$、至多五条连续 branch、$rr$ overlap 与 $R=3\ \mu$m 的本征态主成分；`params.completed_cases=105`、`status=complete`，ARC 3.10.2；由同一脚本生成，2026-08-10 |
| `*.png` | 三张门标定图与十五张方案 1 pair-potential 图（均未入 git，可由脚本回放） |

生成脚本 `scripts/calibrate_to_297.py`（分支 `laser-phase-noise`，本次运行时
尚未提交）；2026-08-04 在 DGX（40 核）上运行，标定段 5219 s
（`to_297.json:elapsed_s`；DE 976 s + NM 4243 s，见运行日志，日志在
DGX `/tmp/to297_run.log`，临时文件）。双光子对照数字取自
`results/cz_gate/to_calibration/to_{mp,pm}.json:tight_tol_infidelity` 与
notebook 01 cell 11 表（SE、t_gate、blockade）。

对相互作用审计由 `scripts/check_297_pair_channels.py` 在提交 `7634116` 的实现生成；
设计依据见 `docs/designs/2026-08-09-full-pair-channel-diagonalization-design.md`
（提交 `b1aa67d`），执行步骤见
`docs/work/2026-08-09-full-pair-channel-diagonalization.md`。最终生产运行耗时
220.6 s；连续两次运行删除耗时字段后 JSON 逐字一致。

新增 $R$–方向扫描由同一脚本在当前工作树提交 `f3f51b7` 之上的未提交实现生成，
2026-08-10；Zeeman 扩展设计与执行计划分别见提交 `1f553cb`、`2ac06ba2`。因此计算
provenance 是“基准提交 + 本 README 所述工作树修改”，尚无可引用的最终实现提交。
最终 105 例全量运行的
`pair_potential_curves.json:params.elapsed_s=503.872` s、`wall_s_this_run=523.652` s、
`completed_cases_this_run=105`。完整对角化要求单线程 BLAS 才有合理的小矩阵性能。

## 复现

```bash
# 回放：读取已有 to_297.json，重算紧容差轨迹 + 三张图 + summary.json（~1 分钟，已验证）
uv run python scripts/calibrate_to_297.py

# 全量重标定（覆盖记录；本次 32 worker 用时 ~87 分钟）
uv run python scripts/calibrate_to_297.py --force --workers 32

# 稳健性检查：冻结脉冲 B 扫描 + B=160 G 重抛光（确定性，~80 分钟）
uv run python scripts/check_297_garb_leg.py

# 最便宜：只从 sidecar 重画十五张方案 1 pair-potential 图（已验证不会改写 JSON）
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl297 \
  .venv/bin/python scripts/check_297_pair_channels.py --plot-only

# 断点续算/重算缺失的 R–方向案例并作图；去掉 --resume 可从头覆盖（本次约 8.7 分钟）
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl297 \
  .venv/bin/python scripts/check_297_pair_channels.py --pair-potentials --resume

# 原 30 GHz、R=3 μm 的显式 pair-basis 审计（pair_channels.json，约 221 秒）
MPLCONFIGDIR=/tmp/mpl297 .venv/bin/python scripts/check_297_pair_channels.py
```

## 附录：显式 pair-basis 对角化与二阶 C6 对照

### 最终审计对角化的矩阵

本次不再把中间 Förster 对态二阶消去。ARC 先在裸 pair states $|\nu\rangle=|n_1l_1j_1m_{j1};n_2l_2j_2m_{j2}\rangle$ 上建立有限维哈密顿量，再直接对角化

$$
H_{\rm pair}(B,R,\theta)
=\sum_\nu \epsilon_\nu(B)|\nu\rangle\langle\nu|
+V_{dd}(R,\theta).
$$


对裸通道 $|c\rangle$，代码用其对角裸能 $\epsilon_c(B)$ 作参考，并定义频率位移与
可达权重

$$
\bar\Delta_k=\frac{E_k-\epsilon_c}{h},\quad
p_k=|\langle\Psi_k|c\rangle|^2,\qquad
W_{\rm weak}=\sum_{|\bar\Delta_k|<83.07\,{\rm MHz}}p_k.
$$

其中 $83.07\ {\rm MHz}=5\times16.614\ {\rm MHz}$ 只定义一个与最大 Rabi
频率相关的**诊断窗口**，不是 ARC 基组能窗，也不是只对角化这段能量。谱线在窗口内
只表示它相对裸通道位移较弱、可能削弱 blockade；$W_{\rm weak}$ 不是门错误率。

### 显式基结果

下表的“弱窗主谱线”写成“位移 MHz（裸态重叠）”；只列决定
$W_{\rm weak}$ 的谱线及紧邻阈值的主谱线。完整成分在
`pair_channels.json:full_pair.fields.{20.0,160.0}.channels`。

各列物理含义如下：`B` 是量子化轴方向的磁场；“裸通道”是激光 doorway state；
“弱窗主谱线”给本征位移及其 doorway overlap；$W_{\rm weak}$ 是所有弱窗谱线的
overlap 总和；“捕获权重”是局域稀疏求解得到的全部本征态对该裸通道 overlap 之和，
越接近 1 表示未显示在局域谱外的权重越小。

| B/G | 裸通道 | 弱窗主谱线 | $W_{\rm weak}$ | 捕获权重 |
|---:|---|---|---:|---:|
| 20 | `rr` | −84.873 (0.733，刚在阈值外); +45.792 (0.131); +79.058 (0.009) | **0.139987** | 0.996906 |
| 160 | `rr` | −54.543 (0.614); −8.893 (0.304) | **0.918024** | 0.998374 |
| 20 | `rr_garb`（JSON: `r_rgarb`） | −27.993 (0.416); −35.748 (0.374); +51.953 (0.100); +65.417 (0.054) | 0.944124 | 0.998528 |
| 160 | `rr_garb`（JSON: `r_rgarb`） | −13.127 (0.374); +2.928 (0.274); −22.061 (0.215); −0.315 (0.115) | 0.978549 | 0.999592 |

决定性结论是：**160 G 没有恢复单一 scalar blockade channel。** `rr` 的 91.8%
可达谱权重仍在 $\pm83.07$ MHz 内，其中 30.4% 位于 −8.89 MHz。对应本征矢也显示
被旧 16 维 $53P_{3/2}+53P_{3/2}$ 模型排除的 pair sectors：例如 −54.54 MHz
本征态含两种 $53P_{3/2}+53P_{1/2}$ 排列各 14.3%，−8.89 MHz 本征态各含
34.4%。这是显式 $V_{dd}$ 混合产生的成分；当前模型并未加入磁场算符自身在不同
$j$ manifolds 间的非对角混合。

### $R$–方向 pair-potential curves：53P 与 70S benchmark

为展示完整 pair spectrum 随距离与方向的演化，脚本另行扫描并在每个网格点**精确
对角化截断基内的完整矩阵**

$$
H_{\rm pair}(B,R,\theta,\phi)
=H_A(B)+H_B(B)+V_{dd}(R,\theta,\phi),\qquad
H_{\rm pair}|\Psi_k\rangle=E_k|\Psi_k\rangle.
$$

图中纵轴不是绝对能量，而是相对于同一磁场下裸 $|rr\rangle$ 对角能的位移

$$
\Delta_k/h=\frac{E_k-\epsilon_{rr}(B)}{h},\qquad
p_k=|\langle rr|\Psi_k\rangle|^2.
$$

这也给出门动力学中 pair eigenstate 的光学可达性：若第二个原子的驱动项为
$H_L=(\Omega/2)|r\rangle\langle1|+\mathrm{h.c.}$，则

$$
\langle\Psi_k|H_L|1r\rangle
=\frac{\Omega}{2}\langle\Psi_k|rr\rangle.
$$

因此“前五条”按 $R_0=3\ \mu$m 处的 $p_k$ 排名，**不是按 $E_k$ 的代数大小
排名**。从锚点向较大、较小 $R$ 两侧，代码用相邻本征矢重叠
$|\langle\Psi_a(R_i)|\Psi_b(R_{i+1})\rangle|^2$ 的 Hungarian 全局匹配连续追踪；
不会在每个 $R$ 重新排名而把不同分支拼成一条线。只保留锚点
$p_k\ge10^{-6}$ 的亮支，所以 $53P_{3/2}$ 的 $\theta=0^\circ$ 只有一条可见分支，
没有人为补入暗态。

扫描参数来自 `pair_potential_curves.json:params`：$B=20,40,60$ G，
$\theta=0^\circ,15^\circ,\ldots,90^\circ$，$\phi=0$，以及
$R=2.5\text{--}8.0\ \mu$m 的 41 点非均匀网格（含精确的 $R_0=3\ \mu$m）。
量子化轴取 $\mathbf B\parallel\hat z$；当前只有轴向磁场、没有横向外场，故绕
$\hat z$ 的旋转对称性使谱与 $\phi$ 无关，$\phi=0$ 是一般代表。doorway 包括
$53P_{3/2}$ 的全部四个 Zeeman 能级 $m_j=-3/2,-1/2,+1/2,+3/2$，以及 benchmark
$70S_{1/2},m_j=-1/2$。其中只有 $m_j=-3/2$ 是当前 $\sigma^-$ 297 nm 门模型的目标
Rydberg 态；其余三个 53P 图是平行的 pair-spectrum 对照，不是新增的门性能预测。
可视化基组取 $n\pm3$、$l_{\max}=2$、$|\delta|<10$ GHz、只保留 $V_{dd}$；一般方向
维数为 53P 的 164 与 70S 的 544。$\theta=0^\circ$ 时 ARC 利用投影量子数守恒，
$m_j=\pm3/2$ 的 53P 基降为 5 维、$m_j=\pm1/2$ 降为 36 维，70S 降为 111 维。
ARC 3.10.2 会在每个简并 channel 内给对角元加入从 $10^{-8}$ GHz 起递增的数值
tie-breaker；新扫描在对角化前按 ARC 的 channel index 逐项扣除它，以恢复物理裸能
和交换简并。105 例中扣除的最大单态偏置为 $3.60\times10^{-4}$ MHz
（`removed_arc_degeneracy_offset_max_mhz`）。

在 $R=3\ \mu$m 的代表性方向，锚点亮谱如下；每项写为“位移 MHz（$p_k$）”，
完整七角度、41 距离点数据在
`pair_potential_curves.json:manifolds.<state>.fields.<B>.angles.<theta>.curves`。

| 态 | B/G | $\theta$ | $R=3\ \mu$m 的至多五条 $rr$ 亮谱线 | $W_{\rm weak}$ |
|---|---:|---:|---|---:|
| 53P ($m_j=-3/2$) | 20 | 0° | +0.000 (1.000) | 1.000000 |
| 53P ($m_j=-3/2$) | 20 | 45° | −5.346 (0.472); −66.734 (0.276); +25.115 (0.207); +537.356 (0.028); +97.179 (0.010) | 0.959852 |
| 53P ($m_j=-3/2$) | 20 | 90° | −88.379 (0.737); +41.745 (0.125); +537.908 (0.111); +72.468 (0.012); +173.083 (0.006) | 0.136756 |
| 53P ($m_j=-3/2$) | 40 | 0° | +0.000 (1.000) | 1.000000 |
| 53P ($m_j=-3/2$) | 40 | 45° | −37.842 (0.627); +27.141 (0.337); +595.357 (0.023); +183.124 (0.008); +100.561 (0.003) | 0.963697 |
| 53P ($m_j=-3/2$) | 40 | 90° | −73.101 (0.831); +593.919 (0.091); +122.123 (0.061); +257.870 (0.004); −424.406 (0.004) | 0.830566 |
| 53P ($m_j=-3/2$) | 60 | 0° | +0.000 (1.000) | 1.000000 |
| 53P ($m_j=-3/2$) | 60 | 45° | −24.823 (0.839); +61.575 (0.133); +656.127 (0.012); +652.217 (0.006); +292.162 (0.005) | 0.972193 |
| 53P ($m_j=-3/2$) | 60 | 90° | −61.754 (0.863); +654.311 (0.071); +196.408 (0.035); −289.457 (0.013); +407.240 (0.007) | 0.862629 |
| 70S | 20 | 0° | +852.940 (0.532); −499.362 (0.223); −1722.574 (0.173); −378.893 (0.063); −2602.217 (0.002) | 0.000000 |
| 70S | 20 | 45° | +851.378 (0.533); −1730.923 (0.153); −516.426 (0.113); −477.199 (0.102); −426.960 (0.021) | 0.000000 |
| 70S | 20 | 90° | +849.805 (0.534); −1735.188 (0.161); −521.714 (0.122); −453.521 (0.096); −464.848 (0.028) | 0.000000 |
| 70S | 40 | 0° | +861.293 (0.528); −493.130 (0.234); −1709.681 (0.174); −361.911 (0.054); −2586.038 (0.002) | 0.000000 |
| 70S | 40 | 45° | +857.474 (0.530); −1713.523 (0.172); −491.937 (0.079); −491.551 (0.055); −516.945 (0.051) | 0.000000 |
| 70S | 40 | 90° | +853.869 (0.533); −1716.887 (0.171); −502.804 (0.153); −421.248 (0.073); −494.131 (0.022) | 0.000000 |
| 70S | 60 | 0° | +869.768 (0.523); −487.014 (0.244); −1696.897 (0.176); −344.595 (0.048); −2569.875 (0.002) | 0.000000 |
| 70S | 60 | 45° | +864.488 (0.526); −1700.008 (0.175); −502.887 (0.160); −445.784 (0.049); −380.815 (0.034) | 0.000021 |
| 70S | 60 | 90° | +859.733 (0.529); −1703.121 (0.175); −472.756 (0.141); −535.400 (0.094); −378.640 (0.036) | 0.000000 |

四个 53P Zeeman doorway 的紧凑对照如下。各数值都直接取完整本征谱在
$R=3\ \mu$m 处的 $W_{\rm weak}$；它是 10 GHz 可视化基内落入诊断弱移窗的 doorway
overlap 总和，不是门错误率，也不是基组收敛证明。

| $m_j$ | B/G | $\theta=0^\circ$ | $\theta=45^\circ$ | $\theta=90^\circ$ |
|---:|---:|---:|---:|---:|
| $-3/2$ | 20 | 1.000000 | 0.959852 | 0.136756 |
| $-3/2$ | 40 | 1.000000 | 0.963697 | 0.830566 |
| $-3/2$ | 60 | 1.000000 | 0.972193 | 0.862629 |
| $-1/2$ | 20 | 0.272727 | 0.915849 | 0.899985 |
| $-1/2$ | 40 | 0.272727 | 0.909110 | 0.873786 |
| $-1/2$ | 60 | 0.272727 | 0.931604 | 0.917907 |
| $+1/2$ | 20 | 0.889907 | 0.848974 | 0.860419 |
| $+1/2$ | 40 | 0.895354 | 0.920549 | 0.918044 |
| $+1/2$ | 60 | 0.900465 | 0.909708 | 0.844033 |
| $+3/2$ | 20 | 1.000000 | 0.900799 | 0.522692 |
| $+3/2$ | 40 | 1.000000 | 0.885857 | 0.689038 |
| $+3/2$ | 60 | 1.000000 | 0.884511 | 0.016634 |

对应本征态不是单个裸 pair state。下表列出 $B=20$ G、$\theta=45^\circ$、
$R=3\ \mu$m 时当前门目标 $53P_{3/2},m_j=-3/2$ 与 70S benchmark 的前五条分支，
以及每个本征矢中权重最大的两个裸成分；A+B 顺序保留，
完整前四成分见每条 branch 的 `anchor_top_components`。

| 态 | rank | $\Delta_k/h$ / MHz | $p_k$ | 最大两个裸成分（权重） |
|---|---:|---:|---:|---|
| 53P | 1 | −5.346 | 0.472307 | $|53P_{-3/2};53P_{-3/2}\rangle$ (0.472); $|53P_{-1/2};53P_{-1/2}\rangle$ (0.093) |
| 53P | 2 | −66.734 | 0.275828 | $|53P_{-3/2};53P_{-3/2}\rangle$ (0.276); $|53P_{-3/2};53P_{-1/2}\rangle$ (0.209) |
| 53P | 3 | +25.115 | 0.206656 | $|53P_{-3/2};53P_{-3/2}\rangle$ (0.207); $|53P_{-3/2};53P_{-1/2}\rangle$ (0.159) |
| 53P | 4 | +537.356 | 0.027769 | $|53S_{-1/2};54S_{-1/2}\rangle$ (0.398); $|54S_{-1/2};53S_{-1/2}\rangle$ (0.398) |
| 53P | 5 | +97.179 | 0.010014 | $|53P_{+1/2};53P_{+1/2}\rangle$ (0.154); $|53P_{-3/2};53P_{+1/2}\rangle$ (0.149) |
| 70S | 1 | +851.378 | 0.533249 | $|70S_{-1/2};70S_{-1/2}\rangle$ (0.533); $|69S_{-1/2};71S_{-1/2}\rangle$ (0.036) |
| 70S | 2 | −1730.923 | 0.152704 | $|70S_{-1/2};70S_{-1/2}\rangle$ (0.153); $|71S_{-1/2};69S_{-1/2}\rangle$ (0.065) |
| 70S | 3 | −516.426 | 0.113159 | $|69S_{-1/2};71S_{-1/2}\rangle$ (0.140); $|71S_{-1/2};69S_{-1/2}\rangle$ (0.140) |
| 70S | 4 | −477.199 | 0.102028 | $|70S_{-1/2};70S_{-1/2}\rangle$ (0.102); $|69S_{-1/2};71S_{-1/2}\rangle$ (0.087) |
| 70S | 5 | −426.960 | 0.021289 | $|70S_{-1/2};70S_{+1/2}\rangle$ (0.070); $|70S_{+1/2};70S_{-1/2}\rangle$ (0.070) |

**方案 1：固定 $\phi=0$ 的七个方向切片。** 中性灰点保留
$p_k\ge10^{-5}$ 的完整局部亮谱背景；彩色线是在 $R=3\ \mu$m 选出的至多五条连续
分支，颜色只表示锚点 rank。灰点与彩色分支点使用同一个面积映射表示
$p_k=|\langle rr|\Psi_k\rangle|^2$，右下角 size legend 给出 $p_k=0.1,0.5,1.0$
的对应点面积；图中不再另设 overlap 颜色条。灰带是
$|\Delta_k/h|<83.07$ MHz，竖虚线为锚点，纵轴采用对称对数尺度。四个 53P Zeeman
doorway 共用同一纵轴范围，70S 三个磁场图另用一个共同范围，便于同族直接比较。

#### $53P_{3/2},m_j=-3/2$（当前 $\sigma^-$ 门目标）

![53P mj=-3/2 pair-potential curves，B=20 G](pair_potential_53P3_2_B20G.png)

![53P mj=-3/2 pair-potential curves，B=40 G](pair_potential_53P3_2_B40G.png)

![53P mj=-3/2 pair-potential curves，B=60 G](pair_potential_53P3_2_B60G.png)

#### $53P_{3/2},m_j=-1/2$

![53P mj=-1/2 pair-potential curves，B=20 G](pair_potential_53P3_2_mj_m1_2_B20G.png)

![53P mj=-1/2 pair-potential curves，B=40 G](pair_potential_53P3_2_mj_m1_2_B40G.png)

![53P mj=-1/2 pair-potential curves，B=60 G](pair_potential_53P3_2_mj_m1_2_B60G.png)

#### $53P_{3/2},m_j=+1/2$

![53P mj=+1/2 pair-potential curves，B=20 G](pair_potential_53P3_2_mj_p1_2_B20G.png)

![53P mj=+1/2 pair-potential curves，B=40 G](pair_potential_53P3_2_mj_p1_2_B40G.png)

![53P mj=+1/2 pair-potential curves，B=60 G](pair_potential_53P3_2_mj_p1_2_B60G.png)

#### $53P_{3/2},m_j=+3/2$

![53P mj=+3/2 pair-potential curves，B=20 G](pair_potential_53P3_2_mj_p3_2_B20G.png)

![53P mj=+3/2 pair-potential curves，B=40 G](pair_potential_53P3_2_mj_p3_2_B40G.png)

![53P mj=+3/2 pair-potential curves，B=60 G](pair_potential_53P3_2_mj_p3_2_B60G.png)

#### $70S_{1/2},m_j=-1/2$ benchmark

![70S benchmark pair-potential curves，B=20 G](pair_potential_70S1_2_B20G.png)

![70S benchmark pair-potential curves，B=40 G](pair_potential_70S1_2_B40G.png)

![70S benchmark pair-potential curves，B=60 G](pair_potential_70S1_2_B60G.png)

**直接比较：** 在该 10 GHz 截断内，70S 在这些代表点的
$W_{\rm weak}\le2.12\times10^{-5}$，更接近强移 benchmark。53P 不存在一个对所有
磁场和方向都单调更优的 Zeeman 选择：例如 $m_j=+3/2$ 在 60 G、
$\theta=90^\circ$ 时为 0.016634，但同一磁场的 $\theta=0^\circ$ 仍为 1；
$m_j=+1/2$ 在表中九个代表点则均为 0.844--0.921。这说明 53P 的可达弱移谱权重
强烈依赖方向、磁场和 doorway，但不构成 70S 或任一 53P 基组的收敛证明。

### 仍然保留的近似与对门结果的含义

- 30 GHz 定量审计仍只计算 $R=3\ \mu$m，且尚未对 `n_range`、$l_{\max}$ 与能窗
  做收敛扫描。新增曲线虽扫描 $R=2.5\text{--}8.0\ \mu$m，却采用更小的
  $n\pm3$/10 GHz 可视化基组；它用于展示分支、avoided crossings 与角度趋势，
  **不替代**上面的 30 GHz 单点数值。比如 20 G、$\theta=90^\circ$ 的主线在两者中
  分别为 −84.873 与 −88.379 MHz。
- 只保留 dipole-dipole coupling。ARC 使用弱场线性 paramagnetic Zeeman；忽略
  diamagnetic、hyperfine，以及磁场算符自身在不同 $j$ manifolds 间的非对角混合。
- 求的是目标裸能附近的稀疏局域谱，不是全部 2276 个本征态；不过弱窗已完整包围，
  且四个裸通道捕获权重都超过 0.996。
- 新增 10 GHz 曲线在每个网格点对 5/36/164（53P，依 $m_j$ 与方向而异）或
  111/544（70S）维截断矩阵做完整对角化，因而该截断内 $rr$ 捕获权重为 1；
  105 个案例的最大本征方程残差为
  $5.23\times10^{-11}$ MHz，最小相邻分支匹配重叠为 0.438
  （分别见 `curves.max_eigensystem_residual_mhz` 与
  `curves.branches[*].min_adjacent_match_overlap`）。这只验证数值对角化和
  branch tracking，不验证物理截断收敛。
- 门动力学仍是原四能级、对角 scalar-interaction 模型；本次只审计 pair spectrum，
  尚未把显式通道接入时间演化或重新优化脉冲。

因此主表 20 G 的 $9.173\times10^{-4}$ 与四能级 160 G 重抛光的
$2.369\times10^{-6}$ 都应标记为**相互作用模型未验证的优化输出**，不能解释为显式
pair physics 下的门误差。下一步必须先把这些可达 pair eigenchannels 接入动力学，再
重新评估或优化门；单纯继续增大 $B$ 不是本次结果支持的修复方案。
