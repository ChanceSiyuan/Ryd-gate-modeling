# 297_laser_noise — 两台 297 nm 激光器的实测频率噪声谱，及其上建立的门误差模型

## 这是什么

297 nm 单光子 CZ 门要求激光的相位噪声小到不破坏门保真度。两台候选光源在**基频**
（1180/1187 nm）下做了表征，其相噪图被数字化为单边频率噪声幅度谱密度
$\sqrt{S_{\delta\nu}}$（单位 Hz/$\sqrt{\text{Hz}}$）。

本目录同时存放**输入数据**（数字化的谱）和消费这些数据的研究的**分析记录**。研究**尚未完成**：
代码已建成并验证，试点已跑，但完整网格尚未运行。下文标注"实测"的都是真实运行得到的数字，
标注"估计"的不是。

---

# 第一部分 — 模型（end-to-end）

## 1.1 噪声如何进入哈密顿量

带光学相位噪声 $\varphi_n(t)$ 的激光驱动 $|1\rangle \to |r\rangle$ 支路：

$$
\Omega(t)\,\exp\!\big[\,i\,(\varphi_c(t) + \varphi_n(t))\,\big]
$$

其中 $\varphi_c$ 是**指令相位**（协议的脉冲整形）。做幺正变换
$V = \exp\!\big[+i\,\varphi_n(t)\,\hat{N}_r\big]$ 把 $\varphi_n$ 从驱动项中移除，代价是留下一个对角项：

$$
\boxed{\;\hat H(t) \;\longrightarrow\; \hat H_0(t) \;+\; 2\pi\,\delta\nu(t)\,\hat N_r\;},
\qquad
\delta\nu(t) \equiv \frac{1}{2\pi}\frac{d\varphi_n}{dt}
$$

$\hat N_r$ 数的是处于 Rydberg 态的原子数 —— **$r$ 和 $r_{\rm garb}$ 都要算**，因为噪声移动的是
物理能级，与模型给它贴什么基矢标签无关。$\delta\nu(t)$ 是瞬时频率偏移（Hz），其单边功率谱为
$S_{\delta\nu}(f) = f^2 S_\varphi(f)$。

这一步是**严格的**，不是近似：频率噪声只耦合到 Rydberg 布居算符，不耦合到别的东西。

## 1.2 误差度量

把含噪态在无噪态附近展开，$|\psi\rangle = |\psi_0\rangle + |\chi_1\rangle + |\chi_2\rangle + \cdots$，其中

$$
|\chi_1\rangle \;=\; -2\pi i \int_0^T \!dt\; \delta\nu(t)\,|A(t)\rangle,
\qquad
|A(t)\rangle \equiv \hat U_0(T,t)\,\hat N_r\,|\psi_0(t)\rangle
$$

我们报告的量是相对无噪末态的**噪声诱导保真度损失**，并对四个逻辑输入取最大：

$$
\varepsilon_{\rm phase} \;=\; \max_s\Big[\,1 - \big|\langle \psi_0^s(T)\,|\,\psi^s(T)\rangle\big|^2\,\Big]
$$

展开到二阶后，它会收缩成一个非常干净的形式。两个事实起了关键作用：

**(i) 不存在一阶项。** 因为
$\langle\psi_0(T)|A(t)\rangle = \langle\psi_0(t)|\hat N_r|\psi_0(t)\rangle$
是**实数**（它是一个布居数），所以 $\langle\psi_0|\chi_1\rangle$ 是纯虚数，
$\mathrm{Re}\,\langle\psi_0|\chi_1\rangle = 0$ —— 而且是对**每一次实现**成立，不只是系综平均。

**(ii) 幺正性免费提供了 $\chi_2$。** 模长守恒在 $\delta\nu^2$ 阶给出
$2\,\mathrm{Re}\langle\psi_0|\chi_2\rangle = -\|\chi_1\|^2$。把两者代入 $|\langle\psi_0|\psi\rangle|^2$：

$$
\boxed{\;\varepsilon \;=\; \|\chi_1\|^2 - \big|\langle\psi_0(T)|\chi_1\rangle\big|^2 \;=\; \big\|\,\hat Q\,\chi_1\big\|^2\;},
\qquad
\hat Q \equiv \mathbb{1} - |\psi_0(T)\rangle\langle\psi_0(T)|
$$

也就是说，这个度量**按构造就是二阶严格的，完全不需要 $\chi_2$ 的任何机制**。这正是选它的理由。

## 1.3 为什么那个"显然的"替代方案是错的

最初的设计用的是**泄漏**可观测量的二阶增量 $\Delta\langle L\rangle$，其中
$\hat Q_L = \sum_q |q\rangle\langle q|$ 对非逻辑态求和。这是**可证明错误**的：

$$
\Delta\langle L\rangle \;=\; \underbrace{\langle\chi_1|\hat Q_L|\chi_1\rangle}_{\text{保留了}}
\;+\; \underbrace{2\,\mathrm{Re}\langle\psi_0|\hat Q_L|\chi_2\rangle}_{\textbf{被丢掉了，同阶}}
$$

一个**不需要任何模拟**的反驳：取 $\hat Q_L = \mathbb{1}$。模长守恒使真实变化恒为零，
而该公式返回 $\|\chi_1\|^2 > 0$。

两个文献校验都没抓到它，因为它们测的都是相对无噪末态的 infidelity，即
$\hat Q|\psi_0\rangle = 0$ —— 恰好湮灭了被丢掉的那一项。这个错误是靠真实门上的直接蒙特卡洛
发现的（20 点中只过 7 点，最坏比值为**负**），并通过采用 §1.2 修复。
**这是整个研究中后果最严重的一处修正。**

## 1.4 滤波函数 —— 这件事为什么算得动

对 $|A(t)\rangle$ 在门窗口内做傅里叶变换：

$$
|G(f)\rangle \;=\; \int_0^T |A(t)\rangle\, e^{-2\pi i f t}\,dt
$$

对噪声系综求平均，得到闭式：

$$
\boxed{\;
\varepsilon_{\rm phase} \;=\; 2\pi^2 \int_{-\infty}^{\infty} S_{\delta\nu}(|f|)\,
\Big[\, \big\|G(f)\big\|^2 - \big|\langle\psi_0(T)|G(f)\rangle\big|^2 \,\Big]\,df
\;}
$$

有两条性质让整个 campaign 变得可负担：

- **$G(f)$ 与激光无关。** 它是**门本身**的属性。所以每个网格点只求解**一次**，把分箱后的核
  $K_b = \int_{\rm bin}\big(\|\hat QG(f)\|^2 + \|\hat QG(-f)\|^2\big)df$ 存下来；此后每一组
  （激光 × 外推 × $f_{\min}$）组合都只是对已存分箱做一次加权求和 —— **秒级，不动求解器**。
  直接蒙特卡洛则需要每点约 200 次求解 × 12168 点。
- **投影项是免费的。** $\langle\psi_0(T)|A(t)\rangle = \langle\psi_0(t)|\hat N_r|\psi_0(t)\rangle$
  已经是前向求解的副产品。

计算 $G$ 需要伴随态 $|\phi_q(t)\rangle = \hat U_0(t,T)|q\rangle$ 覆盖**完整的** 16 维基矢，
由一次 16 列的后向求解加上 4 个逻辑输入的前向求解得到 —— 共 20 列，对比无噪 pass 的 3 列。

传播子被**刻意地从不构造**。$\phi_q$ 与 $\psi_s$ 服从同一个方程，且 $\hat N_r$ 是对角的，
所以 $e^{-iD_i t}$ 因子逐点相消，只剩下驱动尺度（$\lesssim 50$ MHz）的结构。若改由传播子构造
$A(t)$，就会重新引入 $e^{i(D_j - D_s)t}$ 交叉项 —— GHz 量级的 Rydberg 对相互作用和 6.8 GHz 的
$|0\rangle$ 超精细偏移 —— 从而要求约 70 ps 的采样。

## 1.5 会改变结果的约定

| 约定 | 取值 | 弄错的后果 |
|---|---|---|
| **单边 vs 双边谱** | 本 repo 为**单边**：$\sigma_\nu^2 = \int_0^\infty S_{\delta\nu}df$。PRA 107.042611 为双边。 | 直接照抄论文的 $h_0$ 会让**所有**结果高一倍。已对其 Eq. 79 闭式解确认。 |
| **倍频** | 297 nm 是 ~1188 nm 的**四倍频**。光学相位 ×4 $\Rightarrow S_\varphi \times 16 \Rightarrow S_{\delta\nu}\times 16$。 | 差 16 倍。 |
| **Rydberg 计数** | $\hat N_r$ 同时数 $r$ **和** $r_{\rm garb}$。 | 低估噪声耦合。 |
| **后向支路相位** | 伴随支路需要**共轭**的相位恢复（$e^{+ic\tau}$，而前向支路恢复 $e^{-ict}$）。 | 重叠不变量从 $2.4\times10^{-11}$ 劣化到 $2.5\times10^{-5}$。 |

## 1.6 不可省略的数值要求

**条纹分辨。** $\|\hat QG(f)\|^2$ 带有宽度为 $1/T$ 的 sinc 条纹结构。每十倍频程 $p$ 个点的
对数网格间距为 $\Delta f = f\ln 10 / p$，因此要在频带顶端分辨条纹就要求

$$
p \;\ge\; \ln 10 \cdot f_{\max} \cdot T
$$

即 $T = 1\,\mu$s 时 461 点/十倍频程，$T = 4.5\,\mu$s 时升到 **2073**。原始设计里那个固定的
200 会让 $4.5\,\mu$s 的核**高出约 13%**。这一点由一个**不需要跑 ODE** 的 Parseval 检验把关：
对单频音，$\sum_b K_b$ 必须严格等于 $T$。旧的默认值在 50 MHz 处返回 $1.41\,T$（网格落在条纹峰上），
在 150 MHz 处返回 $0.093\,T$（落在零点上）—— 这个检验对两个方向都灵敏。

**准静态分割。** 轨迹生成把 $f_{\rm split}$ 以下的频带塌缩成一个静态偏移。这相当于用 $|G(0)|^2$
替代真实的 $|G(f)|^2$，只在 $f \ll 1/T$ 时成立，所以分割点取 $0.01/t_{\rm gate}$，**而不是**
$1/t_{\rm gate}$。取 $1/t_{\rm gate}$ 时蒙卡/闭式比值在 $n_{\rm rot}=0.5$ 处是 1.617、
在 $n_{\rm rot}=1.0$ 处是 0.394 —— **变号**，因为在整数转数处 $|G(0)|^2 = 0$。
取 $0.01/t_{\rm gate}$ 后比值为 1.0002 / 1.0001。

## 1.7 验证

| 校验 | 结果 |
|---|---|
| PRA 107.042611 Eq. 79 白噪声闭式解 $\varepsilon = \pi^3 h_0 N/\Omega_0$（单边取半） | 符合到 **−0.008% / +0.001%** |
| 静态失谐响应 $G(0)$ 对比门的实测响应 | **7 位有效数字** |
| 真实门上的直接蒙特卡洛，20 网格点 × 200 次实现 | **20/20 通过**，比值 **0.895–1.091** |

蒙卡的接受判据是"4 倍标准误 或 10%，取较宽者"。一个诚实的说明：由于
$\mathrm{se}/\mathrm{pred}\approx 7\%$，40 个 cell 中 4σ 那一支始终是较宽的，所以**实际生效**的
门限是约 25–30%。10% 量级的一致性是真实测得的结果，但它是一个观察，而不是测试所强制的。

---

# 第二部分 — 目前的结果

**状态：仅试点。** 32 个网格点，$n=60$，$T \in \{1.0, 4.5\}\,\mu$s。完整的 12168 点网格尚未运行。
本部分全部来自这 32 个点和那份 20 点蒙卡记录，**必须在完整网格上重新推导**。

## 2.1 量级

来自 20 点蒙卡记录（ECDL，flat 外推 —— 最保守的角落）：

$$
\varepsilon_{\rm phase} \;=\; 7.6\times10^{-3} \;\ldots\; 1.3\times10^{-1}
$$

与同一批点上的无噪声相干泄漏相比，这是**中位数 26 倍**，最坏 **2200 倍**。
在这个噪声模型下，297 的误差预算由相位噪声主导，相干泄漏已经不再是瓶颈。

在这些点上 $\varepsilon_{\rm phase}$ 大致按 $T^{0.69}$ 增长，而无噪声泄漏随 $T$ **下降** ——
两者反向，因此存在真实的最优点。无噪声最优点（$n=64$，$T=2.5\,\mu$s，$9.1\times10^{-6}$）
只在最乐观的噪声模型下存活；在另外三个模型下它移向**更短**的 $T$
（$n=73$，$T=1.0\,\mu$s，$\Omega/2\pi = 16.5$ MHz，$D_{\rm sw} = 25$ MHz）。

## 2.2 开放问题 A —— 答案被一个无法检验的假设括住

两条谱都止于 **1 MHz**。而门的滤波函数峰值在 $0.9\text{–}2.8\times10^{7}$ Hz，
比测量边界高一个数量级。从 1 MHz 到 200 MHz 全部是外推，两种括号选择 ——
`flat`（保持边界值）与 `power`（延续拟合的末段斜率）—— 在核峰处相差约 15 倍。

核峰（$\sim 2\times10^7$ Hz）处的 $S_{\delta\nu}$，以及由此折算的 $\varepsilon_{\rm phase}$：

| 模型 | $S_{\delta\nu}$ | $\varepsilon$ 相对值 | 折算的 $\varepsilon_{\rm phase}$ |
|---|---|---|---|
| ECDL / flat | 2444 | 1 | $7.6\times10^{-3} \ldots 1.3\times10^{-1}$ ← **实测** |
| ECDL / power | 175 | 0.072 | $5.5\times10^{-4} \ldots 9.4\times10^{-3}$ |
| seed / flat | 103 | 0.042 | $3.2\times10^{-4} \ldots 5.6\times10^{-3}$ |
| seed / power | 4.4 | 0.0018 | $1.4\times10^{-5} \ldots 2.4\times10^{-4}$ |

只有第一行是算出来的，其余是按 $\varepsilon \propto S_{\delta\nu}$ 折算。跨度达**四个数量级**。

在试点的这些点上，$\varepsilon_{\rm phase}$ 中真正来自**实测**谱的比例：

| 模型 | 最小 | 中位数 | 最大 |
|---|---|---|---|
| ECDL / flat | 0.00% | **0.14%** | 51.8% |
| ECDL / power | 0.01% | 2.02% | 94.1% |
| seed / flat | 0.02% | 9.51% | 99.4% |
| seed / power | 0.40% | **74.2%** | 100.0% |

这是自洽的而非矛盾的：`flat` 假设了一个高的宽带底噪，于是外推频段主导；
`power` 假设了一个低的底噪，于是实测频段主导。令人不安的是它的推论 ——
**括号的保守端几乎完全是一个假设**，而且在这份数据上再怎么计算也无法收窄它。

> **唯一能解决这件事的是把 $S_{\delta\nu}$ 测到 1 MHz 以上**，或者给出一个上界。
> 一次做到 ~50 MHz 的拍频测量，就能把四个数量级的不确定性压缩到一个。

## 2.3 开放问题 B —— 低频截断决定**哪台激光器赢**

在真实试点核上计算，$\varepsilon_{\rm phase}$ 随积分截断 $f_{\min}$ 的变化：

| 模型 | $f_{\min}=1$ Hz | 10 Hz | 100 Hz | 1 kHz | @10 Hz 的变化 |
|---|---|---|---|---|---|
| ECDL / flat | $3.271\times10^{-2}$ | $3.271\times10^{-2}$ | $3.271\times10^{-2}$ | $3.271\times10^{-2}$ | **−0.002%** |
| ECDL / power | $2.452\times10^{-3}$ | $1.559\times10^{-3}$ | $1.439\times10^{-3}$ | $1.439\times10^{-3}$ | −36.4% |
| seed / flat | $1.095\times10^{-2}$ | $1.383\times10^{-3}$ | $1.383\times10^{-3}$ | $1.383\times10^{-3}$ | **−87.4%** |
| seed / power | $1.068\times10^{-2}$ | $3.210\times10^{-4}$ | $1.810\times10^{-4}$ | $8.842\times10^{-5}$ | **−97.0%** |

机制是两台激光器在频域上**交叉**了：

$$
\begin{aligned}
S_{\delta\nu}(1\ \text{Hz}): &\quad \text{ECDL } 8.17\times10^{5} \;<\; \text{seed } 2.01\times10^{6}
&&\text{（seed 差 2.5 倍）}\\
S_{\delta\nu}(1\ \text{MHz}): &\quad \text{ECDL } 197.7 \;>\; \text{seed } 40.65
&&\text{（seed 好 4.9 倍）}
\end{aligned}
$$

所以在 $f_{\min} = 1$ Hz 时，seed 的慢漂移把它真正的优势盖住了 —— 在 `flat` 下 seed/ECDL 只有
**3.0 倍**。在 $f_{\min} = 10$ Hz 时是 **23.6 倍**，正好等于核峰处 $S_{\delta\nu}$ 比值的预测。
**截断的选择使激光器对比移动约 8 倍，对 seed 而言这比 flat/power 括号本身还大。**

从物理上看，$1/T$ 为 0.22–1.0 MHz，所以约 100 kHz 以下的一切对这个门都是**准静态**的 ——
那是一个静态失谐误差，并不真的是"相位噪声"，伺服锁会把它去掉。1 Hz 那一端到底该不该计入
$\varepsilon_{\rm phase}$，取决于锁的带宽以及实验上门是怎么标定的。

> **这是一个实验输入，不是模型能自行裁定的。** 两种截断都会出图；结果笔记不得默默选一个。

## 2.4 成本 —— 实测，非估计

在 $T$ 轴两端各跑 16 点，20 workers，`--batch-size 15`：

| | $T = 1.0\,\mu$s | $T = 4.5\,\mu$s | 比值 |
|---|---|---|---|
| 16 点墙钟 | 1.8 min | 7.8 min | **4.3×** |
| 单点（batch 内） | 7.2 s | 31 s | |
| 单 worker 峰值 RSS | — | **3.58 GB** | |

预测的 `kernel_fine_per_decade` 比值是 $2073/461 = 4.5$，因此 $T$ 依赖的成本模型在真实门上
**得到确认**。按 9 点 $T$ 轴外推（均值 17.9 s/点）乘以 $13\times13\times72 = 12168$ 点，
得 ~60 core-h $\Rightarrow$ **20 workers 约 3 小时**，峰值约 72 GB / 可用 244 GB。

给后来者的提醒：panel 索引是 $(n_{\rm idx}, t_{\rm idx})$，所以计划里那个示例 panel `3,0`
是网格中**最便宜**的格子，在那儿跑什么也验证不了成本。

---

# 第三部分 — 功率换算

$\Omega \propto \sqrt{P/A}$，于是由 ARC 计算的"1 W 打在标称光斑面积上的单原子 Rabi 频率"可得

$$
P_{\rm at\,atoms} \;=\; \Big(\frac{\Omega}{\Omega_{1\rm W}}\Big)^{2}\cdot 1\ \text{W},
\qquad
P_{\rm nominal} \;=\; \frac{P_{\rm at\,atoms}}{1 - \text{loss}}
$$

其中 beam area $=420\ \mu\text{m}^2$（$=7\,\mu\text{m}\times 20\times$ spacing），
optics loss $=0.8$，即标称功率只有 20% 到达原子。结果缓存在 `omega_per_watt.npz`，
使画图永不调用 ARC。

**达到给定峰值 $\Omega_{297}/2\pi$ 所需的标称功率（W）：**

| $n$ | $\Omega_{1\rm W}$ (MHz) | 9.0 | 11.0 | 13.5 | 15.0 | 16.5 | 18.0 MHz |
|---|---|---|---|---|---|---|---|
| 50 | 23.53 | 0.73 | 1.09 | 1.65 | 2.03 | 2.46 | 2.93 |
| 53 | 21.45 | 0.88 | 1.32 | 1.98 | 2.45 | 2.96 | 3.52 |
| 56 | 19.66 | 1.05 | 1.57 | 2.36 | 2.91 | 3.52 | 4.19 |
| 60 | 17.63 | 1.30 | 1.95 | 2.93 | 3.62 | 4.38 | 5.21 |
| 64 | 15.92 | 1.60 | 2.39 | 3.59 | 4.44 | 5.37 | 6.39 |
| 68 | 14.48 | 1.93 | 2.89 | 4.35 | 5.37 | 6.49 | 7.73 |
| 71 | 13.53 | 2.21 | 3.30 | 4.98 | 6.14 | 7.43 | 8.85 |
| 73 | 12.96 | 2.41 | 3.60 | 5.43 | 6.70 | 8.11 | 9.65 |

已与 `docs/superpowers/specs/2026-07-24-max-leakage-297-sweep-design.md` 中的独立记录交叉核对
（"1–3 W 标称，0.8 损耗 / 420 $\mu$m² $\to$ 53P 上约 9.6–16.6 MHz"）：本表给出 9.59 / 16.61 MHz。

**seed 那台传闻中只有一半功率，就是在这里起作用的。** 功率减半在 $\Omega$ 上的代价是 $\sqrt{2}$：
在 $n=73$ 上，16.5 MHz 需要 8.11 W 标称，其一半只能给出 11.7 MHz。seed 的 24 倍噪声优势
是否扛得住它 2 倍的功率劣势 —— 这正是 campaign 的图要裁定的问题，**无法只从这张表读出**，
因为 $\varepsilon_{\rm phase}$ 和相干泄漏都依赖于 $\Omega$。

---

# 第四部分 — 数据与溯源

## 已被推翻的结论

本文件上一版中的两条主张**撤回**：

- *"seed 激光器赢约 38 倍"*，出自 `scripts/laser_noise_psd.py` 打印的 `required power` 代理量。
  该代理量早于滤波核研究。真实倍数是 **3.0 倍或 23.6 倍**，取决于 $f_{\min}$（§2.3）——
  不说明截断和外推，这个问题本身就不是良定义的。
- 认为 $\sigma_\nu(<1\ \text{MHz})$ 能衡量哪台激光器更好的那个框架。它不能：门的加权是
  $\|\hat QG(f)\|^2$，峰值在 $\sim 2\times10^7$ Hz，而非在 1 MHz 以下均匀分布。
  $\sigma_\nu(<1\ \text{MHz})$ 为 172.7 kHz（ECDL）对 383.9 kHz（seed），恰恰就是 §2.3 证明可以
  被去除的那个低频漂移。

该版中仍然准确的部分：拟合指数（ECDL 0.4626，seed 0.5536）、边界频率、六个数量级的实测范围、
以及数字化误差带列。

## 文件

| 路径 | 内容 | 是否跟踪 |
|---|---|---|
| `psd_ECDL.csv` | 1586 行：`f_Hz, asd_mean_Hz_per_rtHz, asd_lo_…, asd_hi_…` | 是 |
| `psd_seed.csv` | 1161 行，同样的列 | 是 |
| `psd_model.json` | 每台激光：`csv`、`f_edge_hz`、`harmonic`、`power_law_exponent`、`s_dnu_edge_297` | 是 |
| `omega_per_watt.npz` | ARC Rabi-per-watt 缓存，`ryd_n` (8,) 与 `omega_mhz_at_1w` (8,) | 是 |
| `ECDL_phasenoise.png`, `seed_lasernoise.png` | 数字化源图 —— **无法再生成** | 是 |
| `psd_model.pdf`, `psd_model.png` | 生成的图 | 否 |

本 repo 中所有谱密度都是**单边**的：$\sigma_\nu^2 = \int_0^\infty S_{\delta\nu}(f)\,df$。

$\sigma_\nu(1\ \text{Hz}, 200\ \text{MHz})$：ECDL **718.4 kHz**（flat）/ 213.7 kHz（power）；
seed 409.8 / 384.4 kHz。对 13.5 MHz 的驱动，ECDL/flat 那个数给出 $\sigma_\nu/\Omega = 0.053$ ——
§1.2 的微扰展开是二阶的，所以预测值高于约 0.1 时已在其适用区之外，会被**标记**而不是直接引用。

## 图

![digitized ECDL phase noise](ECDL_phasenoise.png)
ECDL 的源图，数字化为 `psd_ECDL.csv`。

![digitized seed laser noise](seed_lasernoise.png)
seed 激光器的源图，数字化为 `psd_seed.csv`。

![fitted PSD model](psd_model.png)
两条实测谱及其拟合的幂律延拓，已缩放到 297 nm。

## 代码

| 组件 | 位置 |
|---|---|
| 谱模型、轨迹生成、滤波核 | `src/ryd_gate/phase_noise.py` |
| 数字化与作图 | `scripts/laser_noise_psd.py` |
| `filter` pass、$\varepsilon_{\rm phase}$ 作图、功率表 | `scripts/max_leakage_297_sweep.py` |
| 蒙特卡洛验证 | `scripts/phase_noise_mc_check.py` |
| 验证记录 | `results/max_leakage_297/a3.0/reports/phase_noise_mc.json` |

设计与计划记录：
`docs/superpowers/specs/2026-07-30-laser-phase-noise-design.md`、
`docs/superpowers/plans/2026-07-30-laser-phase-noise.md`。

## 复现

```bash
# 数字化谱与模型图（不接受任何参数；会重写全部三个数据文件，
# 因为数字化的采样点写在脚本内部）。
uv run python scripts/laser_noise_psd.py

# 单个 panel 的滤波核（n 索引, T 索引）—— panel 成本见 §2.4。
uv run python scripts/max_leakage_297_sweep.py filter --level 4 --panels 3,8 \
    --workers 20 --batch-size 15

# 出一张图，指定激光器 / 外推 / 截断。
uv run python scripts/max_leakage_297_sweep.py plot --metric eps_phase \
    --laser seed --extrapolation power --f-min 10
```

2026-07-31 验证：重跑 `laser_noise_psd.py` 会逐字节复现全部三个数据文件，
因此即使输出是被跟踪的，这条命令也可以安全执行。
