# 297_laser_noise — 297 nm 激光频率噪声对单光子 CZ 门保真度的影响

## 问题与模型

297 nm 单光子 CZ 门用一个整形脉冲直接驱动 $|1\rangle\leftrightarrow|r\rangle$。
两台候选光源在基频 1180/1187 nm 下完成表征，源图被数字化为单边频率噪声谱
$S_{\delta\nu}(f)$。

本研究回答一个问题：同一个门过程在关闭和加入实测激光频率噪声时，末态保真度相差多少。
相位噪声滤波核目前只覆盖 32 个试点门点，因此结果是试点结论，不是完整网格最优值。

带噪激光的驱动项为

$$
\Omega(t)\exp\!\big[i\big(\varphi_c(t)+\varphi_n(t)\big)\big] .
$$

$\varphi_c$ 是控制相位，$\varphi_n$ 是随机光学相位。变换
$V=\exp[i\varphi_n(t)\hat N_r]$ 消去驱动项中的 $\varphi_n$，并得到

$$
\boxed{\;
\hat H(t)=\hat H_0(t)+2\pi\delta\nu(t)\hat N_r
\;},
\qquad
\delta\nu(t)=\frac{1}{2\pi}\frac{d\varphi_n}{dt} .
$$

$\hat N_r$ 统计处于 $r$ 和 $r_{\rm garb}$ 的原子数。物理上，频率噪声产生随机 Rydberg 失谐；
脉冲暂时布居 Rydberg 态时积累随机相位，回到逻辑空间后便表现为相对无噪末态的保真度损失。
$|00\rangle$ 不参与 297 nm 驱动，因此对该噪声是暗态。

| 约定 | 本仓库采用的定义 |
|---|---|
| 频谱 | 单边谱：$\sigma_\nu^2=\int_0^\infty S_{\delta\nu}(f)\,df$ |
| 倍频 | 297 nm 是约 1188 nm 的四倍频，因此 $S_{\delta\nu}$ 乘 16 |
| 噪声算符 | $\hat N_r$ 同时统计 $r$ 与 $r_{\rm garb}$ |
| 适用区 | 模型保留噪声二阶项；$\varepsilon_{\rm phase}>0.1$ 时不再引用微扰结果 |

## 理论推导与实际计算

对逻辑输入 $s$，无噪态与含噪态从同一初态出发：

$$
i\,\partial_t|\psi_0^s(t)\rangle
=\hat H_0(t)|\psi_0^s(t)\rangle,
\qquad
i\,\partial_t|\psi^s(t)\rangle
=\big[\hat H_0(t)+2\pi\delta\nu(t)\hat N_r\big]|\psi^s(t)\rangle .
$$

令 $|\psi^s\rangle=|\psi_0^s\rangle+|\chi_1^s\rangle+O(\delta\nu^2)$。保留噪声的一阶项：

$$
i\,\partial_t|\chi_1^s(t)\rangle
=\hat H_0(t)|\chi_1^s(t)\rangle
+2\pi\delta\nu(t)\hat N_r|\psi_0^s(t)\rangle,
\qquad
|\chi_1^s(0)\rangle=0 .
$$

用无噪传播子把每个时刻产生的扰动传播到末态：

$$
|\chi_1^s(T)\rangle
=-2\pi i\int_0^T\!dt\;\delta\nu(t)|A_s(t)\rangle,
\qquad
|A_s(t)\rangle
=\hat U_0(T,t)\hat N_r|\psi_0^s(t)\rangle .
$$

定义无噪末态正交补上的投影

$$
\hat Q_s
=\mathbb 1-|\psi_0^s(T)\rangle\langle\psi_0^s(T)| .
$$

对任意归一化的含噪末态都有

$$
1-\big|\langle\psi_0^s(T)|\psi^s(T)\rangle\big|^2
=\langle\psi^s(T)|\hat Q_s|\psi^s(T)\rangle
=\big\|\hat Q_s|\psi^s(T)\rangle\big\|^2 .
$$

由于 $\hat Q_s|\psi_0^s(T)\rangle=0$，最低非零阶为

$$
\boxed{\;
\varepsilon_s^{(2)}
=\big\|\hat Q_s\chi_1^s(T)\big\|^2
=\|\chi_1^s(T)\|^2
-\big|\langle\psi_0^s(T)|\chi_1^s(T)\rangle\big|^2
\;} .
$$

$\hat Q_s\chi_1^s$ 是噪声把末态推出理想态射线的分量；一阶平行分量只改变全局相位。
因此 $\varepsilon_s^{(2)}$ 正是该噪声引起的保真度损失。

对门响应作傅里叶变换：

$$
|G_s(f)\rangle
=\int_0^T|A_s(t)\rangle e^{-2\pi i f t}\,dt .
$$

对零均值平稳噪声作系综平均，并使用单边谱约定：

$$
\boxed{\;
\bar\varepsilon_s
=2\pi^2\int_{-\infty}^{\infty}
S_{\delta\nu}(|f|)\big\|\hat Q_sG_s(f)\big\|^2\,df
\;},
\qquad
\varepsilon_{\rm phase}=\max_s\bar\varepsilon_s .
$$

实际计算先为每个门点和逻辑输入生成与激光谱无关的分箱核

$$
K_{s,b}
=\int_{\rm bin}
\left(
\|\hat Q_sG_s(f)\|^2+\|\hat Q_sG_s(-f)\|^2
\right)df ,
$$

再计算

$$
\bar\varepsilon_s
\approx2\pi^2\sum_bS_{\delta\nu}(f_b)K_{s,b} .
$$

因此每个门点只求解一次；更换激光器或带外假设时，只重新加权已存核。
实现按逻辑输入把相干泄漏、散射和 $\bar\varepsilon_s$ 相加，再取最坏输入。

## 核心结果

两台激光器都使用 1 Hz–1 MHz 的数字化实测谱、四倍频缩放和固定的 $f_{\min}=1$ Hz。
测量上边界以上采用 `power` 带外假设：延续最后一个实测十倍频程的下降斜率；
若幅度谱 $\sqrt{S_{\delta\nu}}\propto f^{-p}$，则功率谱 $S_{\delta\nu}\propto f^{-2p}$。
拟合指数为 ECDL $p=0.4626$、seed $p=0.5536$，边界频率分别为 995.7 kHz 与 1.000 MHz。

![两条实测谱及带外延拓](psd_model.png)

（图中同时画出了平坦延拓作为对照；本报告的所有数值均取 `power`。）

2026-07-31 只读重算将 32 个滤波试点与硬件限制 $D_{\rm sw}\le20$ MHz 相交，得到 24 个可实施点。
下表对每种噪声假设选择 `total_error_phase` 最小的试点门点。

$F_0$ 是不含激光相位噪声的最坏输入保真度预算；$F_{\rm noise}$ 在同一点加入相位噪声。
两者均含原有相干泄漏和散射，$\Delta F=F_0-F_{\rm noise}$。

| 激光器 | 试点最优门点 $(n,T,\Omega/2\pi,D_{\rm sw})$ | $F_0$ | $F_{\rm noise}$ | $\Delta F$ |
|---|---|---:|---:|---:|
| ECDL | $(60,1.0\,\mu\mathrm{s},12\,\mathrm{MHz},20\,\mathrm{MHz})$ | 99.5608% | 99.5133% | 0.0476 百分点 |
| seed | $(60,1.0\,\mu\mathrm{s},12\,\mathrm{MHz},20\,\mathrm{MHz})$ | 99.5608% | 99.5585% | 0.0023 百分点 |

**在当前试点中，两台激光器给出同一个最优门点；ECDL 使保真度下降 0.0476 个百分点，
seed 下降 0.0023 个百分点，相差约 20 倍。相位噪声在此假设下不再是主导误差项 ——
即使 ECDL，其相位噪声贡献也小于同点 0.44 百分点的相干泄漏与散射预算。**

这些数值是最坏逻辑输入的加性误差预算，近似使用 $F=1-\mathrm{total\ error}$；
它们不是完整过程层析得到的平均门保真度。

`power` 假设带外噪声按实测斜率一直下降到 200 MHz。真实激光在高偏移频率处通常趋于
自发辐射决定的白噪声底，因此该假设更可能偏乐观；这是本结论的主要不确定性，
只能由 1 MHz 以上的直接测量收窄。由于滤波核与激光谱无关，改用其他带外假设时
只需对已存核重新加权，无需重新求解。

## 独立验证与适用范围

`results/max_leakage_297/a3.0/reports/phase_noise_mc.json` 用 200 次噪声实现验证了
20 个真实门点：20/20 通过，蒙特卡洛与滤波函数预测的比值为 0.895–1.091。
验证在平坦带外假设下进行 —— 那里噪声幅度更大，是对同一套滤波核更严格的检验；
核本身与带外假设无关，所以该验证同样适用于本报告的 `power` 结果。

同一实现还通过白噪声闭式解、静态失谐响应和时间采样收敛检查。
本报告只引用 $\varepsilon_{\rm phase}<0.1$ 的试点结果。

## 数据与溯源

核心结果表直接来自以下字段：

| 路径 | 字段与用途 |
|---|---|
| `../max_leakage_297/a3.0/chunks/chunk_*.npz` | `leakage`：无相位噪声的逐逻辑输入相干泄漏 |
| `../max_leakage_297/a3.0/scatter/scatter_*.npz` | `p_ryd`, `p_r_garb`：逐逻辑输入散射预算 |
| `../max_leakage_297/a3.0/filter/filter_*.npz` | `kernel`, `f_bins`：逐逻辑输入频率噪声滤波核 |
| `psd_ECDL.csv`, `psd_seed.csv` | 两台激光器的数字化单边频率噪声谱 |
| `psd_model.json` | 测量边界、倍频和 power 外推指数 |
| `../max_leakage_297/a3.0/reports/phase_noise_mc.json` | 直接蒙特卡洛验证记录 |

表中的 $F_0$ 由 `leakage + p_ryd + p_r_garb` 逐输入求和后取最大；
$F_{\rm noise}$ 再加入由 `kernel` 与所选 PSD 加权得到的 `eps_phase`。
门点选择与组合逻辑实现于 `scripts/max_leakage_297_sweep.py:phase_noise_values` 和
`scripts/sweeplib/plotting.py:plot_metric_values`。

![ECDL 数字化源图](ECDL_phasenoise.png)

![seed 数字化源图](seed_lasernoise.png)

源图是不可再生成的测量输入；`psd_model.png` 由 `scripts/laser_noise_psd.py` 生成且不进入 Git。
设计与实现记录位于
`docs/superpowers/specs/2026-07-30-laser-phase-noise-design.md` 和
`docs/superpowers/plans/2026-07-30-laser-phase-noise.md`。

## 复现

从既有存储重绘无相位噪声结果：

```bash
uv run python scripts/max_leakage_297_sweep.py plot \
    --output results/max_leakage_297/a3.0 --metric total_error
```

重绘指定噪声假设下的总误差：

```bash
uv run python scripts/max_leakage_297_sweep.py plot \
    --output results/max_leakage_297/a3.0 --metric total_error_phase \
    --laser ECDL --extrapolation power --f-min 1
```

重新生成频谱模型图：

```bash
uv run python scripts/laser_noise_psd.py
```

以上绘图命令本轮未执行；本轮只读取既有 NPZ/JSON/CSV 重算核心表，并运行报告与 Markdown 验证。
