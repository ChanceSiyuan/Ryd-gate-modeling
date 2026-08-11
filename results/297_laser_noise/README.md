# 297_laser_noise — 297 nm 激光频率与强度噪声对单光子 CZ 门保真度的影响

> **交付判断：** 如果假设seed 的频率噪声更低，但原子处可用光强约为 ECDL 的一半，则：

- 低功率区选择 **ECDL**；配置接近 ECDL 6 W / seed 3 W 时保守选择 **ECDL**；
- 达到约 ECDL 8 W / seed 4 W 后选择 **seed**。

## 1. 激光器噪声如何设置

比较两台候选激光器：**ECDL** 和 **seed**。输入数据是它们在基频 1180/1187 nm 下的实测单边频率噪声谱；四倍频到 297 nm 后，频率噪声功率谱放大 16 倍：$S_{\delta\nu}^{297}(f)=16S_{\delta\nu}^{1180/1187}(f)$。
噪声作为随机失谐加入门哈密顿量：$\hat H(t)=\hat H_0(t)+2\pi\,\delta\nu(t)\hat N_r$,
其中 $\hat N_r$ 统计 $r$ 和 $r_{\rm garb}$ 态布居。默认从 $f_{\min}=1$ Hz 开始积分，另用 $f_{\min}=10$ Hz 检查慢漂移敏感性。重建的噪声模型先在实测谱的 100 kHz–1 MHz 区间，对幅度谱作 log-log 最小二乘拟合：
$\sqrt{S_{\delta\nu}(f)}\propto f^{-p}$.
超过最后一个实测频率 $f_{\rm edge}$ 后，不再使用测量点，而是以边界 PSD 保持连续并延伸同一斜率：

$$
S_{\delta\nu}^{297}(f)=S_{\delta\nu}^{297}(f_{\rm edge})
\left(\frac{f}{f_{\rm edge}}\right)^{-2p},
\qquad f>f_{\rm edge}.
$$

| 激光器 | $f_{\rm edge}$ | ASD 指数 $p$ | $S_{\delta\nu}^{297}(f_{\rm edge})$ |
|---|---:|---:|---:|
| ECDL | 995.654 kHz | 0.46265 | $2.444\times10^3\ \mathrm{Hz^2/Hz}$ |
| seed | 1.000 MHz | 0.55359 | $1.033\times10^2\ \mathrm{Hz^2/Hz}$ |

四倍频的 16 倍 PSD 因子在外推前后统一施加。`power` 模型没有额外加入白噪声底或伺服峰，而是让该幂律一直下降到滤波核上限约 192 MHz；因此 **1 MHz 以上是模型外推，不是实测数据**，也是本结论的主要系统误差来源。

![两台激光器的频率噪声谱及幂律外推](psd_model.png)

## 2. 模拟了什么物理过程，保真度如何定义

模拟的是两个 $^{87}\textbf{Rb}$ 原子的 **297 nm 单光子门**。同一束全局 $\sigma^-$ 激光同时照射两个原子，直接驱动时钟态 $|1\rangle\leftrightarrow|r\rangle$；同一束光还按模型中的偶极矩比耦合到 $|r_{\rm garb}\rangle$。主扫描不抽取单次随机噪声波形，而是用滤波核直接计算激光频率噪声引起的系综平均相位误差。
下图给出当前最佳已计算点

$$
(n,T,\Omega_{\rm pk}/2\pi,D_{\rm sw}/2\pi)
=(73,1.0\ \mu\mathrm{s},14.25\ \mathrm{MHz},17.5\ \mathrm{MHz})
$$

实际使用的 297 nm 脉冲。

**图中四个纵轴分别表示：**

- 上图左轴（橙色）是原子位置的瞬时激光强度 $I(t)$，单位为 $\mathrm{kW/cm^2}$，峰值为 288 $\mathrm{kW/cm^2}$。
- 上图右轴（绿色虚线）是目标跃迁 $|1\rangle\leftrightarrow|r\rangle$ 的瞬时 Rabi 频率 $\Omega(t)/2\pi$，单位为 MHz，峰值为 14.25 MHz。
- 下图左轴（蓝色）是展开的确定性控制相位 $\varphi_c(t)$，单位为 rad。它是实验主动施加的 phase modulation，不是随机激光噪声。
- 下图右轴（黄色虚线）是 $\dot\varphi_c(t)/2\pi$，单位为 MHz，即控制相位对应的瞬时激光频率偏移（扫频失谐），范围为 $-17.5$ 至 $+17.5$ MHz。

图中的两条竖直虚线位于 0.15 和 0.85 µs，分别标出上升段、平顶段和下降段的边界。上图的两条曲线描述同一个光场，但强度与电场振幅平方成正比，Rabi 频率与电场振幅成正比，因此

$$
I(t)=I_{\rm pk}E(t/T),
\qquad
\Omega(t)=\Omega_{\rm pk}\sqrt{E(t/T)}.
$$

下图也描述同一个确定性控制的两个量：蓝线是相位，黄线是相位的时间导数。真实单次实验的总相位为 $\varphi_c(t)+\varphi_n(t)$；随机部分 $\varphi_n(t)$ 不画在这张确定性脉冲图中，报告的 $F_{\rm noise}$ 是对 $\varphi_n(t)$ 随机过程取系综平均后的结果。

![最佳 297 nm 门点的强度、Rabi 振幅、控制相位和扫频](optimal_pulse_time_dependence.png)


## 3. 结果是什么
对每个逻辑输入 $s$，无噪声和有噪声的误差预算分别为

$$
F_0=1-\max_s\left(\varepsilon_{\rm leakage}^s+\varepsilon_{\rm scatter}^s\right),
$$

$$
F_{\rm noise}=1-\max_s\left(\varepsilon_{\rm leakage}^s+\varepsilon_{\rm scatter}^s+\varepsilon_{\rm phase}^s\right).
$$

该点的直接结果为：

| 激光器 | 无噪声 $F_0$ | 有噪声 $F_{\rm noise}$ | 噪声造成的下降 $\Delta F$ |
|---|---:|---:|---:|
| ECDL | 99.7693% | 99.7169% | 0.0523 百分点 |
| seed | 99.7693% | **99.7650%** | **0.0042 百分点** |

**影响不大，但两台激光器差异明确。** ECDL 噪声使原有无噪误差增加约 23%；seed 只增加约 1.8%，噪声造成的保真度下降约为 ECDL 的 1/12。总误差仍主要来自相干泄漏和散射，而不是激光噪声。把积分下限改为 10 Hz 后，下降量分别为 0.0522 和 0.0014 百分点，结论不变。主要不确定性是高频谱：门的滤波权重主要在 9–28 MHz，而实测谱只到约 1 MHz；

上述数值依赖偏乐观的幂律下降外推，若真实高频噪声存在白噪声底，损失会更大。每个频率分箱对相位噪声保真度损失的贡献不是由 PSD 单独决定，而是

$$
\Delta\varepsilon_{s,b}=2\pi^2S_{\delta\nu}(f_b)K_{s,b},
$$

即激光谱与门滤波核的乘积。$K_{s,b}$ 来自 Rydberg 布居敏感度的傅里叶响应；驱动态在 Rabi 时间尺度上振荡，所以核的高频峰位于 $f\sim\Omega/2\pi$，而不是测量边界 1 MHz。对第 3 节最佳已计算点 $\Omega/2\pi=14.25$ MHz，定义 Rabi 附近一个倍频程为 7.125–28.5 MHz，逐分箱积分得到：

| 激光器与低频截止 | 最坏逻辑输入 | $>1$ MHz 占总相位误差 | 7.125–28.5 MHz 占 $>1$ MHz 误差 | 高频贡献峰值 |
|---|---:|---:|---:|---:|
| ECDL，$f_{\min}=1$ Hz | 01/10 | 98.50% | 97.31% | 16.50 MHz $=1.16\,\Omega/2\pi$ |
| seed，$f_{\min}=1$ Hz | 11 | 14.24% | 93.38% | 16.50 MHz $=1.16\,\Omega/2\pi$ |
| seed，$f_{\min}=10$ Hz | 11 | 83.24% | 93.38% | 16.50 MHz $=1.16\,\Omega/2\pi$ |

![幂律外推谱与门滤波核加权误差的频率重叠](noise_extrapolation_filter_overlap.png)

图左是两台激光器的 297 nm 频率噪声 PSD：约 1 MHz 的虚线是实测终点，其右侧为幂律外推；阴影是 Rabi 附近的 7.125–28.5 MHz。图右是 $S_{\delta\nu}(f_b)K_{s,b}$ 对 `>1 MHz` 相位误差的归一化分箱贡献，每条曲线在该频段内合计为 100%，因此曲线高度不用于比较两台激光器的绝对误差。

**两台激光器的高频误差都集中在 Rabi 频率附近：93% 以上来自阴影区，峰值为 16.50 MHz，即 $1.16\,\Omega/2\pi$。** ECDL 的总误差也由该高频段主导；seed 在 $f_{\min}=1$ Hz 时仍由低频慢漂移主导，采用 $f_{\min}=10$ Hz 后高频占比升至 83.24%。


**采用默认 $f_{\min}=1$ Hz：**

| ECDL 标称功率（损耗前） | seed 标称功率（损耗前） | ECDL 最佳保真度 | seed 最佳保真度 | 选择 |
|---:|---:|---:|---:|---|
| 2 W | 1 W | 99.5919% | 99.3890% | ECDL |
| 4 W | 2 W | 99.6983% | 99.6431% | ECDL |
| 6 W | 3 W | **99.7021%** | 99.6918% | ECDL，优势很小 |
| 8 W | 4 W | 99.7169% | **99.7466%** | seed |

**若 1–10 Hz 慢漂移被锁频和门标定消除，采用 $f_{\min}=10$ Hz：**

| ECDL 标称功率（损耗前） | seed 标称功率（损耗前） | ECDL 最佳保真度 | seed 最佳保真度 | 选择 |
|---:|---:|---:|---:|---|
| 2 W | 1 W | 99.5964% | 99.4452% | ECDL |
| 4 W | 2 W | 99.6985% | 99.6593% | ECDL |
| 6 W | 3 W | 99.7022% | **99.7210%** | seed |
| 8 W | 4 W | 99.7171% | **99.7483%** | seed |

数据说明，seed 的低噪声优势必须先补偿其 Rabi 频率较低造成的泄漏和散射代价。最终选择应按实际功率区间决定：

- ECDL/seed 约为 2 W/1 W 或 4 W/2 W：选择 **ECDL**。
- ECDL/seed 约为 6 W/3 W：结论依赖 1–10 Hz 慢漂移是否被消除，当前没有稳健赢家；交付判断保守选择 **ECDL**。
- ECDL/seed 达到约 8 W/4 W：两种低频处理都支持选择 **seed**。
- seed 若能在原子处提供约 6.1 W：选择 **seed**，并可达到第 3 节的当前网格最佳点。

因此，如果实际配置接近 ECDL 6 W / seed 3 W，**当前优先选择 ECDL**。只有当 seed 光强提高到约 4 W，或实验确认 1–10 Hz 慢漂移已被有效抑制后，现有数据才支持改选 seed。

该判断仍受高频谱外推限制：噪声谱只实测到约 1 MHz，而门主要敏感于 9–28 MHz，采购前应补测该频段。

![seed 光强为 ECDL 一半时的功率受限最优保真度](power_tradeoff.png)

图中的阴影是当前离散功率采样能够确定的交叉区；曲线只是连接已计算点，不代表在区间内做了连续优化。

## 4. 光强噪声（RIN）的影响：可以忽略，且有定量依据

**输入数据。** `RIN_RMS.png` 是 1180/1187 nm 基频的实测相对强度噪声（RIN）谱，覆盖 10 Hz–10 MHz，约 100 kHz 以上为 $-148$ dBc/Hz 平底。本分析用像素提取把曲线数字化为 `rin_fundamental.csv`；数字化校验：对该谱做 10 Hz–10 MHz 梯形积分得积分 RMS 0.0117%，与原图自带积分 RMS 面板的终点约 0.012% 一致（2026-08-06 核对）。该图对应两台候选激光器中的哪一台没有记录；由下文的余量结论，这一不确定不影响任何判断。

**如何进入模型。** 强度涨落 $\rho(t)=\delta I/I$ 通过 $\Omega\propto\sqrt I$ 使驱动幅度变为 $\Omega(1+\rho/2)$，模型中没有其他随强度变化的项（$r_{\rm garb}$ 的非共振 AC-Stark 由驱动动力学自动包含），因此

$$
\hat H(t)=\hat H_0(t)+\rho(t)\,\tfrac12\hat H_{\rm drive}(t),
$$

与附录的推导逐字同构：把 $2\pi\hat N_r$ 换成 $\tfrac12\hat H_{\rm drive}(t)$，同一滤波核机制给出

$$
\Delta\varepsilon^{\rm int}_{s,b}=2\pi^2S_\rho(f_b)K^{\rm int}_{s,b},
$$

其中 $S_\rho$ 是 297 nm 处的单边 RIN PSD。基频到 297 nm 的缩放：两级未耗尽 SHG 每级使 $\delta P/P$ 翻倍，PSD 共 ×16（与频率噪声同为 16 倍，原因不同）；这是**下界**——倍频腔与功率伺服附加的强度噪声在基频测量中看不到。10 MHz 实测终点以上按 $-148$ dBc/Hz 平底延伸。$|00\rangle$ 不被驱动，核为零、误差为零，与相位噪声相同。

**结果。** 参数点同第 2 节最佳已计算点，全部误差取各自最坏逻辑输入：

| 噪声源（最坏逻辑输入） | 误差 $\varepsilon$ | 相对强度噪声的倍数 |
|---|---:|---:|
| 相干泄漏+散射（无噪声基线，01/10） | $2.31\times10^{-3}$ | 697× |
| ECDL 相位噪声（01/10） | $5.23\times10^{-4}$ | 158×（22.0 dB） |
| seed 相位噪声（11） | $8.23\times10^{-5}$ | 24.8×（13.9 dB） |
| **强度噪声，×16 后（11）** | $\mathbf{3.31\times10^{-6}}$ | — |

表中相位噪声是单独的最坏输入相位误差 $\max_s\varepsilon_{\rm phase}^s$（`intensity_noise_kernel.npz` 的 `eps_phase_*` 字段），与第 3 节的预算差 $\Delta F$ 定义不同（后者是三项和的最坏输入变化），所以 seed 行的 $8.23\times10^{-5}$ 大于第 3 节的 0.0042 百分点。强度噪声各输入为 $00=0$、$01/10=2.27\times10^{-6}$、$11=3.31\times10^{-6}$，对 $f_{\min}=1$ 或 10 Hz 不敏感（$3.313$ vs $3.311\times10^{-6}$）——与相位噪声不同，慢漂移几乎不贡献。计入误差预算最多使任何配置的保真度再下降 $3.3\times10^{-6}$（0.00033 百分点）：比 seed 相位噪声的 0.0042 百分点小一个数量级以上，而第 3 节功率受限表中最小的选型差距是 0.0103 百分点（6 W/3 W、$f_{\min}=1$ Hz 行）。**强度噪声不改变任何一行的 ECDL/seed 选择，也不进入误差预算的前三项，予以忽略。**

**误差的频率结构与主要系统误差。** 强度噪声误差同样集中在高频：50%/90% 来自 24.2/56.4 MHz 以下，98.6% 来自 10 MHz 实测边界以外的平底延伸。结论依赖"RIN 平底延伸到几十 MHz"这一假设：半导体激光器的弛豫振荡峰通常在 GHz、不落入敏感带，但若倍频腔或功率伺服在 10–60 MHz 有强度噪声峰则需复核。余量表述：要让强度噪声升到 seed 相位噪声的量级，297 nm 处的等效 RIN 平底需抬高约 14 dB（相当于基频平底从 $-148$ 升到约 $-134$ dBc/Hz），或 SHG 链附加同量级的噪声。

**验证。** 脚本每次运行自动执行三项交叉验证（2026-08-06 运行）：(1) 同一管线以 $\hat N_r$ 为算符复算相位噪声核，与存量 `filter` 系列的最大相对偏差 $3.5\times10^{-13}$；(2) 轨迹采样 $n_t=4096$ 与 $16384$ 的 $\varepsilon_{\rm int}$ 相对偏差 $<0.01\%$；(3) 以 $\Omega\to\Omega\sqrt{1+\rho_0}$（$\rho_0=\pm10^{-3},3\times10^{-3}$）整段重新传播的有限差分误差与核的 $f\to0$ 极限 $4\pi^2\rho_0^2\|Q\,G(0)\|^2$ 偏差 0.08–0.20%，钉死噪声算符与全部前置因子。

![RIN 谱与强度噪声滤波核加权误差](intensity_noise_filter_overlap.png)

图左是 ×16 到 297 nm 的 RIN PSD（虚线为 10 MHz 以上的平底延伸）；图右按分箱比较三种噪声对保真度损失的绝对贡献（各取最坏输入），紫色强度噪声曲线在全频段比两条相位噪声低 2–4 个数量级，三者都峰在 Rabi 频率附近。

**数据与溯源。** 实测输入图 `RIN_RMS.png`、`ECDL_phasenoise.png` 和 `seed_lasernoise.png` 均为本地数据，不由 git 跟踪（`*.png` 忽略规则）；它们是测量输入，不能由本仓库重新生成。`rin_fundamental.csv`（像素数字化自 `RIN_RMS.png`，含中位线与上下包络列）与 `intensity_noise_kernel.npz`（`f_bins`、`kernel_int` (4,249)、`kernel_phase`、`eps_int_per_input`、`eps_phase_ecdl/seed` 及参数字段）由 `scripts/intensity_noise_band_analysis.py` 生成；图 `intensity_noise_filter_overlap.png` 同样仅在本地生成，重生成命令见下节。本节数字均读自该脚本 2026-08-06 的输出与上述 npz 字段。

# 数据与可复现性

频率噪声的输入与产物是 `psd_ECDL.csv`、`psd_seed.csv`、`psd_model.json` 与 `../max_leakage_297/a3.0/filter/` 的核系列；强度噪声的输入与产物见第 4 节末尾的"数据与溯源"段。

功率受限表格和图片由以下命令从存储扫描结果重新生成：

```bash
uv run python scripts/laser_power_tradeoff.py
```

无功率限制的核心表可用以下只读命令复算：

```bash
uv run python scripts/phase_noise_summary.py
uv run python scripts/phase_noise_summary.py --f-min 10
```

第 4 节的强度噪声分析（含三项自检、`intensity_noise_kernel.npz` 与图）由以下命令重算：

```bash
uv run python scripts/intensity_noise_band_analysis.py
```

## 附录：滤波核与相位噪声误差的关系

消去驱动项中的随机光学相位后，激光频率噪声表现为随机 Rydberg 失谐：

$$
H(t)=H_0(t)+2\pi\delta\nu(t)N_r,
$$

其中 $N_r$ 统计 $|r\rangle$ 和 $|r_{\rm garb}\rangle$ 布居。对逻辑输入 $s$，无噪声轨迹为 $|\psi_0^s(t)\rangle=U_0(t,0)|s\rangle$。在时刻 $t$ 加入一个单位失谐扰动并传播到门末端，得到

$$
|A_s(t)\rangle=U_0(T,t)N_r|\psi_0^s(t)\rangle.
$$

频率噪声产生的一阶末态扰动为

$$
|\chi_1^s(T)\rangle
=-2\pi i\int_0^T\delta\nu(t)|A_s(t)\rangle dt.
$$

沿无噪声末态方向的扰动只改变整体相位，不降低末态重叠，因此用

$$
Q_s=1-|\psi_0^s(T)\rangle\langle\psi_0^s(T)|
$$

投影到无噪声末态的正交补。最低非零阶的系综平均保真度损失是

$$
\varepsilon_{\rm phase}^s
=\mathbb E\!\left[\|Q_s\chi_1^s(T)\|^2\right].
$$

定义门响应的傅里叶变换

$$
|G_s(f)\rangle=\int_0^T|A_s(t)\rangle e^{-2\pi ift}dt.
$$

对零均值平稳噪声采用单边 PSD 约定

$$
\mathbb E[\delta\nu(t)\delta\nu(t')]
=\int_0^\infty S_{\delta\nu}(f)
\cos[2\pi f(t-t')]df,
$$

代入上式得到

$$
\varepsilon_{\rm phase}^s
=2\pi^2\int_0^\infty S_{\delta\nu}(f)
\left[\|Q_sG_s(f)\|^2+\|Q_sG_s(-f)\|^2\right]df.
$$

其中 $2\pi^2$ 来自哈密顿量中的 $(2\pi)^2$ 和单边 PSD 余弦分解的 $1/2$。对第 $b$ 个频率分箱定义

$$
K_{s,b}=\int_{\mathrm{bin}\ b}
\left[\|Q_sG_s(f)\|^2+\|Q_sG_s(-f)\|^2\right]df.
$$

$K_{s,b}$ 是门对该频率区间中单位频率噪声 PSD 的敏感度，只由无噪声门动力学决定，与选择 ECDL 还是 seed 无关。它已经包含分箱积分和分箱宽度。用分箱中心的 PSD 代表箱内噪声后，单个分箱贡献为

$$
\boxed{
\Delta\varepsilon_{s,b}
=2\pi^2S_{\delta\nu}(f_b)K_{s,b}
},
$$

逐输入相位误差和报告使用的最坏输入相位误差分别为

$$
\varepsilon_{\rm phase}^s
\approx\sum_b\Delta\varepsilon_{s,b},
\qquad
\varepsilon_{\rm phase}=\max_s\varepsilon_{\rm phase}^s.
$$

因此 $S_{\delta\nu}(f_b)$ 表示激光器在该频率附近有多少噪声，$K_{s,b}$ 表示门对该频率有多敏感，两者的乘积才是该频率区间实际造成的保真度损失。输入 $|00\rangle$ 没有 Rydberg 布居，满足 $N_r|00\rangle=0$，所以其滤波核和相位噪声损失均为零。



数据来源：`psd_ECDL.csv`、`psd_seed.csv`、`psd_model.json` 和 `../max_leakage_297/a3.0/filter/filter_000321.npz`。上述频带分解和图片由 `scripts/laser_noise_band_analysis.py` 重新生成。
