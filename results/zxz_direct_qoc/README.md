# zxz_direct_qoc — 直接最优控制合成三原子 ZXZ 门(arXiv:2508.19075 Fig. 3 复现)

## 问题与模型

三原子 1r 链(d = 8.9 µm,ARC 70S:$C_6/2\pi = 862.7\ \mathrm{GHz\,\mu m^6}$,
$V_{\mathrm{NN}}/2\pi = 1.736$ MHz),全局驱动:

$$H(t) = \frac{\Omega(t)}{2}\sum_j\big(|1\rangle\langle r|_j + \mathrm{h.c.}\big)
\;-\;\Delta(t)\sum_j n_j\;+\;\sum_{j<k}\frac{C_6}{r_{jk}^6}\,n_j n_k$$

目标是有效三体演化 $U_{\rm target}=e^{-i\tau H_{\rm ZXZ}}$,
$H_{\rm ZXZ}=J\sum_j Z_{j-1}X_j Z_{j+1}$(三原子只有 $Z_1X_2Z_3$ 一项),$\tau J=0.8$;
保真度 $F=\big|\mathrm{Tr}\,U_{\rm target}^\dagger U\big|^2/64$。

**直接法**(`qoc.direct`,IPOPT):把每个时间片的传播子和控制链全部作为决策变量,

$$\min_{\{U_k,\,u_k,\,\dot u_k,\,\ddot u_k\}} 1-F(U_K)
\quad \mathrm{s.t.}\quad
U_k = e^{-iH(\bar u_k)\,\delta t}\,U_{k-1},\;\;
u_{k+1}=u_k+\dot u_k\delta t,\;\;
\dot u_{k+1}=\dot u_k+\ddot u_k\delta t$$

硬件限制全部是变量盒约束(非惩罚):$\Omega/2\pi\in[0,2.4]$ MHz、
$|\dot\Omega|\le 250\ \mathrm{rad/\mu s^2}$、$\Delta/2\pi\in[-20,20]$ MHz、
$|\dot\Delta|\le 2500\ \mathrm{rad/\mu s^2}$、端点为零;$\delta t=0.05$ µs
(K=24 @ 1.2 µs,K=72 @ 3.6 µs)。GRAPE 基线(同一模型、同一约束数值)按论文用惩罚:
$\mathcal{L}=1-F+\lambda\sum_k g_k + r\,\langle(\mathrm{d}^2u/\mathrm{d}t^2)^2\rangle$,
$\lambda=100$,$r\in\{0,10^{-8},10^{-7},10^{-6}\}$,每档 100 随机种子。

## 核心结果(离散 knot 模型)

| | T = 1.2 µs | T = 3.6 µs |
|---|---|---|
| **direct(本仓库)** | **0.9276**(32 种子,29 收敛) | **0.9922**(8/8 收敛) |
| 论文 | 0.894 | 0.945 |
| GRAPE 中位数(r=0 → 1e-6) | 0.655 / 0.487 / 0.565 / 0.343 | — |

direct 远高于所有 r 档的 GRAPE 中位数(GRAPE 百种子最好尾部 0.9314,见小提琴图);
r 越大脉冲越平滑但保真度塌陷,与论文现象学一致。1.2 µs 明显更接近该约束下的速度极限:
收敛所需迭代数量级上升、多起点成功率下降。

![Fig. 3b — direct vs GRAPE 保真度分布](plots/fig3b_violin.png)

![Fig. 3c — 最优脉冲波形](plots/fig3c_pulses.png)

## 独立 ODE 验证与波形告示

`validate_*.npz` 用 `exact_ode` 回放同一脉冲的两种波形(ZOH 中点采样 = 优化器所见;
分段线性 = 硬件实际输出):

| 脉冲 | F_discrete | F_ode(ZOH) | F_ode(线性) |
|---|---|---|---|
| direct pulse1 | 0.9276 | 0.9276 ✓ | 0.692 |
| direct pulse2 | 0.9922 | 0.9922 ✓ | 0.679 |
| GRAPE 最优 (r=0) | 0.9314 | 0.9314 ✓ | 0.777 |

ZOH 一致到 1e-5(接线正确);**线性波形下保真度大幅跌落且排序反转**——
$\delta t=0.05$ µs 时所有优化器都在利用中点链的离散化误差
(Δ 通道一个时间片可横扫全量程)。引用这些数字时必须成对给出;
面向硬件需要更细的 $\delta t$ 或波形精确的目标函数。

## 迁移测试:3 原子脉冲 → 更大晶格

把最优脉冲(ZOH 波形)原样打到 N 原子链和 3×3 方格上,与
$e^{-i\,0.8\,H_{\rm ZXZ}^{(N)}}$(1D)及行内-ZXZ / 2D cluster
($\sum_j X_j\prod_{k\in\mathrm{NN}(j)}Z_k$)对照(`transfer/`):

| N(链) | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| pulse1 F(酉/基态) | 0.928 | 0.801 | 0.698 | 0.609 | —/0.535 | —/0.449 |
| pulse2 F(酉/基态) | 0.992 | 0.008 | 0.015 | 0.005 | —/0.014 | —/0.001 |
| pulse1 $\max_i|\Delta\langle Z_i\rangle|$ | 0.31 | 0.25 | 0.22 | 0.22 | 0.23 | 0.23 |

![迁移衰减曲线](transfer/transfer_decay.png)

- **短脉冲 (1.2 µs) 可迁移**:全局保真度按 ≈0.906/原子几何衰减,
  而 $\langle Z_i\rangle$ 边缘钉扎剖面 N=3→10 全程持平(局域观测量稳健,论文实验的定量依据)。
- **长脉冲 (3.6 µs) 不可迁移**:N=4 即崩——0.9922 过拟合了 N=3 的有限尺寸细节
  (NNN 相位 $V_{\rm NNN}T\approx0.61$ rad)。
- **2D 3×3 两种目标都不成立**(F ~ 1e-3–1e-2):配位数 2→4 改变阻塞结构,2D 需重新优化。

## 数据与复现

| 路径 | 内容 |
|---|---|
| `direct_pulse{1,2}_seed{N}.npz` / `_summary.json` | 每种子波形 + `fidelity`(决策变量)与 `fidelity_rollout`(物理滚动链,择优排序用) |
| `grape_T1.2.npz` | GRAPE 基线:`fidelities (4,100)`、最优脉冲、违约量 |
| `validate_*.npz` | 三种传播模型的交叉验证 |
| `transfer/transfer_metrics.npz` | 迁移测试全部指标 |

Schema 注:11/40 个种子文件早于 `7d61c48`/`14295d6`,缺 `fidelity_rollout`/`warm_started`
字段;所有消费端都有回退,已于 2026-07-31 复算核对(pulse2 重算 rollout 与记录一致
≤1.3e-10),文件保持原样。图不进 git;重新生成:

```bash
# DGX 上,ARC 需 HOME=/tmp/arc297home 前缀
uv run --extra dev --extra qoc-direct python scripts/zxz_direct_qoc.py plot
uv run --extra dev --extra qoc-direct python scripts/zxz_transfer_test.py --full
uv run --extra dev python scripts/plot_transfer_decay.py
# 续算某时长(keep-best 护栏保证只会更好)
uv run --extra dev --extra qoc-direct python scripts/zxz_direct_qoc.py direct --tag pulse1 --seeds 32 --warm-start
```

**产出脚本** `scripts/zxz_direct_qoc.py`、`scripts/zxz_transfer_test.py`、
`scripts/plot_transfer_decay.py`;设计
`docs/designs/2026-07-28-direct-qoc-zxz-design.md`;引擎 ADR-0026。
