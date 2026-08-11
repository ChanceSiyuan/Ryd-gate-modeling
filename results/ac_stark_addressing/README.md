# ac_stark_addressing — 用 AC Stark 光束单原子寻址的工作点选择

## 问题与模型

要在不扰动近邻的前提下寻址一对原子中的一个，用一束聚焦的**远失谐**光去移动目标原子的能级。
这需要同时满足两件互相矛盾的事：光移要**大**（才能把目标从共振上拉开），
散射要**小**（散射直接变成损耗）。二者都随功率上升，所以存在一个真实的折中。

两组扫描分别回答两个层次的问题：

**① 单原子景观**（`ac_stark/`）—— 在 (波长, 功率) 平面上同时给出光移与散射率。
标定点记录在文件头注释里：$\lambda = 784.0$ nm，$P = 160.0$ µW。

**② 双原子寻址优化网格**（`optimization_grids/`）—— 把候选工作点按真正要紧的代价打分：

$$\text{total\_cost} \;=\; \text{pinning\_leak} \;+\; \text{crosstalk} \;+\; \text{scatter\_penalty}$$

其中 `pinning_leak` 是目标原子被"钉住"过程中漏出去的份额，`crosstalk` 是近邻被误动的
份额（记为 $P_{gg}$ 通道），`scatter_penalty` 是散射折算的代价。同时记录
$\delta_A/\Omega$ 与 $\delta_A/V$ 两个无量纲比 —— 前者要足够大（远离驱动共振），
后者要足够小（不破坏阻塞）。

## 核心结果

### 景观：这段窗口里光移会变号

| 量 | 范围 |
|---|---|
| 波长 | 780.5 – 794.5 nm |
| 功率 | 1 – 500 µW |
| 光移 `shift_mhz` | $-659.84$ … $+174.31$ |
| 散射 `scatter_hz` | 0.078 … $2.18\times10^{4}$ |

**光移在窗口内跨越零点**，这是可用的关键特征：存在一个波长使差分光移可以调过零，
于是"移动目标而不移动近邻"有解，而不是只能靠功率硬拉。

### 优化网格：最优工作点在 785.6 nm / 205 µW

三个网格在 781–786 nm × 140–500 µW 上各扫 2400 点。取 $\arg\min(\text{total\_cost})$ 行：

| 网格 | 几何 / 门时长 | λ (nm) | P (µW) | total_cost | P_target | pinning_leak | crosstalk | scatter_penalty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `exp1` | 4.0 µm, 4.5 µs | 785.576 | 204.6 | **2.55e-4** | 0.999066 | 1.29e-5 | 1.33e-4 | 1.10e-4 |
| `exp2` | 4.0 µm, 9.0 µs | 786.000 | 204.6 | 5.90e-4 | 0.998817 | 2.74e-4 | 1.20e-4 | 1.96e-4 |
| `exp5` | 4.5 µm | 785.237 | 444.6 | 9.79e-4 | 0.998762 | 4.47e-4 | 2.36e-6 | 5.30e-4 |

最优点（`exp1`）的目标原子失谐 $\delta_A = -9.51$ MHz，散射仅 24.4 Hz，且
$\delta_A/\Omega = 2.50$、$\delta_A/V = 0.045$ —— **既远离驱动共振，又远小于阻塞能标**，
正是想要的分离。

### 门时长翻倍，代价涨 2.3 倍，而且涨在泄漏上

`exp1` → `exp2` 只改门时长（4.5 → 9.0 µs）：

| | pinning_leak | crosstalk | total_cost |
|---|---|---|---|
| 4.5 µs | 1.29e-5 | 1.33e-4 | 2.55e-4 |
| 9.0 µs | **2.74e-4**（21×） | 1.20e-4（几乎不变） | 5.90e-4 |

**光束开着的时间越长，目标漏得越多，而串扰基本不动** —— 所以缩短门时长对寻址质量的
收益几乎全部来自压制 pinning leak。

`exp5`（4.5 µm 间距）走的是另一条路：串扰被压到 2.36e-6（间距大了 2 个数量级的改善），
但为了在更远处拿到同样的光移，功率要到 444.6 µW，散射代价随之涨到 5.30e-4 并反过来主导总代价。

## 图

![AC Stark 景观](ac_stark/ac_stark_landscape.png)
(波长, 功率) 平面上的光移与散射率。

![光移剖面](ac_stark/ac_stark_profiles.png)
沿工作波长的切片。

![单位光移的散射](ac_stark/ac_stark_scatter_per_shift.png)
真正决定工作点的品质因子：每 MHz 有用光移所付的散射率。

![偏振敏感度](ac_stark/ac_stark_pol_sensitivity.png)
光移随光束偏振的漂移。

![矢量光移优化](ac_stark/ac_stark_vector_opt.png)
矢量光移下的工作点优化。

图不进 git；用下方命令重生成。

## 数据与溯源

| 路径 | 内容 |
|---|---|
| `ac_stark/ac_stark_landscape.csv` | 80 000 行，**有正规表头** `wavelength_nm, power_uw, shift_mhz, scatter_hz` |
| `optimization_grids/exp{1..5}_addressing_opt_grid.csv` | 各 2400 行，**17 列且无表头行** |
| `ac_stark/*.png` | 5 张图 |

### 告示：优化网格 CSV 没有表头，列名只存在于 notebook 里

列序（来自 `scripts/notebooks/02_ac_stark_addressing.ipynb` 的 `dtype_legacy`）：

```
wl, power_uw, delta_A_mhz, delta_B_mhz, scatter_A_hz,
P_gg, P_gr, P_rg, P_rr, pinning_leak,
crosstalk, scatter_penalty, total_cost, P_target, other_residual,
delta_A_over_Omega, delta_A_over_V
```

用这些文件前有两条必须知道：

- 存在一个 19 列的新格式，末尾追加 `delta_half_mhz` 与 `t_gate_us`。对这里的 17 列文件，
  notebook 会**硬编码** `delta_half_mhz = 40.0`、`t_gate_us = 4.5`。这对 `exp1` 正确，
  但对 `exp2` **是错的** —— 它自己的 `#` 注释写的是 `t_gate_us=9.0`，而注释不会被解析。
  **以注释块为准，不要信默认值。**
- notebook 按 `exp5, exp4, exp3, exp2, exp1` 的顺序取第一个存在的文件。所以一张没有注明
  来源文件的图，画的是 `exp5`。

## 复现

```bash
# 出图（读缓存 CSV，不重扫）
uv run jupyter nbconvert --to notebook --execute --inplace \
    scripts/notebooks/02_ac_stark_addressing.ipynb
```

重新生成 CSV 本身需要在 notebook 内重跑扫描 cell，**没有独立脚本**。

**产出** `scripts/notebooks/02_ac_stark_addressing.ipynb`（末次提交 `4e08089`，
2026-07-16，那是一次仓库清理提交，所以物理内容早于该日期）。
