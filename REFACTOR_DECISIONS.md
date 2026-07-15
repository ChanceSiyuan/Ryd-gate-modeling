# Ryd-gate 简化重构决策日志

最后更新：2026-07-15（Asia/Shanghai）
适用仓库：`/home/chance/Ryd-gate-modeling`
状态：持续更新；用于后续 grill、实现和审查

## 0. 本文件的地位与维护规则

1. 本文件记录用户在简化重构讨论中逐项确认的决定，是当前实现与审查的决策源。
2. 本文件优先于 2026-07-13 生成的临时 Claude handoff；该 handoff 中与本文冲突的内容已经过期。
3. `simplify.md` 只保留为最初方向文档，不再修改；后续决定写入本文。
4. 新决定如果推翻旧决定，不删除旧历史：在“被替代的决定”中标明旧结论和替代它的新结论。
5. “已确认”才可以直接实现；“待 grill”不得由实现者自行扩大或猜测。
6. API 可以破坏性修改，不提供 deprecated alias、compatibility wrapper 或并行旧接口。
7. 仓库内所有仍有价值的 scripts/notebooks 物理工作流必须迁移到新 API；允许改写调用方式，不允许物理能力静默退化。

## 1. 仓库清理与研究产物

### R01 — `results/` 是唯一研究结果根目录（已确认）

- 删除 `data/`。
- 原 `data/` 中仍有价值的内容迁入 `results/`。
- 按实验/研究主题建立子目录，而不是把文件平铺在 `results/` 根目录。
- 已采用或计划采用的分类包括：
  - `results/ac_stark_addressing/`
  - `results/cz_gate/`
  - `results/error_budget/`
  - `results/lattice_dynamics/`
  - `results/max_leakage_ode/`
- 用户列出的 AC Stark、error budget、addressing optimization、quench benchmark 和 TFIM 文件都应进入对应主题目录。

### R02 — 删除 `figs/`，但先做来源审计（已确认）

- 删除独立的 `figs/` 目录。
- 先追溯每张图由哪个代码生成。
- 重要且可复现研究结果所需的生成代码保留到对应 script 或 results 主题目录。
- 仅为一次性操作、没有复现价值的生成代码删除。

### R03 — 删除七个 scratch 文件（已确认）

- `_scratch_bfield.py`
- `_scratch_percfg.py`
- `_scratch_pop2.py`
- `_scratch_scan.py`
- `_scratch_tex4th.py`
- `_scratch_tex43.py`
- `_scratch_zzresid.py`

### R04 — backup 与正式 max-leakage 结果（已确认）

- `results/backup_ode_grid8_20260710_0206` 若没有独立价值则删除；审计结论是不需要保留。
- `results/max_leakage_ode/` 是正式结果目录，应保留，不重新分类。
- `scripts/max_leakage_ode_sweep.py` 的专用数值逻辑属于研究脚本，不应被 src 简化顺手删除。

### R05 — 保护论文文件（已确认）

- 不要修改 `algsummary.tex`。
- 当前 dirty worktree 中该文件若显示为删除，也不要由重构代理擅自恢复、暂存或解释；交由用户处理。

## 2. 总体架构边界

### A01 — `src` 的唯一职责（已确认）

`src/ryd_gate` 只负责：

- 原子能级、物理参数、register 几何和相互作用；
- fully specified protocols；
- 单初态、多初态以及随机参数 realization 下的量子演化；
- 明确请求的 observable expectations；
- 最终 basis amplitude；
- 最终测量 sampling；
- 构建或表征模拟系统所需的真实物理函数，例如 Rabi、Zeeman、VdW、ARC 和 effective-theory lowering。

### A02 — 演化后处理不属于 `src`（已确认）

以下内容必须写在使用它们的 script/notebook 中：

- CZ fidelity、conditional phase、logical leakage、gate report；
- ensemble mean/std、error budget；
- decay population integral 和 branching 汇总；
- connected correlation、FFT、structure factor、correlation ratio；
- Bell/SSS/state infidelity；
- optimizer objective、bounds 和结果解释；
- 结果绘图、持久化和报告生成。

用户应当能直接看到自己计算的物理公式，而不是调用一个不透明的 src report helper。

### A03 — 删除派生分析层（已确认）

删除：

- `src/ryd_gate/analysis/` 整个包；
- `src/ryd_gate/gates.py`；
- `cz_gate_report()`、`cz_gate_ensemble_report()`、`CZGateReport`；
- gate metrics、error-budget helpers、population postprocessing；
- src optimizers、objective、projection helpers；
- 所有只消费 evolution result、population array 或 ensemble aggregate 的函数。

纯物理 lowerer 和系统表征函数不因删除 analysis 而删除。

### A04 — 删除 serialization/schema 层（已确认）

删除：

- `core/serialization.py`；
- `src/ryd_gate/schemas/`；
- Register/LevelStructure/Noise 的 `to_dict()` / `from_dict()`；
- schema-only validation framework、测试和依赖。

构造函数直接抛出清楚的 `TypeError` / `ValueError`。

## 3. 顶层公共 API 与导入边界

### API01 — `ryd_gate` 顶层只公开六个名字（已确认）

```python
from ryd_gate import (
    Register,
    RydbergSystem,
    NoiseModel,
    level_structure,
    simulate,
    simulate_ensemble,
)
```

不再顶层导出：

- 任何 Protocol class；
- `EvolutionResult` / `EnsembleResult`；
- `LevelStructure`；
- `InteractionSpec`；
- `DEFAULT_C6`；
- backend/compiler/IR/operator helpers。

### API02 — Protocol 只从子模块导入（已确认）

```python
from ryd_gate.protocols import (
    CZProtocol,
    TOProtocol,
    ARProtocol,
    SweepProtocol,
    DigitalAnalogProtocol,
)
```

Direct-297 的 user-selectable protocols 也只从 `ryd_gate.protocols` 导入。

### API03 — 结果类型只用于专门类型标注（已确认，由 API06 具体化）

- `EvolutionResult`、`GroundStateResult` 和 `EnsembleResult` 是稳定返回类型，但不在顶层导出。
- 若用户需要类型标注，只从 API06 冻结的 `ryd_gate.results` 导入。

### API04 — `LevelStructure` 不在顶层（已确认）

- 用户只调用顶层 `level_structure(...)` factory。
- 用户不能直接构造 `LevelStructure(...)` 绕过 preset 解析与校验。
- 类型标注可从专门 system 子模块导入；不增加第二套构造入口。

### API05 — `NoiseModel` 保留顶层（已确认）

它是用户主动构造随机 realization 所需的核心输入，因此与被动返回的 result types 不同。

### API06 — `ryd_gate.results` 是唯一 result 类型入口（已确认）

```python
from ryd_gate.results import (
    EvolutionResult,
    GroundStateResult,
    EnsembleResult,
)
```

- `ryd_gate.results` 的 public surface 只有上述三个 immutable result types。
- 三个类型均不从顶层 `ryd_gate` 导出。
- 不增加 public `Result` base class、typing protocol、factory、conversion helper 或 analysis helper。
- `EvolutionResult` 与 `GroundStateResult` 可以复用 private amplitude/sample machinery，但不为了 implementation reuse 制造用户可见继承层次。
- `EnsembleResult` 继续是 N11 定义的 raw evolution-result container，不扩张为 ground-state ensemble。
- result 定义从 `ir.py`、`noise.py` 等实现模块移入该单一入口；IR、noise、backend 不再拥有 public result type。
- internal backend-state holders、lazy evaluator callbacks 和 contraction details 仍为 private implementation，不因类型集中而公开。

## 4. Protocol 设计

### P01 — 标准 CZ family 只保留三个类（已确认）

- `CZProtocol`
- `TOProtocol`
- `ARProtocol`

删除 dedicated：

- `ARPProtocol`
- `DoubleARPProtocol`
- `AdiaProtocol`
- `EffectiveCZProtocol`

文档中 “builders + Double-ARP” 等过期 class 声称直接删除。物理上某个 CZ waveform 可以被描述为 ARP/double-ARP，但不因此新增 public class。

### P02 — 任意 callable CZ 属于 `CZProtocol`（已确认）

- TOP/ARP 等可以视为给定参数函数族后的特殊 CZ protocol。
- 任意 callable 及其参数都必须在 `CZProtocol(...)` 构造时固定。
- 不在 `simulate()`、`plot()` 时再传入 `x`。
- callable protocol 是 runtime-only，不支持 serialization。

### P03 — Protocol 构造时 fully specified（已确认）

删除 runtime parameter layer：

- `x` / `params` runtime input；
- `n_params`、`validate_params`、`unpack_params`；
- protocol optimizer bounds；
- `theta_index`、`t_gate_index`；
- 无生产调用的 `phase_420`；
- protocol solver `n_steps`。

TO/AR 的优化向量必须先在 script 中解释为具名构造参数，然后创建 protocol。

### P04 — Protocol 决定真实时长（已确认）

- Protocol 构造时包含 duration 信息。
- 绑定到 system 后解析出物理 `t_gate`。
- `system.t_gate` 是只读真实时长。
- `simulate()` 不允许覆盖 `t_gate`。

### P05 — Protocol runtime interface 保持最小（已确认，由 P17 具体化）

- Protocol 的用户界面只保留 fully specified constructor 与 `plot(system)`。
- system binding、duration resolution、drive lowering 全部通过 P17 的单一 private resolver 完成。
- 不公开、也不保留 protected 的 channel 列表、coefficient dict 或 generic context hooks。
- optimizer metadata 不属于 Protocol。

### P06 — pulse 输入绘图的最终接口（已确认）

```python
figure = protocol.plot(system, n_points=400)
axes = figure.axes
```

- `Protocol.plot()` 只画 protocol 输入控制，不画 simulation result，也不运行量子演化。
- `plot()` 对所有 protocol 始终只返回一个 `matplotlib.figure.Figure`。
- 用户需要进一步定制时从 `figure.axes` 取得 axes。
- `plot()` 不调用 `show()`，不保存文件；notebook/script 决定显示与保存。
- `pulse_traces()` 不再是 public 用户方法，改为仅供 `plot()` 使用的 protected 小型数值 hook。
- 删除历史 `x/params/layout/unit/save/show` 等兼容参数和 subclass-only 返回差异。

### P07 — 多阶段确定性流程用一个 piecewise protocol（已确认）

- local-addressing 的 sweep → hold 不再进行两次演化并传递第一段终态。
- 使用一个 fully specified、分段定义的 time-dependent protocol 完成完整演化。
- 不允许 `simulate(initial_state=EvolutionResult)`。
- 不新增 public `continue_simulation()`。
- 不保留给 scripts 使用的 backend continuation seam。

### P08 — Protocol 时间连续性（已确认）

- coefficient functions 在内部区间 `0 < t < t_gate` 必须连续。
- 分段连接点的函数值必须匹配。
- 一阶导数可以不连续。
- 不支持真正的 interior bang-bang 跳变。
- 因为不公开 ODE `max_step` 或 discontinuity breakpoint API，所以不能接受 solver 可能跨过的真实跳变。

### P09 — DigitalAnalog 完整支持 3×3 `01r`（已确认）

```python
DigitalAnalogProtocol(
    t_gate=...,
    coupling_10=...,
    coupling_r0=...,
    coupling_r1=...,
    energy_1=...,
    energy_r=...,
)
```

- 五个控制彼此独立。
- 每个输入是 `t` 的函数；返回 uniform scalar 或 length-N site profile。
- 三个 coupling 可为 complex。
- 两个 diagonal energy 必须为 real。
- 未提供的项表示该 channel 不存在。
- `|0>` energy 是 gauge zero。
- exact、MPS、PEPS 都必须支持完整 `0↔1`、`0↔r`、`1↔r` Hamiltonian。

### P10 — DigitalAnalog coefficient 采用直接矩阵元（已确认）

- `coupling_10(t) = H[1,0](t)`；
- `coupling_r0(t) = H[r,0](t)`；
- `coupling_r1(t) = H[r,1](t)`；
- compiler 只补 Hermitian conjugate，不再额外乘 `1/2`。
- 从旧 Rabi-frequency 语义迁移时必须显式处理原来的 `1/2`，防止二倍误差。

### P11 — 保留其他有实际物理用途的 protocols（已确认，由 P16 收窄）

- `SweepProtocol`
- `DigitalAnalogProtocol`
- Direct-297 的 Pi/CZ/TO 物理能力

“CZ family 只保留三类”不表示删除这些不同领域的 protocols。

### P12 — Protocol 显式声明 laser groups（已被 P17 替代）

不再实现早期提议的 protected laser-group mapping hook。物理 laser 身份改由 P17 的 typed private drive IR 直接表达，不再从 channel 集合交集反推。

### P13 — `phase_from_chirp` 保留为 public pulse helper（由已确认边界与生产调用确定）

- 从 `ryd_gate.protocols` 导入，不从顶层 `ryd_gate` 导出。
- 它把用户给定的 chirp 积分为 `CZProtocol` 使用的 optical phase，是 pulse construction，不是 evolution-result 后处理。
- `scripts/gen_error_budget_g20.py` 与 CZ/effective-theory benchmarks 有实际调用，不能作为无用 helper 删除。
- 不建立 dedicated ARP protocol class；用户用 `phase_from_chirp(...)` + `CZProtocol(...)` 表达该 waveform。

### P14 — 冻结 Direct-297 public class 名称（已确认）

- `Direct297PiProtocol`
- `Direct297CZProtocol`
- `Direct297TOProtocol`

三类分别表达自动校准的 pi pulse、任意 297-nm CZ pulse 和 TO pulse family。`Direct297` 清楚标识直接单光子 297-nm 路径；不改成只调换词序的短名，也不把单激光 protocol 硬塞进面向 420/1013 的通用 CZ implementation。

### P15 — 删除 `SweepProtocol.plot_address_map()`（已确认）

- 删除这套带 `params/n_sites/grid_shape/address_time/savefig/show` 的第二绘图 API。
- 它没有 production caller，并与 P06 的统一 `Protocol.plot() -> Figure` 规则冲突。
- `Protocol.plot()` 只承担快速查看时间控制波形。
- 需要 local-addressing 空间 heatmap 时，script/notebook 使用已知 `address_fn` 与 `Register.coords` 显式绘制。

### P16 — TFIM quench/anneal 统一为 `SweepProtocol`（已确认，收窄 P11）

删除：

- `TFIMQuenchProtocol`；
- `TFIMAnnealProtocol`；
- `TFIMRydbergControls`；
- `tfim_to_rydberg_controls()`；
- `interaction_longitudinal_shifts()`。

不增加 `tfim_sweep()`、`SweepProtocol.from_tfim()` 或 internal replacement builder。TFIM 是调用工作流对同一个两通道 Rydberg Hamiltonian 的解释，不是独立 evolution protocol：

```text
SweepProtocol / TFIM:
    E[r,1] = Omega(t)/2 = h_x(t)
    E[r,r] = -Delta(t)
```

需要 target-TFIM 参数时，script/notebook 显式写出：

```text
s_i = (1/4) sum_j V_ij
Delta_i(t) = 2 [s_i - h_z,i(t)]
delta_fn(t) = mean_i Delta_i(t)
address_fn(t, i) = Delta_i(t) - delta_fn(t)
```

- quench 使用常数 `h_x(t)` / `h_z(t)`；anneal 使用 script-local piecewise functions。
- 调用者从其已经明确构造的 lattice/interaction 输入计算 `V_ij`；不为此恢复 public `system.interaction_pairs` 或 interaction query API。
- 相关 scripts/notebooks 必须迁移成显式 `SweepProtocol`，保持 exact/MPS/PEPS 接收到的 Hamiltonian 不变。
- TFIM-specific plotting、critical-field analysis、ground-state analysis 和参数解释继续留在 scripts/notebooks。

### P17 — 用 typed private resolved-drive IR 统一 protocol/compiler seam（已确认，替代 P12 并具体化 P05）

Protocol 到所有 backend 的唯一 lowering seam 是：

```python
protocol._resolve(system) -> _ResolvedProtocol

_ResolvedProtocol(
    t_gate=...,
    drives=(
        _LaserDrive(group="420", coefficient=...),
        _ChannelDrive(channel=..., coefficient=...),
    ),
)
```

这些类型全部是 private implementation，不是用户 API。`_ResolvedProtocol` 是 immutable container；每个 drive 只能是以下两种之一：

- `_LaserDrive`：保留物理 laser group 身份与 waveform，供 physical preset 展开、laser noise 注入和绘图使用；
- `_ChannelDrive`：直接表达一个 effective-Hamiltonian primitive channel，不假装它对应某束物理 laser。

system preset 在内部保存 `_LaserLeg(group, channel, factor)`，负责物理 laser 到 primitive transition 及 Clebsch–Gordan/dipole factor 的映射。删除 public `level_structure.laser_channel_ratios`；这些物理数据仍保留，但只作为 compiler implementation。

各类 protocol 的 lowering 规则：

- `CZProtocol` / `TOProtocol` / `ARProtocol` 产生 bound physical system 支持的 `420` / `1013` laser drives；
- Direct-297 protocols 产生 `297` laser drive；
- `SweepProtocol` / `DigitalAnalogProtocol` 产生 direct channel drives。

compiler 对 `_LaserDrive` 先按 realization 注入该 group 的 amplitude/frequency noise，再用 system preset 的 `_LaserLeg` 展开并聚合同组 operator；`_ChannelDrive` 直接进入 Hamiltonian。对只产生 `_ChannelDrive` 的 effective protocol 请求 named physical-laser noise，preflight 必须报 capability error；用户若需要 effective-control noise，应按 N13 写入 realization-specific coefficient functions。

exact、MPS、PEPS 都消费同一份 canonical resolved-drive IR。删除并禁止重新建立以下平行 seams：

- `required_channels`；
- `drive_channels(system)`；
- `get_drive_coefficients(t, ctx)`；
- `resolve_t_gate(system)`；
- generic protocol context dict；
- `TNProtocolContext`；
- backend 直接调用 protocol coefficient dict。

`Protocol.plot(system)` 继续遵守 P06；它可以复用 resolved drives 或一个仅供绘图的窄 internal trace evaluator，但不会重新公开上述 lowering 细节。effective-theory 内部代码和 `scripts/max_leakage_ode_sweep.py` 迁移到 canonical compiled/grouped terms，不以此恢复 public laser-ratio 或 compiler API。

### P18 — Blackman waveform 只保留一个 public helper（已确认，落实 PH01）

唯一 public waveform surface：

```python
from ryd_gate.protocols import blackman_pulse, phase_from_chirp
```

- `blackman_pulse(t, t_rise, t_gate)` 保留现有 continuous flat-top rise/fall envelope 能力，并供 protocols 与 scripts/notebooks 共同使用。
- `blackman_window()` 改为 protocol-internal numerical kernel，不进入 `ryd_gate.protocols.__all__`。
- 删除没有 production caller 的 `blackman_pulse_sqrt()`；调用者需要时显式写 `np.sqrt(blackman_pulse(...))`，不为一行组合增加 public alias。
- 所有 Blackman helpers 从 `ryd_gate.physics` 移出；不保留跨模块兼容 import。
- `phase_from_chirp` 继续遵守 P13，不受本决定改变。

### P19 — 七能级 intermediate detuning 完全归 protocol（已确认）

`CZProtocol`、`TOProtocol`、`ARProtocol` 在构造时显式接收同一个 signed angular-frequency 参数：

```python
intermediate_detuning_rad_s=...
```

- `Delta` 是 laser carrier/rotating-frame control，不是 Rb-87 level structure 的本征属性。
- 从 `level_structure("rb87_7_mp")` 与 `level_structure("rb87_7_pm")` 删除 `detuning_sign` 和 `Delta_Hz`；不保留旧参数兼容层。
- protocol 只接受一个 signed `rad/s` 值，不再把 detuning 拆成 sign 与 unsigned Hz magnitude。
- binding/lowering 时，protocol 通过 private resolved IR 把该 `Delta` 加到所有 intermediate-state diagonal channels；preset 只提供各 intermediate hyperfine offsets 等 atomic structure。
- 同一个 `Delta` 用于 `Omega_eff = omega_420 * omega_1013 / (2 * abs(Delta))` 与 duration resolution，避免 Hamiltonian、time scale 使用不同 detuning。
- `system.t_gate` 仍按 P04 在 protocol 绑定后只读解析，不把 detuning 复制回 system metadata。
- pulse phase/chirp 与 quasi-static laser-frequency noise 继续通过 `_LaserDrive` phase 表达；不能偷偷再修改 preset 的 `Delta`，也不能在两个路径重复计入同一 frequency offset。
- 所有 scripts/notebooks/tests 在 protocol 构造点显式写出自己使用的 signed detuning；删除依赖 preset 隐式 `9.1 GHz/7.8 GHz` defaults 的调用。

### P20 — CZ/TO/AR 的两束 peak Rabi 强制显式（已确认）

三类 constructors 都必须接收：

```python
omega_420_max_rad_s=...
omega_1013_max_rad_s=...
intermediate_detuning_rad_s=...  # P19
```

- 两个 peak Rabi values 必须 finite、strictly positive，单位固定为 rad/s。
- 删除 optional `omega_420_max=None` / `omega_1013_max=None` 与 internal canonical `491 MHz / 185 MHz` defaults。
- protocol 不从 `LevelStructure`、system metadata、manifold tag 或 hidden physical-model fields 读取 Rabi amplitude。
- 用户可直接给实验值，或先用 PH04 的 `rb87_7_mp_rabi_frequencies(...)` 由 laser power/beam area 计算。
- 同一组显式 values 用于 `_LaserDrive` coefficients、`Protocol.plot()`、`Omega_eff` 与 duration resolution；禁止各路径各自采用不同 fallback。
- `with_protocol()` 替换 pulse 时，新 protocol 自带完整 Rabi/detuning scale，不继承旧 protocol 的物理控制参数。
- 所有 scripts/notebooks/tests 显式迁移；不保留旧参数名或默认值兼容层。

### P21 — TO/AR 始终使用 Blackman envelope（已确认）

```python
TOProtocol(..., rise_time_s=20e-9)
ARProtocol(..., rise_time_s=20e-9)
```

- Blackman rise/fall 是 TO/AR family invariant，不是 optional feature。
- `rise_time_s` 是 mandatory、finite、strictly positive 的 protocol constructor parameter，单位为 seconds。
- resolved physical duration 必须满足 `2 * rise_time_s <= t_gate`，否则在 system 构造/binding preflight 报错。
- 删除 `blackman: bool`、flat-envelope branch，以及从 LevelStructure/system 读取 `t_rise` 的路径。
- TO/AR 的 420-nm amplitude 使用 P18 的 Blackman envelope；1013-nm amplitude/phase 在该 family 中保持 constant unity/zero。
- 需要 flat 420 pulse、非 Blackman shape 或任意 420/1013 envelopes 的调用者使用 `CZProtocol`。
- 仓库中历史 `blackman=False` 的短 pulse 必须等价迁移为 `CZProtocol`，保持 Hamiltonian 功能而不在 TO/AR 留例外。

### P22 — TO/AR 保留 dimensionless optimization family（已确认）

TO/AR constructors 同时接收 P19/P20/P21 的 physical scale inputs 与具名 dimensionless family parameters；真实尺度只按以下公式解析：

```text
Omega_eff = omega_420_max_rad_s * omega_1013_max_rad_s
            / (2 * abs(intermediate_detuning_rad_s))

t_gate = duration_ratio * 2*pi / Omega_eff
omega_mod = modulation_frequency_ratio * Omega_eff
delta_phase = frequency_offset_ratio * Omega_eff
```

- `duration_ratio`、`modulation_frequency_ratio`、`frequency_offset_ratio` 是 dimensionless family coordinates；名称由 P23 冻结。
- TO 的 phase family 仍是 `A*cos(omega_mod*t + phi0) + delta_phase*t`。
- AR 的 phase family仍是 fundamental/second-harmonic dual-sine 加同一个 linear phase ramp。
- `intermediate_detuning_rad_s` 是大的 one-photon intermediate detuning；`frequency_offset_ratio` 只是 420 phase 的 linear frequency-offset ratio，二者禁止重复计入。
- 三个 physical scale inputs 都已在 protocol constructor 中显式给出，因此 `t_gate` 在 protocol construction 后唯一确定；binding system 只验证 level/preset compatibility。
- optimization script 必须先把向量元素解释成具名 constructor arguments；不允许把 raw `x` 传给 protocol、`plot()` 或 `simulate()`。
- `system.t_gate` 返回由上式得到的 physical seconds；用户无需在 simulation call 重复给 duration。

### P23 — TO/AR family 参数使用无歧义名称（已确认，修正 P22）

TO 参数：

```python
phase_amplitude_rad=...
modulation_frequency_ratio=...
phase_offset_rad=...
frequency_offset_ratio=...
duration_ratio=...
```

AR 参数：

```python
modulation_frequency_ratio=...
phase_amplitude_1_rad=...
phase_offset_1_rad=...
phase_amplitude_2_rad=...
phase_offset_2_rad=...
frequency_offset_ratio=...
duration_ratio=...
```

- `modulation_frequency_ratio = omega_mod / Omega_eff`。
- `frequency_offset_ratio = delta_phase / Omega_eff`，其中 `delta_phase*t` 是 420 optical phase 的 linear ramp。
- phase amplitudes 与 offsets 的单位通过 `_rad` 明确；所有 ratios 无量纲。
- 删除 `frequency_ratio`、`detuning_ratio`、无单位后缀的 `phase_amplitude*` / `phase_offset*` 旧名。
- 不保留 aliases、deprecated kwargs 或双拼写 parser；scripts 将 optimization-vector positions 显式映射到这些具名参数。

### P24 — generic `CZProtocol` 使用物理时间与显式 duration（已确认，由 P26 再确认）

```python
phase = phase_from_chirp(chirp_fn, t_gate_s)

protocol = CZProtocol(
    t_gate_s=t_gate_s,
    intermediate_detuning_rad_s=delta,
    omega_420_max_rad_s=omega_420,
    omega_1013_max_rad_s=omega_1013,
    envelope_420=lambda t: blackman_pulse(t, rise_time_s, t_gate_s),
    phase_420_rad=phase,
    envelope_1013=None,
    phase_1013_rad=None,
)
```

- generic CZ 的四个 waveform callables 接收 physical time `t`，单位 seconds，定义域为 `[0, t_gate_s]`。
- envelopes 返回 finite real values in `[0,1]`；phases 返回 finite radians。
- generic CZ 直接接收 finite positive `t_gate_s`，不使用 `duration_ratio` 或 system time scale。
- `envelope_1013=None` 是唯一 shorthand，精确定义为 constant `1.0`；`phase_1013_rad=None` 精确定义为 constant `0.0`。
- `envelope_420` 与 `phase_420_rad` mandatory；不对主控制 leg 提供 hidden waveform defaults。
- TO/AR 仍按 P22/P23 使用 dimensionless family，内部生成相同 physical-time `_LaserDrive` coefficients。
- Sweep/DigitalAnalog/generic CZ 的 user callables 因此统一为 `f(t)`；不再支持 generic CZ 的 normalized `s=t/t_gate` callable convention。
- `phase_from_chirp()` 的 physical-time output 可直接传给 `phase_420_rad`，删除历史 `lambda s: phase(s*t_gate)` 适配层。

### P25 — Direct-297 protocols 只接收显式 target Rabi（已确认）

```python
omega_r, omega_r_garb = rb87_297_clock_rabi_frequencies(
    power_297_w,
    beam_area_um2,
    ryd_level=53,
)

protocol = Direct297PiProtocol(
    omega_297_max_rad_s=omega_r,
    ...,
)
```

- `omega_297_max_rad_s` 明确定义为 target `|1> <-> |r>` branch 的 finite positive peak angular Rabi frequency。
- garbage branch relative coupling 由 `rb87_297_clock_4` preset 的 private `_LaserLeg`/dipole ratio 展开；protocol 不接收第二个独立 garbage Rabi，避免制造不一致物理输入。
- 删除三个 Direct-297 protocols 的 `power_at_atoms_w`、`beam_area_um2` 与 lazy ARC/power-to-Rabi cache。
- laser power、optics loss 与 beam geometry 留在 script/notebook；PH03/PH04 的 physics helper 是唯一 public conversion seam。
- Direct297Pi 的 pulse duration 由显式 target Rabi 与 normalized envelope area 在 protocol construction 时确定，不等待 system binding。
- Direct297CZ/TO 的 plotting、duration 与 drive coefficients 使用同一个显式 target Rabi。
- 修改 `ryd_level` 时，调用者用同一 `ryd_level` 重新计算 physics-helper output 并构造新 protocol；不允许 protocol 偷读 system 后静默改变自身 duration。

### P26 — 放弃 CZ normalized-time 提议，保持真实时间（已确认）

- generic `CZProtocol` 的 `envelope_420`、`phase_420_rad`、`envelope_1013`、`phase_1013_rad` 全部继续接收 physical `t` in seconds。
- 不允许 phase 单独接收 normalized `s`，也不把四个 callables 统一改成 `s=t/t_gate`。
- `phase_from_chirp(chirp_fn, t_gate_s)` 继续返回 physical-time callable `phase(t)`；`chirp_fn(t)` 的输入也是 seconds、输出 rad/s。
- P24 的示例与全部规则恢复为最终实现依据；此前讨论中的 normalized-CZ 方案从未确认，不得实现。
- TO/AR 的 dimensionless ratios 只属于 P22 的特殊 optimization families，不扩散到 generic CZ callable contract。

### P27 — `Direct297PiProtocol` 始终自动校准 pi area（已确认）

```python
Direct297PiProtocol(
    omega_297_max_rad_s=...,
    rise_fraction=0.15,
)
```

- 删除 public `t_gate` / `t_gate_s` override。
- protocol 根据 normalized envelope area `F = integral_0^1 A(s) ds` 唯一计算 `t_gate = pi / (omega_297_max_rad_s * F)`。
- `system.t_gate` 返回该自动校准 duration；system binding 不重新计算或覆盖。
- 若允许任意 duration，该对象将不再保证 pi pulse，违反 class semantic；因此任意固定时长 297 waveform 使用 `Direct297CZProtocol`。
- calibration 使用 target branch Rabi；garbage branch 仍只是同一 laser 经 preset private leg ratio 产生的物理 leakage，不参与 target pi-area 定义。
- 不增加 `auto_t_gate` boolean、manual/auto mode 或 cached-by-ryd-level 分支。

### P28 — `Direct297CZProtocol` 使用真实时间 callable（已确认）

```python
protocol = Direct297CZProtocol(
    t_gate_s=...,
    omega_297_max_rad_s=...,
    envelope_297=lambda t: ...,
    phase_297_rad=lambda t: ...,
)
```

- `envelope_297` 与 `phase_297_rad` 接收 physical `t` in seconds，定义域 `[0, t_gate_s]`。
- envelope 返回 finite real `[0,1]`；phase 返回 finite radians。
- `t_gate_s` 与 `omega_297_max_rad_s` 必须 finite、strictly positive。
- `envelope_297` mandatory；`phase_297_rad=None` 是唯一 shorthand，精确定义为 constant zero phase。
- 删除旧 `A_297` / `phi_297` normalized-`s` API、无单位 `t_gate` 名称和 compatibility wrapper。
- P24 的 two-laser generic CZ 与本类共享同一 physical-time callable convention；single-laser 与 two-laser compiler 最终都产生 canonical `_LaserDrive`。
- `Direct297TOProtocol` 的 dimensionless optimization family 不改变 generic Direct297CZ 的 callable contract。

### P29 — `Direct297TOProtocol` 与 TO/AR 使用一致的参数族和 Blackman 约束（已确认）

```python
protocol = Direct297TOProtocol(
    omega_297_max_rad_s=...,
    rise_time_s=...,
    phase_amplitude_rad=...,
    modulation_frequency_ratio=...,
    phase_offset_rad=...,
    frequency_offset_ratio=...,
    duration_ratio=...,
)
```

- 删除旧 `power_at_atoms_w` / `beam_area_um2`；实验功率到 Rabi frequency 的换算由 `ryd_gate.physics.rb87_297_clock_rabi_frequencies(...)` 显式完成。
- 删除 `blackman` boolean 和 `t_rise_fraction`。
- 297-nm TO 始终采用 Blackman flat-top envelope；`rise_time_s` 必填、finite、strictly positive，并要求 `2 * rise_time_s <= t_gate`。
- 参数使用与 `TOProtocol` 相同的无量纲优化族和命名：

  ```text
  t_gate     = duration_ratio * 2π / omega_297_max_rad_s
  omega_mod  = modulation_frequency_ratio * omega_297_max_rad_s
  delta      = frequency_offset_ratio * omega_297_max_rad_s
  phase(t)   = phase_amplitude_rad * cos(omega_mod*t + phase_offset_rad)
               + delta*t
  ```

- 所有角度为 radians，`omega_297_max_rad_s`、`omega_mod` 与 `delta` 为 rad/s，phase 中的 `t` 为 physical seconds。
- 需要 flat envelope 或任意 envelope/phase 的用户改用 `Direct297CZProtocol`；TO 类只表达固定的优化函数族。
- 删除旧 `phase_amplitude` / `frequency_ratio` / `phase_offset` / `detuning_ratio` 名称及 compatibility aliases。

### P30 — `Direct297PiProtocol.rise_fraction` 是自动定时协议的唯一比例时间例外（已确认）

```python
protocol = Direct297PiProtocol(
    omega_297_max_rad_s=...,
    rise_fraction=0.15,
)
```

- `rise_fraction` 必须 finite 且满足 `0 <= rise_fraction <= 0.5`。
- `rise_fraction=0` 精确定义为 square pulse，此时 `t_gate = pi / omega_297_max_rad_s`。
- `0 < rise_fraction <= 0.5` 定义相对于自动求得 `t_gate` 的对称 Blackman rise/fall；默认 `0.15`。
- 不增加 `blackman` boolean 或 `rise_time_s`。这里采用 fraction 是因为 π-area calibration 自己决定总时长；physical rise time 会使 envelope 与未知 `t_gate` 形成不必要的隐式耦合。
- 这是 protocol API 中唯一的 fraction-of-total-time pulse 参数；所有显式总时长的 protocols 均使用 physical seconds。

### P31 — Protocol 构造 API 的剩余机械设计由本重构计划统一冻结（已授权）

- 用户授权计划维护者直接决定保留 protocols 的 constructor 参数、命名、单位、默认值及 validation，不再逐项 grill。
- 决策准则：真实物理量使用带单位后缀的名称；优化坐标只在固定函数族中保持无量纲；不保留 aliases、compatibility wrappers、隐式实验参数换算或重复 mode booleans。
- 若选择只改变 API 表达而不改变物理 Hamiltonian/协议函数族，可直接记录决定。
- 若选择会删除或改变物理模型、可实现的 waveform family 或模拟语义，仍须单独向用户确认。

### P32 — 冻结完整 protocol constructor surface（依据 P31 决定）

`ryd_gate.protocols.__all__` 精确包含八个 concrete protocols 与两个 pulse helpers：

```python
CZProtocol
TOProtocol
ARProtocol
SweepProtocol
DigitalAnalogProtocol
Direct297PiProtocol
Direct297CZProtocol
Direct297TOProtocol
blackman_pulse
phase_from_chirp
```

- `Protocol` ABC、private pulse builders、resolved-drive types 与任何 backend adapter 均不导出。
- 所有 constructors 均为 keyword-only；不接受 positional compatibility、旧名 aliases、deprecated kwargs 或 `**kwargs` passthrough。
- physical seconds 使用 `_s`，angular frequency/Hamiltonian matrix element 使用 `_rad_s`，phase 使用 `_rad`；fixed optimization-family coordinates 才使用 dimensionless ratios。
- 所有 scalar constructor values 必须 finite；Python/NumPy real 或 complex scalars 按字段语义接受，但 `bool` 不能冒充数值。
- 用户提供的 control callables 全部接收 physical `t` in seconds。每次实际求值验证 numeric dtype、shape、finite 与字段范围；P08 的连续性属于调用者 contract，不能用有限 probe 假装证明。
- public runtime surface 只有 constructor 与 `plot(system, n_points=400) -> Figure`；不公开 `t_gate`/`duration_ratio` properties、`*_at()` evaluators、phase tables 或 coefficient hooks。

精确 signatures：

```python
CZProtocol(
    *,
    t_gate_s,
    intermediate_detuning_rad_s,
    omega_420_max_rad_s,
    omega_1013_max_rad_s,
    envelope_420,
    phase_420_rad,
    envelope_1013=None,
    phase_1013_rad=None,
)

TOProtocol(
    *,
    intermediate_detuning_rad_s,
    omega_420_max_rad_s,
    omega_1013_max_rad_s,
    rise_time_s,
    phase_amplitude_rad,
    modulation_frequency_ratio,
    phase_offset_rad,
    frequency_offset_ratio,
    duration_ratio,
)

ARProtocol(
    *,
    intermediate_detuning_rad_s,
    omega_420_max_rad_s,
    omega_1013_max_rad_s,
    rise_time_s,
    modulation_frequency_ratio,
    phase_amplitude_1_rad,
    phase_offset_1_rad,
    phase_amplitude_2_rad,
    phase_offset_2_rad,
    frequency_offset_ratio,
    duration_ratio,
)

SweepProtocol(
    *,
    t_gate_s,
    omega_half_rad_s,
    detuning_rad_s,
    local_detuning_rad_s=None,
)

DigitalAnalogProtocol(
    *,
    t_gate_s,
    coupling_10_rad_s=None,
    coupling_r0_rad_s=None,
    coupling_r1_rad_s=None,
    energy_1_rad_s=None,
    energy_r_rad_s=None,
)

Direct297PiProtocol(
    *,
    omega_297_max_rad_s,
    rise_fraction=0.15,
)

Direct297CZProtocol(
    *,
    t_gate_s,
    omega_297_max_rad_s,
    envelope_297,
    phase_297_rad=None,
)

Direct297TOProtocol(
    *,
    omega_297_max_rad_s,
    rise_time_s,
    phase_amplitude_rad,
    modulation_frequency_ratio,
    phase_offset_rad,
    frequency_offset_ratio,
    duration_ratio,
)
```

Class-specific validation and lowering：

- `CZProtocol`：`t_gate_s` 与两束 Rabi strictly positive；signed `intermediate_detuning_rad_s` 可为 zero，因为 generic CZ duration 不依赖 effective-theory approximation。mandatory 420 callables 与 optional 1013 callables 遵守 P24；envelopes 返回 real `[0,1]`，phases 返回 real radians。
- `TOProtocol` / `ARProtocol`：两束 Rabi、`duration_ratio`、`rise_time_s` strictly positive；intermediate detuning nonzero；`2*rise_time_s <= t_gate`。所有 phase/ratio fields finite，modulation/frequency-offset ratios 可为 negative 或 zero；waveform 与公式严格遵守 P21–P23。
- `SweepProtocol`：三个 control fields 是 callables；前两个 mandatory，local optional。它们返回 finite real scalar rad/s，允许任意符号：

  ```text
  H[r,1](t)   = omega_half_rad_s(t)
  H[r,r](t)   = -detuning_rad_s(t)
  H_i[r,r](t) += -local_detuning_rad_s(t, i)
  ```

  `local_detuning_rad_s(t, i)` 的 `i` 是 `0 <= i < system.N` 的 integer。`omega_half_rad_s` 已经是 `Omega/2` direct matrix element，compiler 不再乘 `1/2`。complex/full `01r` drive 使用 `DigitalAnalogProtocol`。
- `DigitalAnalogProtocol`：五个 controls 均为 optional callable，但至少一个存在。每次返回 numeric scalar 或 exact shape `(N,)` profile；scalar 广播。couplings 可为 finite complex，energies 必须 finite real。shape 可以随时间在 scalar/profile 间变化，private resolver 每次展开，禁止用若干 probe times 猜测永久 shape。compiler 只补 Hermitian conjugate，严格遵守 P09/P10。
- 三个 Direct-297 classes 严格遵守 P25/P27–P30：只接受 target Rabi；Pi 自动定时；generic CZ 使用 physical-time callable；TO 使用 fixed Blackman family。
- CZ/TO/AR 只兼容 `rb87_7_mp` / `rb87_7_pm`；Direct-297 只兼容 `rb87_297_clock_4`；DigitalAnalog 只兼容 `01r`；Sweep 兼容拥有 `|1>`/`|r>` effective channels 的 `1r` 与 `01r`。不兼容组合在 system construction/`with_protocol()` 时拒绝。
- Sweep/DigitalAnalog 只产生 `_ChannelDrive`，因此继续遵守 N14，拒绝 named physical-laser noise；laser CZ 与 Direct-297 families 分别产生 `420/1013` 与 `297` `_LaserDrive`。

历史措辞的最终解释：

- P19 的 `Omega_eff`/duration 规则只适用于 TO/AR；generic CZ 的 intermediate detuning 只进入 intermediate-state diagonal Hamiltonian，duration 由 `t_gate_s` 决定。
- P03 删除的是旧 public `phase_420()` method/runtime field，不是 P24 mandatory constructor callable `phase_420_rad`。
- P04 的“绑定后解析”只表示 system compatibility check 与 `system.t_gate` exposure；所有保留 protocols 的 duration 在 protocol construction 后已经唯一确定，system 不补隐藏 scale。
- 本条带单位的 spellings 取代 P09/P16 示例中的旧 `t_gate`、`coupling_*`、`energy_*`、`omega_half_fn`、`delta_fn` 与 `address_fn` 名称，不保留 aliases。

### P33 — 冻结两个 public pulse helper 的输入 API（依据 P31 决定）

```python
blackman_pulse(t_s, rise_time_s, t_gate_s)

phase_from_chirp(
    chirp_rad_s,
    t_gate_s,
    *,
    n_samples=1001,
)
```

- `blackman_pulse` 支持 scalar 或 NumPy-array `t_s` 并保持对应输出 shape；`t_gate_s` finite/positive，`rise_time_s` finite/positive 且 `2*rise_time_s <= t_gate_s`。它返回 dimensionless real envelope；square-pulse mode 不塞进该 helper。
- `phase_from_chirp` 要求 callable `chirp_rad_s(t_s)` 每次返回 finite real rad/s；`t_gate_s` finite/positive；`n_samples` 是非-bool integer 且 `>=2`。
- `phase_from_chirp` 返回 physical-time callable `phase_rad(t_s)`，通过固定 grid 的 cumulative trapezoid 与 interpolation 实现 O(1) evaluation；`n_samples` 是公开的 pulse-discretization accuracy knob，不藏进 protocol/backend options。
- 返回 callable 的有效定义域严格是 `[0,t_gate_s]`；超出定义域报错，不以 clipping 隐藏调用方或 backend 的时间错误。
- 不保留旧 `t_gate` keyword、positional third argument、`n_points` alias、normalized-time wrapper 或从 `ryd_gate.physics` 的 compatibility import。

## 5. System、LevelStructure 与 Register

### S01 — `RydbergSystem` 显式构造（已确认）

```python
RydbergSystem(
    level_structure=...,
    register=...,
    protocol=...,
    interaction_cutoff_um=...,
)
```

- 删除 fluent `set_atom_level`、`set_atom_geom`、`set_protocol`。
- 删除 `SystemModel` ABC。
- 删除 system metadata/meta 字典接口。
- 保留 `with_protocol()`，返回绑定另一 fully specified protocol 的新 system。

### S02 — Level structure 只允许内置 preset（已确认）

唯一入口：

```python
level_structure("preset_name", **physical_kwargs)
```

- 删除 public custom `LevelStructureSpec` / `TransitionSpec` DSL。
- 新物理能级结构必须在 src 中实现、验证并成为正式 preset。
- 保留：`1r`、`01r`、`rb87_7_mp`、`rb87_7_pm`、`rb87_297_clock_4`。
- 删除 Hamiltonian 恒等且无生产用途的 `01` preset。
- 删除只有 capability 声称、没有真实实现的 stabilizer 能力。

### S03 — `Register` 是纯几何（已确认）

只存：

```python
Register(coords)
```

- `N` 从坐标数量推导。
- S13 删除 derived `spacing_um` property；factory 的同名输入只用于生成 coordinates。
- site/basis 顺序严格由 `coords` 行顺序定义；S11 删除 `ids`。
- 保留 `chain()`、`rectangle()`、`square()`、`triangular()` 形状工厂。
- 删除 `RegisterLayout`、metadata、重复尺寸字段和 device validation。

### S04 — 删除 `Register.from_coordinates()`（已确认）

- 任意坐标只用 `Register(coords=...)`。
- 直接构造保持用户坐标原样，不偷偷 `center=True`。
- 需要居中时由用户显式处理坐标。

### S05 — 删除 public `system.product_state()`（已确认）

- 用户不再通过 system 构造或取得 dense product-state vector。
- 初态、basis amplitude 与 sample outcomes 始终使用同一种 physical level-label sequence。
- labels 到 exact/MPS/PEPS backend state 的转换是 private state-compiler seam。
- 手动 CZ bra/final-vector overlap 已由 `result.amplitude(labels)` 取代。
- specialized `scripts/max_leakage_ode_sweep.py` 若需要 dense index/vector，在 script 内保留自己的窄逻辑，不扩大 public system API。

### S06 — 用 `01r` 完全取代并删除 `analog_3`（已确认）

`analog_3` 不再是 preset、public 名称或 internal physical-model 分支。原来的三能级 ladder 用 `01r + DigitalAnalogProtocol` 精确表达，physical-label 映射固定为：

```text
analog_3:  |g>  |e>  |r>
01r:       |0>  |1>  |r>
```

忽略已经由 E08 删除的 non-Hermitian 项后，单原子 Hamiltonian 的等价输入为：

```python
DigitalAnalogProtocol(
    t_gate=...,
    coupling_10=lambda t: 0.5 * omega_420 * c_420(t),
    coupling_r1=lambda t: 0.5 * omega_1013,
    coupling_r0=None,
    energy_1=lambda t: Delta,
    energy_r=None,  # 或显式给出实验需要的 Rydberg detuning/profile
)
```

- 上式 coefficient 是 P10 定义的直接 Hamiltonian matrix element；迁移时保留两个 Rabi frequency 的 `1/2`，compiler 不再补该因子。
- `energy_1=Delta` 保留原 `H[e,e]=Delta` 的符号；`|0>` 继续作为 gauge zero。
- 原来的 1013 coupling 不再作为 system static special term，而是 fully specified protocol 中的常数 `coupling_r1(t)`。
- local addressing、AC-Stark/chirp、intermediate/Rydberg populations、Rydberg interaction 以及 exact/MPS/PEPS 演化能力保持不变。
- 原先的 `rabi_eff = omega_420 * omega_1013 / (2 * abs(Delta))`、`time_scale = 2*pi/rabi_eff` 和实验参数 provenance 不进入 generic `01r` preset；使用它们的 script/notebook 必须显式计算并命名。
- 中间态与 Rydberg 自发辐射继续按 E09 在 script 中用 `Gamma_e * integral(n_1 dt)` 与 `Gamma_r * integral(n_r dt)` 做一阶估计。
- 物理 420/1013 激光的时间相关噪声写进 realization-specific coefficient functions，遵守 N13；不能把 `DigitalAnalogProtocol` 的 direct channels 伪装成 named physical-laser drives。
- 删除所有 `analog_3` source builders、fields、preset kwargs、compiler/backend 分支、TN local-block/site 特例、capability guards、tests 和过期文档；不保留 alias、deprecation shim 或自动迁移层。

### S07 — 冻结每个 `level_structure` preset 的 kwargs（已确认）

```python
level_structure("1r", ryd_level=70)
level_structure("01r", ryd_level=70)

level_structure(
    "rb87_7_mp",
    ryd_level=70,
    magnetic_field_G=20.0,
)

level_structure(
    "rb87_7_pm",
    ryd_level=53,
    magnetic_field_G=20.0,
)

level_structure(
    "rb87_297_clock_4",
    ryd_level=53,
    magnetic_field_G=...,
    quantization_axis=(0.0, 0.0, 1.0),
)
```

- `1r` 与 `01r` 只接受 `ryd_level`。
- `rb87_7_mp` 与 `rb87_7_pm` 只接受 `ryd_level` 与 `magnetic_field_G`。
- `rb87_297_clock_4` 只接受 `ryd_level`、`magnetic_field_G` 与 S08 的 `quantization_axis`。
- 默认 Rydberg states 继续遵守 I03；改变 `ryd_level` 必须触发 I04-I06 的 ARC interaction 重算。
- 删除所有 presets 的 `C6_rad_s_um6`、`enable_rydberg_decay`、`enable_intermediate_decay`、`t_rise`、Rabi fields、`Delta_Hz`、`detuning_sign` 以及其他未列明 kwargs。
- `t_rise`、Rabi amplitudes 与 P19 的 intermediate detuning 都属于 fully specified protocol。
- decay rates 与 branching ratios 作为 preset 自带的只读 physical characterization data 保留，但不再控制 non-Hermitian evolution。
- unknown kwargs 在 factory boundary 立即报错；不允许 metadata bag、`**kwargs` passthrough、静默忽略或 deprecated aliases。

### S08 — 297 preset 显式保存 quantization axis（已确认，扩展 S07）

```python
level_structure(
    "rb87_297_clock_4",
    ryd_level=53,
    magnetic_field_G=20.0,
    quantization_axis=(0.0, 0.0, 1.0),
)
```

- `quantization_axis` 是 length-3 finite real vector；零向量非法，preset 在构造时归一化并保存 immutable unit vector。
- 默认 `+z` 保持现有计算结果。
- 该方向统一定义 ARC P-state pair calculation 的 `(theta, phi)`，并用于 `rr`、`r-r_garb`、`r_garb-r_garb` 三类 interactions。
- N05/N15 的 position-noise realization 使用扰动后的 pair displacement 与同一 quantization axis 重算方向；不得退回 hard-coded `+z`。
- `1r`、`01r`、`rb87_7_mp`、`rb87_7_pm` 的当前 S-state interaction 各向同性，因此拒绝无效的 `quantization_axis` kwarg。
- 不把 quantization axis 放入 `Register`：它属于 atomic/field setup，同一 geometry 可以绑定不同量子化轴的 level structure。

### S09 — nominal `Register` 严格为二维平面几何（已确认）

- `Register(coords=...)` 只接受 finite shape `(N, 2)` coordinates，单位固定为 um；shape `(N,3)` 明确报错。
- `Register.chain()`、`rectangle()`、`square()`、`triangular()` 全部只产生二维 coordinates。
- `register.coords` 保持只读 `(N,2)` canonical representation；不增加 dimension flag，也不公开 padded 3D copy。
- 计算 S08 的 297 anisotropic interaction 时，private compiler 仅在角度计算边界把 nominal pair displacement `(dx, dy)` 嵌入为 `(dx, dy, 0)`。
- `quantization_axis` 继续是三维 unit vector，因此可以垂直或倾斜于二维原子平面。
- nominal Register 的二维限制不禁止 N05 的 realization-specific out-of-plane thermal position offsets；它们属于 noise execution state，不写回或替换 public Register。

### S10 — system geometry 只通过 `register` 暴露（已确认）

```python
register = Register.rectangle(4, 4, spacing_um=5.0)
system = RydbergSystem(
    level_structure=...,
    register=register,
    protocol=...,
)
```

- `RydbergSystem(..., register=...)` 是注入 nominal geometry 的唯一 constructor seam。
- public 读取只使用 `system.register`。
- 删除 `system.geometry` attribute/property alias；protocol、compiler、noise、scripts、notebooks 与 tests 全部迁移到 `system.register`。
- 不接受 `geometry=` constructor alias，也不保留 deprecation shim。
- `Register` 仍只负责 S03/S09 的纯几何，不因绑定到 system 而获得 level structure、interaction 或 protocol state。

### S11 — 删除 `Register.ids`，site 只使用整数顺序（已确认，修正 S03）

- 删除 `Register(..., ids=...)` 与 public `register.ids`。
- 删除 shape factories 的 `prefix=` 参数。
- 删除没有 production caller 的 `register.index()` 与 `register.id_at()`。
- site index 唯一为整数 `0 ... N-1`，顺序严格等于 `coords` 的行顺序。
- initial-state physical-label list、observable 的 site 参数、amplitude label sequence 与 sample outcome tuple 全部使用同一 row-major site order。
- compiler/backend 不生成或维护第二套 string site labels；内部若需要 tensor index name，只能由 integer position 私下生成，不能成为用户身份 API。
- 实验设备名、trap labels 或其他外部名称映射留在 script/notebook，不进入 Register metadata。
- 不保留 ids alias、兼容 lookup 或 deprecation shim。

### S12 — 删除 `Register.sublattice`（已确认，再次收窄 S03）

- 删除 `Register(..., sublattice=...)` 与 public `register.sublattice`。
- shape factories 只生成 coordinates，不附带 checkerboard signs 或三角 lattice 的零数组。
- 删除 backend/TN 从 Register 读取 sublattice 来生成 `af1` / `af2` 等隐式初态的路径；E17 要求调用者传入完整 physical-label `initial_state`。
- staggered magnetization、checkerboard correlations、domain coloring 等分析在 script/notebook 中显式构造 weight arrays，再通过 O11 的 `n()` 与 expression algebra 组合 observable 或对 expectation 后处理。
- 不增加 `register.checkerboard()`、`sublattice_weights()` 或等价 helper；矩形 lattice 的 `(-1) ** (row + col)` 属于具体分析定义，不是所有 geometry 都具有的 intrinsic field。
- TN private lattice ordering 只能从 canonical `coords` 与 factory/shape validation 推导，不能重新把 analysis sublattice 塞回 Register metadata。

### S13 — 删除 derived `register.spacing_um`（已确认，再次收窄 S03）

- 删除 public `register.spacing_um` property；对 irregular geometry，“spacing”作为最小 pair distance 既含糊又需要隐式 `O(N^2)` 扫描。
- `Register.chain/rectangle/square/triangular(..., spacing_um=...)` 的构造参数保留，只用于生成 canonical coordinates。
- system interaction 不读取或保存一个 global spacing；对每个 nominal pair 直接计算 `r_ij = norm(coords[i] - coords[j])` 与 `V_ij = C6_ij / r_ij**6`。
- I07 的 `interaction_cutoff_um` 也直接比较每个真实 `r_ij`，不再通过 spacing 推导 `nn/nnn` cutoff。
- 删除旧 `InteractionSpec(mode="nn" | "nnn")` 依赖 `register.spacing_um` 的 fallback；不得用 private alias 把它恢复。
- script/notebook 若需要展示 regular-lattice spacing，使用自己传给 factory 的变量；若研究 irregular pair distances，显式从 `register.coords` 计算所需统计量。

### S14 — 删除 Register 的 public geometry-query 与绘图方法（已确认，完成 S03 收口）

最终 public surface：

```python
Register(coords)
Register.chain(...)
Register.rectangle(...)
Register.square(...)
Register.triangular(...)

register.coords
register.N
```

- 删除 public `distances_um()`；它没有 production caller 且会分配 `N x N` dense matrix。
- 删除 public `distance_pairs()`；compiler/system 使用自己的 private pair iterator 直接消费 canonical coordinates。
- 删除 `blockade_edges()`、`draw()`，且不新增 `Register.plot()`；简单 geometry scatter 由 script/notebook 直接绘制 `register.coords`。
- `plot_spatial_rydberg()` 消费 evolution expectations，因此迁入使用它的 lattice notebook，不能留在 geometry module。
- `is_in_domain()` 等 state/backend helper 移入真正使用它的 private module。
- `nn_nnn_relative_pairs()`、`cylinder_nn_nnn_pairs()` 等 TN topology helpers 移入 TN internal；它们不属于 Register public API。
- `register.coords` 是真正 readonly 的 finite `(N,2)` NumPy array；frozen dataclass 但数组仍可写的半不可变状态不允许保留。
- `register.N` 只由 `coords.shape[0]` 推导，不另存重复字段。
- `ryd_gate.lattice` 若继续作为 implementation module，其 `__all__` 只含 `Register`；API01 仍从顶层导出同一个 `Register`。

### S15 — 冻结 `RydbergSystem` 的最小 public surface（已确认）

唯一 public attributes/methods：

```python
system.level_structure
system.register
system.protocol
system.interaction_cutoff_um
system.N
system.t_gate
system.observables

system.with_protocol(...)
system.ground_state(...)
```

- 前四项暴露构成 immutable system 的显式输入；`N` 从 `register.N` 推导，`t_gate` 从 bound protocol 解析。
- `observables` 是 O11 的最小 `E()` / `n()` factory。
- `with_protocol()` 遵守 S01/P03/P04，返回共享同一 level structure/register/cutoff、绑定另一 fully specified protocol 的新 system。
- `ground_state()` 严格遵守 E12-E21，只对 fully constructed `1r` system 返回 `GroundStateResult`。
- 删除 public `basis`、`operators`、`hamiltonian_channels`、`static_hamiltonian_terms`、`interaction_pairs` 与任何 compiled/lowered term access。
- 删除 public `dim`；需要展示 nominal Hilbert dimension 的 script 可由 `len(system.level_structure.levels) ** system.N` 显式计算，但 simulation 不以 dense dimension 定义可执行能力。
- 删除 public `is_sparse`、`amplitude_scale` 与其他 backend/noise execution state。
- `product_state()`、旧 product-vector `ground_state()`、`geometry` 与 metadata aliases 已分别由 S05/E15/S10/A03 删除，不保留兼容层。
- compiler/backend 可持有 private basis/operator/interaction/cache objects，但不得通过 system public properties 泄漏。

### S16 — `RydbergSystem` 构造时必须绑定 protocol（已确认）

```python
system = RydbergSystem(
    level_structure=...,
    register=...,
    protocol=...,
    interaction_cutoff_um=None,
)
```

- `protocol` 是 mandatory keyword-only input，不允许 `None`。
- `system.protocol` 是 S15 的 non-optional readonly attribute。
- `system.t_gate` 在每个合法 system 上都可解析；删除“无 protocol”导致的 runtime error 分支。
- 删除 `_require_protocol()`、protocol-less base-system state 与 compiler/backend 的 optional-protocol handling。
- `simulate()` 与 `ground_state(at=...)` 都只接受 fully constructed、protocol-bound system。
- optimization/scan workflow 用第一个实际 protocol 构造 system，之后可继续使用 `system.with_protocol(new_protocol)`；不为复用 level structure/register 保留半构造对象。
- `with_protocol()` 必须验证新 protocol 与同一 level structure 的 compatibility，并返回新的 immutable system；原 system 不变。

### S17 — 冻结 `LevelStructure` 的七项 public characterization surface（已确认）

唯一 public readable attributes：

```python
ls.name
ls.levels
ls.ryd_level
ls.magnetic_field_G
ls.quantization_axis
ls.decay_rates_per_s
ls.branching_ratios
```

decay/branching schemas：

```python
ls.decay_rates_per_s["r"]
# {"total": ..., "radiative": ..., "blackbody": ...}

ls.decay_rates_per_s["e1"]
# {"total": ...}

ls.branching_ratios["e1"]
# {"to_0": ..., "to_1": ..., "to_L0": ..., "to_L1": ...}
```

- `decay_rates_per_s` 与 `branching_ratios` 是 deep-immutable、physical-level-label-keyed mappings。
- decay rates 单位明确为 `s^-1`；不是 Hamiltonian angular frequency，不使用 `_rad_s` 命名。
- 每个 decay level 只包含有物理定义的 mechanism；未定义项缺席，不用 `None` 或假 `0` 填充。
- mechanism keys 只允许 `"total"`、`"radiative"`、`"blackbody"`；存在 components 时必须与 total 一致。
- `branching_ratios[source_level]` 只描述 radiative decay 条件下的 normalized destination branches，不把 blackbody loss 混入；只保存 preset 实际计算的 source levels。
- 删除旧 `ryd_state_decay_rate`、`ryd_RD_rate`、`ryd_BBR_rate`、`ryd_garb_decay_rate`、`mid_state_decay_rate`、`ryd_branch`、`mid_branch` scalar/nested aliases，并迁移 scripts/notebooks。
- `rydberg_levels`、transitions/`Transition`、detuning-channel maps、local static matrices/couplings、hyperfine offsets、Zeeman/garbage detunings、`_LaserLeg`/CG/dipole ratios、pair-C6 resolver/cache 与 interaction channel classification 全部 private。
- 删除 public `physical_model`、`Delta`、`t_rise`、`rabi_eff`、`time_scale`、`rabi_420`、`rabi_1013`、decay-enable flags、`default_c6`、`rydberg_indices`、`initial_level`、`local_dim`、`index()`、`initial_level_or_default()`、`supports_backend()`。
- preset 默认 initial level 仍可作为 private state-compiler data 服务 E02 的 `initial_state=None`，但不作为第二套 public initial-state API。

## 6. Rydberg interaction 与 ARC

### I01 — 删除 `InteractionSpec`（已确认）

- `InteractionSpec` 不再是 public 或 internal user configuration。
- 相互作用物理由具体 level-structure preset 决定。
- register 只提供几何。
- system 根据 preset + register 生成内部 interaction terms。

### I02 — 用户选择 Rydberg 态，不注入任意 C6（已确认）

采用：

```python
level_structure("01r", ryd_level=70)                 # 70S
level_structure("rb87_7_mp", ryd_level=70)          # 70S
level_structure("rb87_297_clock_4", ryd_level=60)   # 60P3/2
```

- orbital/fine-structure 由 preset 决定。
- 用户只指定主量子数 `ryd_level`。
- 从所有 public presets 删除 `C6_rad_s_um6`。
- 不允许用户注入任意 scalar C6。

### I03 — 默认 Rydberg 态（已确认）

- `1r`、`01r` 是默认 Rb-87 70S 的有效模型。
- `rb87_7_mp` 默认 70S。
- `rb87_7_pm` 按其 preset 的 S-state 默认主量子数（当前为 53S），并允许 `ryd_level` 覆盖。
- `rb87_297_clock_4` 默认 53P3/2，绝不使用 70S fallback。
- 297 改变 `ryd_level` 始终表示另一个 `nP3/2`，不是 S state。

### I04 — ARC 是 C6 的唯一来源（已确认）

- preset 只保存默认量子态，不保存运行时 fallback C6。
- 所有 C6 根据完整原子态由 ARC 计算。
- S-state 结果按量子数缓存。
- P-state 结果还按 pair orientation `(theta, phi)` 缓存。
- 删除 `DEFAULT_C6` 作为 public 常量和运行时 fallback。
- 禁止“改了 `ryd_level`，C6 仍沿用旧硬编码值”。

### I05 — 七能级的 `r` / `r_garb` interaction（已确认）

- 两者是同一 `nS1/2` 的不同磁子能级。
- 在当前各向同性 S-state 有效模型中共享相同 C6。
- `rr`、`r-r_garb`、`r_garb-r_garb` 使用同一个 S-state pair coefficient。

### I06 — 297 的 channel-resolved interaction（已确认）

- 297 的 `r` 与 `r_garb` 是 `nP3/2` 的不同磁子能级。
- ARC 分别计算 `rr`、`r-r_garb`、`r_garb-r_garb` 三类角度依赖 pair interaction。
- 禁止把 target `rr` 的 C6 复制给全部 P-state channels。

### I07 — `interaction_cutoff_um` 的完整语义（已确认）

```python
interaction_cutoff_um=None  # 保留所有物理 pair
interaction_cutoff_um=a     # 保留 nominal distance <= a 的 pair
interaction_cutoff_um=0.0   # 关闭所有 pair interactions
```

- 负数非法。
- cutoff 比较包含小的数值容差，规则 lattice 可直接写 `a`，不需要 `1.01*a`。
- 该参数替代含糊的 `mode="all" | "nn" | "nnn"`。

### I08 — Interaction representation 完全内部化（已确认）

- 删除 public `system.interaction_pairs`。
- 不新增 `system.interaction_strength()`。
- backend/compiler/noise 通过 private IR 使用解析后的 interaction terms。
- 若分析脚本单独研究 C6/Vij，应显式调用相应物理计算，而不是扩大 RydbergSystem API。

## 6A. 物理计算 helper 的模块边界

### PH01 — `ryd_gate.physics` 是受支持但非顶层的专家模块（已确认）

- `ryd_gate.physics` 只提供从实验/原子输入正向计算 Hamiltonian 参数或系统物理数据的函数，例如 Rabi frequency、AC-Stark/scattering、Zeeman shift 与 ARC C6。
- 这些名字不加入 API01 的 `ryd_gate` 顶层六项导出；用户需要时显式从 `ryd_gate.physics` 导入。
- waveform/pulse construction 不属于 atomic physics；Blackman pulse 等 helper 归入 `ryd_gate.protocols`，与 P13 的 `phase_from_chirp` 位于同一层。
- 任何消费 `EvolutionResult`、expectation、amplitude、samples 或 ensemble statistics 的计算都不得进入 `ryd_gate.physics`，继续留在 scripts/notebooks。
- system/preset materialization 所需的 ARC cache、branching-ratio builder、laser-leg ratios 和 physical-model fields 可以保留，但必须是 private implementation。
- `ryd_gate.physics` 必须使用显式 `__all__`；NumPy/SciPy imports、缓存函数、校准常量和其他 module globals 不能因为缺少 export boundary 而成为 accidental public API。
- 具体 public helper 名称继续逐项 grill；实现者不得把当前模块中所有非下划线名字自动视作已确认保留。

### PH02 — Manovitz AC-Stark/scattering 校准模型迁入 notebook（已确认）

- 删除 public `ryd_gate.physics.compute_shift_scatter()`。
- 将其 D1/D2 scalar/vector shift、scattering 公式，以及 `784 nm / 160 uW / 1 um waist` 的 Manovitz calibration assumptions 和 constants，完整迁入 `scripts/notebooks/02_ac_stark_addressing.ipynb` 的用户可见代码单元。
- notebook 继续生成原有 AC-Stark landscape/sensitivity/profile 图与数值；不得只复制历史输出而丢失可执行公式。
- 删除仅服务该模型的 public/dynamic `FREQ_D1`、`FREQ_D2`、`LAMBDA_D1`、`LAMBDA_D2`、calibration constants 与 module `__getattr__` machinery；必要的 ARC lookup 可在 notebook-local helper 中显式完成。
- 该模型不是通用 system characterization API：当前函数没有把 power/intensity、beam geometry 和 atomic state 全部作为输入，却以通用名称返回 Hz。
- 未来若确有多个 production workflows 需要公共 AC-Stark 模型，应另行设计一个把 intensity/power、beam geometry、atomic state、polarization 和单位全部显式化的新函数；本次不预留兼容 shim。

### PH03 — 冻结 `ryd_gate.physics.__all__` 为五个函数（已确认）

唯一 public surface：

```python
from ryd_gate.physics import (
    single_photon_rabi,
    rb87_7_mp_rabi_frequencies,
    rb87_297_clock_rabi_frequencies,
    zeeman_shift_rad_s,
    arc_pair_c6_rad_s_um6,
)
```

- `our_laser_rabis` 重命名为 `rb87_7_mp_rabi_frequencies`；名称直接对应 physical preset/manifold，不再使用仓库作者视角的 `our`。
- `direct_297_rabis` 重命名为 `rb87_297_clock_rabi_frequencies`；返回 target 与 garbage branch 的 Rabi frequencies，并保留 clock-state `1/sqrt(2)` 因子。
- `single_photon_rabi` 保留为任意明确原子跃迁的 generic ARC calculation。
- `zeeman_shift_rad_s` 保留为通用线性 Zeeman calculation。
- `arc_pair_c6_rad_s_um6` 保留给 I08 所述的独立 pair-physics 研究与 preset internal reuse。
- `electric_field_uniform_beam`、`lande_gj`、`rydberg_zeeman_shift_rad_s`、branching builders、ARC/cache/calibration helpers 和 physical-model field builders 全部 internal。
- `RYD_LEVEL_OUR`、`RYD_LEVEL_297` 等默认常量不公开；Rydberg 主量子数通过 preset/函数参数显式指定。
- 不保留旧函数名 alias、deprecation shim 或 wildcard accidental exports；所有 scripts/notebooks/tests 同步迁移到新名字。

### PH04 — public physics helper 用参数名显式编码单位（已确认）

不引入 `pint`、quantity wrapper 或新的 unit class。三个 Rabi helpers 的 public signatures 冻结为：

```python
single_photon_rabi(
    power_w,
    beam_area_um2,
    *,
    n1, l1, j1, mj1,
    n2, l2, j2, q,
) -> float  # rad/s

rb87_7_mp_rabi_frequencies(
    power_420_w,
    power_1013_w,
    beam_area_um2,
    *,
    ryd_level=70,
) -> tuple[float, float]  # omega_420, omega_1013; rad/s

rb87_297_clock_rabi_frequencies(
    power_297_w,
    beam_area_um2,
    *,
    ryd_level=53,
) -> tuple[float, float]  # omega_r, omega_r_garb; rad/s
```

- 删除含糊的 `beam_area` 参数名；面积单位固定为 `um^2`，功率固定为 W。
- `rb87_7_mp_rabi_frequencies` 的 tuple 顺序固定为 `(omega_420, omega_1013)`。
- `rb87_297_clock_rabi_frequencies` 的 tuple 顺序固定为 `(omega_r, omega_r_garb)`。
- 所有 Rabi outputs 固定为 angular frequency `rad/s`，不能在不同 helper 中混用 Hz 与 rad/s。
- 负 power、非正 beam area、非法量子数或 polarization `q` 必须在 public boundary 报清楚的 validation error。

## 7. Simulation 与 backend

### E01 — 只有两个公共演化入口（已确认）

- `simulate()`
- `simulate_ensemble()`

backend compiler、TN compiler、specialized DMRG 和求解器类不从顶层公开。

### E02 — 一个或多个初态的自然输入（已确认）

- `initial_state=None`：按 E27 统一使用 `|1...1>`，不读取 preset-specific default。
- `initial_state="plus"`：保留明确的 plus-state shorthand。
- flat physical level labels：一个 product state，例如 `["0", "1"]`。
- nested labels：多个 product states，例如 `[["0","0"], ["0","1"]]`。
- 输入一个就演化一个；输入多个就共享 Hamiltonian 编译并分别演化多个。
- 不接受任意 public dense vector 或 arbitrary backend-native state。
- 删除 `all_ground/all_0/all_zero/all_1/all_r` aliases。

### E03 — exact user backend 只保留 `exact_ode`（已确认）

删除：

- `exact_dense`
- `exact_sparse`
- bare `exact`
- dense/sparse expm backends

唯一 exact user name：

```python
backend="exact_ode"
```

### E04 — dense/sparse 是用户可选 Hamiltonian format（已确认）

```python
backend_options={
    "hamiltonian_format": "auto" | "dense" | "sparse",
    "rtol": ...,
    "atol": ...,
}
```

- format 是 storage/matvec 策略，不是不同 backend。
- 必须允许用户强制选择，因为某些规模 dense 更快、另一些 sparse 才可运行。
- sparse 路径不能偷偷 `.toarray()`。
- `auto` 必须根据结构/内存估算选择安全路径。

### E05 — exact solver 固定 DOP853（已确认）

- ODE solver 完全根据 `rtol/atol` 自适应选步。
- 不公开 `max_step`。
- unknown backend option 报错。
- 必须检查 solver success。
- 默认 `rtol/atol` 已由 E11 冻结为 `1e-8` / `1e-12`。

### E06 — `t_eval` 的严格语义（已确认）

- `t_eval=None` 是唯一 final-time shorthand。
- `t_eval=[]` 报错，并提示使用 `None`。
- 显式 `t_eval` 必须一维、finite、严格递增、无重复，且在 `[0, system.t_gate]`。
- 显式 `t_eval` 表示恰好这些 expectation measurement times，不自动追加 public 终点。
- 提供显式 `t_eval` 但没有 observables 时拒绝调用。
- `t_eval=None` 且没有 observables 是合法的，因为用户仍可读取最终 `amplitude()` 或调用 `sample()`。
- backend 可内部演化到 `t_gate`，但不能把内部追加点暴露为 measurement time。
- MPS/PEPS 必须让 requested times 成为真实 step boundaries，不能 round 后谎报原时间。

### E07 — 不保存完整 state trajectory（已确认）

- exact/MPS/PEPS 都只保留实现最终读取所需的 private final backend state。
- requested times 的中间 state 只在求解期间用于计算 expectations，随后丢弃。
- 不提供 `store_states` escape hatch。
- decay integral 通过请求足够密的 population expectations，在 script 中数值积分。

### E08 — 完全删除 non-Hermitian evolution（已确认）

- src 中所有 Hamiltonian 必须 Hermitian。
- 删除 `enable_rydberg_decay`、`enable_intermediate_decay`。
- 删除 imaginary decay diagonals、norm-loss 语义和对应测试。
- sample 不再需要 conditional-on-survival 或 loss outcome 规则。
- 若一阶近似失效，未来应实现完整 Lindblad/quantum trajectories；不保留当前 no-jump 模型冒充它。

### E09 — decay physical data 保留，后处理在 scripts（已确认）

- lifetime、decay rate、branching ratio 仍是 system/level preset 的物理数据。
- scripts 使用 Hermitian dynamics 计算一阶 decay budget：

  ```text
  p_decay^(1) = sum_k Gamma_k * integral <n_k(t)>_(Gamma=0) dt
  ```

- 积分点由 script 通过 `t_eval` 明确请求。

### E10 — `simulate()` 的返回基数与初态一致（已确认）

- 单初态输入直接返回一个 `EvolutionResult`。
- 多初态 nested-label 输入返回 `tuple[EvolutionResult, ...]`。
- 不返回 mutable list，也不为单初态额外套一层长度为 1 的容器。
- 多结果 tuple 可索引、迭代和解包，但不能增删。
- 空的初态 batch 报错。

### E11 — exact ODE 稳定默认容差（已确认）

```python
rtol = 1e-8
atol = 1e-12
```

- 两项都有稳定默认值，普通 `simulate()` 不强迫用户重复填写。
- 用户进行高精度审计时通过 `backend_options` 显式覆盖。
- `rtol`、`atol` 必须 finite 且严格为正。
- SciPy 自带的宽松默认值不用于本项目。

### E12 — ground-state solver 只接受带明确 geometry 的 `1r` system（已确认；energy-only 输出由 E15 替代）

- ground-state search 不是 `simulate()` backend，也不返回 `EvolutionResult`。
- 它只接受 fully constructed `RydbergSystem`；geometry、site ordering、boundary 与 interaction 都来自该 system 的 `Register` 和 `1r` preset。
- level structure 必须严格为 `1r`；所有其他 presets 均在 preflight 拒绝。
- `TenpyDMRGBackend` / `YASTNPEPSBackend` 及其 `find_ground_state()` 不作为用户界面；solver implementations 是 private adapters。
- 早期“standalone `ground_state_energy()` 只返回 float”的输出设计已被 E15 替代。

### E13 — ground state 强制指定 instantaneous time（已确认，适用于 E15）

```python
result = system.ground_state(
    at=...,
    method=...,
)
```

- `at` 是 mandatory keyword-only input；不提供默认值。
- `at` 必须 finite，并位于闭区间 `[0, system.t_gate]`。
- solver 使用 P17 canonical lowering 在 `at` 处冻结完整 Hamiltonian；geometry、local drives 和 interactions 都包含在同一 snapshot 中。
- 不把空 `at` 解释为 final time，也不尝试判断任意 callable protocol 是否“实际上恒定”。

### E14 — DMRG 与 PEPS imaginary-time 属于同一个 ground-state solver family（已确认，由 E15/E16 具体化）

- `method` 必须显式为 `"dmrg"` 或 `"peps_imaginary_time"`；不设置隐藏默认算法。
- 两种 method 都只接受 E12 的 `1r` system，并遵守 E13 的 mandatory `at` snapshot。
- MPS-DMRG 与 PEPS imaginary-time implementation 是同一 deep module 后面的两个 private adapters。
- 删除 public `TenpyDMRGBackend.find_ground_state()`、`YASTNPEPSBackend.find_ground_state()` 以及 `simulate_tn(method="dmrg")`；PEPS real-time evolution 仍由 `simulate(..., backend="peps")` 提供。
- exact ground-state diagonalization 继续由需要它的 notebook/script 显式调用 `eigsh`，不增加第三个 method。
- 两种 method 的旧 energy-only output 均已替代；DMRG result 见 E15，PEPS 对称 result 见 E16。

### E15 — `system.ground_state()` 返回 stateful `GroundStateResult`（已确认）

```python
result = system.ground_state(
    at=...,
    method="dmrg",
    observables={"C_ij": z_i @ z_j},
    method_options={...},
)

result.expectation("energy")
result.expectation("C_ij")
result.amplitude(labels, phase_reference=reference_labels)
result.sample(shots=1000, seed=0)
```

- 当前旧 `system.ground_state()` 的“返回 all-ground product vector”语义删除；初态继续使用 E02 的 physical level labels。
- `GroundStateResult` 是独立的 specialized result type，不是带 fake `times=[0]` 的 `EvolutionResult`。
- public 读取面只有 `expectation(name)`、`amplitude(labels, phase_reference=...)` 和 `sample(shots, seed)`；不公开 backend-native MPS、generic metadata、solver diagnostics、`times` 或 `.energy` alias。
- Hamiltonian energy 是 observable；solver 已经必然计算它，因此 reserved `expectation("energy")` 始终存在并返回 real scalar。
- 其他 expectation 只计算调用者在 `observables={...}` 中显式请求的 Hermitian scalar expressions；返回 real scalar，不返回 length-1 array。
- backend state 可以由 result 私有持有，以便 amplitude 和 sampling 懒计算，但用户不能直接取得它。
- `sample()` 继续要求显式 positive integer `shots` 与 integer `seed`，返回 physical-level-label counts。
- ground-state eigenvector 的 global phase 不唯一，因此 complex amplitude 必须在读取时提供 physical-label `phase_reference`。内部旋转 gauge，使参考 basis amplitude 为 non-negative real；参考 amplitude 为零或数值上不可分辨时明确报错。
- expectation、energy 和 sampling 与 global phase 无关，不接受也不需要 `phase_reference`。
- standalone `ground_state_energy()` 删除；其 DMRG 用途由 `system.ground_state(...).expectation("energy")` 完全覆盖。
- TN notebook 的 `C_ij`、structure factor 和 correlation ratio 通过 O11/O13 observable algebra 显式组合并请求，不再读取 raw MPS。

### E16 — PEPS imaginary-time 返回与 DMRG 相同的 `GroundStateResult`（已确认，具体化 E14/E15）

```python
result = system.ground_state(
    at=...,
    method="peps_imaginary_time",
    observables={...},
    method_options={...},
)
```

- PEPS 与 DMRG 的 public result surface 完全一致：`expectation(name)`、`amplitude(labels, phase_reference=...)`、`sample(shots, seed)`。
- requested observables 通过 PEPS observable lowering 测量；reserved `expectation("energy")` 始终存在。
- basis amplitude 使用 product-state PEPS contraction，并遵守 E15 的 explicit phase-reference gauge；不要求 YASTN 暴露 backend-native state。
- sampling 使用 YASTN 的 Boundary-MPS、CTM 或 BP environment sampler；environment/contraction 选择及精度全部由求解时显式传入的 `method_options` 固定。
- amplitude 与 sampling 都保持 lazy；用户未调用时不构造相应 contraction/environment。
- PEPS contraction/sampling 是受 method options 控制的数值近似，但不产生第二套 result type，也不通过 generic result metadata 报告设置。
- contraction、sampling 或 convergence 未达到调用者指定条件时抛出清楚的异常，不静默返回看似精确的值。
- PEPS real-time evolution 继续由 `simulate(..., backend="peps")` 提供，与本 imaginary-time ground-state path 分离。

### E17 — 显式 `initial_state` 选择简并 ground-space 代表（已确认）

```python
result = system.ground_state(
    at=...,
    method=...,
    initial_state=["1", "r", "1", "r", ...],
    observables={...},
    method_options={...},
)
```

- `initial_state` 是 mandatory input，使用与 E02 相同的 flat physical level labels；长度必须等于 `system.register.N`，且每项必须属于 `1r` levels。
- 同一输入语义适用于 DMRG 与 PEPS imaginary-time；不接受 backend-native MPS/PEPS seed。
- 删除 `"af1"`、`"af2"`、`"all_ground"` 等 hidden pattern aliases；checkerboard、domain 或其他 seed 由 script/notebook 显式生成 labels。
- 该 seed 同时负责数值初始化和选择 degenerate / nearly-degenerate ground space 中的一个代表态。
- solver 不自动构造 cat state，也不把返回态声称为数学上唯一的 ground state。
- `expectation("energy")` 在精确简并空间中保持相同；其他 expectations、amplitudes 和 samples 明确属于实际收敛的那个代表态。
- E15 的 `phase_reference` 只固定该代表态的 global phase，不能也不试图选择简并子空间中的方向。
- 未满足显式 convergence 条件时抛错；不以“可能简并”为理由静默接受未收敛结果。

### E18 — ground-state solver 使用单一 validated `method_options` mapping（已确认，由 E19/E20 具体化）

- `system.ground_state(..., method_options={...})` 是 DMRG 与 PEPS imaginary-time 唯一的数值选项入口。
- `method_options` 是 mandatory keyword-only mapping；method 决定其严格 schema。
- DMRG 与 PEPS 可以拥有不同 key 集合，但每个 key 必须表达调用者真正需要控制的 accuracy target 或 resource limit。
- unknown、拼错、重复表达或不适用于所选 method 的 key 立即报错；不静默忽略。
- 不导出 `DMRGOptions`、`PEPSOptions`、TeNPy/YASTN engine option classes 或第二套 kwargs surface。
- mixer、warmup strategy、environment implementation class、内部 optimizer iteration 等 engine plumbing 留在 private adapter，除非后续 grill 证明它们是无法隐藏的物理/数值控制。
- result 不重复公开 method options、solver diagnostics 或 convergence metadata；未满足显式目标时直接抛错。
- `bond_dimension`、`truncation_tolerance`、`energy_tolerance`、`max_iterations` 目前只是讨论示例，不能在完成后续 schema grill 前视为冻结名称。

### E19 — DMRG `method_options` 只公开五个 keys（已确认，具体化 E18）

```python
method_options={
    "bond_dimension": 128,
    "discarded_weight_tolerance": 1e-10,
    "relative_energy_tolerance": 1e-8,
    "entropy_tolerance": 1e-6,
    "max_sweeps": 20,
}
```

- 五个 keys 全部 mandatory；不提供隐藏 accuracy/resource defaults。
- `bond_dimension` 与 `max_sweeps` 必须是 positive integers，分别表示 MPS bond-dimension cap 与 sweep hard cap。
- 三个 tolerances 必须 finite 且严格为正；`discarded_weight_tolerance < 1`。
- 返回结果前必须同时满足：相邻 sweeps 的 relative energy change、entanglement-entropy change，以及实际 discarded Schmidt weight 均不超过调用者阈值。
- 达到 `max_sweeps` 仍未同时满足三项条件时抛错，提示调用者增加 bond dimension/sweeps 或重新审视 tolerances。
- `svd_min`、`trunc_cut`、`chi_max` 等 TeNPy names 只在 private adapter 内映射，不泄漏到 public schema。
- mixer、Lanczos options、minimum sweeps、chi growth schedule 和 normalization cleanup 均为 private implementation；不接受 passthrough engine kwargs。

### E20 — PEPS imaginary-time `method_options` 只公开七个 keys（已确认，具体化 E18）

```python
method_options={
    "bond_dimension": 8,
    "environment_bond_dimension": 32,
    "discarded_weight_tolerance": 1e-8,
    "relative_energy_tolerance": 1e-6,
    "environment_tolerance": 1e-8,
    "environment_max_iterations": 50,
    "imaginary_time_schedule": (
        (0.10, 30),
        (0.03, 30),
        (0.01, 40),
    ),
}
```

- 七个 keys 全部 mandatory；不提供隐藏 accuracy/resource defaults。
- `bond_dimension`、`environment_bond_dimension`、`environment_max_iterations` 必须是 positive integers。
- 三个 tolerances 必须 finite 且严格为正；`discarded_weight_tolerance < 1`。
- `imaginary_time_schedule` 必须是非空 immutable sequence of `(dtau, max_steps)`；每个 `dtau` finite、positive 且严格递减，每个 `max_steps` 是 positive integer。
- schedule 同时规定 imaginary-time Trotter refinement 与 hard resource cap；adapter 不在调用者不知情时追加 stages。
- 只有 relative energy change、实际 discarded weight 和 contraction-environment residual 全部满足调用者阈值，并且最终最小-`dtau` stage 已执行，才返回 `GroundStateResult`。
- schedule 用尽或 environment 达到 iteration cap 而未同时满足条件时抛错。
- NTU/CTM implementation、warmup strategy、bond optimizer、SVD initialization、dtype 和 engine-specific kwargs 均为 private implementation。
- expectation 与 sampling 可使用按 `environment_bond_dimension` / `environment_tolerance` 收敛的 double-layer environment；complex amplitude 必须另做 single-layer product-bra contraction。
- 三种读取复用相同的显式 accuracy settings，而不是错误地复用同一个 environment object；result 不公开 environment、contraction object 或 diagnostics。

### E21 — `"energy"` 是 `GroundStateResult` 唯一 reserved observable name（已确认）

- `result.expectation("energy")` 始终表示 E13 snapshot Hamiltonian `H(at)` 在实际收敛 ground-state representative 上的 expectation。
- ground-state solver 本来就需要该能量进行优化和收敛判断，因此它始终存在，不属于额外 eager analysis。
- 用户传入 `observables={"energy": expr}` 时在求解前报 reserved-name error；不得覆盖或影藏 solver energy。
- 不增加 `"H"`、`"E0"`、`"ground_energy"` 等 aliases，也不恢复 `.energy` property。
- 其他 observable labels 仍完全由调用者定义。
- `EvolutionResult` 不保留 `"energy"` 特殊语义；含时 Hamiltonian expectation 若需要，必须作为普通显式 observable 请求。

### E22 — MPS/PEPS time evolution 使用 mandatory validated `backend_options`（已确认）

- `simulate(..., backend="mps" | "peps", backend_options={...})` 是 TN time-evolution 数值选项的唯一入口。
- `backend_options` 必须是显式、非空的 keyword-only mapping；backend 决定严格 schema。
- unknown、拼错、不适用或 passthrough engine keys 在演化前报错，不静默忽略。
- public keys 只表达调用者真正需要控制的 time discretization、state truncation、bond-dimension resource cap 与 contraction accuracy。
- 不导出 `TenpyOptions`、`YASTNOptions` 或其他 engine option classes，也不接受 arbitrary TeNPy/YASTN kwargs。
- backend options 不通过 result metadata 重复返回；未满足显式 accuracy 条件时抛错。
- `exact_ode` 不受本条 mandatory-mapping 规则影响，继续使用 E04/E11 的 schema 与稳定默认 tolerances。
- MPS 与 PEPS 的准确 key schemas 由后续决定分别冻结，在此之前实现者不得自行选取。

### E23 — MPS TDVP `backend_options` 只公开三个 keys（已确认，具体化 E22）

```python
backend_options={
    "time_step": 0.01,
    "bond_dimension": 256,
    "discarded_weight_tolerance": 1e-8,
}
```

- 三个 keys 全部 mandatory；不提供隐藏 accuracy/resource defaults。
- `time_step` 必须 finite 且 strictly positive，表示 TDVP step upper bound。
- backend 必须按每个 requested `t_eval` anchor 与 `system.t_gate` 分段；可缩短局部末步以精确命中 anchor，但绝不能超过 `time_step` 或谎报时间。
- `bond_dimension` 必须是 positive integer，表示 MPS bond-dimension hard cap。
- `discarded_weight_tolerance` 必须 finite、strictly positive 且 `< 1`，表示整段演化累计允许丢弃的 Schmidt weight。
- 累计 discarded weight 超标时抛错，并提示减小 `time_step` 或增加 bond dimension；不通过 result diagnostics 让用户事后判断。
- two-site TDVP、Krylov tolerance、per-step SVD cutoff、canonicalization 与 normalization cleanup 均为 private implementation。

### E24 — PEPS real-time 显式选择 BP 或 CTM measurement（已确认）

```python
backend_options={
    # ...
    "measurement_method": "belief_propagation",  # 或 "ctm"
}
```

- `measurement_method` 是 PEPS public numerical choice，因为 BP 与 CTM 在 cost、accuracy 和 observable capability 上存在真实差异。
- public values 精确为 `"belief_propagation"` 与 `"ctm"`；不接受 YASTN class names、缩写 aliases 或 arbitrary plugin objects。
- backend 必须在演化前根据 requested observables preflight 所选 method 的 capability。
- 若 BP 不支持某个 pair/multi-site contraction，立即报 capability error；禁止静默 fallback 到 CTM。
- expectation 使用所选 double-layer measurement environment；sampling 使用同一 method 的 conditional environment sampler。
- complex amplitude 与 global phase 需要 single-layer contraction，因此固定走 private boundary-MPS product-bra contraction，不受 `measurement_method` 控制。
- notebook/script 可以继续显式比较 BP 与 CTM numerical behavior，而无需直接操作 backend-native state。

### E25 — PEPS real-time `backend_options` 只公开八个 keys（已确认，具体化 E22/E24）

```python
backend_options={
    "time_step": 0.01,
    "bond_dimension": 8,
    "discarded_weight_tolerance": 1e-8,
    "measurement_method": "belief_propagation",
    "environment_bond_dimension": 32,
    "environment_tolerance": 1e-8,
    "environment_max_iterations": 50,
    "device": "cpu",
}
```

- 八个 keys 全部 mandatory；不提供隐藏 accuracy/resource defaults。
- `time_step` finite 且 strictly positive；backend 可缩短局部末步以精确命中 anchors，但绝不能超过它。
- `bond_dimension`、`environment_bond_dimension`、`environment_max_iterations` 必须是 positive integers。
- `discarded_weight_tolerance` 与 `environment_tolerance` 必须 finite、strictly positive；前者 `< 1` 并按整段 real-time evolution 累计检查。
- `measurement_method` 严格遵守 E24。
- `device` 精确为 `"cpu"` 或 `"cuda"`；请求 CUDA 而 PyTorch/CUDA/YASTN support 不可用时立即报错，禁止 silent CPU fallback。
- environment accuracy settings 同时约束 requested expectations、lazy sampling 与 lazy single-layer amplitude contraction；各算法可以使用不同 private objects，但不能偷偷换用更宽松设置。
- cumulative discarded weight 或 environment convergence 未达标时抛错，不通过 metadata 返回失败结果。
- 删除旧 `use_cuda`、`yastn_backend`、`require_gpu`、`device` passthrough、`dtype`、`update_environment`、`initialization`、`max_iter`、`tol_iter`、`ctm_*` 等 engine-shaped options。
- update algorithm 固定为 private NTU，numeric dtype 固定为 `complex128`。

### E26 — 冻结各 backend 的 observable term capability，禁止 PEPS measurement fallback（已确认）

- `exact_ode` 与 MPS 支持每个 `ObservableExpr` term 作用于任意有限数量的 distinct sites；实际可计算规模仍由用户承担。
- PEPS real-time 使用 `measurement_method="belief_propagation"` 时，每个 term 最多作用于一个 distinct site；一个 expression 可以是任意多个 one-site terms 的和。
- PEPS real-time 使用 `measurement_method="ctm"` 时，每个 term 最多作用于两个 distinct sites，因此支持 `C_ij = <Z_i Z_j>` 等 pair correlations。
- BP 遇到 two-site term 必须在演化开始前报 capability error，绝不暗中构造 CTM environment 或切换 measurement method。
- PEPS ground-state solver 固定使用 private CTM measurement 计算 reserved `"energy"` 与所有 requested expectations，从而保证 E15/E16 要求的 pair-correlation 能力；不为 E20 增加第八个 `method_options` key。
- `amplitude()` 与 `sample()` 使用各 backend 的 specialized contractions，不属于 `ObservableExpr` measurement，因此不受本条 term-site-count 上限约束。
- 所有 observable capability 检查必须在 ODE/TN evolution 或 ground-state iteration 开始前完成；不能计算完成后才因 measurement 失败。

### E27 — `initial_state=None` 对所有保留 presets 统一为 `|1...1>`（已确认，具体化 E02）

```python
simulate(system, initial_state=None)
# 精确等价于
simulate(system, initial_state=["1"] * system.N)
```

- `1r`、`01r`、`rb87_7_mp`、`rb87_7_pm` 与 `rb87_297_clock_4` 都使用同一规则；五个保留 presets 均必须包含 physical level label `"1"`。
- 删除 preset/compiler 中的 `initial_level`、`initial_level_or_default()` 与每个 preset 的 initial-state metadata；默认行为不再存在两套来源。
- 默认选择实际被 Rydberg drive 激发的 `|1>`；尤其避免 Direct-297 默认落在完全 dark 的 `|0>` 而产生误导性的静止结果。
- CZ computational-basis comparison、checkerboard/domain seeds 与其他有研究含义的初态仍由 script/notebook 显式传入 physical labels。
- `simulate_ensemble()` 完全委托 `simulate()`，因此继承同一默认且不单独实现 initial-state logic。

## 8. Observable expression 与 expectation

### O01 — 所有仓库 observable 必须由 system factory 组合得到（已确认）

- system/preset 提供与物理 level labels 一致的 immutable observable factory。
- scripts/notebooks 需要的 populations、projectors、coherences、pair products 等必须能由基础 expression 组合。
- 不为每个分析结果在 src 注册一个具名 postprocessing function。

### O02 — `simulate()` 只接受 structured scalar observables（已确认）

目标形式：

```python
simulate(
    system,
    observables={
        "n_r": expr,
        "rr": expr2,
    },
)
```

- label 唯一。
- 不接受 arbitrary user matrices、callables 或 backend-native operators。
- 不引入 public vector-valued observable/bundle。
- per-site profile 用多个 scalar expressions 表达；backend 可内部 batch/coalesce。

### O03 — 最小 expression algebra（已确认，由 O11 收敛 factory surface）

- factory primitives 精确只有 O11 的 `E(ket, bra, site)` 与 `n(level, site)`。
- expression algebra 精确支持 `+`、`-`、一元 `-`、scalar `*`、operator product `@`、`.dagger()` 与 O13 的 Python `sum()`。
- level sums、weighted sums、finite product projectors 和所有 finite scalar combinations 都由上述两个 primitives 与 algebra 显式构造，不对应额外方法名。
- scalar 可为 complex intermediate；传给 solver 的最终 expression 必须按 O12 Hermitian。
- 不引入第二套 matrix/callable API，也不在实现阶段自行增加 convenience aliases。

### O04 — expectation 只读取明确请求的数据（已确认）

- 删除 public `.expectations` mapping。
- `result.expectation(name)` 只读取 simulation 已记录的值。
- 未请求/未知 name 抛出 `KeyError`，提示重新调用 `simulate()`。
- result 不保存 system 来事后测量新 observable。
- 禁止 lazy expectation。

### O05 — expectation 始终与 `result.times` 对齐（已确认）

- `result.times` 始终是一维数组。
- `t_eval=None` 时 `result.times == [system.t_gate]`。
- `result.expectation(name)` 始终返回等长一维数组，即使只有终点也是 shape `(1,)`。
- 不使用 scalar/array 联合返回类型。

### O06 — observable 必须 Hermitian，expectation 始终 real（已确认，替代旧结论）

- observable expression 必须 Hermitian。
- expectation 返回 `float64` 数组。
- 容差内的数值虚部丢弃；超出容差报错。
- 非 Hermitian coherence 应拆成 Hermitian X/Y observables，在 script 中组合。
- 这条决定替代旧 handoff 中“expectations 必须保留 complex dtype”的结论。

### O07 — 删除完整 basis probabilities（已确认）

- 删除 `.probabilities()`。
- 不构造指数规模的完整计算基概率分布。
- sampling 必须按支持 backend 的合理算法延迟执行，不能把 probabilities 作为公共中间结果。

### O08 — 删除无价值的 instantaneous eigenbasis diagnostic（已确认）

删除 `scripts/notebooks/02_ac_stark_addressing.ipynb` 中默认关闭、无输出、无下游消费的逐时刻重对角化 `H(t)` 并取最大 overlap 的 F1/F2 诊断。

### O09 — backend capability 在演化前检查（已确认）

- initial state、protocol、t_eval、observable expression 和 backend options 必须 preflight。
- unsupported expression 立即报 capability error。
- 禁止耗时演化后才发现不支持。
- 禁止 TN 静默忽略、漏 key 或自动 dense fallback。

### O10 — 删除 public identity/norm observable（已确认）

- 删除 `system.observables.identity()`。
- 所有 Hamiltonian 已限定为 Hermitian，归一化态的 identity expectation 恒为 1，不再提供物理信息。
- logical/Pauli observables 由真实 level projectors 组合，例如 `Z_i = n_0,i - n_1,i`。
- backend 构造 Hamiltonian 所需的 internal identity operator 继续保留。
- norm preservation 属于 backend 数值验证或未来明确的 solver diagnostics，不伪装成用户 observable。

### O11 — ObservableFactory 只保留 `E()` 与 `n()`（已确认）

public factory 精确限定为两个物理原语：

```python
obs.E(ket, bra, site)   # |ket><bra|_site
obs.n(level, site)      # |level><level|_site
```

- `n(level, site)` 数学上等价于 `E(level, level, site)`，但保留为最常用的 occupation/projector 物理词汇。
- 用户在 scripts/notebooks 中通过 `+`、`-`、一元 `-`、scalar `*`、operator product `@` 与 `.dagger()` 显式组合所有其他标量算符。
- 删除 `level_sum()`、`weighted_level_sum()`、`product_projector()`、`site_populations()`；不保留 compatibility aliases。
- `identity()` 已由 O10 删除。
- 例如两原子总 Rydberg 布居直接写成 `obs.n("r", 0) + obs.n("r", 1)`；product-state projector 直接写成 `obs.n("1", 0) @ obs.n("r", 1)`。
- `system.observables` 不是 preset named registry，也不返回自动命名的 bundle；传给 `simulate(..., observables={...})` 的结果名称始终由调用者明确指定。
- backend 可在 lowering 时识别并合并这些表达式，但不得把便利别名重新暴露给用户。

### O12 — 允许 complex intermediate scalar，最终 observable 必须 Hermitian（已确认）

```python
A = obs.E("0", "1", site=0)
X = A + A.dagger()
Y = -1j * (A - A.dagger())
```

- `ObservableExpr` 的 scalar multiplication 接受 real 或 complex numbers。
- 允许组合过程中出现 non-Hermitian intermediate expression。
- 只在完整 expression 传入 `simulate(..., observables=...)` 时验证其 Hermiticity；最终 non-Hermitian expression 在演化前报错。
- `result.expectation(name)` 仍按 O06 返回 real `float64` 数组。
- 不增加 `X()`、`Y()`、`real_part()`、`imag_part()` 等 factory aliases。

### O13 — `ObservableExpr` 支持标准 Python `sum()`（已确认）

```python
n_r = sum(obs.n("r", i) for i in range(system.N))
```

- `ObservableExpr.__radd__` 仅把 Python `sum()` 的 additive seed `0 + expr` 解释为 `expr`。
- 除整数/实数零这一特例外，number 与 `ObservableExpr` 相加仍报 `TypeError`；不把任意 scalar 隐式提升为 identity operator。
- 这只是 expression algebra 的组合语义，不新增 `level_sum()`、`zero()` 或其他 factory alias。
- 空 generator 的 Python 结果仍是整数 `0`，不能作为 observable 传给 `simulate()`；此时 preflight 报清楚的类型错误。

## 9. EvolutionResult 的唯一公共读取面

### ER01 — 结果只提供三类物理读取（已确认）

```python
result.expectation(name)
result.amplitude(level_labels)
result.sample(shots=..., seed=...)
```

`result.times` 是 expectation 的时间坐标，不是第四类物理结果。

### ER02 — final backend state 完全私有（已确认）

- 删除 public `.psi_final`。
- 删除 public `.final_state`。
- 删除 public `.states` trajectory。
- backend state 仅作为实现 `amplitude()` / `sample()` 的私有数据。
- scripts 不能把它传入下一段演化；多阶段过程用一个 piecewise protocol。

### ER03 — `amplitude(labels)` 的输入与初态标签统一（已确认）

```python
result.amplitude(["0", "0"])
```

- labels 必须是与 initial-state input 相同的 flat physical level sequence。
- 不公开 integer-index wrapper。
- 不公开 `BasisSpec.index()`。
- 不接受拼接字符串 `"00"` alias。
- 只返回真实 `t_gate` 的一个 complex amplitude。

### ER04 — amplitude 支持 exact、MPS、PEPS（已确认）

- exact：直接 coefficient lookup。
- MPS：product-state overlap contraction。
- PEPS：single-layer approximate contraction，不允许隐式 dense conversion。
- TN normalization/compression 必须保留 complex normalization/global-phase factor，否则 amplitude phase 无意义。
- complex CZ overlaps 的 phase 参与 Nielsen fidelity、conditional phase、optimization objective 和 benchmark，不能用 probability 替代。

### ER05 — `sample()` 完全懒惰（已确认，由 ER09 扩展 backend scope）

```python
result.sample(shots=1000, seed=7)
```

- `shots`、`seed` 是 mandatory keyword-only integers。
- 不允许 `seed=None`，不接收外部 RNG object。
- 未调用时不计算 samples 或完整 probability vector。
- 返回 `Counter[tuple[str, ...]]`，例如 `{("0", "r"): 311}`。
- 单原子结果也使用 `("r",)`，不使用字符串拼接。
- exact、MPS、PEPS 的 backend scope 由 ER09 冻结；任何路径都禁止为 sampling 隐式 dense 化。

### ER06 — 只请求 expectation 时不做 sampling（已确认）

- `sample()` 不得在 simulation 阶段 eager 执行。
- 只请求 expectations 的用户不承担 sampling 开销。

### ER07 — PEPS amplitude 的数值设置显式输入（已确认）

- PEPS contraction method、bond dimension、tolerance 必须作为显式 `simulate(..., backend_options=...)` 输入。
- result 私下保留执行 `amplitude()` 所需的这些设置，但不把它们重新包装成 public metadata。
- `result.amplitude(labels)` 在 exact/MPS/PEPS 上仍只返回一个 complex scalar。
- PEPS contraction 达到用户给定 tolerance 才返回；未收敛时抛出清楚的 convergence error，不能静默返回一个来源不明的近似值。
- 用户脚本中的 simulation call 本身就是数值 provenance；不恢复 generic `.metadata`。

### ER08 — 不提供 public solver diagnostics 或 result metadata（已确认）

- `EvolutionResult` 不提供 `.metadata`、`.diagnostics`、`.stats` 或等价的通用逃生口。
- exact ODE 的 `nfev`、内部实际步数、自动选择的 dense/sparse format 等执行细节不进入 public result。
- TN 的 truncation/contraction 诊断也不通过 generic result 字段公开；用户需要控制的数值选择必须在 `simulate(..., backend_options=...)` 中显式输入。
- solver 或 contraction 未满足成功条件时必须抛出明确异常，不能要求用户事后检查 diagnostics 才发现失败。
- 性能与数值回归数据可留在 backend-private instrumentation 和 tests；研究脚本若要 benchmark，应在调用边界自行计时并记录显式输入。
- 因而 `EvolutionResult` 的 public surface 仍严格限定为 `times`、`expectation(name)`、`amplitude(labels)` 与 `sample(...)`。

### ER09 — `EvolutionResult.sample()` 支持 exact、MPS、PEPS（已确认，扩展 ER05）

- 三种 backends 使用同一个 `sample(shots=..., seed=...) -> Counter[tuple[str, ...]]` public interface。
- exact 从 private final state 按计算基概率采样；不得把完整 probabilities 暴露为 public 中间结果。
- MPS 使用 sequential conditional MPS measurement；不得转换成 dense vector。
- PEPS 使用由 simulation `backend_options` 固定精度的 environment-based conditional sampling；不得转换成 dense tensor。
- MPS sampling 对所存 MPS state 精确；PEPS sampling 是受显式 contraction/environment tolerances 控制的数值近似。
- sampling 始终 lazy；未调用时不构造 sampler、environment 或 samples。
- 相同 private state、`shots` 与 `seed` 必须产生相同 counts；physical labels 和 register ordering 与 ER05 保持一致。
- PEPS contraction/environment 未达到调用者指定条件时抛 convergence error，不返回未经验证的 counts。
- result 不公开 sampler、environment、概率表、truncation diagnostics 或 sampling metadata。

### ER10 — sampling seed 统一为 non-negative integer（最终确认）

- exact、MPS、PEPS 的 `EvolutionResult.sample()` 与 DMRG/PEPS 的 `GroundStateResult.sample()` 都要求 `seed` 是 non-negative Python/NumPy integer；`bool`、负数与其他类型在 result boundary 拒绝。
- 所有 backends 使用同一个已经校验的 seed 初始化各自的 local RNG；禁止 global NumPy/Torch RNG mutation 或 backend-specific signed-seed canonicalization。
- 相同 private state、`shots` 与 `seed` 的可复现语义不变；PEPS evidence 保存用户传入的这个 non-negative seed。

## 10. NoiseModel 与 ensemble

### N01 — NoiseModel 的统计边界（已确认）

- 只支持 quasi-static noise：一个 shot 内保持不变。
- 不同 shots 独立。
- 只支持零均值 Gaussian 分布。
- 不支持 arbitrary sampler callback、时间相关噪声、任意分布或 correlation matrix。
- 更复杂随机过程由研究 script 构造 fully specified protocol/system realizations。

### N02 — 按物理 laser group 建模准静态激光噪声（已确认）

```python
NoiseModel(
    laser_amplitude_sigma={
        "420": 0.01,
        "1013": 0.005,
    },
    laser_frequency_sigma_rad_s={
        "420": ...,
        "1013": ...,
    },
)
```

每个 laser group、每个 shot 独立采样：

```text
Omega_L(t) -> (1 + epsilon_L) Omega_L(t) exp(-i delta_L t)
```

- 噪声不改变 nominal gate duration。
- 删除含义过宽的 global amplitude/rabi scale field。
- 297 使用 key `"297"`。

### N03 — 删除匿名 detuning noise（已确认）

- 删除 `detuning_sigma_rad_s`。
- 激光频率/失谐噪声使用具名 `laser_frequency_sigma_rad_s`。
- 不用匿名字段猜测 intermediate、target `r` 或 `r_garb` 应如何移动。
- 未来若增加 atomic-level environment shift，必须使用新的、物理含义明确的 preset-aware 模型。

### N04 — 删除 local-addressing noise 特例（已确认）

- 删除 `local_rin_sigma`。
- 不新增 `local_addressing_scale_sigma`。
- 该能力没有 production script/notebook 调用，只被旧 analysis helper、测试和注释文档使用。
- 当前实现通过 shallow-copy/duck typing 修改 protocol，违背 fully specified 原则。
- 需要该研究噪声时，script 为每个 realization 显式构造 noisy addressing functions。

### N05 — position noise 支持任意 N（已确认）

```python
NoiseModel(position_sigma_um=(sigma_x, sigma_y, sigma_z))
```

- 每个原子独立采样三维 Gaussian 位移。
- nominal `Register` 按 S09 始终是二维；sampling 时内部嵌入 `z=0`，再加 `(dx,dy,dz)`，因此 `sigma_z` 表示平面阵列的面外热运动而不是三维 nominal layout。
- nominal register 先决定 pair topology/cutoff；抖动不让边随机出现或消失。
- 对保留边按新距离重算 `1/R^6`。
- 297 还按新方向重算缓存的 `C6(theta, phi)`。
- 不再限制两原子。

### N06 — realization 只保存原始随机样本（已确认）

每个 shot 保存：

- laser amplitude scales；
- laser frequency offsets；
- shape `(N,3)` 的 position offsets。

不保存：

- pair distances；
- derived C6/Vij；
- matrices、Hamiltonian terms 或 callables。

因此每个 shot 的记录保持 `O(N)`。

### N07 — 禁止空 `NoiseModel()`（已确认）

- 至少一个 sigma 必须非零。
- sigma 必须 finite、non-negative。
- 空模型报错并提示使用 deterministic `simulate()`。
- 删除 `.any_active`。

### N08 — `simulate_ensemble()` 完全委托 `simulate()`（已确认）

- ensemble 只采样 fully specified realization、调用 `simulate()`、收集结果。
- 不直接实例化 exact compiler/solver。
- 不自行处理 initial state、t_eval 或 expectation。
- 原则上支持所有能执行该 realization 的 backends；unsupported noise/backend 组合在 preflight 报 capability error。

### N09 — 同一 shot 的多个初态共享 realization（已确认）

用于 CZ 等比较时，batch 中所有初态必须经历完全相同的 laser/position noise sample。

### N10 — shots 与 seed 强制显式（已确认）

- `shots`、`seed` 是 mandatory keyword-only integers。
- seed 不允许 `None`。
- 相同 inputs/noise/seed 必须复现相同 realizations。

### N11 — EnsembleResult 是纯 raw container（已确认）

只保留：

```python
EnsembleResult(
    results=...,
    realizations=...,
    seed=...,
)
```

删除：

- generic metadata；
- mean/std/fidelity/error-budget methods；
- 重复的 shots、backend、noise configuration 字段；
- plot/save/load。

### N12 — EnsembleResult 与 simulate 返回形状一致（已确认）

- 单初态：`results[shot]` 是一个 `EvolutionResult`。
- 多初态：`results[shot][initial_state_index]` 是对应结果。
- 不强迫单初态用户写 `results[shot][0]`。
- 容器使用 tuple，而不是 mutable list。

### N13 — pulse 内时间相关激光噪声不属于 NoiseModel（已确认）

- 用户把连续随机 amplitude/phase functions 直接写入 protocol。
- 每条随机时间轨迹对应一个 fully specified protocol realization。
- 函数仍必须满足内部时间连续性规则。

### N14 — named laser noise 严格匹配 active `_LaserDrive`（已确认）

`simulate_ensemble()` 在开始采样前绑定 system/protocol，并根据 P17 的 `_ResolvedProtocol` 做严格 preflight：

- `NoiseModel` 中每个显式 laser key 都必须对应本次 protocol 实际产生的 `_LaserDrive.group`；
- active laser group 没有出现在 `NoiseModel` 中，表示该 laser 不加噪声，不要求用户为它填写零值；
- unknown、system 不支持或本次 protocol 未启用的 laser key 一律报 capability/configuration error，不能静默忽略；
- `CZProtocol` / `TOProtocol` / `ARProtocol` 只接受其实际产生的 `"420"` / `"1013"` groups；
- Direct-297 protocols 只接受实际产生的 `"297"` group；
- `SweepProtocol` / `DigitalAnalogProtocol` 只产生 `_ChannelDrive`，因此拒绝所有 named physical-laser noise；包括 S06 中取代 `analog_3` 的 `01r + DigitalAnalogProtocol` 工作流；
- effective-control noise 与时间相关 physical-laser noise 都由调用者写进 realization-specific coefficient functions，遵守 N13。

该规则对 exact、MPS、PEPS 相同，并在任何 realization 或 backend 演化开始前完成；不能让错误配置运行若干 shots 后才失败。

### N15 — position noise 只扰动 interaction geometry（已确认）

- 每个 realization 按 N05 的 position offsets 更新原子坐标，并仅据此重算 nominal topology 上的 pair distance、`1/R^6` interaction，以及 297-nm preset 的方向相关 `C6(theta, phi)`。
- protocol 已给出的 uniform 或 length-N site drive profiles 保持不变；position noise 不把它们重新解释为空间光场，也不隐式改变 Rabi amplitude、phase、detuning 或 addressing shift。
- 不新增 beam waist、beam center、wave vector、spatial field callback 或“坐标到 drive”之类的 public API。
- 若研究目标包含原子位移穿过非均匀光束导致的 drive 变化，script 为每个 realization 显式构造新的 fully specified coefficient functions，并自行决定它与 position sample 的 correlation。
- 该边界对 `_LaserDrive` 与 `_ChannelDrive` 相同，并在 exact、MPS、PEPS 上保持一致。

### N16 — 冻结 `EnsembleResult.realizations` 的 public schema（已确认）

`EnsembleResult.realizations` 是 shot-major immutable tuple；其中每个 shot record 固定为以下深只读 mapping：

```python
{
    "laser_amplitude_scales": Mapping[str, float],
    "laser_frequency_offsets_rad_s": Mapping[str, float],
    "position_offsets_um":
        tuple[tuple[float, float, float], ...] | None,
}
```

- 三个 key 始终存在，不能因某类 noise 未启用而改变 record shape。
- 两个 laser mappings 只包含 `NoiseModel` 显式配置的 groups；没有相应配置时是空的只读 mapping。
- amplitude mapping 保存实际 `1 + epsilon_L`，frequency mapping 保存实际 `delta_L`，不保存 sigma 或未缩放的标准正态 draw。
- position noise 未配置时为 `None`；配置时是长度严格等于 `system.N` 的 immutable 3-tuples，单位为 um。
- outer record、nested mappings 与 position tuples 均不可变；不能把 mutable `dict`/`list` 直接泄漏给用户。
- 不新增第四个 public `NoiseRealization` type；具体 frozen-mapping implementation 保持 private。
- 不保存 pair distances、angles、`C6`、`Vij`、Hamiltonian terms 或其他 derived quantities。
- 删除旧的 flat `detuning_rad_s`、`amplitude_scale`、`local_scales`、`pair_distances_um` 等 record keys，并同步迁移直接检查这些 keys 的 tests/scripts。

### N17 — realization sampling 与 simulation 请求细节解耦（已确认，具体化 N10）

固定 `system + NoiseModel + shots + seed` 时，`EnsembleResult.realizations` 必须相同，不受以下因素影响：

- laser sigma mappings 的 insertion order；
- exact/MPS/PEPS backend 选择及其 options；
- `t_eval` 与 observables；
- 单初态或多初态及其顺序；
- 后续是否调用 `amplitude()` 或 `sample()`。

实现必须先验证并 canonicalize NoiseModel，再按稳定排序的 laser group keys 一次性采样完整 shot-major realizations，最后把同一批 realization values 交给选定 backend。随机数生成不能散落在 compiler、solver、result lazy methods 或逐初态循环中。

- 同一 shot 的所有初态继续共享 N09 的 realization。
- exact/MPS/PEPS 交叉验证可以用相同 seed 获得相同物理样本。
- backend 失败不能通过提前或额外消耗 RNG 改变其他 shot 的定义。
- `EvolutionResult.sample(seed=...)` 使用自己的显式 seed，不消费 ensemble realization RNG。

### N18 — `EnsembleResult` 保留多个完整 results（已确认）

- `EnsembleResult.results` materialize 并保留所有 shots 的多个完整 `EvolutionResult`；不压缩为 mean/std、expectation tensor 或其他 summary。
- 单初态与多初态的 shot-major shape 继续严格遵守 N12。
- 每个 `EvolutionResult` 保留实现 `expectation()`、`amplitude()` 与 lazy `sample()` 所需的 private final backend state；这些方法不能因结果来自 ensemble 而缺失或报“state not retained”。
- 不增加 `retain_state`、partial/lightweight result mode、streaming iterator 或 generator public API。
- 因此内存上界显式为 `O(shots * final_backend_state_size)`，另加已请求 expectation 数据；这是保持 result surface 完整一致的有意取舍。
- 大规模 TN expectation-only ensemble 若无法一次 materialize，应在 script/notebook 中分批调用、读取 expectation 并释放每批结果；不为此把 aggregation/streaming policy 放回 src。

## 11. scripts/notebooks 的能力保持

### W01 — CZ complex overlaps 必须保留（已确认）

迁移为：

```python
result.amplitude(["0", "0"])
```

不能只返回 probability，因为 phase 实际参与：

- Nielsen average fidelity；
- conditional/ZZ phase；
- TO/AR optimization objectives；
- error-budget scoring；
- effective-theory parity；
- gate benchmark pass/fail。

### W02 — postprocessing 公式直接写回调用者（已确认）

- `cz_gate_report()`、`error_budget()` 等由 script/notebook 显式实现。
- 用户必须能阅读并知道自己计算的量是什么。

### W03 — 所有有价值 observables 必须可迁移（已确认）

- population、pair projector、correlation、logical projectors 等由 system observable algebra 组合。
- connected quantities、积分和 FFT 在 script 中完成。
- 只允许删除 O08 指明的 F1/F2 nonlinear diagnostic。

### W04 — local-addressing 两阶段脚本迁移为 piecewise protocol（已确认）

- 不再直接读取或传递 final backend state。
- 不新增 continuation API。

### W05 — AC-Stark/local-addressing notebook 迁移到 `01r`（已确认，落实 S06）

- `scripts/notebooks/02_ac_stark_addressing.ipynb` 不再构造 `analog_3`，改为 `level_structure("01r", ryd_level=70)` 与 fully specified `DigitalAnalogProtocol`。
- notebook 在用户可见单元中显式计算 `Delta`、`omega_420`、`omega_1013`、`rabi_eff`、`time_scale`、AC-Stark shift 和 piecewise addressing waveform。
- observable 使用 `n("0")`、`n("1")`、`n("r")`，并在 notebook 中明确注释其物理对应关系 `0 -> g`、`1 -> e`、`r -> r`。
- notebook 的相干动力学与图表不得退化；已由 O08 确认删除的 instantaneous-eigenbasis F1/F2 diagnostic 仍不恢复。

## 12. 被后续决定替代的历史结论

下表用于阻止实现者误读旧 handoff。

| 旧结论 | 状态 | 当前结论 |
|---|---|---|
| 顶层保守导出 Protocol/result types/InteractionSpec | 已替代 | 顶层只有 API01 的六个名字 |
| 保留 `InteractionSpec` 顶层并可能重命名 | 已替代 | 完全删除；preset + register + cutoff 负责 interaction |
| `1r/01r` 允许 `C6_rad_s_um6=custom` | 已替代 | 只用 `ryd_level`，C6 来自 ARC |
| runtime `DEFAULT_C6` fallback | 已替代 | ARC 是唯一 C6 来源 |
| public `system.interaction_pairs` | 已替代 | interaction representation 完全 internal |
| 新增 `system.interaction_strength()` | 已替代 | 不新增查询 API |
| `Register.from_coordinates()` 用于 arbitrary traps | 已替代 | 直接 `Register(coords=...)` |
| 两阶段演化保留 private continuation seam | 已替代 | 一个连续 piecewise protocol；不允许 continuation |
| non-Hermitian expectation 使用 raw norm bookkeeping | 已替代 | non-Hermitian evolution 整体删除 |
| expectations 保留 complex dtype | 已替代 | 只接受 Hermitian observable，返回 real arrays |
| public `final_state` 或 `psi_final` | 已替代 | final backend state 完全 private |
| scripts 用 `system.product_state()` + final vector 算 overlap | 已替代 | 使用 label-based `result.amplitude()`；public `system.product_state()` 删除 |
| NoiseModel 有 `detuning_sigma_rad_s` | 已替代 | 按物理 laser group 的 frequency noise |
| NoiseModel 有一个 global `amplitude_sigma` | 已替代 | 按 laser group 的 independent amplitude noise |
| 保留或重命名 local RIN | 已替代 | local-addressing noise 特例删除 |
| Noise ensemble 初版 exact-only | 已替代 | 委托 `simulate()`，原则上 backend-agnostic |
| realization 记录 derived pair effects | 已替代 | 只记录 O(N) 原始随机样本 |
| 只给 target `r-r` 加 interaction | 已修正 | 七能级 S channels 共享；297 P channels 分别计算 |
| `.expectations` mapping + lazy fallback | 已替代 | 只有 `expectation(name)`，且仅读取显式请求值 |
| PEPS amplitude 通过 generic result metadata 报告 contraction 设置 | 已替代 | contraction 设置是显式 backend input；result 不重复公开 metadata |
| public identity observable 用于 survival norm | 已替代 | non-Hermitian 已删除；public identity observable 删除 |
| public result 暴露 `nfev`、自动 storage format 或 generic solver diagnostics | 已替代 | 数值选择由显式输入决定；失败报错；执行统计仅留内部测试/调试 |
| PEPS 未达到内部 convergence tolerance 就必须报错 | 已替代 | PEPS02/PEPS03：返回最后一个数学有效的 estimate 与数值证据；只有 validity failure 才报错 |
| PEPS 的 NTU error 称为 discarded Schmidt weight | 已替代 | PEPS04/PEPS17：使用 `NTU truncation error`；MPS 才继续使用 discarded weight |
| PEPS 用 `discarded_weight_tolerance` / `truncation_error_tolerance` 同时控制 SVD 并 gate NTU error | 已替代 | PEPS17：删除两者；`svd_tolerance` 只控制局部 SVD，NTU error 只进入 evidence |
| PEPS 把 NTU optimizer 的 `max_iter=4`、`tol_iter=1e-10` 固定为 private policy | 已替代 | PEPS18：公开 `ntu_max_iterations` 与 `ntu_iteration_tolerance`，忠实传给 real-time/ground-state NTU |
| PEPS public options 直接镜像 YASTN `method` / `initialization` / `fix_metric` / `pinv_cutoffs` | 已拒绝 | PEPS19：这些共同定义一个受测试的 private NTU algorithm；public 只保留数值控制，不接受 engine kwargs |
| PEPS imaginary-time ground state 固定在 CPU | 已替代 | PEPS20：ground-state `method_options` 也显式要求 `device="cpu" | "cuda"`，整个 result 生命周期保持同一设备 |
| E25 的 PEPS real-time 八 key schema | 已替代 | PEPS21：冻结包含 `time_step_s`、SVD/NTU controls 的十 key mandatory schema |
| E20 的 PEPS ground-state 七 key schema | 已替代 | PEPS22：冻结包含 SVD/NTU controls、dimensionless schedule 与 device 的九 key mandatory schema |
| PEPS evidence 字段留给实现者自行扩张 | 已替代 | PEPS23/PEPS24：统一一个 immutable type，固定九个顶层字段与两个最小 lazy records |
| PEPS27 允许直接 `Register(coords)` 只要可推断为 Cartesian product | 已替代 | PEPS31：当前 YASTN PEPS 只接受 `Register.chain/rectangle/square` 创建并私下标记的 grid register |
| API06 规定 `ryd_gate.results.__all__` 只有三个 result types | 已替代 | PEPS25：追加三个 PEPS evidence record types；顶层 `ryd_gate` 六项仍不变 |
| PEPS amplitude evidence 被理解为 public amplitude 调用日志 | 已修正 | PEPS33：它记录实际成功执行的 distinct product-bra coefficient contractions，ground phase reference 也是普通 labels record |
| PEPS 不允许 result 暴露任何数值 trace | 已替代 | PEPS05：PEPS 参数与数值轨迹属于稳定 evidence；具体 public shape 继续 grill |
| PEPS evidence 保存完整 per-step/per-iteration traces | 已替代 | PEPS14：只保存 numerical provenance 与免费产生的有界摘要；不为 evidence 增加 contraction |
| MPS/PEPS real-time option 使用无单位后缀的 `time_step` | 已替代 | PEPS08：统一改为 physical-seconds `time_step_s`，不保留 alias |
| PEPS ground-state schedule 直接乘 physical `rad/s` Hamiltonian | 已替代 | PEPS09：先按固定局域能标归一化 Hamiltonian，schedule 保持无量纲 |
| PEPS amplitude 内部逐级增加 contraction bond dimension 并自动比较 | 已替代 | PEPS15：只执行用户指定 bond dimension 的一次 estimate；跨 dimension 比较由 caller 重跑完成 |
| PEPS ground-state 在 schedule 中按 energy tolerance 自动收敛/提前停止 | 已替代 | PEPS16：完整执行 schedule，只在末尾做一次必要 CTM；删除 relative-energy gate/check interval |
| ObservableExpr 只允许 real scalar multiplication | 已替代 | complex intermediate scalar 合法；传给 `simulate()` 的最终 expression 必须 Hermitian |
| 保留 `TFIMQuenchProtocol` / `TFIMAnnealProtocol` 与 TFIM mapping helpers | 已替代 | 只保留 `SweepProtocol`；TFIM mapping 与 schedule 在使用它的 scripts/notebooks 中显式书写 |
| Protocol 用 protected mapping hook 声明 laser group，或用 system/protocol channel 集合交集推断 | 已替代 | P17 的 `_LaserDrive` 直接保留物理 laser 身份；`_ChannelDrive` 明确表示 effective control |
| compiler 分别调用 `drive_channels()`、`get_drive_coefficients()`、`resolve_t_gate()` 或 TN 专用 context | 已替代 | 唯一 lowering seam 是 private `protocol._resolve(system) -> _ResolvedProtocol` |
| 保留独立 `analog_3` physical preset 与 backend special cases | 已替代 | S06：使用 `01r + DigitalAnalogProtocol` 精确表达，并完全删除 `analog_3` |
| standalone `ground_state_energy()` 只返回 float | 已替代 | E15 的 `system.ground_state()` 返回 `GroundStateResult`；energy 通过 `expectation("energy")` 读取 |
| `system.ground_state()` 表示 all-ground product vector | 已替代 | E15 将该名称用于真正的 many-body ground-state search；product initial state 使用 physical labels |
| `EvolutionResult.sample()` 只支持 exact | 已替代 | ER09 要求 exact、MPS、PEPS 都提供 lazy native sampling，禁止 TN dense fallback |

## 13. 尚待逐项 grill 的问题

当前已列出的 PEPS option/evidence public-shape 问题均由 PEPS21-PEPS26 冻结。后续闭环审计若发现新的物理能力或数值语义缺口，必须先追加待 grill 项，不能由实现者自行决定。

## 14. 后续每次 grill 的记录格式

每确认一个新决定，应在本文件相应主题中追加：

- 唯一 decision ID；
- 状态：已确认 / 被替代 / 待 grill；
- public API 示例；
- 明确删除什么；
- 对 exact/MPS/PEPS/scripts/notebooks 的影响；
- 若替代旧决定，注明旧 decision ID。

每次实现审查都应先检查第 12 节与第 15 节，避免按旧 handoff 或旧 fail-closed PEPS 条目的过期语义工作。

## 15. PEPS 数值语义修订（2026-07-15）

本节是 E16、E20、E23-E25、ER07-ER09 中 PEPS 数值语义的后续修订。发生冲突时以本节为准；旧条款作为历史保留，不得继续实现其 fail-closed 要求。

### PEPS01 — PEPS 仍是本轮必须完整交付的 public backend（已确认）

- 不把 PEPS 降级为 experimental API，也不以 guarded notebook、`NotImplementedError` 或 `xfail` 代替实现。
- real-time PEPS 必须支持当前 `1r` / `01r` capability；PEPS imaginary-time ground state 继续只接受带明确二维 geometry 的 `1r` system。
- `expectation()`、lazy `amplitude()` 与 lazy `sample()` 的 public 能力必须存在；禁止 dense fallback。
- 本轮重构在 PEPS amplitude、数学有效的 sampling、调用方迁移和端到端测试完成前不能验收或提交。
- “完整交付”表示功能与数值证据完整，不表示 `src` 自动认证 estimate 已收敛；后者由 PEPS02 修订。

### PEPS02 — PEPS 返回 estimate 与 evidence，不替用户判定收敛（已确认，替代旧 fail-closed 语义）

- 用户显式设置 PEPS state、NTU、BP/CTM、boundary contraction 与 imaginary-time 参数。
- backend 忠实执行这些参数，并返回最后一个数学有效的近似结果。
- CTM/BP 达到 iteration cap、NTU error 较大、energy 尚未稳定、amplitude 随 contraction bond dimension 仍变化，均不得仅因“看起来未收敛”而抛错。
- schedule 或 resource cap 用尽时，只要仍有数学有效 estimate，就返回 estimate 与完整 evidence。
- caller 在 scripts/notebooks 中改变 bond dimensions、schedule、step size、environment controls 或 initial state，收集多次 estimates 并自行完成 convergence study。
- `src` 不提供“已物理收敛”或“已找到全局基态”的 certificate。

### PEPS03 — 只保留数学有效性与 capability 检查（已确认）

以下情况仍必须在边界处报错，不能作为低精度 estimate 返回：

- 输入类型、shape、单位、option key、register geometry 或 backend capability 不合法；
- 请求不支持的 observable term 或非最近邻 PEPS interaction；
- 请求 CUDA 但环境不可用；
- YASTN tensor operation 本身失败；
- tensor、energy、expectation、amplitude、norm 或 probability 出现 `NaN` / `Inf`；
- PEPS norm 非正，或 imaginary residual 大到无法解释为 real norm；
- sampling conditional probabilities 显著为负、非 real、无法归一化或总权重非正。

“没有达到 convergence tolerance”不是 validity failure；“没有数学意义的数值”才是。

### PEPS04 — PEPS 使用 NTU truncation error，不冒充 discarded weight（已确认）

- YASTN `Evolution_out.truncation_error` 是 NTU environment metric 下的 relative norm error，不是 Schmidt discarded weight。
- MPS 继续使用 `discarded_weight_tolerance`；PEPS option 使用 `truncation_error_tolerance`。
- real-time evidence 保存每一步的 worst-bond NTU error 与沿整段演化的累计曲线。
- ground-state evidence 保存各 stage/window 的实际 NTU error；warmup 与 final-stage 数据都保留。
- 这些值用于 caller 比较 runs，不作为 PEPS02 禁止的自动 acceptance gate。

### PEPS05 — PEPS 数值参数与轨迹属于 result evidence（已确认，具体 shape 待 grill）

- PEPS result 必须保留运行时实际使用的 algorithm parameters 与 convergence-study 所需的数值轨迹。
- evidence 至少覆盖：NTU errors、BP/CTM residuals 与 iterations、imaginary-time energy history、amplitude contraction estimates/norm estimates，以及 sampling contraction 的有界统计。
- 不暴露 YASTN tensor、environment、sampler 或任意 engine object。
- 不恢复无约束 generic metadata bag；字段必须稳定、具名、只读，并能由 scripts 直接保存或画图。
- eager evolution/expectation 与 lazy amplitude/sample 的 report 如何进入 public result，仍列于第 13 节继续 grill。

### PEPS06 — PEPS amplitude 是 lazy normalized product-bra estimate（已确认，由 PEPS15 冻结 contraction schedule）

- 真正计算发生在 `result.amplitude(labels)`，而不是 `simulate()`；未调用时不支付 contraction 成本。
- 使用 single-layer product-bra boundary-MPS contraction，禁止 `Peps.to_tensor()` 或其他 dense conversion。
- YASTN NTU 会丢失正实 global normalization scale，因此 raw product coefficient 不能直接作为 physical amplitude。
- 每个 result 第一次 amplitude 调用时，用受相同用户 contraction controls 约束的独立 double-layer contraction 计算 finite、real-positive norm；norm 只在 private reader 内缓存一次。
- 返回 `raw_amplitude / sqrt(norm)`；positive-real normalization 不删除 complex global phase。
- ground-state target/reference amplitudes 复用同一 norm，再应用 explicit physical-label `phase_reference` gauge。
- contraction 未表现出 convergence 时仍按 PEPS02 返回最后一个数学有效 estimate 与 evidence；fixed bond-dimension strategy 由 PEPS15 冻结。

### PEPS07 — `t_eval` 是 expectation measurement cost 的唯一控制（已确认）

- expectation expression 在 `simulate(..., observables=...)` 中显式请求，并在每个 `t_eval` 时刻计算；`result.expectation(name)` 只读取已记录数值。
- amplitude/sample 继续是只作用于私有终态的 lazy readouts。
- 不增加 `measurement_stride`、自动抽稀或隐藏时间网格。
- PEPS 每个 measurement time 可能需要一次 BP/CTM contraction；文档必须明确成本大致随 `len(t_eval)` 线性增长。
- 同一时刻的多个 requested observables 应共享该时刻的 measurement environment。

### PEPS08 — real-time TN step 使用 physical-seconds 名称（已确认，修订 E23/E25）

- MPS 与 PEPS real-time `backend_options` 都把 `time_step` 改名为 `time_step_s`。
- 不保留旧 `time_step` alias、deprecated key 或 compatibility parser。
- real-time gate 始终使用原始 physical `rad/s` Hamiltonian 与 seconds step；不得用 ground-state normalization 改变物理演化。
- scripts/notebooks/tests 同步迁移。

### PEPS09 — ground-state 使用 normalized Hamiltonian 与 dimensionless multi-stage schedule（已确认，修订 E20）

- snapshot 仍由 `system.ground_state(at=...)` 冻结真实 physical Hamiltonian；ground-state method 只接受 `1r`。
- solver 计算固定局域能标

  ```text
  Lambda = max(max_i ||h_i(at)||_2, max_(i,j) |V_ij|)
  ```

  并使用 `H_tilde = H / Lambda` 做 imaginary-time evolution。
- `Lambda == 0` 表示完全零 Hamiltonian，无法选择唯一 ground-state representative，属于 validity failure。
- `imaginary_time_schedule` 保持无量纲 sequence of `(dtau, max_steps)`；multi-stage capability 保留，单一 stage 也是合法的保守用法。
- 较大 `dtau` stage 只是 warmup；最小 `dtau` stage 称为 final refinement stage。后者给出最细 estimate，但 PEPS02 禁止把它自动声明为 converged。
- 每个 stage 精确执行其声明的 steps，但不做逐-stage CTM、energy measurement 或 history；只把整个 schedule 已自然产生的 maximum NTU summary 写入 PEPS24 的 bounded evidence。任何 non-finite 或 tensor failure 仍按 PEPS03 报错。
- `GroundStateResult.expectation("energy")` 始终返回原始未缩放 Hamiltonian 的 real energy，单位 `rad/s`；report 保存 `hamiltonian_scale_rad_s=Lambda`。

### PEPS10 — PEPS 调用方与验证要求（已确认）

- `examples/demo_local_addressing_tn.py`、`scripts/bench_quench_check.py`、notebooks 03/04/05 必须迁移到最终 PEPS schemas。
- 删除“backend under rewrite”、以宽泛 `except Exception` 隐藏旧参数错误、以及把 `NotImplementedError` 当作通过条件的测试。
- 快速测试覆盖参数/shape/validity checks、lazy behavior、normalized complex amplitude、无 dense fallback 与 result evidence。
- 小系统真实 YASTN tests 比较 exact/MPS/PEPS estimates，并至少改变一次 bond/environment dimensions 或 step controls，使 convergence-study workflow 可执行。
- DGX 验证 4x4 CUDA smoke 与仓库实际保留的较大二维 workflow；slow physics tests 和完整 docs render 仍是最终审查门槛。

### PEPS11 — numerical evidence 只属于 PEPS（已确认）

- PEPS02 的 report-but-don't-gate 语义只扩张 PEPS result，不顺带扩张 exact/MPS diagnostics API。
- exact 继续由 `rtol` / `atol` 与 solver success 控制；不公开 ODE step trace、`nfev` 或 storage-format diagnostics。
- MPS real-time 与 DMRG 继续执行既有 discarded-weight / convergence gates；不新增 TDVP/DMRG public trace。
- `EnsembleResult` 不新增 aggregate evidence；其中每个 PEPS `EvolutionResult` 自己携带对应 shot 的 PEPS evidence。
- PEPS evidence 的访问入口必须显式带有 `peps` 语义，不能伪装成所有 backend 都支持的 generic result metadata；精确入口由 PEPS12 冻结。

### PEPS12 — evidence 入口是只读 `result.peps_evidence` property（已确认）

```python
result = simulate(..., backend="peps")
evidence = result.peps_evidence
```

- `EvolutionResult` 与 `GroundStateResult` 都提供 `peps_evidence` property。
- PEPS result 返回具名、只读的 evidence object；exact/MPS result 返回 `None`。
- property 只读取已经产生的数据，绝不能因访问 property 而触发 BP/CTM、amplitude、norm 或 sampling contraction。
- 不增加 generic `result.metadata`、`result.diagnostics`、`result.convergence` 或任意 string-key escape hatch。
- `EnsembleResult` 不汇总 evidence；其每个 PEPS child result 各自携带 shot-specific evidence。
- evidence object 的 lazy operation 行为由 PEPS13 冻结；精确字段继续在第 13 节 grill。

### PEPS13 — evidence ledger append-only，property 返回 immutable snapshot（已确认）

- PEPS `simulate()` / `ground_state()` 返回时，ledger 已包含 eager evolution、requested expectation 与 ground-state optimization 产生的 evidence。
- 每个成功的 lazy `amplitude()` / `sample()` 在完成后向 private ledger 追加对应 operation record。
- `result.peps_evidence` 每次返回截至该时刻的 immutable snapshot；此前取得的 snapshot 永不变化。
- property 本身不触发计算；只有用户显式调用 lazy readout 才能生成新 evidence。
- `amplitude(labels)` 继续只返回 `complex`，`sample(shots=..., seed=...)` 继续只返回 `Counter`；不增加 `return_convergence` 或 tuple-return mode。
- 相同 labels 的 amplitude 可复用 private cached norm/amplitude/evidence，不重复追加等价记录。
- validity failure 抛错且不把半完成 operation 伪装成 successful evidence；异常本身应包含足够的 failure context。
- physical expectations、times 与 amplitudes 的既有值不因 ledger 增长而改变；append-only 只描述后来执行了哪些 lazy numerical operations。

### PEPS14 — evidence 只保留 provenance 与免费产生的有界摘要（已确认，收窄 PEPS04/05/09）

- `peps_evidence` 不是完整 diagnostic trace；它的主要用途是记录 estimate 的 numerical provenance，并让 caller 比较不同参数 runs。
- 保存实际使用的 PEPS public algorithm parameters；ground-state 另保存 PEPS09 的 derived `hamiltonian_scale_rad_s`。
- 保存计算本身已经产生、无需额外 contraction 的 bounded summaries，例如 real-time cumulative NTU truncation error、ground-state max NTU truncation error、worst/final BP/CTM residual。
- lazy amplitude 只追加该 labels estimate 的最终 contraction/norm error summary；不保存 adaptive sequence 的全部 intermediate estimates。
- sampling 不保存 `shots * N` conditional trace，也不为 evidence 增加 instrumentation contraction；只保留输入 provenance，数学有效性仍按 PEPS03 检查。
- 不保存完整 per-step NTU curve、per-iteration BP/CTM trace、逐 stage energy history或任意 backend log。
- 不为了生成 evidence 额外运行 BP、CTM、norm、amplitude 或 sampling contraction；没有被实际 readout 需要的数值就不计算。
- caller 判断 convergence 的主要依据是用不同 parameters 重跑后，比较 physical expectations/amplitudes/energies 是否稳定；evidence summary 只提供辅助背景。

### PEPS15 — amplitude 只使用用户指定的固定 contraction bond dimension（已确认，替代早期 adaptive-chi 决定）

- `result.amplitude(labels)` 每次只按 simulation/ground-state options 中的 `environment_bond_dimension` 执行一次 single-layer product-bra contraction。
- 首次 amplitude 所需的 double-layer norm contraction 使用同一 bond dimension / tolerance / iteration cap，并按 PEPS06 在 private reader 中缓存。
- backend 不内部尝试 `chi=...` sequence，不自动增加 bond dimension，也不因不同 chi estimates 未一致而报错。
- contraction 产生 finite numerator、finite real-positive norm 时返回 normalized complex estimate；未达到 engine tolerance 只记录最终 free residual summary。
- caller 通过分别运行 `environment_bond_dimension=16/32/64/...` 并比较 physical amplitudes，自行完成 bond-dimension convergence study。
- `peps_evidence` 不保存不存在的 adaptive sequence，只保存实际固定参数与该次 contraction 已产生的最终摘要。

### PEPS16 — ground-state 完整执行 schedule，只在末尾做一次必要 CTM（已确认，替代旧 convergence loop）

- 每个 `imaginary_time_schedule` entry 的 `(dtau, max_steps)` 精确执行 `max_steps`；不因 energy change、NTU error 或 environment residual 提前停止。
- 删除 PEPS ground-state public `relative_energy_tolerance`；不增加 `convergence_check_interval`。
- coarse/final stages 之间不为 convergence evidence 运行 CTM，也不计算逐 stage energy history。
- schedule 全部完成后，只构造一次最终 CTM environment，用于 reserved physical energy 与 caller-requested expectations。
- 最终 CTM 达到 iteration cap 但仍产生 finite、数学有效的 energy/expectations 时，按 PEPS02 返回 estimate，并在 evidence 中记录实际 final residual/iterations；不因 residual 高于 tolerance 拒绝结果。
- caller 通过改变 schedule、最小 `dtau`、state bond dimension 与 environment bond dimension 重跑并比较最终 physical results。
- tensor failure、non-finite energy、invalid norm 等仍是 PEPS03 的 validity failures。

### PEPS17 — PEPS 的截断输入只表达真实的 SVD cutoff（已确认，修订 PEPS04/E20/E25）

- PEPS real-time 与 ground-state options 删除 `discarded_weight_tolerance` 和 `truncation_error_tolerance`；不保留 alias 或 deprecated compatibility key。
- 新增 `svd_tolerance`，其唯一含义是传给 YASTN 局部 SVD 的 singular-value cutoff。
- `bond_dimension` 是 PEPS state bond dimension 的硬上限；`svd_tolerance` 与它共同决定局部截断，但二者都不声明最终 physical estimate 已收敛。
- YASTN 返回的 NTU truncation error 只按 PEPS14 汇总进 `peps_evidence`；backend 不把它与任何 tolerance 比较，也不据此提前停止或拒绝结果。
- 禁止用 `min(svd_tolerance, 1e-12)` 或其他隐藏 clamp 改写用户输入；通过边界 validation 后，应忠实传给 engine。
- MPS 的 `discarded_weight_tolerance` 及其既有 fail-closed 语义保持不变。

### PEPS18 — NTU 局部优化停止控制由用户显式提供（已确认，修订 E20/E25）

- PEPS real-time 与 ground-state options 都新增 `ntu_max_iterations` 和 `ntu_iteration_tolerance`。
- `ntu_max_iterations` 是每个 bond truncation optimization 的正整数迭代上限；`ntu_iteration_tolerance` 是相邻局部 truncation-error objective 变化的 finite、strictly-positive 停止阈值。
- 两个值分别原样传给 YASTN `evolution_step_(max_iter=..., tol_iter=...)`；删除 private `_NTU_MAX_ITER` / `_NTU_TOL_ITER` policy，不保留第二套隐藏 override。
- 局部 optimizer 达到 iteration cap 但返回 finite tensor/update 时，不抛 convergence error；按 PEPS02 继续并返回最终数学有效 estimate。
- 两个输入值进入 PEPS provenance；若 YASTN 本次计算已经返回 iteration counts，evidence 可以保存整个 operation 的 bounded maximum，不能保存逐 bond/逐 step trace，也不能为此增加计算。
- 这些 controls 只属于 PEPS；MPS/DMRG API 不改变。

### PEPS19 — 固定 NTU algorithm，拒绝 YASTN-shaped passthrough（已确认，延续 P01/API03）

- `method="mpo"`、`initialization="EAT_SVD"`、`fix_metric=0` 与受测试的 pseudo-inverse cutoff ladder 共同定义当前 `ryd_gate` PEPS NTU algorithm，保持 private。
- public PEPS options 不暴露 `method`、`initialization`、`fix_metric`、`pinv_cutoffs`、`opts_post_truncation` 或任意 `**yastn_kwargs`。
- 用户仍显式控制用于 convergence study 的 physical step、state/environment bond dimensions、SVD cutoff、NTU iteration controls 与 contraction controls。
- 固定算法选择必须集中定义并由测试覆盖，不能散落为调用点之间不一致的 magic values。
- 将来若需要不同更新算法，应设计新的具名 method/algorithm capability 并单独审查，而不是把第三方 engine surface 泄漏进当前 schema。
- 不影响 exact/MPS/DMRG public options。

### PEPS20 — ground-state PEPS 与 real-time 一样显式选择 device（已确认，修订 E20）

- `method="peps_imaginary_time"` 的 mandatory `method_options` 新增 `device`，public values 精确为 `"cpu"` 或 `"cuda"`。
- imaginary-time NTU、最终 CTM、requested expectations 以及 private final state 上的 lazy `amplitude()` / `sample()` 全部保持在同一用户指定设备；禁止读出时偷偷搬到 CPU。
- 请求 `"cuda"` 但 PyTorch、CUDA 或 YASTN CUDA support 不可用时，在开始演化前报 capability error；禁止 silent CPU fallback 或自动设备选择。
- `peps_evidence` provenance 保存实际 `device`；访问 evidence 不执行 device probe 或 tensor transfer。
- real-time PEPS 的同名 `device` key 与上述语义完全一致；exact/MPS/DMRG API 不改变。

### PEPS21 — 冻结 real-time PEPS 十 key schema（已确认，替代 E25）

```python
backend_options={
    "time_step_s": 1e-9,
    "bond_dimension": 8,
    "svd_tolerance": 1e-12,
    "ntu_max_iterations": 20,
    "ntu_iteration_tolerance": 1e-10,
    "measurement_method": "ctm",  # 或 "belief_propagation"
    "environment_bond_dimension": 32,
    "environment_tolerance": 1e-8,
    "environment_max_iterations": 50,
    "device": "cuda",             # 或 "cpu"
}
```

- 十个 keys 全部 mandatory；mapping 必须精确匹配，unknown/missing key 在 tensor allocation 前报错。
- `time_step_s` 是 finite、strictly-positive physical-seconds Trotter substep upper bound；局部末步可缩短以命中 anchors。
- `bond_dimension`、`ntu_max_iterations`、`environment_bond_dimension`、`environment_max_iterations` 是 positive integers；`svd_tolerance`、`ntu_iteration_tolerance`、`environment_tolerance` 是 finite、strictly-positive floats。
- `measurement_method` 严格保留 E24 的 `"belief_propagation" | "ctm"`；`device` 严格遵守 PEPS20。
- 前五项控制 real-time Trotter/NTU state evolution；measurement/environment controls 约束显式 expectation 与 private final-state lazy contractions。
- 未请求 observables 时不构造 measurement environment；未调用 `amplitude()` / `sample()` 时不执行对应 lazy contractions。mandatory 参数声明不会触发不需要的计算。
- 删除旧 `time_step`、`discarded_weight_tolerance`、`truncation_error_tolerance` 与所有 aliases/defaults；不接受 engine kwargs。
- 所有输入及实际执行 device 进入 PEPS provenance，但仍按 PEPS14 只保存有界免费摘要。

### PEPS22 — 冻结 ground-state PEPS 九 key schema（已确认，替代 E20）

```python
method_options={
    "bond_dimension": 8,
    "svd_tolerance": 1e-12,
    "ntu_max_iterations": 20,
    "ntu_iteration_tolerance": 1e-10,
    "environment_bond_dimension": 32,
    "environment_tolerance": 1e-8,
    "environment_max_iterations": 50,
    "imaginary_time_schedule": (
        (0.10, 30),
        (0.03, 30),
        (0.01, 40),
    ),
    "device": "cuda",  # 或 "cpu"
}
```

- 九个 keys 全部 mandatory；mapping 必须精确匹配，unknown/missing key 在 tensor allocation 前报错。
- `bond_dimension`、`ntu_max_iterations`、`environment_bond_dimension`、`environment_max_iterations` 是 positive integers；三个 tolerances 是 finite、strictly-positive floats。
- `imaginary_time_schedule` 是唯一 imaginary-time step/resource control：必须是非空 immutable sequence of `(dtau, max_steps)`，`dtau` finite、positive 且严格递减，`max_steps` 是 positive integer。
- schedule 作用于 PEPS09 的 dimensionless normalized Hamiltonian，并严格遵守 PEPS16 的完整执行语义；adapter 不追加、删除或提前终止 stage。
- ground-state 不接受 `time_step_s` 或 `measurement_method`；最终 reserved energy、requested expectations 与 lazy sampling 固定使用 CTM，lazy amplitude 使用 PEPS06/PEPS15 的 product-bra contraction。
- 删除旧 `discarded_weight_tolerance`、`truncation_error_tolerance` 与 `relative_energy_tolerance`，不保留 aliases/defaults。
- `device` 严格遵守 PEPS20；所有输入与 derived `hamiltonian_scale_rad_s` 进入 provenance。
- 最终 CTM 达到 iteration cap 但产生 finite valid estimate 时返回；只把免费 residual/iteration summary 写入 evidence，不执行 convergence gate。

### PEPS23 — real-time 与 ground-state 共用一个 `PEPSEvidence` 类型（已确认，具体化 PEPS12/13）

```python
evidence = result.peps_evidence  # PEPSEvidence | None
```

- PEPS `EvolutionResult` 与 PEPS `GroundStateResult` 都返回同一个 immutable `PEPSEvidence` public type；exact/MPS/DMRG 返回 `None`。
- 不创建 `RealTimePEPSEvidence` / `GroundStatePEPSEvidence` inheritance hierarchy，也不暴露 backend-native report 类型。
- `parameters` 是对应 PEPS strict option schema 的 immutable exact copy；禁止添加 schema 外 metadata keys 或把它变成任意扩展 bag。
- mode-specific quantity 使用少量明确的 optional field：例如 `hamiltonian_scale_rad_s` 对 real-time 为 `None`，对 ground-state 保存 PEPS09 的 derived scale。
- lazy amplitude/sample records 只允许两个小型 immutable record types；不为 eager evolution、BP、CTM、NTU 各创建 public class hierarchy。
- 每次 property access 仍遵守 PEPS13 的 snapshot 语义；统一类型不意味着返回同一个可变 object。
- 精确字段与两个 lazy record 的字段继续逐项 grill；实现者不能自行扩张。

### PEPS24 — 冻结 `PEPSEvidence` 的九个字段与两个最小 lazy records（已确认，具体化 PEPS14/23）

```python
PEPSEvidence(
    parameters=...,
    hamiltonian_scale_rad_s=...,
    max_ntu_truncation_error=...,
    cumulative_ntu_truncation_error=...,
    environment_residual=...,
    environment_iterations=...,
    norm_contraction_error=...,
    amplitudes=(...),
    samples=(...),
)

PEPSAmplitudeEvidence(
    labels=("1", "r", ...),
    contraction_error=...,
)

PEPSSampleEvidence(
    shots=1000,
    seed=123,
)
```

- `parameters` 是 PEPS21 或 PEPS22 对应 strict schema 的 immutable exact copy，不接受额外 keys。
- `hamiltonian_scale_rad_s` 对 real-time 为 `None`，对 ground-state 为 PEPS09 的 finite positive derived scale。
- `max_ntu_truncation_error` 保存整次 state evolution 中已经产生的最大 finite non-negative NTU error。
- `cumulative_ntu_truncation_error` 只对 real-time 保存；ground-state 为 `None`，避免把 schedule length 依赖的累计值误当基态质量指标。
- real-time 的 `environment_residual` / `environment_iterations` 分别汇总所有 requested expectation times 的 worst residual / maximum iterations；没有 requested observables 时两者为 `None`。
- ground-state 的 `environment_residual` / `environment_iterations` 保存最终必要 CTM 的 residual / iterations。
- `norm_contraction_error` 在尚未调用 amplitude 时为 `None`；首次成功 normalized amplitude 后保存 cached double-layer norm contraction 已产生的最终 error summary。
- 每个成功且此前未缓存的 labels amplitude 追加一个 `PEPSAmplitudeEvidence`；只保存 physical labels 与 single-layer final contraction error，不重复保存 value 或 norm error。
- 每次成功 lazy sampling 追加 `PEPSSampleEvidence(shots, seed)`；不保存 conditional trace、Counter 副本或额外 sampling diagnostics。
- 不保存 physical expectations、amplitudes、samples、`converged` boolean、完整 curves、逐 bond/逐 iteration data、timestamps 或 backend logs。
- 所有 records 与 nested tuples/mapping views 必须深度不可变，并遵守 PEPS13 的 snapshot 行为。

### PEPS25 — evidence types 只从 `ryd_gate.results` public 导入（已确认，延续 API01/API06）

```python
from ryd_gate.results import (
    PEPSEvidence,
    PEPSAmplitudeEvidence,
    PEPSSampleEvidence,
)
```

- 三个 immutable record types 加入 `ryd_gate.results.__all__`，与 `EvolutionResult`、`GroundStateResult`、`EnsembleResult` 位于同一个 result contract module。
- 不加入 `ryd_gate.__all__`；API01 的顶层六个名字保持不变。
- 普通调用者无需显式 import，直接读取 `result.peps_evidence`；显式 import 仅用于 type annotations、tests 或 tooling。
- 不创建第二个 `ryd_gate.peps_results` module，也不从 backend implementation module 导出 public records。

### PEPS26 — `PEPSEvidence.to_dict()` 是唯一 serialization helper（已确认）

```python
payload = result.peps_evidence.to_dict()
json.dump(payload, file)
```

- 只在顶层 `PEPSEvidence` 提供 `to_dict()`；两个 lazy record types 不各自增加转换方法。
- 每次调用返回全新的 mutable deep copy，只包含 JSON-compatible `dict`、`list`、`str`、`int`、finite `float` 与 `None`。
- immutable parameter mapping、schedule/labels tuples、amplitude/sample record tuples 都递归转换为普通 dict/list；不泄漏 snapshot 内部容器。
- 不提供 `from_dict()`、`.json()`、`.save()`、文件路径处理、I/O、analysis 或 convergence judgement。
- 不在 evidence 中引入 schema-version field；public field names 本身受 PEPS24 的普通 API compatibility 约束。

### PEPS27 — `Register` 是唯一 geometry，PEPS 只验证 Cartesian capability（已确认）

- geometry 只能通过 `Register` 构建并由 `system.register` 传给所有 backends；PEPS 不增加 geometry factory、lattice object、coordinate option 或第二套 site ordering。
- PEPS real-time 与 imaginary-time 共用同一个 preflight，只接受坐标构成完整 axis-aligned Cartesian product 的 `Register`。
- 支持 `1×N`、`N×1` 与一般 `Nx×M`；两个坐标轴可非均匀 spacing，Register site order 任意。
- 每个 Cartesian cell 必须恰好对应一个 site；preflight 必须验证完整 bijection，不能只检查 `len(xs) * len(ys) == N`。
- capability 按 coordinates 的结构判断，不按 factory provenance 判断：直接 `Register(coords)` 手写出的完整 Cartesian grid 合法；不是只有 `Register.chain/rectangle/square` 的返回值才合法。
- `Register.triangular(...)`、rotated/skewed grid、带洞 grid 与一般 irregular `Register(coords)` 仍是合法 public geometry，但本轮 PEPS 明确不支持；选择 PEPS 时在 tensor allocation 前报 capability error，exact/MPS 不受影响。
- 不新增 `peps_geometry`、`lattice_shape` 或 backend-specific remapping 参数。

### PEPS28 — PEPS 只使用有限 Register 的 open boundary（已确认）

- real-time 与 imaginary-time PEPS 都把 `system.register` 解释为 finite open-boundary Cartesian lattice。
- 不在 `Register`、`RydbergSystem`、`backend_options` 或 `method_options` 增加 `boundary`、`periodic`、`cylinder` 等 public input。
- 禁止只切换 YASTN lattice boundary 而仍沿用 open-plane distances/interactions；这种实现会让物理 Hamiltonian 与 tensor topology 不一致。
- periodic/cylindrical capability 若未来需要，必须同时设计 wrapped physical distance、pair topology、position noise、site ordering 与相应 tests，作为独立 API 决策。
- 当前 exact/MPS 也继续模拟 `Register` 给出的有限坐标和 system 解析出的 interactions；本条不改变它们。

### PEPS29 — PEPS nearest neighbour 是 Cartesian tensor-graph edge（已确认，具体化 PEPS03）

- 在 PEPS27 验证后的 lattice indices 上，两个 sites 仅当一个 index 相差 `1` 且另一个相同时才是 PEPS graph neighbours。
- system canonical lowering 产生的所有 nonzero pair terms 必须属于这些 graph edges；允许只包含 edge subset，也允许完全没有 interaction。
- 不按最短 Euclidean distance shell 重新解释 neighbour；因此非均匀 Cartesian spacing 不会改变 tensor adjacency，interaction strength 仍来自真实坐标距离。
- graph 外 diagonal/long-range pair 在 tensor allocation 前报 capability error；PEPS 禁止自动丢弃、近似合并或静默截断。
- 不新增 PEPS-specific cutoff/range/topology input；`RydbergSystem(..., interaction_cutoff_um=...)` 仍是唯一 public pair-selection control。
- `interaction_cutoff_um=None` 本身合法；只有它实际产生 graph 外 nonzero pair 时 PEPS 才拒绝。exact/MPS 继续接受 system 的完整 pair set。
- position-noise realization 只改变 nominal graph 上已选 pairs 的 distance/direction/weight，不改变 pair topology。
- real-time 与 imaginary-time PEPS 必须复用同一个 geometry/topology preflight，不能分别解释 neighbour。

### PEPS30 — amplitude/norm 使用一次确定性的短边 boundary sweep（已确认，具体化 PEPS06/15/19）

- normalized amplitude numerator 与 double-layer norm 都只执行一次完整 boundary contraction；不做正反方向 convergence check、重复平均或 hidden adaptive sweep。
- 设 validated Cartesian grid shape 为 `(Nx, Ny)`：`Nx <= Ny` 时逐列收缩，使 boundary chain 长度为 `Nx`；`Nx > Ny` 时逐行收缩。
- 正方形 tie 固定逐列；所有 sweeps 始终从低 lattice coordinate 向高 coordinate。
- numerator 与 norm 必须使用完全相同的 orientation、order 和 boundary controls，避免 normalization 混合两种近似 convention。
- orientation/order 是 PEPS19 固定算法的一部分，不增加 public option，也不写入每条 evidence；同一 register/options/library version 必须可复现。
- 不允许为了 evidence 或“验证方向一致”额外收缩；caller 仍只通过提高 environment bond dimension/tightening controls 后重跑来研究稳定性。

### PEPS31 — 当前 YASTN PEPS 只接受三个 grid factories；任意 unit-disk graph TN 独立设计（最终确认，替代 PEPS27 的 structural inference）

- geometry 仍且只能由 public `Register` 构建；PEPS 不增加 geometry 参数或第二套坐标对象。
- 当前 `backend="peps"` 与 `method="peps_imaginary_time"` 只接受由 `Register.chain(...)`、`Register.rectangle(...)` 或 `Register.square(...)` 创建的 register。
- `Register(coords)` 与 `Register.triangular(...)` 对 exact/MPS 仍完全合法，但当前 YASTN PEPS 一律在 tensor allocation 前报 capability error；即使 direct coordinates 恰好排成矩形，也不做 shape inference。
- `Register` 增加 private-only grid provenance，例如 `_peps_grid_shape: tuple[int, int] | None`：chain 为 `(N, 1)`，rectangle/square 为 `(rows, cols)`，direct/triangular 为 `None`。该信息不进入 public `Register` surface、`__all__`、repr/serialization 或 system API。
- PEPS site-to-grid mapping 直接使用 factory 保证的 stable row order 与 private shape；同时验证 `rows * cols == register.N`。禁止通过 rounded unique coordinates 猜测 topology。
- PEPS28 的 open boundary 与 PEPS29 的 graph-edge interaction rule 继续有效；graph adjacency 由 private shape 和 register site order 唯一定义，真实 interaction weights 仍来自 system canonical lowering。
- 任意二维 Rydberg coordinates 上的 unit-disk-graph tensor network 是未来独立 backend/algorithm：它应复用 `RydbergSystem`、canonical Hamiltonian IR 与 result contract，但不能作为 `backend="peps"` 的 geometry mode、YASTN kwargs 或当前 refactor 的附带功能。
- 本轮不得创建 future backend name、placeholder class、experimental flag 或空实现；只在 architecture/non-goals 中记录 seam。

### PEPS32 — boundary controls、算法与 `contraction_error` 完全冻结（最终 pin-down，具体化 PEPS15/17/30）

- amplitude numerator 与 double-layer norm 固定使用 YASTN `transfer_mpo + mps.zipper + one-site mps.compression_ + mps.vdot`；删除当前手写逐行 SVD contraction。
- `environment_bond_dimension` 原样映射到 zipper 的 `D_total`；`environment_tolerance` 同时映射到 boundary SVD cutoff 与 compression `overlap_tol`；`environment_max_iterations` 映射到每个 absorbed layer 的 `max_sweeps`。
- zipper 与 variational compression 都使用 `normalize=False`，完整保留 complex contraction factor；numerator 与 norm 逐字使用相同 controls、PEPS30 orientation/order。
- norm 直接使用同一个 `environment_bond_dimension` 数值，不平方；`svd_tolerance` 只控制 NTU state truncation，绝不复用于 readout contraction。
- iteration cap 用尽仍返回 finite estimate；不得比较 contraction summary 与 `environment_tolerance`，不得 reverse sweep、average、adaptive chi 或 dense fallback。
- 单次 contraction 的 `contraction_error` 唯一定义为所有 absorbed boundary layers 已产生的 zipper discarded ratios 与 `abs(compression_out.doverlap)` 的 maximum；没有 compression 时为 `0.0`。
- `contraction_error` 必须 finite、non-negative，但只是 dimensionless heuristic evidence，不是 physical error bound、discarded Schmidt weight 或 certificate。

### PEPS33 — normalized coefficient cache、ground phase gauge 与 evidence transaction 冻结（最终 pin-down，修订 PEPS13/24）

- simulation/ground-state 完成时的 PEPS34 local validity scan 同时保存每个 site tensor 的同设备 positive Frobenius scale；zero/non-finite scale 是 validity failure。lazy amplitude 复用这些 scales，使 numerator 与 norm 从同一临时 rescaled PEPS view 构造，不修改 private final state，也不计算可能 overflow 的 scale product或重复扫描。
- double-layer norm lazy 计算一次并 private cache；必须 finite、按 PEPS34 real up to roundoff、strictly positive。normalized coefficient 为 `raw / sqrt(norm)`。
- private coefficient cache 按 physical `labels` tuple；real-time amplitude 直接读取 coefficient。
- ground-state amplitude 先取得 `phase_reference` coefficient，再取得 target coefficient，并返回 `target / (reference / abs(reference))`；reference normalized magnitude `<= sqrt(float64 epsilon)` 时要求用户更换 reference。
- `PEPSEvidence.amplitudes` 不是 public-call pairing log，而是成功执行过的 distinct product-bra coefficient contractions：reference labels 与 target labels 都使用 PEPS24 已冻结的两字段 `PEPSAmplitudeEvidence(labels, contraction_error)`，不新增 `phase_reference` 字段。
- reference/target 相同只 contraction/record 一次；换 reference 只计算未缓存 labels；重复 cached labels 不追加 record。
- norm、coefficient cache 与 evidence 必须事务性提交：一次 public amplitude 调用任一步失败时，不发布 `norm_contraction_error` 或半完成 amplitude records；旧 snapshot 不变。
- normalized amplitude 只要求 finite；即使 approximation 给出 `abs(amplitude) > 1` 也原样返回，不 clamp、不据此 gate。caller 通过改变 controls 重跑判断稳定性。

### PEPS34 — 统一 complex128 数学有效性边界，不恢复 convergence gate（最终 pin-down，具体化 PEPS03）

- private validity relative scale 固定为 `sqrt(np.finfo(np.float64).eps)`；对理论上 real 的 `z`，允许的 imaginary roundoff 为该值乘 `max(1, abs(z))`。
- finite Hermitian expectation、ground energy 与 double-layer norm 在 slack 内取 real part；超过 slack 是无法满足 public real-result contract 的 validity failure。禁止用 `environment_tolerance` 或 contraction evidence 充当此 threshold。
- ground energy 必须以 complex 累加完整 physical Hamiltonian expectation，最后统一做一次 reality check；禁止逐项 `np.real()` 隐藏问题。
- PEPS state 每个 local tensor norm 必须 finite、strictly positive；这是 O(N) local scan，不触发 global norm contraction。
- amplitude numerator/normalized coefficient 必须 finite；除 PEPS33 的 reference-zero gauge check 外不施加物理域或 convergence gate。
- CUDA/CPU tensors、projected networks、environments 与 boundary MPS 全程留在用户指定 device；禁止 tensor/network/environment 整体搬到 host。只允许数学有效性 reduction、conditional sampling 的 scalar RNG branching、最终 public scalar与免费 evidence summary通过 `.item()` 变为 host Python `complex/float`。

### PEPS35 — tolerance 区间与 BP/CTM residual 的 report-only 语义冻结（最终 pin-down，修订 PEPS21/22/24）

- `0 < svd_tolerance < 1`、`0 < environment_tolerance < 1`；YASTN relative SVD cutoff 在 `>=1` 时会删除全部 singular values，因此边界 validation 必须拒绝。
- `ntu_iteration_tolerance` 只要求 finite、strictly positive，不设置 `<1` 上限；所有 integer caps 拒绝 bool 且为 positive integer。
- BP 的 `environment_tolerance` 是 message residual stopping target；CTM 中同时是 corner stopping target 与 environment SVD cutoff；boundary readouts 遵守 PEPS32。所有映射无 hidden clamp。
- 环境运行到 engine early stop 或用户 iteration cap；`.converged` 不控制返回。finite high residual 与 cap exhaustion 都正常返回 estimate。
- BP summary 使用 finite non-negative `max_diff`；BP expectation/sampling 的 message dimension 不受 `environment_bond_dimension` 控制，该 key 在 BP workflow 仍供 amplitude/norm 使用。
- CTM 第一次 sweep 因没有前一组 corner spectra而产生 structurally unavailable `max_dsv` 时，evidence `environment_residual=None`；两次及以上 history 后 non-finite/negative residual 才是 validity failure。
- real-time requested measurement times 的 `environment_iterations` 取 maximum；residual 全部可用时取 maximum，只要任一不可用则整体为 `None`。无 observables 时两者均为 `None`。
- ground-state 保存最终唯一 CTM 的 residual/iterations。CTM actual `max_D` 不得超过用户 cap；低于 cap 合法。

### PEPS36 — sampling 使用经过数学验证的 private conditional sampler（最终 pin-down，具体化 PEPS03/ER09）

- 禁止直接把 YASTN `env.sample()` 的输出视为已验证结果，因为其内部可能先取 `.real` 且不暴露全部 candidate weights。
- 分别实现 private BP 与 CTM sequential conditional adapters；严格使用 selected method，禁止 BP silent fallback CTM，禁止 dense conversion。
- 每个 site 的所有 physical-level raw weights 必须 finite；imaginary parts 使用 PEPS34 slack。负权重 slack 必须逐 candidate 由其自身 magnitude 计算，不能由其他大权重放宽；各自只在 roundoff slack 内从微负 clamp 到 zero，显著 non-real/negative 立即 validity failure。
- conditional total 必须 finite、strictly positive；概率固定做两次显式有限归一化并以 finite cumulative array 抽样。使用 `np.searchsorted(cdf, u, side="right")`，选择的概率必须 strictly positive，避免 `u == 0` 落入 leading zero-probability bin；RNG 固定为 local `np.random.default_rng(seed)`，禁止修改 global NumPy/Torch RNG state。
- BP 与 CTM 都复用 PEPS30 的短边遍历：`Nx <= Ny` 时逐列，否则逐行；同一 layer 内低坐标到高坐标。输出 tuple 仍重排为 Register site order，candidate labels 固定按 `terms.levels`。
- CTM conditional boundary 只需要概率比，固定使用 `zipper/compression normalize=True` 以稳定 conditioned boundary scale；PEPS32 的 `normalize=False` 只适用于必须保留 complex global factor 的 amplitude/norm contraction。
- CTM sampling 的 boundary update 使用 `environment_bond_dimension`、`environment_tolerance`、`environment_max_iterations`；cap exhaustion 仍按 report-only 返回，只要 conditional distribution 合法。
- 输出保持 `Counter[tuple[str, ...]]`，tuple 顺序严格为 Register site order。任一步失败不追加 evidence；成功后只追加 PEPS24 的 `(shots, seed)` record，不保存 `shots*N` trace或 Counter 副本。

## 16. Documentation 简化（2026-07-15）

### DOC01 — 用户文档只使用普通 Markdown（已确认）

- 删除 Quarto、quartodoc、`.qmd` 页面、generated API reference、capability-matrix generator 与 GitHub Pages workflow。
- 删除 `docs` optional dependency；文档不再需要独立 build/render 步骤。
- public signature 以源码定义为唯一来源，用户可通过 IDE 或 `help(...)` 查看；实际 constructor/function 的 validation 与 error message 是最终准据，不声称 docstring 重复了每条校验规则。
- README 与 `docs/*.md` 必须能由 GitHub 直接渲染；数学公式使用 GitHub Markdown 支持的 LaTeX 语法。

### DOC02 — 公开文档冻结为 README 加三页（已确认）

```text
README.md
docs/
├── model.md
├── simulation.md
└── gates.md
```

- `README.md`：项目定位、安装、一个最短 exact 示例，以及三页文档入口。
- `model.md`：回答“模拟的 Hamiltonian 是什么、怎样构造”，唯一负责 Register、presets、interaction、Hamiltonian 与 Protocol 共同规则。
- `simulation.md`：回答“怎样求解、返回什么”，唯一负责 initial state、observables、`t_eval`、backend options、results、ground state、PEPS evidence 与 noise ensemble。
- `gates.md`：回答“怎样从原始演化结果研究 gate”，唯一负责 gate protocol 选择、logical-basis evolution、Nielsen fidelity、conditional phase、leakage 与一阶 decay budget。

### DOC03 — 文档事实只有一个 owner（已确认）

- Hamiltonian、物理单位与 interaction convention 只在 `model.md` 定义。
- solver/result/backend capability 只在 `simulation.md` 定义；不再维护第二份 capability matrix。
- gate metrics 与 error-budget 公式只在 `gates.md` 定义；src 仍遵守 A02/A03，不恢复 report helper。
- 完整可执行研究流程、优化、扫描、绘图与持久化继续放在 `examples/`、`scripts/` 和 notebooks，不复制进稳定文档。
- 稳定文档不保存具体 optimum、benchmark pass/fail 或可能随 ARC/solver/研究结果变化的数值声称。

### DOC04 — 内部记录与文档验收不进入公开 docs（已确认）

- `REFACTOR_DECISIONS.md` 是架构决策源，不渲染为用户文档。
- 一次性数值调查进入对应 `results/<topic>/`；`find_phase` 微扰调查迁入 `results/effective_theory/`。
- notebook 验收脚本迁至 `scripts/check_notebooks.py`，不伪装成文档构建步骤。
- `docs/` 最终只包含 DOC02 的三份 Markdown 文件，不保留 cache、HTML、reference、ADR 或 research 子目录。
