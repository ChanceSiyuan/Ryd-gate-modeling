# ryd-gate

[![CI](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Universal Rydberg-atom simulation toolkit for neutral-atom quantum computing.

Models two-photon excitation in ⁸⁷Rb (hyperfine structure, Rydberg blockade, spontaneous decay, AC Stark shifts) and N-atom lattice systems. Provides time-optimal (TO) and amplitude-robust (AR) CZ gate protocols, adiabatic sweep protocols, Monte Carlo noise analysis, local addressing simulation, and an optional tensor-network path for large lattices.

## Installation

```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# With optional tensor-network support
uv pip install -e ".[dev,tn]"
```

## Quickstart

### Two-atom CZ gate

```python
import ryd_gate as rg
from ryd_gate.lattice import make_chain

# 1. Choose protocol
protocol = rg.TOProtocol()

# 2. Create system with the protocol bound
system = rg.RydbergSystem.from_lattice(
    make_chain(2, spacing_um=3.0),
    "rb87_7",
    param_set="our",
    protocol=protocol,
)

# 3. Run simulation
import numpy as np
psi0 = np.zeros(49, dtype=complex)
psi0[8] = 1.0  # |11> state

X_TO = [0.1122, 1.0431, -0.72565603, 0.0, 0.452, 1.219096]
result = rg.simulate(system, X_TO, psi0)

# 4. Analyze
print(f"Final state norm: {np.linalg.norm(result.psi_final):.6f}")
```

### N-atom lattice sweep

```python
import ryd_gate as rg
import numpy as np
from ryd_gate.lattice import make_chain

delta_start = 2 * np.pi * -40e6
delta_end = 2 * np.pi * 40e6
t_sweep = 10e-6
Omega = 2 * np.pi * 4e6

# 1. Function-defined sweep protocol
protocol = rg.SweepProtocol(
    t_gate=t_sweep,
    omega_half_fn=lambda t: 0.5 * Omega,
    delta_fn=lambda t: delta_start + (delta_end - delta_start) * t / t_sweep,
)

# 2. Create lattice system with the protocol bound
system = rg.RydbergSystem.from_lattice(
    make_chain(4, spacing_um=5.0),
    "1r",
    protocol=protocol,
)

# 3. Simulate (psi0 = ground state |gggg>)
psi0 = system.ground_state()
result = rg.simulate(system, [], psi0)
```

### 全流程示例：2D g–r 晶格演化（exact + ttn）

构建一个 3×3 的 ground–Rydberg (`1r`) 方格晶格，用失谐扫描把原子从全基态驱动进
Rydberg 有序相，并跟踪 Rydberg 占据均值 `<n_r>` 的时间演化。同一段代码只需切换
`backend=` 即可在**精确态矢量**和**树张量网络 (TTN)** 之间切换。

```python
import numpy as np
import ryd_gate as rg
from ryd_gate import InteractionSpec
from ryd_gate.lattice import make_square_lattice

# 1. 几何 + 能级结构：3x3 方格，每个格点是 |1>(基态) / |r>(Rydberg) 两能级
geom = make_square_lattice(3, 3, spacing_um=1.0)          # 无量纲间距

# 2. 失谐扫描协议（detuning 从 -8 扫到 +8，用时 6）
delta_start, delta_end, t_sweep = -8.0, 8.0, 6.0
Omega = 1.0
protocol = rg.SweepProtocol(
    t_gate=t_sweep,
    omega_half_fn=lambda t: 0.5 * Omega,
    delta_fn=lambda t: delta_start + (delta_end - delta_start) * t / t_sweep,
    n_steps=60,
)
system = rg.RydbergSystem.from_lattice(
    geom, "1r",
    interaction=InteractionSpec(C6=6.0, mode="nn"),       # 近邻 VdW ~6；TN 勿用默认 all（全连接极慢）
    protocol=protocol,
)
t_eval = np.linspace(0.0, 6.0, 7)

# 3a. 精确后端：states 是态矢量，用 system.expectation 读 Rydberg 均值
res = rg.simulate(system, [], "all_ground", backend="exact", t_eval=t_eval)
for t, psi in zip(res.times, res.states):
    print(f"t={t:.1f}  <n_r>={system.expectation('sum_nr', psi) / system.N:.4f}")

# 3b. 树张量网络后端（需要 `pip install ryd-gate[tn-ttn]`）：
#     同样的调用，TN 后端把 <n_r> 记录在 result.metadata["obs"]["n_mean"] 里
res_ttn = rg.simulate(
    system, x, "all_ground",
    backend="ttn", t_eval=t_eval, observables=["n_mean"],
    backend_options={"chi_max": 12, "dt": 0.2},
)
print("TTN <n_r>:", np.round(np.asarray(res_ttn.metadata["obs"]["n_mean"]), 4))
```

精确后端的输出（TTN 在这个小体系上应当一致）：

```
t=0.0  <n_r>=0.0000
t=1.0  <n_r>=0.0100
t=2.0  <n_r>=0.0305
t=3.0  <n_r>=0.0992
t=4.0  <n_r>=0.3209
t=5.0  <n_r>=0.2977
t=6.0  <n_r>=0.4123
```

`<n_r>` 随失谐扫描从 0 升到约 0.41——原子被驱动进 Rydberg 态，而近邻阻塞把均值压在
0.5 以下。TTN 后端请用 `mode="nn"` 并适当减小 `chi_max`、增大 `dt`（上例 `chi_max=12`,
`dt=0.2`）；默认 `mode="all"` 全连接相互作用会让 TTN 慢 orders of magnitude。把
`backend` 换成 `"tenpy"`/`"gputn"`/`"gputtn"`/`"2dtn"` 即可切到其它张量网络后端；
`"gputtn"` 使用 Julia ITensorNetworks.jl 的 GPU TTN-TDVP，需要本仓库 Julia project
已实例化且 `CUDA.functional()` 为 true。

## 项目结构 (Project structure)

所有代码都在单一命名空间 `ryd_gate` 下：**物理模型**（systems / protocols / IR）与
**算法后端**（backends）分离。核心从不构造算法专属的矩阵——每个后端把同一份
`HamiltonianIR` 降阶成自己的表示。数据流见 [`docs/architecture.md`](docs/architecture.md)。

```
src/ryd_gate/
├── __init__.py            # 包门面：re-export 常用类型 + 统一入口 simulate
├── simulate.py            # 统一仿真入口 simulate(system, x, psi0, backend=...)
├── pulse.py               # Blackman 脉冲包络工具
│
├── core/                  # ── Rydberg 系统模型核心 ──
│   ├── system.py              # RydbergSystem 主类（几何 + 能级结构 + 协议）
│   ├── level_structures.py    # 能级/跃迁/相互作用 spec + 预设 (1r/01r/ger/rb87_7)
│   ├── rb87_params.py         # Rb87 七能级物理参数 (our / lukin)
│   ├── local_blocks.py        # 单原子 Hamiltonian 矩阵块
│   ├── factories.py           # from_lattice 构造逻辑
│   ├── interactions.py        # 范德华相互作用
│   ├── states.py              # 多体初态构造
│   ├── channel_lowering.py    # 协议通道降阶
│   ├── system_model.py        # 系统抽象基类（通用框架）
│   ├── basis.py / blocks.py / observables.py / operator_spec.py / operators.py
│                              # 通用量子框架抽象：希尔伯特空间、块/可观测量注册表、符号算符
│
├── ir/                    # ── 统一中间表示（算法无关）──
│   ├── hamiltonian.py         # HamiltonianIR（静态项 + 驱动项 + basis/geometry/metadata）
│   └── evolution.py           # EvolutionResult（所有后端的统一输出）
│
├── protocols/             # ── 脉冲/控制协议 ──
│   ├── base.py / channels.py  # 协议基类、驱动通道命名约定
│   ├── gate_cz_to.py          # 时间最优 (TO) CZ 门
│   ├── gate_cz_ar.py          # 幅度鲁棒 (AR) CZ 门
│   ├── sweep.py               # 失谐扫描 + 局部寻址
│   ├── lattice_dynamics.py    # TFIM ↔ Rydberg 控制映射（淬火/退火）
│   └── digital_analog.py      # 数字-模拟混合协议
│
├── lattice/               # 纯几何：晶格形状/坐标/子格标签 + 绘图（不含能量）
├── physics/               # Rb87 原子物理：AC Stark 频移、ARC 衰变分支比
├── analysis/              # 结果后处理：门保真度/误差预算、局部寻址、粗化、对称性等
│
├── backends/              # ── 算法后端（统一命名空间）──
│   ├── _options.py            # 共享 as_backend_options()（dict/dataclass 归一化）
│   ├── exact/                 # 精确态矢量：sparse_expm / dense_ode / monte_carlo + legacy/
│   ├── tn_common/             # 所有 TN 后端共享：TN IR / lattice_spec / simulate_tn 分发
│   ├── tenpy_mps/             # TeNPy MPS DMRG/TDVP（最完整的 TN 后端）   [extra: tn]
│   ├── gputn/                 # CUDA / cuQuantum 张量网络                 [extra: gputn-cu12]
│   ├── itensor/               # Julia ITensors / ITensorNetworks / TNQS 桥接
│   ├── ttn/                   # 树张量网络（用 _vendor/pytreenet）        [extra: tn-ttn]
│   ├── nqs/                   # NQS/tVMC 外部求解器边界                   [extra: nqs]
│   └── peps2d/                # 2D PEPS/BP 外部求解器边界                 [extra: tn-2d]
│
└── _vendor/               # 内嵌第三方 PyTreeNet（见 _vendor/NOTICE.md）
```

**怎么读这个结构**：用户通常只接触三样东西 —— 一个 `protocols/` 里的协议、
`core/` 里的 `RydbergSystem`、以及顶层的 `simulate(...)`。其余 (`ir/`、`backends/`、
`_vendor/`) 是分发与求解的内部机制，按需深入即可。

## API Reference

All commonly used symbols are available from the top-level package:

```python
import ryd_gate as rg
```

### Systems

| Symbol | Description |
|--------|-------------|
| `rg.RydbergSystem.from_lattice(geometry, "rb87_7", param_set="our", protocol=...)` | 87Rb 7-level model with "our" parameters |
| `rg.RydbergSystem.from_lattice(geometry, "rb87_7", param_set="lukin", protocol=...)` | 87Rb 7-level model with Lukin parameters |
| `rg.RydbergSystem.from_lattice(geometry, "ger", param_set="analog_3", protocol=...)` | 3-level analog ger model |
| `rg.RydbergSystem.from_lattice(geometry, "1r", protocol=...)` | N-atom 2-level lattice |

### Protocols

| Symbol | Description |
|--------|-------------|
| `rg.TOProtocol()` | Time-optimal CZ gate — cosine phase, 6 params |
| `rg.ARProtocol()` | Amplitude-robust CZ gate — dual-sine phase, 8 params |
| `rg.SweepProtocol(...)` | Function-defined global Rydberg sweep |

### Simulation

| Symbol | Description |
|--------|-------------|
| `rg.simulate(system, x, psi0)` | Exact compile + evolve (default backend) |
| `rg.compile_hamiltonian_ir(system, params)` | Core unified Hamiltonian output |
| `rg.EvolutionResult` | Result dataclass: psi_final, times, states |
| `rg.HamiltonianIR` | Solver-agnostic Hamiltonian IR |

### Analysis

| Symbol | Description |
|--------|-------------|
| `rg.average_gate_infidelity(system, protocol, x)` | Nielsen average gate infidelity |
| `rg.error_budget(system, protocol, x)` | Error decomposition by channel |
| `rg.AddressingEvaluator` | Local addressing error analysis |

### Pulse Utilities

| Symbol | Description |
|--------|-------------|
| `rg.blackman_pulse(t, t_rise, t_gate)` | Blackman-windowed flat-top pulse |
| `rg.blackman_pulse_sqrt(t, t_rise, t_gate)` | Square-root Blackman envelope |
| `rg.blackman_window(t, t_rise)` | Raw Blackman window function |

## Physical Model

The two-atom solver models two ⁸⁷Rb atoms with:

| Parameter | Value | Description |
|-----------|-------|-------------|
| n | 70 | Rydberg principal quantum number |
| Ω_eff | 2π × 7 MHz | Effective two-photon Rabi frequency |
| Δ | 2π × 9.1 GHz | Intermediate state detuning |
| d | 3 μm | Interatomic distance |
| C₆ | 2π × 874 GHz·μm⁶ | van der Waals coefficient |

The 7-level basis per atom: `|0⟩, |1⟩, |e₁⟩, |e₂⟩, |e₃⟩, |r⟩, |r_garb⟩`
Two-atom Hilbert space: 7² = 49 dimensions.

## Examples

Self-contained, runnable demos live in `examples/` (a good place to start):

| Example | Description |
|---------|-------------|
| `examples/demo_local_addressing.py` | Exact local-addressing experiments (domain shrinking, Higgs mode) |
| `examples/demo_local_addressing_tn.py` | Same workflow through the tensor-network backend |

```bash
uv run python examples/demo_local_addressing.py --Lx 2 --Ly 2 --experiment domain
```

## Notebooks

Experimental, batch, and plotting workflows live in `scripts/notebooks/`:

| Notebook | Description |
|----------|-------------|
| `01_cz_gate_validation_and_errors.ipynb` | CZ gate validation, population evolution, deterministic/MC error budgets, and sensitivity analysis |
| `02_ac_stark_local_addressing.ipynb` | AC Stark landscapes, local-addressing dynamics, addressing scans, noise sensitivity, and adiabaticity diagnostics |
| `03_lattice_dynamics_and_annealing.ipynb` | Route-2 pulse cells, exact 3-level lattice dynamics, and tensor-network annealing configurations |

## Testing

```bash
# Run fast tests only (~1 second)
uv run pytest

# Run all tests including slow ODE-based tests (~15 min)
uv run pytest -m ""

# Run with coverage
uv run pytest -m "" --cov=ryd_gate --cov-report=html
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | Data flow, backend table, and core layout |
| [Getting Started](docs/getting_started.md) | Installation and basic usage |
| [Schrodinger Solver](docs/schrodinger_solver.md) | 7-level model theory and API |
| [Error Budget](docs/error_budget_methodology.md) | Error decomposition methodology |
| [Validation](docs/validation.md) | Test suite documentation |

## References

* Evered *et al.*, "High-fidelity parallel entangling gates on a neutral-atom quantum computer", *Nature* **622**, 268 (2023).
* Ma *et al.*, "Benchmarking and fidelity response theory of high-fidelity Rydberg entangling gates", *PRX Quantum* **6**, 010331 (2025).

## License

MIT
