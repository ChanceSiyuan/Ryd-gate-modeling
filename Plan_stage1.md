# Refactor Plan: Product API for Neutral-Atom Simulation

## 目标

把当前仓库从“研究代码 + backend 实验平台”逐步整理成一个面向中性原子平台的经典模拟器 SDK。目标用户可以用接近 Pulser 的抽象描述：

```python
register = Register.square(side=4, spacing_um=5.0)
device = DeviceSpec.virtual_rb87()
atom_model = AtomModel.preset("01r")

seq = Sequence(register, device=device, atom_model=atom_model)
seq.declare_channel("rydberg", "rydberg_global")
seq.add(Pulse.constant(...), "rydberg")

result = simulate(seq, backend="exact")
```

但内部不要推翻现有架构。当前仓库已经形成了清晰的数据流：

```text
RydbergSystem + Protocol
        |
        v
HamiltonianIR
        |
        v
exact / MPS / GPUTN / PEPS / other backends
```

重构方向应是新增一层稳定的用户 API，并把它编译到现有的 `RydbergSystem`、`Protocol`、`HamiltonianIR`。不要把 `Register`、`DeviceSpec`、`Sequence` 直接变成 backend 输入，也不要把 `RydbergSystem` 改成新的 God class。

## 设计原则

1. `Register` 只描述原子位置、id 和顺序，不描述能级、相互作用、设备、pulse。
2. `AtomModel` 描述局域能级、transition、detuning channel 和物理参数视图，不描述具体几何。
3. `DeviceSpec` 描述硬件/虚拟设备约束，不代表 backend，也不代表真实 QPU job。
4. `Waveform` 和 `Pulse` 是可验证、可序列化的用户层 pulse 描述，不直接负责 Hamiltonian 矩阵构造。
5. `Sequence` 是用户层调度对象，负责编排 pulse/channel/target/timing，最终编译成现有 `Protocol`。
6. `NoiseModel` 是声明式噪声配置，底层可以先接入现有 Monte Carlo 和非厄米 decay 机制。
7. 现有 `RydbergSystem.from_lattice(...)`、`Protocol`、`compile_hamiltonian_ir(...)`、backend dispatcher 应继续作为核心内核存在。

## 当前 repo 已有能力映射

| 目标 API | 当前相似实现 | 已经有的能力 | 缺口 |
|---|---|---|---|
| `Register` | `src/ryd_gate/lattice/geometry.py::LatticeGeometry` | 坐标、N、sublattice、spacing、chain/square/triangular/custom factory | atom ids、稳定顺序的公开承诺、distance matrix、blockade graph、device validation、JSON |
| `RegisterLayout` | 无显式类；`make_square_lattice` 等隐式表达 layout | 能生成常用几何 | 没有可复用 layout 对象、trap layout、qubit-to-trap 映射 |
| `AtomModel` | `LevelStructureSpec`, `TransitionSpec`, `rb87_params`, `local_blocks` | levels、rydberg levels、transition channel、detuning channel、Rb87 7-level 参数和 local blocks | 缺 public model object、initial level、species、noise/collapse 描述、model capability |
| `InteractionSpec` | `core.level_structures.InteractionSpec`, `core.interactions.vdw_couplings` | C6、range cutoff、all/nn/nnn VdW pair lowering | 缺 C3/XY/custom interactions、device 默认系数、单位 schema |
| `ChannelSpec` | `protocols/channels.py`, `LevelStructureSpec.transitions`, string channel convention | channel name convention 已经被 compiler 和 backend 使用 | 缺 channel 约束、addressing mode、duration/amplitude/detuning limits、retarget rules |
| `DeviceSpec` | 无单独对象；约束散落在 `InteractionSpec`、TN spec、backend options | 可以手动传 C6、spacing、backend 参数 | 缺硬件约束验证、虚拟设备、真实设备 profile、capability matrix |
| `Waveform` | `pulse.py` blackman helpers；`SweepProtocol`/`DigitalAnalogProtocol` callable schedules | 能表达任意函数型连续 schedule | 缺不可变 waveform 数据结构、采样、拼接、序列化、duration validation |
| `Pulse` | `Protocol.get_drive_coefficients` 隐式表达 | Hamiltonian coefficient lowering 已经存在 | 缺 amplitude/detuning/phase/post_phase_shift 的用户对象 |
| `Sequence` | `Protocol` 子类；`DigitalAnalogProtocol`；`SweepProtocol` | 已能把 schedule 降成 channel coefficients | 缺 channel declaration、operation list、targeting、delay、measure、validate、serialize |
| `NoiseModel` | `MonteCarloRunner`, `enable_rydberg_decay`, `enable_intermediate_decay`, static overlays | detuning/amplitude/position/local RIN、非厄米 decay、branching analysis 的部分能力 | 缺统一声明式 schema、backend capability validation、collapse ops/trajectory IR |
| `SimulationResult` | `EvolutionResult`, `MonteCarloResult` | psi_final、times、states、metadata、MC fidelity samples | 缺 user-level sample bitstrings、observables schema、job-like result object |

## 目标目录结构

新增用户层 API，不移动现有核心模块：

```text
src/ryd_gate/api/
    __init__.py
    register.py
    atom_model.py
    device.py
    waveform.py
    pulse.py
    sequence.py
    noise.py
    result.py

src/ryd_gate/compiler/
    __init__.py
    sequence_compiler.py
    noise_compiler.py

src/ryd_gate/compat/
    __init__.py
    pulser_import.py
    pulser_export.py
```

现有模块继续保留：

```text
src/ryd_gate/lattice/geometry.py        # pure geometry kernel
src/ryd_gate/core/level_structures.py   # local levels/transitions/interactions
src/ryd_gate/core/system.py             # RydbergSystem core model
src/ryd_gate/protocols/                 # backend-facing Protocol interface
src/ryd_gate/ir/                        # HamiltonianIR/EvolutionResult
src/ryd_gate/backends/                  # exact/TN/GPU/PEPS engines
```

## API 1: Register

### 职责

`Register` 是用户层原子 register。它只负责：

- atom/qubit ids；
- 坐标，单位公开为 `um`；
- 稳定顺序；
- layout metadata；
- 与当前 `LatticeGeometry` 的桥接；
- 几何 validation；
- distance/pair/blockade graph 等几何工具。

### 不承担的职责

`Register` 不应包含：

- `AtomModel`；
- `DeviceSpec`；
- C6/C3 等相互作用强度；
- pulse/channel；
- backend option；
- Hamiltonian term。

### 建议数据结构

```python
@dataclass(frozen=True)
class Register:
    ids: tuple[str, ...]
    coords_um: tuple[tuple[float, ...], ...]
    layout: RegisterLayout | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_coordinates(cls, coords, ids=None, prefix="q", center=True): ...

    @classmethod
    def square(cls, side: int, spacing_um: float, prefix="q"): ...

    @classmethod
    def rectangle(cls, rows: int, cols: int, spacing_um: float, prefix="q"): ...

    @classmethod
    def chain(cls, n_atoms: int, spacing_um: float, prefix="q"): ...

    @property
    def n_atoms(self) -> int: ...

    @property
    def dim(self) -> int: ...

    @property
    def coords_array(self) -> np.ndarray: ...

    def index(self, atom_id: str) -> int: ...
    def id_at(self, index: int) -> str: ...
    def distances_um(self) -> np.ndarray: ...
    def interaction_pairs(self, cutoff_um=None) -> tuple[tuple[int, int, float], ...]: ...
    def blockade_graph(self, radius_um: float) -> "nx.Graph": ...
    def validate(self, device: "DeviceSpec") -> None: ...
    def to_geometry(self) -> "LatticeGeometry": ...
```

### 当前 repo 对应实现

`LatticeGeometry` 已经是正确的内核边界：它只包含 `N`、`coords`、`sublattice`、`spacing_um`。`make_chain`、`make_square_lattice`、`make_triangular_lattice`、`make_geometry_from_coords` 已经能生成几何。

### 差异

`LatticeGeometry` 是内部几何对象，不适合作为产品 API：

- 没有 atom ids，当前 `BasisSpec.site_labels` 在 `build_from_lattice` 中直接生成 `"0"..."N-1"`；
- 没有公开保证 qubit order；
- 没有 validation；
- 没有序列化；
- shape factory 命名偏研究脚本，不够用户层。

### 迁移策略

第一阶段新增 `Register.to_geometry()`，不修改 `RydbergSystem.from_lattice`。后续可以加：

```python
RydbergSystem.from_register(register, atom_model, interaction=..., protocol=...)
```

它内部仍调用 `from_lattice(register.to_geometry(), ...)`。

## API 2: RegisterLayout

### 职责

`RegisterLayout` 描述 trap layout 或规则几何模板。它是可选元数据，不替代 `Register`。

```python
@dataclass(frozen=True)
class RegisterLayout:
    name: str
    trap_coords_um: tuple[tuple[float, ...], ...]
    kind: Literal["chain", "square", "rectangular", "triangular", "custom"]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def make_register(self, trap_ids, atom_ids=None) -> Register: ...
```

### 当前 repo 对应实现

当前只有函数式 layout factory，没有独立 layout 对象。

### 第一阶段是否必须实现

不是必须。可以先把 `layout` 字段设为 `None` 或简单 metadata，等到需要 trap reuse、device calibration 或 Pulser import/export 时再实现。

## API 3: AtomModel

### 职责

`AtomModel` 是用户层“局域物理模型”描述。它应回答：

- 单原子 Hilbert space 有哪些 levels；
- 初态默认 level；
- 哪些 levels 是 Rydberg；
- 哪些 transitions 可被 channel 驱动；
- detuning channel 作用在哪个 level；
- 使用哪套物理参数；
- 该模型支持哪些噪声和 backend。

### 建议数据结构

```python
@dataclass(frozen=True)
class AtomModel:
    name: Literal["01", "1r", "01r", "ger", "analog_3", "rb87_7"]
    levels: tuple[str, ...]
    initial_level: str
    rydberg_levels: tuple[str, ...] = ()
    transitions: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    detuning_levels: Mapping[str, str] = field(default_factory=dict)
    interaction: Literal["none", "ising_c6", "xy_c3", "custom"] = "ising_c6"
    species: str | None = "Rb87"
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def preset(cls, name: str, **params) -> "AtomModel": ...

    def to_level_structure(self) -> "LevelStructureSpec": ...
    def validate_channel(self, channel: "ChannelSpec") -> None: ...
    def supported_backends(self) -> frozenset[str]: ...
```

### 预设建议

| Preset | 含义 | 当前支持程度 |
|---|---|---|
| `01` | 两能级 computational model，主要给 circuit/stabilizer 使用 | 当前没有显式 preset；需要新增 |
| `1r` | 两能级 Rydberg analog model，`|1> <-> |r>` | 当前已有 `level_structure("1r")` |
| `01r` | 三能级模型，`|0>` spectator，`|1> <-> |r>`，可选 `|0> <-> |1>` | 当前已有 `level_structure("01r")` 和 `DigitalAnalogProtocol` |
| `ger` | 三能级 ladder，`|g> <-> |e> <-> |r>` | 当前已有 `level_structure("ger")` 和 `analog_3` physical blocks |
| `analog_3` | 带 Rb87 参数的三能级有效/ladder 模型 | 当前通过 `level_structure="ger", param_set="analog_3"` 表达 |
| `rb87_7` | 7-level Rb87 精细模型 | 当前已有 `level_structure("rb87_7"), param_set in {"our", "lukin"}` |

### 当前 repo 对应实现

- `LevelStructureSpec` 已有 `name`、`levels`、`rydberg_levels`、`transitions`、`detuning_levels`。
- `TransitionSpec` 已有 transition-to-channel 映射。
- `rb87_params.py` 已有 Rb87 7-level 物理参数。
- `local_blocks.py` 已有 `analog_3` 和 `rb87_7` local Hamiltonian blocks。

### 差异

`LevelStructureSpec` 是 compiler-facing 内核对象，适合 Hamiltonian lowering；`AtomModel` 是 user-facing model choice，应额外包含：

- `initial_level`；
- species；
- parameter set；
- physical constants；
- noise/collapse metadata；
- backend capability；
- user-friendly aliases。

### 迁移策略

不要把 `LevelStructureSpec` 改名成 `AtomModel`。新增 `AtomModel` wrapper，并让：

```python
AtomModel.preset("01r").to_level_structure()
```

返回当前 `LevelStructureSpec`。`RydbergSystem.from_lattice` 暂时继续接受 string 或 `LevelStructureSpec`；后续可扩展接受 `AtomModel`。

## API 4: InteractionSpec

### 职责

`InteractionSpec` 描述 pair interaction 的物理形式和截断策略。

### 当前 repo 实现

当前已有：

```python
@dataclass(frozen=True)
class InteractionSpec:
    C6: float = DEFAULT_C6
    max_range_um: float | None = None
    mode: Literal["all", "nn", "nnn"] = "all"
```

并且 `build_from_lattice` 会把它降成 `interaction_pairs`，`H_vdw` 用 `RydbergPairInteractionSpec` 注册。

### 建议扩展

短期保持兼容，增加字段时要保守：

```python
@dataclass(frozen=True)
class InteractionSpec:
    kind: Literal["vdw_c6", "dipole_c3", "xy_c3", "custom"] = "vdw_c6"
    C6: float | None = DEFAULT_C6
    C3: float | None = None
    max_range_um: float | None = None
    mode: Literal["all", "nn", "nnn", "cutoff"] = "all"
    angular: Mapping[str, Any] = field(default_factory=dict)
```

不要在第一阶段改动现有 `InteractionSpec`，可以先新增 `api.InteractionSpec` 或只补文档。等 `DeviceSpec` 稳定后再统一。

## API 5: ChannelSpec

### 职责

`ChannelSpec` 是 `DeviceSpec` 的一部分，描述硬件或虚拟通道能力：

- 通道类型；
- 连接哪个 atom transition；
- global/local addressing；
- amplitude/detuning/phase limits；
- duration/clock constraints；
- modulation bandwidth；
- target/retarget 规则。

### 建议数据结构

```python
@dataclass(frozen=True)
class ChannelSpec:
    name: str
    kind: Literal["rydberg", "raman", "microwave", "dmm", "custom"]
    transition: str
    addressing: Literal["global", "local"]
    max_abs_amplitude_rad_per_us: float | None = None
    max_abs_detuning_rad_per_us: float | None = None
    min_duration_ns: int = 0
    max_duration_ns: int | None = None
    clock_period_ns: int = 1
    max_targets: int | None = None
    retarget_time_ns: int | None = None
    modulation_bandwidth_mhz: float | None = None
    phase_reference: Literal["global", "per_channel", "per_target"] = "per_channel"

    def validate_pulse(self, pulse: "Pulse") -> list["ValidationIssue"]: ...
```

### 当前 repo 对应实现

当前 channel 是 string convention：

- `global_X`, `global_n` for `1r`;
- `drive_R`, `drive_hf`, `delta_R`, `delta_hf` for `01r`;
- `drive_420`, `drive_420_dag`, `H_1013`, `lightshift_zero` for 7-level/analog paths。

这些 string 已经被 `Protocol.drive_channels`、`get_drive_coefficients`、`compile_hamiltonian_ir`、`channel_lowering.py` 使用。

### 差异

现在的 channel 只说明“compiler 应该找哪个 block”，不说明“设备是否允许这个 pulse”。`ChannelSpec` 应该在用户层 validation 负责硬件约束，compiler 仍然使用现有 channel name。

## API 6: DeviceSpec

### 职责

`DeviceSpec` 描述设备约束。它不代表 backend，不代表 QPU job，也不应该直接保存 wavefunction。

它负责：

- geometry constraints；
- atom species；
- allowed atom models；
- interaction coefficients；
- channel specs；
- timing constraints；
- run/shots limits；
- default noise；
- validation。

### 建议数据结构

```python
@dataclass(frozen=True)
class DeviceSpec:
    name: str
    dimensions: Literal[2, 3]
    atom_species: str
    allowed_atom_models: tuple[str, ...]
    default_atom_model: str
    min_atom_distance_um: float
    max_atom_num: int | None = None
    max_radial_distance_um: float | None = None
    interaction_coeffs: Mapping[str, float] = field(default_factory=dict)
    channels: Mapping[str, ChannelSpec] = field(default_factory=dict)
    supports_slm_mask: bool = False
    max_sequence_duration_ns: int | None = None
    max_runs: int | None = None
    default_noise: "NoiseModel | None" = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def virtual_rb87(cls, atom_model="01r", **relaxed_constraints): ...

    def validate_register(self, register: Register) -> list["ValidationIssue"]: ...
    def validate_atom_model(self, atom_model: AtomModel) -> list["ValidationIssue"]: ...
    def validate_pulse(self, pulse: "Pulse", channel: str) -> list["ValidationIssue"]: ...
    def validate_sequence(self, sequence: "Sequence") -> list["ValidationIssue"]: ...
    def to_virtual(self, **relaxed_constraints) -> "DeviceSpec": ...
```

### 为什么不直接包含一个 `atom_model`

同一台中性原子平台可以用不同模拟层级描述：

- `1r` 两能级 analog；
- `01r` 三能级 pulse simulation；
- `rb87_7` 精细模型；
- 未来的 `01` circuit/stabilizer 子集。

所以 `DeviceSpec` 应该有 `allowed_atom_models` 和 `default_atom_model`，而不是只能绑定一个固定 `AtomModel`。具体仿真使用哪个模型，由 `Sequence(..., atom_model=...)` 或 `simulate(..., atom_model=...)` 决定。

### 当前 repo 对应实现

当前没有单独的 `DeviceSpec`。硬件/物理约束分散在：

- `InteractionSpec.C6/mode/max_range_um`；
- `RydbergSystem.from_lattice(..., param_set=...)`；
- `rb87_params.py`；
- protocol constructors；
- TN backend options。

### 迁移策略

第一阶段实现 virtual device，不要试图声明真实硬件完整参数：

```python
DeviceSpec.virtual_rb87_01r()
DeviceSpec.virtual_rb87_1r()
DeviceSpec.virtual_rb87_7()
```

这些只用于 validation 和默认 C6/channel limits，不改变现有 backend。

## API 7: Waveform

### 职责

`Waveform` 是可序列化、可采样的时间函数，公开单位建议：

- duration: `ns`；
- amplitude/detuning samples: `rad/us` 或明确 unit 字段；
- 内部 compiler 可转换到当前 repo 使用的 SI seconds / rad/s。

### 建议数据结构

```python
@dataclass(frozen=True)
class Waveform:
    duration_ns: int
    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)
    samples: tuple[float, ...] | None = None
    unit: str = "rad_per_us"

    @classmethod
    def constant(cls, duration_ns: int, value: float, unit="rad_per_us"): ...

    @classmethod
    def ramp(cls, duration_ns: int, start: float, stop: float, unit="rad_per_us"): ...

    @classmethod
    def blackman(cls, duration_ns: int, area: float | None = None, peak: float | None = None): ...

    @classmethod
    def interpolated(cls, duration_ns: int, times_ns, values, unit="rad_per_us"): ...

    @classmethod
    def custom(cls, samples, dt_ns: int = 1, unit="rad_per_us"): ...

    def sample(self, dt_ns: int = 1) -> np.ndarray: ...
    def value_at(self, t_ns: float) -> float: ...
    def integral(self) -> float: ...
    def first_value(self) -> float: ...
    def last_value(self) -> float: ...
    def concat(self, other: "Waveform") -> "CompositeWaveform": ...
```

### 当前 repo 对应实现

当前 `pulse.py` 有 Blackman helper；`SweepProtocol` 和 `DigitalAnalogProtocol` 接收 Python callable，因此表达能力很强。

### 差异

callable schedule 不适合作为产品 API 的唯一形式：

- 不易 serialize；
- 不易 validate duration/limits；
- 不易 export/import；
- 不易做 UI/plot/sampling；
- 不易 hash/reproduce。

### 迁移策略

新增 `Waveform` 后，`SequenceProtocol.get_drive_coefficients(t, params)` 通过 `waveform.value_at(t_ns)` 得到当前值。旧 `SweepProtocol` 和 `DigitalAnalogProtocol` 保留，供高级用户直接写函数。

## API 8: Pulse

### 职责

`Pulse` 组合 amplitude、detuning、phase。它不保存 target，target 属于 `Sequence.add(...)` 的 operation。

### 建议数据结构

```python
@dataclass(frozen=True)
class Pulse:
    amplitude: Waveform
    detuning: Waveform
    phase_rad: float = 0.0
    post_phase_shift_rad: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def constant(
        cls,
        duration_ns: int,
        amplitude: float,
        detuning: float,
        phase_rad: float = 0.0,
    ) -> "Pulse": ...

    @classmethod
    def constant_amplitude(cls, amplitude: float, detuning: Waveform, phase_rad=0.0): ...

    @classmethod
    def constant_detuning(cls, amplitude: Waveform, detuning: float, phase_rad=0.0): ...

    @property
    def duration_ns(self) -> int: ...

    def validate(self, channel: ChannelSpec) -> list["ValidationIssue"]: ...
```

### 当前 repo 对应实现

Pulse 当前被 `Protocol.get_drive_coefficients` 隐式表达。比如：

- `SweepProtocol` 输出 `global_X = Omega(t)/2`, `global_n = -Delta(t)`；
- `DigitalAnalogProtocol` 输出 `drive_R = Omega_R(t)/2`, `delta_R = -Delta_R(t)` 等。

### 差异

用户现在必须理解 Hamiltonian coefficient convention。`Pulse` 应该让用户描述物理 amplitude/detuning/phase，compiler 再处理：

- factor `1/2`；
- detuning sign；
- hermitian conjugate；
- per-site/global lowering。

## API 9: Sequence

### 职责

`Sequence` 是产品 API 的中心，负责把 register、device、atom model、channel、pulse schedule 组织起来。

### 建议数据结构

```python
@dataclass(frozen=True)
class PulseOp:
    channel: str
    pulse: Pulse
    t_start_ns: int
    targets: tuple[str, ...] | None = None

@dataclass(frozen=True)
class DelayOp:
    channel: str
    duration_ns: int
    t_start_ns: int

@dataclass(frozen=True)
class TargetOp:
    channel: str
    targets: tuple[str, ...]
    t_start_ns: int

@dataclass(frozen=True)
class MeasureOp:
    basis: str
    t_start_ns: int

Operation = PulseOp | DelayOp | TargetOp | MeasureOp
```

```python
class Sequence:
    def __init__(
        self,
        register: Register,
        device: DeviceSpec,
        atom_model: AtomModel | str | None = None,
    ): ...

    def declare_channel(self, name: str, channel_id: str, initial_target=None): ...
    def add(self, pulse: Pulse, channel: str, targets=None, protocol="min-delay"): ...
    def delay(self, duration_ns: int, channel: str): ...
    def target(self, targets, channel: str): ...
    def phase_shift(self, phi_rad: float, *atom_ids: str): ...
    def measure(self, basis: str = "ground-rydberg"): ...

    @property
    def duration_ns(self) -> int: ...

    def validate(self) -> list[ValidationIssue]: ...
    def sample(self, dt_ns: int = 1) -> "SampledSequence": ...
    def draw(self, ...): ...
    def to_dict(self) -> dict: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Sequence": ...

    def compile_protocol(self) -> "Protocol": ...
    def compile_system(self, interaction: InteractionSpec | None = None) -> "RydbergSystem": ...
```

### 当前 repo 对应实现

`Protocol` 已经是 backend-facing schedule interface。`SweepProtocol` 和 `DigitalAnalogProtocol` 说明这个抽象已经跑通：

- `unpack_params(...)`；
- `drive_channels(system)`；
- `get_drive_coefficients(t, params)`；
- optional plotting；
- exact/TN backend 共用 channel lowering。

### 差异

当前 `Protocol` 是研究/内核接口，不是用户产品接口：

- 用户需要直接写 Python function；
- 没有 operation list；
- 没有 device validation；
- 没有 target retargeting；
- 没有 serialization；
- 没有统一 pulse object。

### 编译策略

新增 `SequenceProtocol(Protocol)`，不要让 backend 直接读 `Sequence`：

```python
class SequenceProtocol(Protocol):
    def __init__(self, sequence: Sequence): ...

    @property
    def n_params(self) -> int:
        return 0

    def unpack_params(self, x, system) -> dict:
        return {"t_gate": sequence.duration_ns * 1e-9, "n_sites": system.N}

    def drive_channels(self, system) -> frozenset[str]:
        return compiled channel names

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        return coefficients at time t using existing channel conventions
```

`Sequence.compile_system()` 应该做：

```text
Register.to_geometry()
AtomModel.to_level_structure()
DeviceSpec/default InteractionSpec
Sequence.compile_protocol()
RydbergSystem.from_lattice(...)
```

### 第一阶段支持范围

优先支持 `01r` 和 `1r`：

- global Rydberg channel；
- optional local Rydberg channel；
- amplitude/detuning/phase；
- delay；
- measure marker；
- exact backend；
- MPS backend for supported two-level lowering。

不要第一阶段就实现完整 Pulser feature set。

## API 10: NoiseModel

### 职责

`NoiseModel` 统一描述噪声，不直接运行 MC。

它应支持：

- quasi-static detuning noise；
- amplitude noise；
- local RIN；
- position noise；
- state preparation/readout error；
- Rydberg/intermediate spontaneous emission；
- dephasing/relaxation；
- leakage；
- backend capability validation。

### 建议数据结构

```python
@dataclass(frozen=True)
class NoiseModel:
    runs: int = 1
    seed: int | None = None

    state_prep_error: float = 0.0
    p_false_pos: float = 0.0
    p_false_neg: float = 0.0

    temperature_uK: float | None = None
    position_sigma_um: float | tuple[float, float, float] | None = None

    amp_sigma: float | None = None
    detuning_sigma_rad_per_us: float | None = None
    local_rin_sigma: float | None = None

    relaxation_rates: Mapping[str, float] = field(default_factory=dict)
    dephasing_rates: Mapping[str, float] = field(default_factory=dict)
    rydberg_decay: bool = False
    intermediate_decay: bool = False
    branching: bool = False

    def validate(self, atom_model: AtomModel, backend: str) -> list[ValidationIssue]: ...
    def sample_quasistatic(self, register: Register, rng: np.random.Generator) -> "NoiseSample": ...
    def to_monte_carlo_runner(self, system: "RydbergSystem", x: list[float]) -> "MonteCarloRunner": ...
```

### 当前 repo 对应实现

`MonteCarloRunner` 已经支持：

- detuning noise；
- amplitude noise；
- local RIN；
- position noise；
- gate fidelity statistics；
- branching/error budget 部分逻辑。

`local_blocks.py` 和 `rb87_params.py` 已经有：

- `enable_rydberg_decay`；
- `enable_intermediate_decay`；
- `enable_0_scattering`；
- decay rates；
- branching metadata。

### 差异

当前噪声是 procedure API：

```python
runner.setup_detuning_noise(...)
runner.setup_amplitude_noise(...)
runner.run_gate_fidelity(...)
```

产品 API 应是 declarative：

```python
noise = NoiseModel(detuning_sigma_rad_per_us=..., position_sigma_um=...)
result = simulate(seq, noise=noise, backend="exact")
```

### 迁移策略

第一阶段只让 `NoiseModel` 编译到现有 `MonteCarloRunner`。更复杂的 Lindblad/MCWF 后续再加 `LindbladIR` 或 `TrajectoryIR`。

## API 11: Simulation Entry Point

### 当前状态

当前入口：

```python
simulate(system, x, psi0="all_ground", backend="exact", **kwargs)
```

要求用户传 `RydbergSystem` 和 protocol 参数 `x`。

### 建议扩展

保留旧入口，新增 overload：

```python
def simulate(obj, x=None, psi0="all_ground", *, backend="exact", noise=None, **kwargs):
    if isinstance(obj, Sequence):
        system = obj.compile_system()
        protocol_x = []
    elif isinstance(obj, RydbergSystem):
        system = obj
        protocol_x = [] if x is None else x
    else:
        raise TypeError(...)
```

短期也可以新增更明确的函数：

```python
simulate_sequence(sequence, backend="exact", noise=None, ...)
```

这样不会破坏现有测试。

## API 12: Result / Sampling

### 当前实现

`EvolutionResult` 已有：

- `psi_final`；
- `times`；
- `states`；
- `metadata`。

`MonteCarloResult` 已有 fidelity/noise sample statistics。

### 建议用户层结果

```python
@dataclass
class SimulationResult:
    evolution: EvolutionResult
    sequence: Sequence | None = None
    register: Register | None = None
    atom_model: AtomModel | None = None

    def expectation(self, observable): ...
    def sample(self, n_shots: int, basis="computational", seed=None) -> Counter[str]: ...
    def populations(self, level: str) -> np.ndarray: ...
    def correlations(self, level: str = "r") -> np.ndarray: ...
```

不要第一阶段替换 `EvolutionResult`。可以先提供 wrapper。

## Compiler Bridge

新增 `sequence_compiler.py`，职责是把用户 API 降到现有内核：

```text
Register
  -> LatticeGeometry

AtomModel
  -> LevelStructureSpec + physical params

DeviceSpec
  -> validation + default InteractionSpec

Sequence
  -> SequenceProtocol

SequenceProtocol + RydbergSystem
  -> HamiltonianIR through existing compile_hamiltonian_ir
```

关键点：compiler bridge 不能绕开 `compile_hamiltonian_ir`，否则 exact/TN/GPU backend 会分叉。

## Backend Compatibility

### exact backend

第一优先级。它能支持：

- `1r`；
- `01r`；
- `ger/analog_3`；
- `rb87_7` 小系统；
- quasi-static MC；
- 非厄米 decay。

### MPS backend

当前 MPS/TN path 支持 `1r` 和部分 `01r` lowering。注意：

- `omega_hf != 0` 的 `DigitalAnalogProtocol` 当前会被 TN lowering 拒绝；
- `01r` 在 TN 中很多时候被映射到 effective `1r` 子空间；
- 7-level 不适合当前 TeNPy MPS path，除非新增 local site/operator 支持并控制维度增长。

产品 API 的 `DeviceSpec.validate_sequence` 应提前给出 capability warning/error，而不是让 backend 深处报错。

### Stabilizer backend

Stabilizer 不适合任意 analog laser waveform。它只应支持：

- `AtomModel("01")`；
- Clifford circuit 或 pulse-compiled Clifford subset；
- measurement/readout noise；
- 可能的 Pauli noise。

不要宣称 stabilizer 可以模拟一般 `01r` Rydberg pulse。

## 分阶段实施计划

### Phase 0: 文档和边界冻结

目标：避免继续把职责塞进 `RydbergSystem` 或 `Protocol`。

任务：

- 在 docs 中明确当前内核架构；
- 新增本计划；
- 记录 public API 目标；
- 标出 legacy/research APIs 和 product APIs。

验收：

- `Plan.md` 存在；
- README 或 docs index 能链接到 architecture/product plan；
- 没有代码行为变更。

### Phase 1: Register + AtomModel wrapper

目标：先建立最小用户层对象，复用现有 geometry 和 level structure。

任务：

- 新增 `api/register.py`；
- 新增 `api/atom_model.py`；
- `Register.to_geometry()`；
- `AtomModel.preset(...).to_level_structure()`；
- tests 覆盖 order、coords、level preset roundtrip。

验收：

```python
reg = Register.square(2, 5.0)
model = AtomModel.preset("01r")
system = RydbergSystem.from_lattice(reg.to_geometry(), model.to_level_structure())
assert system.N == 4
assert system.basis.local_levels == ("0", "1", "r")
```

### Phase 2: DeviceSpec + ChannelSpec validation

目标：建立硬件约束层，但不改 backend。

任务：

- 新增 `api/device.py`；
- 定义 `ValidationIssue`；
- virtual Rb87 devices；
- register validation；
- atom model validation；
- pulse limit validation 的 skeleton。

验收：

- 太近的 atoms 被 `validate_register` 报错；
- 不支持的 atom model 被 `validate_atom_model` 报错；
- 现有 `RydbergSystem.from_lattice` 路径不受影响。

### Phase 3: Waveform + Pulse

目标：用可序列化对象替代大多数用户 callable schedule。

任务：

- 新增 `api/waveform.py`；
- 新增 `api/pulse.py`；
- 实现 constant/ramp/blackman/interpolated/custom；
- 实现 sampling 和 duration checks；
- 明确 unit conversion。

验收：

- waveform sampling deterministic；
- pulse amplitude/detuning duration 必须一致；
- Blackman helper 与旧 `pulse.py` 基本一致。

### Phase 4: Sequence + SequenceProtocol

目标：让用户可以用 sequence API 跑 exact simulation。

任务：

- 新增 `api/sequence.py`；
- 新增 `compiler/sequence_compiler.py`；
- 实现 global `1r` sequence；
- 实现 `01r` Rydberg pulse sequence；
- `Sequence.compile_protocol()` 返回 `SequenceProtocol`；
- `Sequence.compile_system()` 返回 `RydbergSystem`。

验收：

```python
reg = Register.chain(2, 4.0)
seq = Sequence(reg, DeviceSpec.virtual_rb87(), AtomModel.preset("01r"))
seq.declare_channel("ryd", "rydberg_global")
seq.add(Pulse.constant(1000, amplitude=1.0, detuning=0.0), "ryd")
system = seq.compile_system()
result = simulate(system, [], system.product_state("11"))
```

后续再把 `simulate(seq, backend="exact")` 做成便利入口。

### Phase 5: NoiseModel integration

目标：把现有 MC 噪声能力产品化。

任务：

- 新增 `api/noise.py`；
- `NoiseModel.to_monte_carlo_runner(...)`；
- 支持 detuning/amplitude/local RIN/position；
- 对 unsupported backend 给 validation warning/error。

验收：

- 新 API 能复现现有 `MonteCarloRunner.setup_*` 的结果；
- `NoiseModel` 可序列化；
- exact backend tests 通过。

### Phase 6: Results and sampling

目标：给用户统一 observables/sample API。

任务：

- 新增 `api/result.py`；
- wrapper `SimulationResult`；
- bitstring sampling；
- populations/correlations helpers；
- register id order 显式用于 bitstring 输出。

验收：

- `sample(n_shots)` 返回 bitstrings；
- bitstring 顺序和 `Register.ids` 一致；
- `population("r")` 与现有 `system.expectation("n_r_i")` 一致。

### Phase 7: Pulser compatibility subset

目标：导入/导出一部分 Pulser-like sequence，而不是一开始追求全兼容。

任务：

- `compat/pulser_import.py`；
- `compat/pulser_export.py`；
- 支持 Register、constant/ramp/interpolated waveform、basic pulse、global rydberg channel；
- 不支持项给明确错误。

验收：

- 简单 Pulser-style JSON 能导入成本 repo `Sequence`；
- 本 repo `Sequence` 能导出为可读 dict；
- capability matrix 文档说明不支持项。

## 需要避免的重构误区

1. 不要把 `Register` 改成包含 `AtomModel` 的对象。几何和能级必须分离。
2. 不要让 `DeviceSpec` 变成 backend config。backend config 应继续在 backend options 中。
3. 不要删除 `RydbergSystem.from_lattice`。它是当前核心桥接点。
4. 不要让 `Sequence` 直接生成 Hamiltonian matrix。它应该生成 `Protocol` 或 IR。
5. 不要第一阶段追求完整真实硬件约束。先做 virtual device 和 validation skeleton。
6. 不要把 arbitrary analog pulse 宣称为 stabilizer-compatible。
7. 不要把旧 notebooks 一次性改完。先新增 examples，再逐步迁移。

## 建议的第一批 tests

```text
tests/api/test_register.py
tests/api/test_atom_model.py
tests/api/test_device.py
tests/api/test_waveform.py
tests/api/test_pulse.py
tests/api/test_sequence_compile.py
tests/api/test_noise_model.py
```

重点测试：

- `Register.ids` 顺序稳定；
- `Register.to_geometry()` 不改变坐标顺序；
- `AtomModel.preset("01r")` roundtrip 到 `LevelStructureSpec`；
- `DeviceSpec.validate_register` 捕获 minimum distance；
- `Waveform.sample` deterministic；
- `Pulse` duration mismatch 报错；
- `SequenceProtocol` 输出 channel names 与 `LevelStructureSpec` 匹配；
- `Sequence.compile_system()` 能跑现有 exact backend；
- `NoiseModel` 能调用现有 MC runner。

## 里程碑定义

### Milestone A: Minimal Product API

用户可以不直接接触 `RydbergSystem.from_lattice`：

```python
reg = Register.square(2, 5.0)
seq = Sequence(reg, DeviceSpec.virtual_rb87(), AtomModel.preset("01r"))
...
result = simulate_sequence(seq, backend="exact")
```

### Milestone B: Backend parity

同一个 `Sequence` 可以跑：

- exact；
- MPS, if capability allows；
- GPUTN, if installed and capability allows。

### Milestone C: Noise parity

`NoiseModel` 能覆盖当前 MC runner 的主要功能，并能复现旧 notebook 中的 local addressing / detuning / amplitude noise 分析。

### Milestone D: Compatibility and docs

提供：

- getting started；
- API reference；
- capability matrix；
- Pulser subset import/export；
- examples for `1r`, `01r`, `rb87_7`。

## 最终目标架构

```text
User-facing layer
    Register
    AtomModel
    DeviceSpec / ChannelSpec
    Waveform / Pulse / Sequence
    NoiseModel
    SimulationResult

Compiler layer
    Register -> LatticeGeometry
    AtomModel -> LevelStructureSpec + physical params
    Sequence -> SequenceProtocol
    NoiseModel -> MonteCarloRunner / future LindbladIR

Existing core
    RydbergSystem
    Protocol
    HamiltonianIR
    EvolutionResult

Backends
    exact
    mps / tenpy
    gputn
    peps
    future stabilizer
```

这个方案的核心是增量扩展：把已有研究内核当作稳定 engine，把产品 API 做成清晰、可验证、可序列化的前端层。
