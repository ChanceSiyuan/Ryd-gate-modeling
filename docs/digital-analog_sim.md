# Digital-Analog IQP Simulation: 可执行实现计划

本文档把原来的物理可行性分析整理成一个可以直接按阶段实现的工程计划。核心目标不是一次性写出完整模拟器，而是把 Route 1、Route 3A、Route 3B 拆成可验证的小模块：

1. 先实现 ideal IQP 和采样指标，得到不含 pulse error 的理论基准。
2. 再实现 weak-blockade physical sequence，验证 finite pulse、VdW tail 和 loss 对 Route 1 的影响。
3. 然后实现 active subset 版本，判断避开 blockade 后的相位是否还足够非平凡。
4. 最后实现 true constrained-IQP，在 independent-set Hilbert space 里验证 diagonal phase dynamics。

当前代码库已经有以下基础：

- `src/ryd_gate/lattice/geometry.py`：square lattice、physical coordinates、all-pairs VdW coupling。
- `src/ryd_gate/lattice/operators.py`：2-level sparse operators 和现有 3-level cascade operators。
- `src/ryd_gate/lattice/evolution.py`：constant Hamiltonian 和 sweep evolution。
- `src/ryd_gate/lattice/states.py`：product states、checkerboard states。
- `src/ryd_gate/lattice/observables.py`：bit/trit masks、Rydberg occupation。
- `src/ryd_gate/analysis/observable_metrics.py`：state overlap 和 norm。
- `scripts/run_3level_lattice.py`、`scripts/plot_population_evolution_sch.py`：现有脚本风格和作图保存方式。

缺少的是 IQP-specific states、diagonal phase、sampling metrics、constrained subspace、hyperfine-Rydberg sequence runner 和最终作图脚本。下面按实现顺序列出。

---

# 0. 成功标准

第一版完成后，应该能回答四个问题：

1. Route 1 是否存在窗口：`V_NN / Omega_pulse << 1` 但 `V_NN * t_int = O(1)` 且 `P_surv` 可接受。
2. Route 1 的 physical distribution 是否接近 ideal IQP distribution。
3. Route 3A 的 active subset 是否只是太稀疏、太接近 uniform/product distribution。
4. Route 3B 的 constrained diagonal phase 是否能在 allowed subspace 中产生非平凡采样分布。

最终把所有结果压缩成四个 feasibility numbers：

- `F_prep`：初态制备或 map-up/readout 是否成功。
- `D_TV(p_phys, p_ideal)`：物理采样分布是否接近目标分布。
- `P_surv`：loss/postselection 是否可承受。
- `Phi_rms`：diagonal phase 是否足够非平凡。

建议探索阈值：

- `F_prep >= 0.9`
- `D_TV <= 0.1--0.2`
- `P_surv >= 0.5`
- `Phi_rms = O(1)`

这些阈值先作为 proof-of-principle 标准，后续如果目标变成 scalable sampling，再收紧。

---

# 1. 新增和扩展的 `src` 模块

模块边界按现有代码语义划分：

- `src/ryd_gate/lattice/`：几何、VdW、basis state、blockade graph、constrained subspace、lattice Hamiltonian。
- `src/ryd_gate/protocols/`：激光 pulse/control schedule，即“怎么打脉冲”。
- `src/ryd_gate/analysis/`：采样指标、Route benchmark、ideal-vs-physical comparison。
- `scripts/`：参数扫描和作图，不复制核心物理公式。

因此 IQP 的数学对象可以在 `lattice` 下，但 Route 级 benchmark 不应放在 `lattice` 里的 protocol 命名模块中；真正的 pulse sequence 应放到 `protocols`，综合指标函数应放到 `analysis`。

## 1.1 扩展 `src/ryd_gate/lattice/geometry.py`

目的：把几何和 VdW 信息标准化，供三条 route 共同使用。

新增函数：

```python
def distance_matrix(coords_um: np.ndarray) -> np.ndarray:
    """Return pairwise Euclidean distances in micrometers."""

def vdw_matrix(
    coords_um: np.ndarray,
    C6: float,
    max_range_um: float | None = None,
) -> np.ndarray:
    """Return dense V_ij = C6 / R_ij^6 matrix with zero diagonal."""

def blockade_edges(coords_um: np.ndarray, Rb_um: float) -> list[tuple[int, int]]:
    """Return graph edges for pairs inside blockade radius."""

def active_subset_chain(N: int, stride: int, offset: int = 0) -> np.ndarray:
    """Return every stride-th site for 1D active subset studies."""

def active_subset_checkerboard(sublattice: np.ndarray, which: int = 1) -> np.ndarray:
    """Return checkerboard active subset indices."""

def greedy_independent_subset(coords_um: np.ndarray, Rb_um: float) -> np.ndarray:
    """Return a deterministic maximal subset with all pair distances > Rb_um."""
```

测试：

- `distance_matrix` 对称、对角为 0。
- `vdw_matrix[i, j] = C6 / R^6`。
- 1D spacing `a` 且 `Rb > a` 时，`blockade_edges` 包含 nearest-neighbor edges。
- `active_subset_chain(N=6, stride=2)` 返回 `[0, 2, 4]`。
- `greedy_independent_subset` 中任意两点距离都大于 `Rb_um`。

## 1.2 新增 `src/ryd_gate/lattice/iqp_states.py`

目的：提供 ideal IQP 和 Route 3 初态。

新增函数：

```python
def plus_state(N: int) -> np.ndarray:
    """Return |+>^N in the 2-level computational basis."""

def basis_probabilities(psi: np.ndarray) -> np.ndarray:
    """Return normalized computational-basis probabilities."""

def apply_hadamard_all(psi: np.ndarray, N: int) -> np.ndarray:
    """Apply H^N to a state vector."""

def product_plus_on_subset(
    N: int,
    active_sites: np.ndarray,
    background_bit: int = 0,
) -> np.ndarray:
    """Return product |+> on active sites and fixed background elsewhere."""

def enumerate_independent_sets(N: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Return basis indices satisfying all blockade constraints."""

def uniform_independent_set_state(N: int, allowed_indices: np.ndarray) -> np.ndarray:
    """Return uniform coherent superposition over allowed configurations."""

def project_to_indices(psi: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Project a state onto selected computational basis indices."""
```

测试：

- `plus_state(N)` norm 为 1，每个振幅为 `1/sqrt(2^N)`。
- `apply_hadamard_all(apply_hadamard_all(psi)) == psi`。
- `product_plus_on_subset` 的非零 basis 数为 `2^len(active_sites)`。
- 1D chain 的 independent-set 数量符合 Fibonacci：`N=1,2,3,4` 为 `2,3,5,8`。
- `uniform_independent_set_state` 只在 allowed indices 上有非零振幅。

## 1.3 新增 `src/ryd_gate/lattice/iqp_operators.py`

目的：实现 diagonal IQP phase，不要一开始构造大矩阵。IQP 的核心演化是 basis-wise phase multiplication。

新增函数：

```python
def bit_occupations(N: int) -> np.ndarray:
    """Return shape (2^N, N) occupation table in computational basis."""

def diagonal_energy_from_vdw(
    N: int,
    vdw_matrix: np.ndarray,
    detunings: np.ndarray | None = None,
) -> np.ndarray:
    """Return E_z = sum_ij V_ij n_i n_j + sum_i detuning_i n_i."""

def apply_diagonal_phase(
    psi: np.ndarray,
    energies: np.ndarray,
    t_int: float,
) -> np.ndarray:
    """Return exp(-i E_z t_int) * psi_z."""

def phase_matrix(vdw_matrix: np.ndarray, t_int: float) -> np.ndarray:
    """Return Phi_ij = V_ij t_int."""

def phase_rms(phi_matrix: np.ndarray, active_sites: np.ndarray | None = None) -> float:
    """Return RMS two-body phase over selected sites."""

def ideal_iqp_state(
    N: int,
    vdw_matrix: np.ndarray,
    t_int: float,
    detunings: np.ndarray | None = None,
) -> np.ndarray:
    """Return H^N U_diag H^N |0^N>."""

def ideal_iqp_distribution(
    N: int,
    vdw_matrix: np.ndarray,
    t_int: float,
    detunings: np.ndarray | None = None,
) -> np.ndarray:
    """Return computational-basis probabilities after ideal IQP."""
```

测试：

- `t_int=0` 时 `apply_diagonal_phase` 不改变 state。
- `N=2` 时 `|11>` 相位为 `exp(-i V_12 t)`。
- `vdw_matrix=0` 时 `ideal_iqp_distribution` 为 `|0^N>` 的 delta distribution。
- `phase_rms` 随 `t_int` 线性缩放。

## 1.4 新增 `src/ryd_gate/analysis/sampling_metrics.py`

目的：把文档中所有采样指标做成可复用函数。

新增函数：

```python
def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Return 0.5 * sum(abs(p - q))."""

def classical_fidelity(p: np.ndarray, q: np.ndarray) -> float:
    """Return (sum sqrt(p_i q_i))^2."""

def renyi2_entropy(p: np.ndarray) -> float:
    """Return -log(sum p_i^2)."""

def participation_ratio(p: np.ndarray) -> float:
    """Return 1 / sum p_i^2."""

def anti_concentration_fraction(p: np.ndarray, alpha: float = 1.0) -> float:
    """Return fraction of outcomes with p_z > alpha / dim."""

def uniform_distribution(dim: int) -> np.ndarray:
    """Return uniform probability vector."""

def uniform_tv_distance(p: np.ndarray) -> float:
    """Return TV distance to uniform distribution."""

def collision_probability(p: np.ndarray) -> float:
    """Return sum p_i^2."""
```

测试：

- `D_TV(p, p) = 0`。
- `D_TV([1,0], [0,1]) = 1`。
- `classical_fidelity(p, p) = 1`。
- uniform distribution 的 `H2 = log(dim)`，`PR = dim`。
- delta distribution 的 `PR = 1`。

## 1.5 新增 `src/ryd_gate/lattice/constrained.py`

目的：Route 3B 的 allowed subspace、leakage 和 constrained diagonal phase。

新增函数：

```python
def allowed_config_indices(N: int, blockade_edges: list[tuple[int, int]]) -> np.ndarray:
    """Return basis indices with no simultaneous excitations on blockade edges."""

def forbidden_config_indices(N: int, blockade_edges: list[tuple[int, int]]) -> np.ndarray:
    """Return basis indices violating at least one blockade edge."""

def constrained_uniform_state(N: int, allowed_indices: np.ndarray) -> np.ndarray:
    """Return |+_I>."""

def constrained_energy_spectrum(
    N: int,
    allowed_indices: np.ndarray,
    vdw_matrix: np.ndarray,
) -> np.ndarray:
    """Return E_C for allowed configurations."""

def constrained_iqp_state(
    N: int,
    allowed_indices: np.ndarray,
    vdw_matrix: np.ndarray,
    t_int: float,
) -> np.ndarray:
    """Return U_diag |+_I> in the full 2^N vector space."""

def leakage_probability(psi: np.ndarray, allowed_indices: np.ndarray) -> float:
    """Return probability outside allowed subspace."""

def amplitude_uniformity_error(psi: np.ndarray, allowed_indices: np.ndarray) -> float:
    """Return sum_C ||psi_C|^2 - 1/|I|| over allowed configs."""

def phase_spread(psi: np.ndarray, allowed_indices: np.ndarray) -> float:
    """Return phase variance over occupied allowed amplitudes."""
```

测试：

- 1D chain allowed count 符合 Fibonacci。
- `constrained_uniform_state` leakage 为 0。
- forbidden basis vector leakage 为 1。
- nearest-neighbor blockade edges 下，allowed configs 满足 `n_i n_j = 0`。
- `constrained_energy_spectrum` 中 nearest-neighbor interaction 项对 allowed configs 不贡献能量。

## 1.6 新增 `src/ryd_gate/lattice/hyperfine_rydberg.py`

目的：实现文档中的 `|0>, |1>, |r>` lattice Hamiltonian 基础。这个模块只负责 many-body operator、Hamiltonian assembly 和 segment evolution，不负责定义“实验协议”。不要复用现有 cascade 3-level operator 的命名，以免把 `g-e-r` 和 `0-1-r` 混在一起。

新增 dataclass：

```python
@dataclass(frozen=True)
class HyperfineRydbergOps:
    N: int
    dim: int
    H_vdw: csc_matrix
    sum_n1: csc_matrix
    sum_nr: csc_matrix
    drive_hf: csc_matrix
    drive_R: csc_matrix
    n_r_list: list[csc_matrix]
```

新增函数：

```python
def build_hf_rydberg_ops(
    geom: LatticeGeometry,
    ryd_decay: float = 0.0,
    loss_decay: float = 0.0,
) -> HyperfineRydbergOps:
    """Build many-body operators for local levels |0>, |1>, |r>."""

def build_hf_rydberg_hamiltonian(
    ops: HyperfineRydbergOps,
    Omega_hf: float = 0.0,
    Delta_hf: float = 0.0,
    Omega_R: float = 0.0,
    Delta_R: float = 0.0,
    local_delta_R: np.ndarray | None = None,
) -> csc_matrix:
    """Assemble H for one piecewise-constant segment."""

def evolve_hf_segments(
    psi0: np.ndarray,
    ops: HyperfineRydbergOps,
    segments: list[dict],
    store_states: bool = False,
    normalize_each_step: bool = False,
) -> EvolutionResult:
    """Evolve a user-provided list of piecewise-constant pulse segments."""

def integrated_rydberg_time(
    times: np.ndarray,
    states: np.ndarray,
    ops: HyperfineRydbergOps,
) -> float:
    """Return integral dt sum_i <n_i^r(t)>."""
```

测试：

- `N=1`、`V=0` 时 `pi/2` pulse 把 `|1>` 送到 `(|1> - i|r>)/sqrt(2)`，允许相位 convention。
- `pi/2 + -pi/2` echo fidelity 接近 1。
- non-Hermitian decay 打开时，final norm 小于 1。
- `integrated_rydberg_time` 对常量 `|r...r>` state 给出约 `N * T`。

## 1.7 新增 `src/ryd_gate/protocols/digital_analog.py`

目的：定义真正的 laser/control protocol，即 map-up、interaction window、map-down、basis rotation 等 pulse segment。这个模块只描述“怎么打 pulse”，不计算 Route feasibility 指标。

新增 dataclass：

```python
@dataclass(frozen=True)
class PulseSegment:
    duration: float
    Omega_hf: float = 0.0
    Delta_hf: float = 0.0
    Omega_R: float = 0.0
    Delta_R: float = 0.0
    local_delta_R: np.ndarray | None = None
    label: str = ""

@dataclass(frozen=True)
class DigitalAnalogSequence:
    segments: tuple[PulseSegment, ...]
    description: str = ""
```

新增函数：

```python
def rydberg_pi_over_2_segment(
    Omega_R: float,
    sign: int = +1,
    label: str = "rydberg_pi_over_2",
) -> PulseSegment:
    """Return a |1>-|r> pi/2 pulse segment."""

def rydberg_pi_segment(
    Omega_R: float,
    sign: int = +1,
    label: str = "rydberg_pi",
) -> PulseSegment:
    """Return a |1>-|r> pi pulse segment."""

def interaction_segment(t_int: float, label: str = "interaction") -> PulseSegment:
    """Return a drive-off diagonal interaction segment."""

def echo_sequence(Omega_R: float, t_wait: float = 0.0) -> DigitalAnalogSequence:
    """Return pi/2, optional wait, -pi/2 empty-circuit echo."""

def route1_iqp_sequence(Omega_R: float, t_int: float) -> DigitalAnalogSequence:
    """Return Route 1 map-up, interaction, readout sequence."""

def map_up_down_sequence(Omega_R: float) -> DigitalAnalogSequence:
    """Return map-up/map-down calibration sequence."""
```

测试：

- `rydberg_pi_over_2_segment` 的 duration 为 `pi / (2 * Omega_R)`。
- `interaction_segment` 所有 drive amplitudes 为 0。
- `echo_sequence` 至少包含 map-up 和 readout pulse。
- `route1_iqp_sequence` 中 interaction segment duration 等于 `t_int`。

## 1.8 新增 `src/ryd_gate/analysis/iqp_benchmarks.py`

目的：把 route-level benchmark 封装成小函数，脚本只负责扫参和画图。这里可以依赖 `lattice`、`protocols`、`analysis/sampling_metrics.py`，但不定义新的 laser protocol。

新增 dataclass：

```python
@dataclass(frozen=True)
class Route1Params:
    N: int
    spacing_um: float
    C6: float
    Omega_pulse: float
    t_int: float
    ryd_decay: float = 0.0
```

新增函数：

```python
def route1_echo_benchmark(params: Route1Params) -> dict[str, float]:
    """Return F_prep, F_echo, P_surv for empty-circuit echo."""

def route1_phase_loss_tradeoff(
    N: int,
    vnn_values: np.ndarray,
    phi_targets: np.ndarray,
    ryd_decay: float,
) -> dict[str, np.ndarray]:
    """Return t_phi and P_surv grids."""

def route1_distribution_benchmark(params: Route1Params) -> dict[str, object]:
    """Return p_ideal, p_phys, D_TV, F_cl, F_state, P_surv."""

def route3a_subset_summary(
    geom: LatticeGeometry,
    subsets: dict[str, np.ndarray],
    Omega_map: float,
    phi_target: float,
    ryd_decay: float,
) -> dict[str, dict]:
    """Return Vmax, Vrms, t_phi, P_surv, eta for each subset."""

def route3a_complexity_benchmark(
    geom: LatticeGeometry,
    subset: np.ndarray,
    t_int: float,
) -> dict[str, float]:
    """Return H2, PR, uniform TV, anti-concentration for active subset IQP."""

def route3b_constrained_phase_benchmark(
    geom: LatticeGeometry,
    Rb_um: float,
    t_values: np.ndarray,
    ryd_decay: float,
) -> dict[str, np.ndarray]:
    """Return constrained IQP metrics vs t_int."""
```

测试：

- `route1_phase_loss_tradeoff` 中 `t_phi = phi / V_NN`。
- decay 为 0 时 `P_surv = 1`。
- Route 3A `eta = Vmax / Omega_map`。
- Route 3B 输出数组长度等于 `len(t_values)`。

---

# 2. Route 1 实现计划：weak-blockade 标准 IQP

## 2.1 理论目标

Route 1 要验证：

```text
V_NN / Omega_pulse << 1
```

使 preparation/readout 近似 product single-qubit rotation，同时：

```text
V_NN * t_int = O(1)
```

使 diagonal IQP phase 足够大。

理想目标分布：

```text
p_ideal(z) = |<z| H^N U_diag H^N |0^N>|^2
U_diag = exp[-i sum_{i<j} V_ij t_int n_i n_j]
```

物理目标分布来自完整 sequence：

```text
|1>^N
-> pi/2 pulse in |1>-|r>
-> interaction window
-> -pi/2 readout pulse
-> measurement
```

## 2.2 先实现 ideal baseline

实现位置：

- `src/ryd_gate/lattice/iqp_states.py`
- `src/ryd_gate/lattice/iqp_operators.py`
- `src/ryd_gate/analysis/sampling_metrics.py`

验收：

- `pytest tests/test_iqp_states.py tests/test_iqp_operators.py tests/test_sampling_metrics.py`
- `N=2` 的相位和手算一致。
- `V=0` 的 IQP output 回到 `|0^N>`。

## 2.3 再实现 empty-circuit echo

实现位置：

- `src/ryd_gate/lattice/hyperfine_rydberg.py`
- `src/ryd_gate/protocols/digital_analog.py`
- `src/ryd_gate/analysis/iqp_benchmarks.py`

输出：

- `F_prep`
- `F_echo`
- `P_surv`
- `T_r = int dt sum_i <n_i^r(t)>`

扫描：

- `N = 2, 4, 6, 8`
- `V_NN / Omega_pulse = 0.01, 0.03, 0.1, 0.3`
- `ryd_decay = 0` 和实验估计值

验收：

- `V_NN / Omega_pulse -> 0` 时 `F_echo -> 1`。
- 打开 decay 后 `P_surv` 随 `N` 和 pulse duration 单调下降。

## 2.4 再实现 ideal-vs-physical distribution

输出：

- `D_TV(p_phys, p_ideal)`
- `F_cl(p_phys, p_ideal)`
- `F_state`，只在 pure-state normalized comparison 下使用。
- `P_surv`

验收：

- `V_NN/Omega_pulse` 越小，finite-pulse error 越低。
- `t_int=0` 时 physical output 接近 empty echo。
- `ryd_decay=0` 且 fast-pulse 极限下 `D_TV` 接近 0。

## 2.5 两原子 phase calibration

实现位置：

- 先放在 `src/ryd_gate/analysis/iqp_benchmarks.py` 中作为 `two_atom_phase_calibration(...)`。
- 后续如需要 fitting 细化，再拆到 `src/ryd_gate/analysis/phase_fitting.py`。

输出：

- `phi_fitted(t)`
- `V_12 * t`
- `delta_phi(t) = phi_fitted - V_12 * t`

验收：

- `Omega_pulse` 足够大且 decay 关闭时，`delta_phi` 接近 0。
- 当 `V_12/Omega_pulse` 增大，残差变大。

---

# 3. Route 3A 实现计划：active subset product plus

## 3.1 理论目标

Route 3A 不尝试制备完整 independent-set superposition，而是选择 active subset `S`，要求：

```text
R_ij > R_b, for all i,j in S
```

然后只在 `S` 上制备 product plus。它是稀疏子图上的标准 IQP，实验最容易，但风险是 active qubit 太少、相位太弱、输出分布太平凡。

## 3.2 子集选择

实现位置：

- `src/ryd_gate/lattice/geometry.py`
- `src/ryd_gate/analysis/iqp_benchmarks.py`

候选 subset：

- 1D chain：stride 2、stride 3。
- square lattice：checkerboard、稀疏 checkerboard。
- arbitrary coordinates：`greedy_independent_subset(coords_um, Rb_um)`。

输出：

- `N_active`
- `min_distance`
- `Vmax`
- `Vmed`
- `Vrms`
- `eta = Vmax / Omega_map`
- `t_pi4 = (pi/4) / Vmax`
- `P_surv = exp[-Gamma_r * N_active * t_pi4 / 2]`

验收：

- 所有 active pair 的距离都大于 `Rb_um`。
- stride 越大，`Vmax` 越小，`t_pi4` 越大。

## 3.3 active subset IQP complexity

实现位置：

- `src/ryd_gate/lattice/iqp_states.py`
- `src/ryd_gate/lattice/iqp_operators.py`
- `src/ryd_gate/analysis/sampling_metrics.py`

输出：

- `H2(p)`
- `PR(p)`
- `D_TV(p, uniform)`
- `anti_concentration_fraction`
- probability histogram

验收：

- 如果 `V_ij t_int` 全部很小，输出应接近 trivial distribution。
- 当 `Phi_rms = O(1)`，`H2` 和 `PR` 应显示更强 spreading。

## 3.4 map-up/map-down fidelity

第一版可以先用 `src/ryd_gate/protocols/digital_analog.py` 里的 map-up/map-down sequence，并通过 `src/ryd_gate/lattice/hyperfine_rydberg.py` 演化；inactive sites 固定在 background state。

输出：

- `F_map(S)`
- `P_loss(S)`
- `D_TV^{t=0}(S)`

验收：

- `eta << 1` 时 map fidelity 高。
- active subset 比 full lattice 的 echo error 低。

---

# 4. Route 3B 实现计划：true constrained-IQP

## 4.1 理论目标

Route 3B 在 allowed independent-set subspace 中工作：

```text
|+_I> = 1/sqrt(|I|) sum_{C in I} |C>
```

其中 `I` 是没有 blockade-edge 双激发的 configurations。diagonal evolution 为：

```text
U_diag^I = exp[-i t sum_{i<j} V_ij n_i n_j]
```

nearest-neighbor blockade edges 上 `n_i n_j = 0`，所以非平凡相位主要来自 blockade radius 外的 VdW tails。

## 4.2 先实现 constrained exact math

实现位置：

- `src/ryd_gate/lattice/constrained.py`
- `src/ryd_gate/lattice/iqp_operators.py`

输出：

- allowed count `|I|`
- energy spectrum `E_C`
- `Var(E_C)`
- `Phi_rms`
- `H2(p_X(t))`
- `D_TV(p_X(t), p_X(0))`

验收：

- `t=0` 时分布不变。
- `Var(E_C)=0` 时 diagonal phase 不产生非平凡 spreading。
- allowed subspace leakage 保持 0。

## 4.3 constrained-state preparation

第一版不要马上做 GRAPE。建议按复杂度分三步：

1. 直接使用 ideal `|+_I>`，验证 constrained diagonal phase 的物理价值。
2. 用现有 2-level lattice sweep 试探是否能接近 `|+_I>`。
3. 如果 2 失败，再引入 pulse optimization。

需要指标：

- `F_I = |<+_I|psi>|^2`
- `P_I = probability inside allowed subspace`
- `D_amp = sum_C ||psi_C|^2 - 1/|I||`
- `phase_spread = Var(arg psi_C)`

验收：

- 只检查 probability uniform 不够；必须同时检查 phase spread。
- 如果 `F_I` 很低，但 `P_I` 很高，说明制备在 allowed subspace 中但相干结构不对。

## 4.4 final measurement 可行性

第一版先比较 ideal constrained output。第二版再加 map-down：

```text
ideal constrained state
-> diagonal phase
-> map down
-> hyperfine basis rotation
-> measurement
```

输出：

- `F_mapdown^I`
- `P_loss^I`
- `D_TV(p_measured, p_ideal^I)`

验收：

- map-down pulse error 不能主导 `D_TV`。
- 如果 map-down error 过大，说明需要更快 pulse 或 mean-field shift compensation。

---

# 5. 测试矩阵

新增测试文件建议如下。

## `tests/test_iqp_states.py`

覆盖：

- `plus_state`
- `apply_hadamard_all`
- `product_plus_on_subset`
- `enumerate_independent_sets`
- `uniform_independent_set_state`

运行：

```bash
pytest tests/test_iqp_states.py
```

## `tests/test_iqp_operators.py`

覆盖：

- basis occupation table。
- diagonal energy 手算。
- two-qubit phase。
- ideal IQP 在 zero interaction 下的极限。

运行：

```bash
pytest tests/test_iqp_operators.py
```

## `tests/test_sampling_metrics.py`

覆盖：

- TV distance。
- classical fidelity。
- Renyi-2 entropy。
- participation ratio。
- anti-concentration。

运行：

```bash
pytest tests/test_sampling_metrics.py
```

## `tests/test_constrained_iqp.py`

覆盖：

- blockade edges。
- allowed config enumeration。
- leakage probability。
- constrained energy spectrum。
- constrained IQP state。

运行：

```bash
pytest tests/test_constrained_iqp.py
```

## `tests/test_hyperfine_rydberg.py`

覆盖：

- one-atom Hamiltonian/operator convention。
- segment evolution with externally supplied pulse segments。
- non-Hermitian norm decay。
- integrated Rydberg time。

运行：

```bash
pytest tests/test_hyperfine_rydberg.py
```

## `tests/test_digital_analog_protocols.py`

覆盖 pulse/control protocol 小例子：

- pi/2 和 pi pulse duration。
- interaction segment drive 为 0。
- echo sequence segment order。
- Route 1 IQP sequence 包含 map-up、interaction、readout。

运行：

```bash
pytest tests/test_digital_analog_protocols.py
```

## `tests/test_iqp_benchmarks.py`

覆盖 route-level 小例子：

- Route 1 phase-loss formula。
- Route 1 `t_int=0` echo。
- Route 3A subset summary。
- Route 3B constrained benchmark output shape。

运行：

```bash
pytest tests/test_iqp_benchmarks.py
```

完整验证顺序：

```bash
pytest tests/test_iqp_states.py tests/test_iqp_operators.py tests/test_sampling_metrics.py
pytest tests/test_constrained_iqp.py
pytest tests/test_hyperfine_rydberg.py tests/test_digital_analog_protocols.py tests/test_iqp_benchmarks.py
pytest
```

---

# 6. 新增 `scripts` 作图计划

所有脚本默认保存到 `docs/figures`，沿用现有脚本风格。核心计算放在 `src`，脚本只做参数扫描和画图。

## 6.1 共享脚本工具

新增 `scripts/iqp_plot_common.py`：

```python
FIGDIR = "docs/figures"

def ensure_figdir() -> str:
    ...

def default_c6() -> float:
    ...

def default_ryd_decay() -> float:
    ...

def savefig(fig, name: str) -> None:
    ...
```

不要在这里放物理核心逻辑，只放 plotting utilities 和默认参数。

## 6.2 `scripts/plot_iqp_route1_pulse_error.py`

图 1：Route 1 pulse-error phase diagram。

输入扫描：

- `N = 2, 4, 6, 8`
- `V_NN / Omega_pulse = logspace(-2, -0.3)`

输出：

- `docs/figures/iqp_route1_pulse_error.png`

图内容：

- 横轴：`V_NN / Omega_pulse`
- 纵轴：`N`
- 颜色：`1 - F_echo`

## 6.3 `scripts/plot_iqp_route1_phase_loss.py`

图 2：phase-vs-loss tradeoff。

输入扫描：

- `phi_target = pi/8, pi/4, pi/2`
- `V_NN` 或 spacing `a`
- `N`

输出：

- `docs/figures/iqp_route1_phase_loss.png`

图内容：

- `t_phi = phi_target / V_NN`
- `P_surv = exp[-Gamma_r * N * t_phi / 2]`

## 6.4 `scripts/plot_iqp_route1_distribution_distance.py`

图 3：ideal-vs-physical sampling distance。

输入扫描：

- `N = 2, 4, 6, 8`
- `V_NN / Omega_pulse = 0.01, 0.03, 0.1, 0.3`

输出：

- `docs/figures/iqp_route1_distribution_distance.png`

图内容：

- `D_TV(p_phys, p_ideal)` vs `N`
- 可附加 `F_cl` 或 `P_surv` panel。

## 6.5 `scripts/plot_iqp_two_atom_phase.py`

图 4：two-atom Ramsey phase calibration。

输入扫描：

- `t_int`
- `V_12 / Omega_pulse`
- decay on/off

输出：

- `docs/figures/iqp_two_atom_phase.png`

图内容：

- `phi_fitted(t)`
- theory line `V_12 * t`
- residual `delta_phi(t)`

## 6.6 `scripts/plot_iqp_route3a_subset_tradeoff.py`

图 5：active subset tradeoff。

输入扫描：

- chain stride。
- checkerboard subset。
- greedy independent subset。

输出：

- `docs/figures/iqp_route3a_subset_tradeoff.png`

图内容：

- `N_active`
- `Vmax`
- `t_pi4`
- `P_surv`
- `eta = Vmax/Omega_map`

## 6.7 `scripts/plot_iqp_route3a_complexity.py`

图 6：active subset IQP complexity proxy。

输出：

- `docs/figures/iqp_route3a_complexity.png`

图内容：

- `H2`
- `PR`
- `D_TV(p, uniform)`
- probability histogram

## 6.8 `scripts/plot_iqp_route3b_prep.py`

图 7：constrained-state preparation benchmark。

第一版可以只画 ideal target 和 sweep trial，不做 optimization。

输出：

- `docs/figures/iqp_route3b_prep.png`

图内容：

- `F_I`
- `P_I`
- `D_amp`
- `phase_spread`

## 6.9 `scripts/plot_iqp_route3b_full_sequence.py`

图 8：constrained-IQP full sequence benchmark。

输出：

- `docs/figures/iqp_route3b_full_sequence.png`

图内容：

- 横轴：`t_int`
- 纵轴/panels：`D_TV(p_phys, p_ideal^I)`、`P_loss`、`H2(p)`

---

# 7. 推荐实现里程碑

## Milestone A：ideal IQP core

改动：

- 新增 `iqp_states.py`
- 新增 `iqp_operators.py`
- 新增 `sampling_metrics.py`

测试：

```bash
pytest tests/test_iqp_states.py tests/test_iqp_operators.py tests/test_sampling_metrics.py
```

验收：

- 可以生成 `p_ideal`。
- 可以计算 `D_TV`、`F_cl`、`H2`、`PR`。

## Milestone B：Route 1 physical small exact

改动：

- 新增 `hyperfine_rydberg.py`
- 新增 pulse sequence builders 到 `protocols/digital_analog.py`
- 新增 Route 1 benchmark functions 到 `analysis/iqp_benchmarks.py`

测试：

```bash
pytest tests/test_hyperfine_rydberg.py tests/test_digital_analog_protocols.py tests/test_iqp_benchmarks.py
```

验收：

- empty echo 在 weak-interaction 极限高保真。
- `P_surv` 正确反映 non-Hermitian loss。
- 可以比较 `p_phys` 和 `p_ideal`。

## Milestone C：Route 1 figures

改动：

- 新增 `plot_iqp_route1_pulse_error.py`
- 新增 `plot_iqp_route1_phase_loss.py`
- 新增 `plot_iqp_route1_distribution_distance.py`
- 新增 `plot_iqp_two_atom_phase.py`

验收：

- 生成图 1 到图 4。
- 每个脚本可单独运行。
- 图保存到 `docs/figures`。

## Milestone D：Route 3A active subset

改动：

- 扩展 geometry subset helpers。
- 新增 Route 3A benchmark functions 到 `analysis/iqp_benchmarks.py`。
- 新增两个 Route 3A scripts。

测试：

```bash
pytest tests/test_iqp_benchmarks.py
```

验收：

- 能列出不同 subset 的 `N_active, Vmax, t_pi4, P_surv`。
- 能判断 subset IQP 是否过于接近 uniform/product distribution。

## Milestone E：Route 3B constrained exact

改动：

- 新增 `constrained.py`
- 新增 Route 3B benchmark 到 `analysis/iqp_benchmarks.py`
- 新增 Route 3B scripts。

测试：

```bash
pytest tests/test_constrained_iqp.py tests/test_iqp_benchmarks.py
```

验收：

- 能构造 `|+_I>`。
- 能计算 constrained diagonal phase。
- 能给出 `Var(E_C)`、`H2`、`D_TV`、`P_surv`。

## Milestone F：整理和整体验证

运行：

```bash
pytest
python scripts/plot_iqp_route1_pulse_error.py
python scripts/plot_iqp_route1_phase_loss.py
python scripts/plot_iqp_route1_distribution_distance.py
python scripts/plot_iqp_two_atom_phase.py
python scripts/plot_iqp_route3a_subset_tradeoff.py
python scripts/plot_iqp_route3a_complexity.py
python scripts/plot_iqp_route3b_prep.py
python scripts/plot_iqp_route3b_full_sequence.py
```

验收：

- 所有核心测试通过。
- 8 张核心图生成。
- 每张图能对应一个 feasibility question。

---

# 8. 实现时的注意事项

1. 第一版只支持小系统 exact simulation，不要一开始优化到 `N=40`。
2. Diagonal IQP phase 用 vectorized basis energies 实现，不要先构造巨大 dense unitary。
3. `|0>, |1>, |r>` 的 hyperfine-Rydberg 模型和现有 `g-e-r` cascade 3-level 模型分开命名。
4. non-Hermitian loss evolution 不能每步归一化；只有需要 conditional state 时才显式 normalize。
5. `scripts` 里不要复制核心公式；核心公式放进 `src`，脚本只调函数。
6. Route 3B 的 preparation 不要先做最难优化问题；先用 ideal `|+_I>` 判断 constrained diagonal phase 是否值得继续。
7. 所有随机或 greedy subset 选择都要 deterministic，方便图和测试复现。

---

# 9. 原始理论分析和物理动机

以下保留原始理论分析，作为实现计划背后的物理动机和指标来源。

可以把你的数值验证分成两个目标：

[
\textbf{Route 1: weak-blockade / large-spacing 标准 IQP}
]

和

[
\textbf{Route 3: constrained-IQP / blockade-compatible IQP-like sampler}.
]

这两条路线要验证的“可行性”其实不同。Route 1 要证明：**在 preparation / readout pulse 期间 (V_{ij}) 足够小，系统近似完整 (2^n) qubit IQP。** Route 3 要证明：**虽然最近邻 blockade 存在，但我们只在 allowed subspace 里工作，仍然能得到可控、可测、非平凡的 diagonal phase dynamics。**

文章中的基本 Rydberg Hamiltonian 是

[
H/\hbar=
\frac{\Omega(t)}{2}\sum_i\sigma_i^x
-\sum_i n_i(\Delta(t)-\delta_i(t))
+\sum_{i<j}V_{ij}n_in_j,
]

其中 (V_{ij}=V_0/|x_i-x_j|^6)，而 Geim 等人的实验参数 (R_b/a\simeq1.3-1.4) 会禁止最近邻 Rydberg 双激发；这正是我们需要避开或利用的关键约束。

---

# 0. 先统一你要模拟的物理模型

你的 simulator 已经能做 (0-1-r) Rabi，那么建议把一次实验序列写成：

[
|0/1\rangle_{\rm hf}
\overset{\text{prepare}}{\longrightarrow}
\text{initial state}
\overset{\text{map up or direct } \pi/2}{\longrightarrow}
\text{Rydberg qubit}
\overset{H_{\rm diag}}{\longrightarrow}
\text{IQP phase}
\overset{\text{map down / basis rotation}}{\longrightarrow}
\text{measurement}.
]

完整 Hamiltonian 至少包括：

[
H(t)/\hbar =
\frac{\Omega_R(t)}{2}\sum_i
\left(|r_i\rangle\langle 1_i|+\mathrm{h.c.}\right)
-\Delta_R(t)\sum_i n_i^r
+
\frac{\Omega_{\rm hf}(t)}{2}\sum_i
\left(|1_i\rangle\langle0_i|+\mathrm{h.c.}\right)
-\Delta_{\rm hf}(t)\sum_i n_i^1
+
\sum_{i<j}V_{ij}n_i^r n_j^r .
]

再加一个最小损耗模型：

[
H_{\rm nh}
==========

## H

## i\frac{\Gamma_r}{2}\sum_i n_i^r

i\frac{\Gamma_{\rm loss}}{2}\sum_i n_i^r
]

或者用 quantum trajectories。第一版可以先用 non-Hermitian survival probability：

[
P_{\rm surv}(t)=|\psi(t)|^2.
]

Geim 文章里 atom loss 被认为是 analog Rydberg simulation 的主要误差源之一，包括 Rydberg decay、Rydberg evolution 期间关闭 tweezers 导致的 loss、two-photon excitation 的 intermediate-state scattering 等；他们靠 loss-resolved readout 直接检测 missing atoms。

---

# 1. Route 1：weak-blockade 标准 IQP，应该模拟什么？

目标是实现近似标准 IQP：

[
|\psi_{\rm ideal}\rangle
========================

U_{\rm IQP}|+\rangle_R^{\otimes n},
]

其中

[
U_{\rm IQP}
===========

\exp\left[
-i\sum_{i<j}\phi_{ij} n_i n_j
-i\sum_i\theta_i n_i
\right],
\qquad
\phi_{ij}=V_{ij}t_{\rm int}.
]

用 (n_i=(1+Z_i)/2) 或 ((1-Z_i)/2) 只差 convention；本质是 diagonal (Z_iZ_j) phase circuit。

## 1.1 首先模拟 “空线路回波”：没有 interaction time，只看 prep/readout

这是最重要的 baseline。

序列：

[
|1\rangle^{\otimes n}
\xrightarrow{R_y(\pi/2)}
|+\rangle_R^{\otimes n}
\xrightarrow{t=0}
|+\rangle_R^{\otimes n}
\xrightarrow{R_y(-\pi/2)}
|1\rangle^{\otimes n}.
]

或者用 Geim 类似的 hyperfine map-up / map-down：

[
|+\rangle_{\rm hf}^{\otimes n}
\rightarrow
|+\rangle_R^{\otimes n}
\rightarrow
|+\rangle_{\rm hf}^{\otimes n}.
]

要输出的数值结果：

[
F_{\rm prep}
============

|\langle +^{\otimes n}|\psi_{\rm after\ prep}\rangle|^2,
]

[
F_{\rm echo}
============

|\langle 1^{\otimes n}|\psi_{\rm after\ prep+readout}\rangle|^2,
]

[
P_{\rm surv}.
]

扫描参数：

[
\frac{V_{\max}}{\Omega_{\rm prep}},
\qquad
\frac{V_{\max}}{\Omega_{\rm read}},
\qquad
n,
\qquad
a,
\qquad
n_{\rm Rydberg}.
]

其中

[
V_{\max}=\max_{i<j}V_{ij}
]

通常是最近邻。

你希望看到一个相图：

[
F_{\rm echo}
\text{ vs }
(V_{\max}/\Omega_{\rm pulse},\ n).
]

如果

[
V_{\max}/\Omega_{\rm pulse}\gtrsim 0.1
]

通常就会开始明显不是独立单比特 rotation；如果要做到很多 qubit 同时高保真，实际可能需要更小。这里不用先相信经验值，让 simulator 给出阈值。

**判断标准：** Route 1 只有在

[
F_{\rm echo}\approx 1
]

且

[
P_{\rm surv}\approx 1
]

时才有资格继续看 IQP phase。否则连 (H^{\otimes n}) 都没有实现。

---

## 1.2 再模拟 “相互作用窗口”：IQP 相位够不够大、损耗是否可接受

在 prep/readout 之间关掉 drive，只让

[
H_{\rm diag}=\sum_{i<j}V_{ij}n_i n_j+\sum_i\Delta_i n_i
]

演化。

输出：

[
\phi_{ij}=V_{ij}t_{\rm int}.
]

至少要画：

1. 最近邻相位
   [
   \phi_{\rm NN}=V_{\rm NN}t_{\rm int}.
   ]

2. 次近邻相位
   [
   \phi_{\rm NNN}=V_{\rm NNN}t_{\rm int}.
   ]

3. 总 Rydberg 占据时间
   [
   \mathcal T_r
   ============

   \int dt\ \sum_i \langle n_i^r(t)\rangle .
   ]

4. survival
   [
   P_{\rm surv}\approx \exp[-\Gamma_r \mathcal T_r].
   ]

对标准 (|+\rangle_R^{\otimes n})，中间演化时平均 Rydberg 数大约是

[
\langle N_r\rangle \approx n/2.
]

所以损耗会随 (n) 很快变坏：

[
P_{\rm surv}
\sim
\exp[-\Gamma_r n t_{\rm int}/2].
]

要画一个 tradeoff plot：

[
x=a \quad \text{或} \quad V_{\rm NN},
]

[
y=t_{\rm int},
]

颜色：

[
\phi_{\rm NN},\quad
P_{\rm surv},\quad
F_{\rm state}.
]

Route 1 的核心矛盾是：

[
V_{\rm NN}\ll \Omega_{\rm pulse}
]

才能做 product (|+\rangle)，但

[
V_{\rm NN}t_{\rm int}\sim O(1)
]

才有非平凡 IQP 相位。于是 (V_{\rm NN}) 太小会导致 (t_{\rm int}) 太长，Rydberg decay 变严重。

建议固定目标角度，例如：

[
\phi_{\rm NN}=\pi/8,\quad \pi/4,\quad \pi/2
]

分别计算所需时间：

[
t_{\phi}=\frac{\phi}{V_{\rm NN}}.
]

然后画：

[
P_{\rm surv}(\phi)
==================

\exp[-\Gamma_r n\phi/(2V_{\rm NN})].
]

这张图会非常直接告诉你：**为了避免 blockade 拉大 spacing 后，IQP phase 是否还来得及积累。**

---

## 1.3 直接比较 ideal IQP distribution 和 physical distribution

对小系统 (n=4,6,8,10,12)，做完整 exact simulation。

定义 ideal output：

[
p_{\rm ideal}(z)
================

|\langle z|
H^{\otimes n}
U_{\rm IQP}
H^{\otimes n}
|0^n\rangle|^2.
]

你的物理模拟输出：

[
p_{\rm phys}(z)
]

来自完整 (0-1-r) pulse sequence，包括 finite pulses、(V_{ij})、detuning、loss/noise。

计算：

[
D_{\rm TV}
==========

\frac12\sum_z |p_{\rm phys}(z)-p_{\rm ideal}(z)|,
]

[
F_{\rm cl}
==========

\left(\sum_z\sqrt{p_{\rm phys}(z)p_{\rm ideal}(z)}\right)^2,
]

[
F_{\rm state}
=============

|\langle \psi_{\rm ideal}|\psi_{\rm phys}\rangle|^2
]

若你保留 pure-state simulation。

还建议画：

[
D_{\rm TV}
\text{ vs }
n
]

在不同

[
V_{\rm NN}/\Omega_{\rm pulse}
]

下的曲线。

这比只看 state fidelity 更接近实验目标，因为 IQP 最终关心的是采样分布。

---

## 1.4 做 pair-level calibration：两原子 Ramsey phase

在实验上最容易先验证的是两体相位。

模拟：

[
|++\rangle
\rightarrow
e^{-iV_{12}t n_1n_2}|++\rangle
\rightarrow
\text{measure } XX,XY,YX,YY.
]

提取相位 (\phi_{12})，看它是否等于

[
\phi_{12}=V_{12}t.
]

输出：

[
\delta\phi_{12}
===============

\phi_{\rm fitted}-V_{12}t.
]

扫描：

[
V_{12}/\Omega_{\rm pulse},\quad t,\quad \Gamma_r,\quad \Delta.
]

这张图是 Route 1 的最小实验 feasibility benchmark：如果两体 phase gate 都不准，多体 IQP 不需要继续。

---

# 2. Route 3：constrained-IQP，先分清两个版本

你写的第三点有两种可实现含义。

## 版本 3A：active subset product plus

选择一个 active subset (S)，其中任意两个 active sites 都不在 blockade radius 内：

[
(i,j)\in S,\quad i\neq j
\quad\Rightarrow\quad
R_{ij}>R_b.
]

然后只在这些 site 上制备

[
\prod_{i\in S}\frac{|1_i\rangle+|r_i\rangle}{\sqrt2},
]

其余 site 保持 (|1\rangle)。

这不是严格意义上的 “independent-set Hilbert-space uniform superposition”，而是 **稀疏子图上的标准 IQP**。它实验上最容易。

## 版本 3B：true constrained-IQP

目标态是

[
|+_{\mathcal I}\rangle
======================

\frac{1}{\sqrt{|\mathcal I|}}
\sum_{C\in\mathcal I}|C\rangle,
]

其中 (\mathcal I) 是所有没有最近邻 Rydberg 双激发的 independent sets。

然后做

[
U_{\rm diag}
============

\exp\left[-it\sum_{i<j}V_{ij}n_in_j\right]
]

但这个 (U_{\rm diag}) 只作用在 constrained Hilbert space 中。

这个更有物理意义，也更接近 Rydberg blockade 的自然平台；但制备 (|+_{\mathcal I}\rangle) 本身比较难。Geim 文章使用的 Floquet 工程也是建立在 nearest-neighbor blockade 的 PXP constrained dynamics 上，PXP projector

[
P=\prod_{\langle i,j\rangle}(1-n_in_j)
]

会移除最近邻同时 Rydberg 激发的构型。

---

# 3. Route 3A：active subset product plus，应该模拟什么？

这个路线的验证目标是：**找一个 active subset，使 map-up 不受 blockade 破坏，同时剩余 vdW tail 还能提供足够非平凡的 IQP 相位。**

## 3.1 子集选择与相互作用矩阵

给定几何，枚举几个候选 active subset：

* 1D chain：隔一个取一个、隔两个取一个；
* square lattice：checkerboard 子格、稀疏 checkerboard；
* kagome / honeycomb：选距离大于 (R_b) 的最大 independent subset。

对每个 subset (S)，输出：

[
N_{\rm active}=|S|,
]

[
V_{ij}^{(S)}=\frac{C_6}{R_{ij}^6},
]

[
V_{\max}^{(S)},\quad V_{\rm med}^{(S)},\quad V_{\rm rms}^{(S)}.
]

画热图：

[
\Phi_{ij}=V_{ij}^{(S)}t.
]

这一步会告诉你这个 subset 上的 IQP 是否只是很弱、很稀疏的相位图。

---

## 3.2 map-up / map-down fidelity

因为 active sites 互相不 blockade，理想上 map 应该好很多。模拟：

[
|+\rangle_{\rm hf}^{\otimes S}
\rightarrow
|+\rangle_R^{\otimes S}
\rightarrow
|+\rangle_{\rm hf}^{\otimes S}.
]

输出：

[
F_{\rm map}(S),
\quad
P_{\rm loss}(S),
\quad
D_{\rm TV}^{t=0}(S).
]

扫描：

[
\min_{i,j\in S}R_{ij},
\quad
\Omega_{\rm map},
\quad
n_{\rm Rydberg},
\quad
a.
]

这里最重要的无量纲数是：

[
\eta_S=\frac{V_{\max}^{(S)}}{\Omega_{\rm map}}.
]

若

[
\eta_S\ll1
]

则 map 近似局域；若 (\eta_S) 还是不小，说明 subset 仍不够稀疏或 pulse 不够快。

---

## 3.3 非平凡 phase 与 survival 的 tradeoff

由于 active subset 拉大了最小距离，

[
V_{\max}^{(S)}
]

会显著减小。你需要模拟：

[
t_{\pi/4}^{(S)}
===============

\frac{\pi/4}{V_{\max}^{(S)}}.
]

然后计算：

[
P_{\rm surv}^{(S)}
\approx
\exp[-\Gamma_r |S|t_{\pi/4}^{(S)}/2].
]

画：

[
|S|
\text{ vs }
t_{\pi/4}^{(S)}
\text{ vs }
P_{\rm surv}^{(S)}.
]

这张图会告诉你 active subset 方案的核心代价：**越稀疏，map 越好；但相互作用越弱，IQP phase 越慢。**

---

## 3.4 IQP 输出分布是否有足够复杂性

即使实验上可行，也可能太稀疏，输出分布接近 product distribution。建议计算：

1. 二阶 Renyi entropy：

[
H_2(p)
======

-\log\sum_z p(z)^2.
]

2. participation ratio：

[
\mathrm{PR}
===========

\frac{1}{\sum_z p(z)^2}.
]

3. anti-concentration 指标：

[
\Pr_z[p(z)>\alpha/2^{|S|}]
]

或者简单画 output probability histogram，和 Porter-Thomas / uniform 对比。

4. 输出分布与 uniform 的距离：

[
D_{\rm TV}(p_{\rm ideal},p_{\rm unif}).
]

如果 (p_{\rm ideal}\approx p_{\rm unif}) 或几乎 factorized，那么即使实验可行，也不是很有意思的 IQP sampler。

---

# 4. Route 3B：true constrained-IQP，应该模拟什么？

这个路线更接近 Rydberg blockade 的优势。

目标是：

[
|+_{\mathcal I}\rangle
======================

\frac{1}{\sqrt{|\mathcal I|}}
\sum_{C\in\mathcal I}|C\rangle,
]

其中 (C) 是无最近邻 (r r) 的 allowed configurations。

## 4.1 首先只做 constrained-state preparation

你需要先证明能制备接近 (|+_{\mathcal I}\rangle) 的状态。可以尝试三类 protocol：

### Protocol A：global PXP quench / pulse shaping

从 vacuum：

[
|1\rangle^{\otimes n}
]

在

[
H_{\rm PXP}/\hbar=
\frac{\Omega(t)}{2}\sum_iP\sigma_i^xP
-\Delta(t)N
]

下演化，通过优化 (\Omega(t),\Delta(t)) 最大化目标。

### Protocol B：adiabatic / quasi-adiabatic sweep

从大负 detuning vacuum，扫到某个 (\Delta)，寻找哪个 (\Delta) 的态最接近 uniform independent-set state。

### Protocol C：closed-loop / GRAPE / CRAB pulse optimization

直接把目标函数设成：

[
\mathcal C
==========

1-
|\langle +*{\mathcal I}|\psi(T)\rangle|^2
+
\lambda L*{\rm leak}
+
\mu P_{\rm loss}.
]

其中

[
L_{\rm leak}=1-\langle P\rangle
]

是 blockade leakage。

需要输出：

[
F_{\mathcal I}
==============

|\langle +_{\mathcal I}|\psi\rangle|^2,
]

[
P_{\mathcal I}=\langle P\rangle,
]

[
D_{\rm amp}
===========

\sum_{C\in\mathcal I}
\left|
|\psi_C|^2-\frac1{|\mathcal I|}
\right|,
]

[
\mathrm{phase\ spread}
======================

\mathrm{Var}_{C\in\mathcal I}[\arg\psi_C].
]

注意：只让 probability uniform 不够；constrained-IQP 的初态需要 **amplitudes 相干且相位可控**。

---

## 4.2 然后模拟 constrained diagonal phase

在 constrained subspace 中，最近邻项恒为零：

[
n_i n_j=0,\qquad \langle i,j\rangle.
]

所以 diagonal phase 主要来自 blockade radius 外的 tails：

[
H_{\rm diag}^{\mathcal I}
=========================

\sum_{(i,j)\notin E_{\rm blockade}}
V_{ij}n_in_j.
]

输出：

[
\mathrm{Var}*{C\in\mathcal I}
\left[
E_C
\right],
\qquad
E_C=
\sum*{i<j}V_{ij}n_i(C)n_j(C).
]

这个能量方差决定相位是否足够展开。若

[
t,\sqrt{\mathrm{Var}(E_C)}
\ll1,
]

则 diagonal evolution 太弱，输出接近初态；若过大且有 loss/dephasing，可能不可控。

建议扫描：

[
t
]

并画：

[
H_2(p_X(t)),
\quad
D_{\rm TV}(p_X(t),p_X(0)),
\quad
P_{\rm surv}(t),
\quad
F_{\rm physical\ vs\ ideal}(t).
]

这里 (p_X(t)) 是最终在 (X)-basis 或 hyperfine basis-rotated readout 下的分布。

---

## 4.3 模拟 final measurement 是否可行

Route 3B 的优势是：如果你最终 map down 到 hyperfine qubit，再做 hyperfine single-qubit gates，则不需要在 (|1\rangle-|r\rangle) manifold 里做直接 (\pi/2)。Geim 文章的技术主张正是通过 hyperfine digital processing 实现 (Z)-basis 以外的 state preparation and measurement；他们也强调 non-destructive hyperfine readout 能把 loss 信息拿出来。

但 map-down 自身仍有 vdW tail error。文章 Methods 里指出，long-range vdW tails 会 detune Rydberg (\pi)-pulse；他们补偿 mean-field shift，但 microstate-dependent local resonance 无法补偿，2D kagome 的 NNN tail energy 到 670 kHz，因而把 mapping pulse 提高到 12.6 MHz。

所以你要模拟：

[
\text{ideal constrained state}
\rightarrow
\text{map down}
\rightarrow
\text{hyperfine basis rotation}
\rightarrow
\text{measurement}.
]

输出：

[
F_{\rm mapdown}^{\mathcal I},
]

[
P_{\rm loss}^{\mathcal I},
]

[
D_{\rm TV}(p_{\rm measured},p_{\rm ideal}).
]

这会直接回答：**constrained-IQP 的最终 sampling 是否会被 map-down error 毁掉。**

---

# 5. 你最终应该给出哪些“可行性图”

建议最终做成 8 张核心图。

## 图 1：Route 1 的 pulse-error 相图

横轴：

[
V_{\rm NN}/\Omega_{\rm pulse}
]

纵轴：

[
n
]

颜色：

[
1-F_{\rm echo}.
]

目的：判断完整 (|+\rangle_R^{\otimes n}) 是否能在 finite interaction 下被制备和读出。

---

## 图 2：Route 1 的 phase-vs-loss tradeoff

横轴：

[
a
\quad\text{或}\quad V_{\rm NN}
]

纵轴：

[
\phi_{\rm target}=\pi/8,\pi/4,\pi/2
]

颜色：

[
P_{\rm surv}
]

或所需时间

[
t_{\phi}=\phi/V_{\rm NN}.
]

目的：判断拉大 spacing 后，非平凡 IQP phase 是否还来得及积累。

---

## 图 3：Route 1 的 ideal-vs-physical sampling distance

横轴：

[
n
]

纵轴：

[
D_{\rm TV}(p_{\rm phys},p_{\rm ideal})
]

多条曲线：

[
V_{\rm NN}/\Omega_{\rm pulse}=0.01,0.03,0.1,0.3.
]

目的：判断多体 IQP 分布是否仍可解释为理想 IQP。

---

## 图 4：Route 1 的 two-atom phase calibration

横轴：

[
t
]

纵轴：

[
\phi_{\rm fitted}
]

和理论线：

[
V_{12}t.
]

再画残差：

[
\delta\phi(t).
]

目的：给实验上最小可验证 benchmark。

---

## 图 5：Route 3A 的 active subset tradeoff

横轴：

[
N_{\rm active}
]

纵轴：

[
V_{\max}^{(S)}
\quad\text{或}\quad
t_{\pi/4}^{(S)}.
]

颜色：

[
F_{\rm map}(S)
\quad\text{或}\quad
P_{\rm surv}(S).
]

目的：判断稀疏子图路线有没有合适的 active qubit 数和相互作用强度。

---

## 图 6：Route 3A 的 IQP complexity proxy

对不同 active subset，画：

[
H_2(p),\quad
D_{\rm TV}(p,p_{\rm unif}),\quad
\mathrm{PR}.
]

目的：判断稀疏子图的输出分布是否太平凡。

---

## 图 7：Route 3B 的 constrained-state preparation fidelity

横轴：

[
T_{\rm prep}
]

或 pulse optimization iteration。

纵轴：

[
F_{\mathcal I},
\quad
P_{\mathcal I},
\quad
D_{\rm amp}.
]

目的：判断 (|+_{\mathcal I}\rangle) 或近似 constrained plus state 是否能制备。

---

## 图 8：Route 3B 的 constrained-IQP full-sequence benchmark

横轴：

[
t_{\rm int}
]

纵轴：

[
D_{\rm TV}(p_{\rm phys},p_{\rm ideal}^{\mathcal I}),
\quad
P_{\rm loss},
\quad
H_2(p).
]

目的：判断完整 constrained-IQP 采样是否在实验误差下还能保持目标分布。

---

# 6. 最小数值实验路线

建议按下面顺序做，不要一开始就上大系统。

## Stage I：两原子和三原子

验证基础：

[
R_y(\pi/2),
\quad
e^{-iVt n_1n_2},
\quad
R_y(-\pi/2).
]

输出：

[
F,\quad
\phi_{\rm fitted},\quad
P_{\rm surv}.
]

三原子要特别看：

[
|r1r\rangle
]

这种 allowed NNN configuration 的 vdW tail，以及

[
|rr1\rangle
]

这种 blockade-violating configuration 的失败模式。

---

## Stage II：小链 (n=6\sim12)

比较：

1. Route 1 full IQP；
2. Route 3A sparse subset；
3. Route 3B constrained Hilbert space。

输出：

[
F_{\rm state},
\quad
D_{\rm TV},
\quad
P_{\rm loss},
\quad
H_2.
]

---

## Stage III：中等系统 (n=20\sim40)

不再做 full state fidelity，改用：

[
P_{\rm loss},
\quad
\langle X_i\rangle,
\quad
\langle X_iX_j\rangle,
\quad
H_2(p_{\rm sampled}),
\quad
collision probability.
]

可以用 MPS / Krylov / Monte Carlo trajectories，或者只在 constrained Hilbert space 中模拟。

---

# 7. 预期你会看到的结论

我预计 Route 1 会出现一个很窄的窗口：

[
V_{\rm NN}/\Omega_{\rm pulse}\ll1
]

保证 (|+\rangle_R^{\otimes n}) 可制备；但同时

[
V_{\rm NN}t_{\rm int}\sim O(1)
]

要求演化时间不短。这个窗口是否存在，强烈取决于你的 (\Omega_{\rm pulse})、Rydberg lifetime、spacing、主量子数和目标 (n)。

Route 3A 更容易实验实现，因为 active sites 避开 blockade；但它可能变成“少量 active qubits + 很弱 long-range tails”的稀疏 IQP，需要用 (H_2)、PR、anti-concentration 来证明分布非平凡。

Route 3B 最符合 Rydberg blockade 平台的物理优势；但最难的是制备

[
|+_{\mathcal I}\rangle.
]

一旦能制备，后面的 diagonal constrained phase 反而很自然，因为 blockade 已经把非法构型排除了。

---

# 8. 最关键的 feasibility 指标

最后建议把所有结果压缩成四个数字：

[
\boxed{
F_{\rm prep}
}
]

初态制备是否成功。

[
\boxed{
D_{\rm TV}(p_{\rm phys},p_{\rm ideal})
}
]

物理采样分布是否接近目标 IQP。

[
\boxed{
P_{\rm surv}
}
]

实验 postselection 是否可承受。

[
\boxed{
\Phi_{\rm rms}
==============

t\sqrt{\frac{1}{M}\sum_{i<j}\phi_{ij}^2}
}
]

diagonal phase 是否足够非平凡。

如果一组参数同时满足：

[
F_{\rm prep}\gtrsim 0.9,
]

[
D_{\rm TV}\lesssim 0.1\text{--}0.2,
]

[
P_{\rm surv}\gtrsim 0.5
]

且

[
\Phi_{\rm rms}\sim O(1),
]

那就可以说该方案在实验上有探索价值。阈值可以根据你要做的是 proof-of-principle 还是 scalable sampling 再收紧。
