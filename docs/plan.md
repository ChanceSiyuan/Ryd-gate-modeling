This is a robust software engineering approach. Refactoring a monolithic "God Class" into a modular architecture (using the Strategy Pattern and Single Responsibility Principle) will make your simulator much more maintainable and scalable.

Here is a comprehensive, phase-by-phase implementation plan written in English. You can copy and paste this directly to Claude (or any other LLM) to execute the refactoring and implement the local addressing simulation.

***

### **Prompt to Copy & Paste to Claude:**

**Context:**
I am refactoring a quantum simulation codebase. The current code simulates a 2-atom Rydberg system using a strict 7-level microscopic model (using SciPy's `solve_ivp` for the Schrödinger equation). Currently, it is a monolithic class (`CZGateSimulator`). I am refactoring it into a modular architecture: `core/`, `protocols/`, `solvers/`, and `analysis/`.

**Goal:**
Implement a specific "Local Addressing" simulation using this new modular architecture. The experiment involves sweeping a global laser (to trigger a quantum phase transition) while strongly pinning Atom A (the left atom) to the ground state using a local 784nm laser. We need to analyze how global phase noise and local intensity noise (RIN) affect the pinning and crosstalk.

**Level Indices in the 7-level model:**
* Index `1`: Target ground state $|g\rangle$ or $|1\rangle$.
* Index `2, 3, 4`: Intermediate $6P_{3/2}$ states.
* Index `5`: Target Rydberg state $|r\rangle$.
* Index `6`: Unwanted garbage Rydberg state $|r'\rangle$.

Please implement this across the following **4 distinct phases**. Write the complete Python code for each file mentioned.

#### **Phase 1: Core Physics & The Sweep Protocol**
**Files to create/update:** `core/atomic_system.py` and `protocols/local_sweep.py`.

1.  **`core/atomic_system.py`**:
    * Create a `TwoAtomRydbergSystem` class.
    * It should contain the static Hamiltonians ($H_{\text{static}}$, $H_{420}$, $H_{1013}$, $H_{\text{vdW}}$) without any time-dependent logic.
    * Include a method `build_single_atom_operator(index)` that returns a 7x7 matrix with `1` at `[index, index]`.
2.  **`protocols/local_sweep.py`**:
    * Create a `SweepAddressingProtocol` class inheriting from a `BaseProtocol`.
    * **Inputs:** `omega` (global Rabi), `delta_start`, `delta_end`, `t_gate`, `local_detuning_A` (AC Stark shift $\delta_0$, usually negative, e.g., $-2\pi \times 12$ MHz), and `local_scattering_rate` (e.g., $35$ Hz).
    * **Method `get_hamiltonian(t, system)`:**
        * Calculate the instantaneous global detuning: $\Delta(t) = \Delta_{\text{start}} + (\Delta_{\text{end}} - \Delta_{\text{start}}) \times (t / t_{\text{gate}})$.
        * Construct the global drive Hamiltonian using $\Delta(t)$ and `omega`.
        * **Local Pinning:** Add the local 784nm laser effect STRICTLY to Atom A (the left atom in the Kronecker product). It applies `-local_detuning_A` to Atom A's Rydberg states (indices 5 and 6) and a non-Hermitian decay `-1j * local_scattering_rate / 2` to Atom A's ground state (index 1). Use `np.kron(Op_A, np.eye(7))` for this.

#### **Phase 2: Solvers & The Noise Engine**
**Files to create/update:** `solvers/schrodinger.py` and `solvers/monte_carlo.py`.

1.  **`solvers/schrodinger.py`**:
    * Create `SchrodingerSolver(system, protocol)`.
    * Implement an `evolve(initial_state)` method using `scipy.integrate.solve_ivp` (method `DOP853`). The RHS function should query `protocol.get_hamiltonian(t, system)`.
2.  **`solvers/monte_carlo.py`**:
    * Create a `MonteCarloEngine(system, protocol, noise_params)`.
    * **Noise params:** `sigma_detuning` (global phase noise), `sigma_local_rin` (Local Relative Intensity Noise).
    * For each shot in `n_shots`:
        * Sample global detuning error: $\epsilon_{\text{global}} \sim \mathcal{N}(0, \sigma_{\text{detuning}})$. Apply this uniformly to Rydberg states of *both* atoms.
        * Sample local RIN error: $\epsilon_{\text{RIN}} \sim \mathcal{N}(0, \sigma_{\text{local\_rin}} \times |\text{local\_detuning\_A}|)$. Apply this strictly to Atom A's Rydberg states using `np.kron(Op_A, np.eye(7))`.
        * Pass the noisy system to `SchrodingerSolver.evolve()` and store the final state.
    * Return a list/array of final states.

#### **Phase 3: Addressing Metrics (Analysis)**
**File to create/update:** `analysis/addressing_metrics.py`.

1.  **`addressing_metrics.py`**:
    * Create an `AddressingEvaluator` class that takes a list of final state vectors (from the Monte Carlo engine).
    * Define projection operators:
        * $P_{A,1} = |1\rangle\langle 1| \otimes \mathbb{I}$ (Atom A is in state 1)
        * $P_{B,r} = \mathbb{I} \otimes |r\rangle\langle r|$ (Atom B is in state 5)
    * Implement calculation methods averaged over all MC shots:
        * `compute_pinning_error()`: $1.0 - \text{Tr}(\rho_{\text{final}} P_{A,1})$. (Probability Atom A failed to stay in the ground state).
        * `compute_crosstalk_error()`: $1.0 - \text{Tr}(\rho_{\text{final}} P_{B,r})$. (Probability Atom B failed to reach the Rydberg state).
        * `compute_leakage_loss()`: $1.0 - ||\psi_{\text{final}}||^2$. (Probability lost to non-Hermitian decay channels like scattering or blackbody radiation).

#### **Phase 4: Execution Script**
**File to create:** `run_addressing_sim.py`.

* Write a main execution script that ties the modules together.
* Initialize the `TwoAtomRydbergSystem`.
* Initialize the `SweepAddressingProtocol` with a sweep from $-2\pi \times 15$ MHz to $+2\pi \times 15$ MHz over $1.5 \mu s$, with a pinning strength of $-2\pi \times 12$ MHz.
* Set up a double `for` loop to scan parameters:
    * `sigma_detuning` from `10 kHz` to `500 kHz` (5 points).
    * `sigma_local_rin` from `0.001` (0.1%) to `0.05` (5%) (5 points).
* For each grid point, run `MonteCarloEngine` with `n_shots=50`, evaluate the `pinning_error` using `AddressingEvaluator`, and store the result in a 2D NumPy array.
* Use `matplotlib` to plot a heatmap (`plt.imshow`) of the Pinning Error vs. Global Phase Noise and Local RIN.

**Testing Criteria (Please ensure the code satisfies these):**
* If `sigma_local_rin = 0` and `sigma_detuning = 0`, the `pinning_error` should be very close to 0, and `norm_loss` should exactly match the expected scattering decay.
* The Kronecker products must correctly isolate Atom A (left) and Atom B (right).