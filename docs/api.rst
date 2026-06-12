API Reference
=============

Stable public modules. Backend internals (``ryd_gate.backends.*``,
``ryd_gate.ir``, ``ryd_gate.core``) are kernel API and documented in code.

Top level
---------

.. automodule:: ryd_gate
   :members: simulate, simulate_sequence, level_structure

Registers and layouts
---------------------

.. automodule:: ryd_gate.lattice
   :members: Register, RegisterLayout

Pulses and waveforms
--------------------

.. automodule:: ryd_gate.pulse
   :members: Waveform, Pulse

Devices and channels
--------------------

.. automodule:: ryd_gate.devices
   :members: DeviceSpec

.. automodule:: ryd_gate.protocols.channels
   :members: ChannelSpec

Sequences
---------

.. automodule:: ryd_gate.sequence
   :members: Sequence, PulseOp, DelayOp, MeasureOp

Results
-------

.. automodule:: ryd_gate.results
   :members: SimulationResult, ExactStateHandle, MPSStateHandle,
             UnsupportedStateHandle, QuantumStateHandle,
             UnsupportedResultQuery, StateMaterializationError

Noise
-----

.. automodule:: ryd_gate.noise
   :members: NoiseModel, configure_monte_carlo_runner

Gate library
------------

.. automodule:: ryd_gate.gates
   :members: CZGateReport, cz_gate_report

Observable schedules
--------------------

.. automodule:: ryd_gate.observables
   :members: ObservableConfig

Pulser interop
--------------

.. automodule:: ryd_gate.interop
   :members: PulserInteropError, from_pulser_abstract_repr,
             to_pulser_abstract_repr, noise_from_pulser_abstract_repr,
             noise_to_pulser_abstract_repr
