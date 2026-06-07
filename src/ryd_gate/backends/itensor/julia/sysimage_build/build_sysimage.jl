# Build a PackageCompiler sysimage that bakes ITensors / TNQS / CUDA so the 2dtn
# kernel skips JIT on every run (cold-start drops from ~600s to ~warm-run time).
#
# Run from the repo root with the build-only environment active:
#
#   julia --project=src/ryd_gate/backends/itensor/julia/sysimage_build \
#         src/ryd_gate/backends/itensor/julia/sysimage_build/build_sysimage.jl
#
# PackageCompiler is resolved from this build-only project; the baked packages are
# resolved from the RUNTIME project (`../Project.toml`) so versions match the kernel
# Manifest. The resulting sysimage is Julia-version- and machine-specific; rebuild
# after a Julia upgrade. The 2dtn backend auto-detects it at the default path below.

import Pkg
Pkg.add("PackageCompiler")  # idempotent; installs into the build env if missing

using PackageCompiler

const HERE = @__DIR__
const RUNTIME_PROJECT = normpath(joinpath(HERE, ".."))
const SYSIMAGE_DIR = joinpath(RUNTIME_PROJECT, "sysimages")
const SYSIMAGE_PATH = joinpath(SYSIMAGE_DIR, "ryd_tnqs.so")

mkpath(SYSIMAGE_DIR)

create_sysimage(
    [
        :TensorNetworkQuantumSimulator,
        :ITensors,
        :ITensorNetworks,
        :ITensorMPS,
        :NamedGraphs,
        :TensorOperations,
        :CUDA,
        :JSON3,
        :NPZ,
    ];
    sysimage_path = SYSIMAGE_PATH,
    precompile_execution_file = joinpath(HERE, "precompile_workload.jl"),
    project = RUNTIME_PROJECT,
)

println("Wrote sysimage to $SYSIMAGE_PATH")
