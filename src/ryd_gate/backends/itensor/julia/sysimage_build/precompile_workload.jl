# Precompile workload for the TNQS 2D-TN sysimage.
#
# Drives the real kernel code path (`run_from_files` -> `run_2dtn_bp`) on a tiny
# 2x2 CPU payload so PackageCompiler traces the hot methods: the gate layers,
# `apply_gates`, `truncate`, the `bp` `expect`, and JSON3/NPZ I/O. The payload sets
# `use_cuda=false` so the build works on CPU-only machines; GPU kernels still JIT
# on first GPU use at runtime.
#
# `include` of the kernel runs its top-level `using` statements; the kernel's bottom
# `main()` call is guarded by `abspath(PROGRAM_FILE) == @__FILE__`, so including it
# here defines the functions without executing the script.
#
# Regenerate precompile_payload.json with:
#   uv run python src/ryd_gate/backends/itensor/julia/sysimage_build/gen_precompile_payload.py

include(joinpath(@__DIR__, "..", "run_tnqs_2d_bp.jl"))

let payload = joinpath(@__DIR__, "precompile_payload.json")
    mktempdir() do dir
        run_from_files(payload, joinpath(dir, "result.npz"), joinpath(dir, "result.json"))
    end
end
