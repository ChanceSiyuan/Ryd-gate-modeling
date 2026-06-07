#!/usr/bin/env julia
#
# PEPSKit.jl iPEPS real-time simple-update kernel for 2D Rydberg quench dynamics.
#
# Models the infinite-lattice (bulk) limit: a translation-invariant 2x2 iPEPS
# evolved with real-time simple update (exp(-iH dt)) and measured with CTMRG.
# Supports 1r (d=2) and 01r (d=3) local structures, with uniform or A/B sublattice
# driving. On-site drive terms are folded into nearest-neighbour bond terms
# (divided by the coordination number z=4) so the Hamiltonian is purely 2-site and
# the efficient 2-site simple update is used.

using TensorKit
using PEPSKit
using MPSKit
using JSON3
using NPZ
using LinearAlgebra

const Z_COORD = 4  # nearest-neighbour bonds per site on the square lattice

function run_from_files(input_json, output_npz, output_json)
    payload = JSON3.read(read(input_json, String))
    result = run_pepskit_ipeps(payload)
    NPZ.npzwrite(output_npz, result["arrays"])
    open(output_json, "w") do io
        JSON3.write(io, result["metadata"])
    end
end

function main()
    if length(ARGS) != 3
        error("Usage: run_pepskit_ipeps.jl payload.json result.npz result.json")
    end
    input_json, output_npz, output_json = ARGS
    run_from_files(input_json, output_npz, output_json)
end

function build_operators(latspec)
    levels = String.(latspec.levels)
    d = Int(latspec.physical_dim)
    P = ℂ^d
    idx(l) = findfirst(==(l), levels)
    function proj(l)
        m = zeros(ComplexF64, d, d)
        m[idx(l), idx(l)] = 1.0
        return TensorMap(m, P ← P)
    end
    function flip(a, b)
        m = zeros(ComplexF64, d, d)
        m[idx(a), idx(b)] = 1.0
        m[idx(b), idx(a)] = 1.0
        return TensorMap(m, P ← P)
    end
    ops = Dict{String, Any}()
    ops["id"] = id(ComplexF64, P)
    ops["n_r"] = proj("r")
    if String(latspec.level_structure) == "01r"
        ops["n_0"] = proj("0")
        ops["n_1"] = proj("1")
        ops["x_01"] = flip("0", "1")
        ops["x_1r"] = flip("1", "r")
    else
        ops["n_1"] = proj(levels[1])
        ops["x_1r"] = flip(levels[1], "r")
    end
    return P, ops
end

function onsite_term(ops, latspec, dr)
    h = (Float64(dr.omega_R) / 2) * ops["x_1r"] - Float64(dr.delta_R) * ops["n_r"]
    if String(latspec.level_structure) == "01r"
        h += (Float64(dr.omega_hf) / 2) * ops["x_01"] - Float64(dr.delta_hf) * ops["n_1"]
    end
    return h
end

# Build the (purely nearest-neighbour) step Hamiltonian with on-site drive terms
# folded into the bonds. `sd` is the schedule entry for this Trotter step.
function build_step_hamiltonian(lat, sq, ops, latspec, sd, V::Float64)
    nr = ops["n_r"]
    idP = ops["id"]
    # On sublattice A/B is selected by checkerboard parity of (row + col); for a
    # uniform unit cell both sublattices carry the same on-site term.
    local hA, hB
    if String(latspec.unit_cell) == "sublattice"
        hA = onsite_term(ops, latspec, sd.A)
        hB = onsite_term(ops, latspec, sd.B)
    else
        h = onsite_term(ops, latspec, sd)
        hA = h
        hB = h
    end
    terms = Pair[]
    for b in nearest_neighbours(sq)
        i, j = b
        hi = iseven(i[1] + i[2]) ? hA : hB
        hj = iseven(j[1] + j[2]) ? hA : hB
        term = V * (nr ⊗ nr) + (1 / Z_COORD) * (hi ⊗ idP) + (1 / Z_COORD) * (idP ⊗ hj)
        push!(terms, b => term)
    end
    return LocalOperator(lat, terms...)
end

function init_state_vectors(init, levels, Nr, Nc)
    d = length(levels)
    idx(l) = findfirst(==(l), levels)
    function onehot(l)
        v = zeros(ComplexF64, d)
        v[idx(String(l))] = 1.0
        return v
    end
    sv = Matrix{Vector{ComplexF64}}(undef, Nr, Nc)
    if String(init.pattern) == "uniform"
        v = onehot(init.label)
        for r in 1:Nr, c in 1:Nc
            sv[r, c] = v
        end
    else
        vA = onehot(init.A)
        vB = onehot(init.B)
        for r in 1:Nr, c in 1:Nc
            sv[r, c] = iseven(r + c) ? vA : vB
        end
    end
    return sv
end

function measure_env(peps, χ::Int, tol::Float64, maxiter::Int)
    trunc_env = truncerror(; atol = 1e-10) & truncrank(χ)
    env0 = CTMRGEnv(rand, ComplexF64, peps, ℂ^χ)
    env, = leading_boundary(
        env0, peps;
        alg = :sequential, projector_alg = :fullinfinite,
        tol = tol, maxiter = maxiter, trunc = trunc_env,
    )
    return env
end

function bulk_expect(peps, env, lat, op)
    sites = [(r, c) for r in 1:size(peps, 1) for c in 1:size(peps, 2)]
    vals = [real(expectation_value(peps, LocalOperator(lat, (CartesianIndex(r, c),) => op), env))
            for (r, c) in sites]
    return sum(vals) / length(vals)
end

function bulk_nn_corr(peps, env, lat, op)
    sq = InfiniteSquare(size(peps, 1), size(peps, 2))
    bonds = collect(nearest_neighbours(sq))
    vals = [real(expectation_value(peps, LocalOperator(lat, (b[1], b[2]) => op ⊗ op), env))
            for b in bonds]
    return sum(vals) / length(vals)
end

function measure_observables(peps, env, lat, ops, latspec, names)
    out = Dict{String, Float64}()
    is01r = String(latspec.level_structure) == "01r"
    nr = bulk_expect(peps, env, lat, ops["n_r"])
    for name in names
        if name in ("n_r", "n_mean", "sum_nr")
            out[name] = nr
        elseif name == "sigma_z"
            out[name] = 2 * nr - 1
        elseif name == "n_1"
            out[name] = is01r ? bulk_expect(peps, env, lat, ops["n_1"]) : 1 - nr
        elseif name == "n_0"
            out[name] = is01r ? bulk_expect(peps, env, lat, ops["n_0"]) : 0.0
        elseif name == "nn_corr"
            out[name] = bulk_nn_corr(peps, env, lat, ops["n_r"])
        else
            error("Unknown PEPSKit observable: $name")
        end
    end
    return out
end

function run_pepskit_ipeps(payload)
    latspec = payload.lattice
    rt = payload.runtime
    Nr, Nc = Int(latspec.Nr), Int(latspec.Nc)
    P, ops = build_operators(latspec)
    lat = fill(P, Nr, Nc)
    sq = InfiniteSquare(Nr, Nc)
    D = Int(rt.bond_dim)
    χ = Int(rt.env_dim)
    dt = Float64(rt.dt)
    V = Float64(latspec.V_nn)
    levels = String.(latspec.levels)

    sv = init_state_vectors(payload.initial_state, levels, Nr, Nc)
    peps = product_peps(
        randn, ComplexF64, P, ℂ^D;
        unitcell = (Nr, Nc), noise_amp = Float64(rt.init_noise), state_vector = sv,
    )
    wts = SUWeight(peps)

    alg = SimpleUpdate(;
        trunc = truncerror(; atol = Float64(rt.su_trunc_atol)) & truncrank(D),
        imaginary_time = false, bipartite = false,
    )

    record = Set(Int(x) for x in payload.record_steps)
    names = String.(payload.observables)
    obs_acc = Dict(name => Float64[] for name in names)
    times = Float64[]

    function do_measure!(t)
        push!(times, t)
        env = measure_env(peps, χ, Float64(rt.ctmrg_tol), Int(rt.ctmrg_maxiter))
        vals = measure_observables(peps, env, lat, ops, latspec, names)
        for name in names
            push!(obs_acc[name], vals[name])
        end
    end

    (0 in record) && do_measure!(0.0)
    for sd in payload.schedule
        H = build_step_hamiltonian(lat, sq, ops, latspec, sd, V)
        peps, wts, _ = time_evolve(peps, H, dt, 1, alg, wts; check_interval = 0)
        step = Int(sd.step)
        (step in record) && do_measure!(step * dt)
    end

    arrays = Dict{String, Any}("times" => times)
    for (name, vals) in obs_acc
        arrays["obs_" * name] = vals
    end
    metadata = Dict{String, Any}(
        "backend" => "pepskit",
        "method" => "pepskit_ipeps_su",
        "engine_package" => "PEPSKit.jl",
        "unit_cell" => String(latspec.unit_cell),
        "physical_dim" => Int(latspec.physical_dim),
        "level_structure" => String(latspec.level_structure),
        "bond_dim" => D,
        "env_dim" => χ,
        "dt" => dt,
        "n_steps" => length(payload.schedule),
        "V_nn" => V,
        "trotter_order" => Int(rt.trotter_order),
    )
    return Dict("arrays" => arrays, "metadata" => metadata)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
