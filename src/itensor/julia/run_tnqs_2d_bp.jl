#!/usr/bin/env julia

using CUDA
using ITensors
using JSON3
using LinearAlgebra
using NPZ
using TensorNetworkQuantumSimulator

const TNQS = TensorNetworkQuantumSimulator

function main()
    if length(ARGS) != 3
        error("Usage: run_tnqs_2d_bp.jl payload.json result.npz result.json")
    end
    input_json, output_npz, output_json = ARGS
    payload = JSON3.read(read(input_json, String))
    result = run_2dtn_bp(payload)
    NPZ.npzwrite(output_npz, result["arrays"])
    open(output_json, "w") do io
        JSON3.write(io, result["metadata"])
    end
end

function run_2dtn_bp(payload)
    runtime = payload.runtime
    lattice = payload.lattice
    n_sites = Int(lattice.N)
    dt = Float64(runtime.dt)
    chi_max = Int(runtime.chi_max)
    cutoff = Float64(runtime.svd_min)
    use_cuda = Bool(runtime.use_cuda)
    measurement_alg = String(_property_or(runtime, :measurement_alg, "bp"))
    measurement_bond_dim = Int(_property_or(runtime, :measurement_bond_dim, 32))
    chi_2d_prime = _optional_int(_property_or(runtime, :chi_2d_prime, nothing))
    normalize_tensors = Bool(_property_or(runtime, :normalize_tensors, false))

    if use_cuda && !CUDA.functional()
        error("runtime.use_cuda=true but CUDA.functional() is false")
    end

    g = build_interaction_graph(lattice)
    state_eltype = _state_eltype(runtime, use_cuda)
    psi = tensornetworkstate(state_eltype, initial_state_function(lattice, payload), g, "S=1/2")
    psi_bpc = BeliefPropagationCache(psi)
    if use_cuda
        psi_bpc = CUDA.cu(psi_bpc)
    end

    schedule = payload.schedule
    record_steps = Set(Int(x) for x in payload.record_steps)
    observables = Set(String(x) for x in payload.observables)
    apply_kwargs = (; maxdim = chi_max, cutoff = cutoff, normalize_tensors = normalize_tensors)

    obs_sigma_z = Vector{Vector{Float64}}()
    obs_czz_centerline = Vector{Vector{Float64}}()
    recorded_times = Float64[]
    truncation_error = Float64[]

    if 0 in record_steps
        record_observables!(
            psi_bpc,
            lattice,
            0.0,
            observables,
            measurement_alg,
            measurement_bond_dim,
            chi_2d_prime,
            cutoff,
            normalize_tensors,
            obs_sigma_z,
            obs_czz_centerline,
            recorded_times,
        )
    end

    for step_data in schedule
        omega = Float64.(collect(step_data.omega_1d))
        delta = Float64.(collect(step_data.delta_1d))
        step = Int(step_data.step)
        step_errors = Float64[]

        local_gates, local_vertices = local_gate_layer(psi_bpc, lattice, omega, delta, dt / 2)
        psi_bpc = apply_gate_layer(psi_bpc, local_gates, local_vertices, apply_kwargs, step_errors)

        pair_gates, pair_vertices = pair_gate_layer(psi_bpc, lattice, dt)
        psi_bpc = apply_gate_layer(psi_bpc, pair_gates, pair_vertices, apply_kwargs, step_errors)

        psi_bpc = apply_gate_layer(psi_bpc, local_gates, local_vertices, apply_kwargs, step_errors)
        push!(truncation_error, isempty(step_errors) ? 0.0 : maximum(step_errors))

        if step in record_steps
            record_observables!(
                psi_bpc,
                lattice,
                step * dt,
                observables,
                measurement_alg,
                measurement_bond_dim,
                chi_2d_prime,
                cutoff,
                normalize_tensors,
                obs_sigma_z,
                obs_czz_centerline,
                recorded_times,
            )
        end
    end

    final_bpc = measurement_cache(psi_bpc, chi_2d_prime, cutoff, normalize_tensors)
    final_sigma_z = sigma_z_2d(final_bpc, lattice, measurement_alg, measurement_bond_dim)
    arrays = Dict{String, Any}(
        "times" => recorded_times,
        "final_sigma_z" => final_sigma_z,
        "truncation_error" => truncation_error,
    )
    if !isempty(obs_sigma_z)
        arrays["obs_sigma_z"] = _stack_rows(obs_sigma_z, n_sites)
    end
    if !isempty(obs_czz_centerline)
        n_cols = length(obs_czz_centerline[1])
        arrays["obs_czz_centerline"] = _stack_rows(obs_czz_centerline, n_cols)
    end

    metadata = Dict{String, Any}(
        "backend" => "2dtn",
        "method" => "2dtn_bp",
        "engine_package" => "TensorNetworkQuantumSimulator.jl",
        "n_sites" => n_sites,
        "chi_max" => chi_max,
        "dt" => dt,
        "n_steps" => length(schedule),
        "svd_min" => cutoff,
        "use_cuda" => use_cuda,
        "measurement_alg" => measurement_alg,
        "measurement_bond_dim" => measurement_bond_dim,
        "chi_2d_prime" => chi_2d_prime,
        "normalize_tensors" => normalize_tensors,
        "max_truncation_error" => isempty(truncation_error) ? 0.0 : maximum(truncation_error),
        "state_serialized" => false,
    )
    return Dict("arrays" => arrays, "metadata" => metadata)
end

function build_interaction_graph(lattice)
    Lx = Int(lattice.Lx)
    Ly = Int(lattice.Ly)
    g = named_grid((Lx, Ly))
    snake_to_2d = Int.(collect(lattice.snake_to_2d))
    for pair in lattice.vdw_pairs_1d
        i = Int(pair[1])
        j = Int(pair[2])
        if i == j
            continue
        end
        vi = vertex_from_snake(i, snake_to_2d, Ly)
        vj = vertex_from_snake(j, snake_to_2d, Ly)
        edge = TNQS.NamedEdge(vi => vj)
        if !TNQS.has_edge(g, edge)
            g = add_edge(g, edge)
        end
    end
    return g
end

function initial_state_function(lattice, payload)
    Ly = Int(lattice.Ly)
    n_sites = Int(lattice.N)
    snake_to_2d = Int.(collect(lattice.snake_to_2d))
    occ_1d = Int.(collect(payload.initial_occupations_1d))
    occ_2d = zeros(Int, n_sites)
    for pos in 1:n_sites
        occ_2d[snake_to_2d[pos] + 1] = occ_1d[pos]
    end
    return v -> occ_2d[row_major_index0(v, Ly) + 1] == 1 ? "Up" : "Dn"
end

function local_gate_layer(psi_bpc, lattice, omega, delta, dt::Float64)
    Ly = Int(lattice.Ly)
    n_sites = Int(lattice.N)
    snake_to_2d = Int.(collect(lattice.snake_to_2d))
    site_dict = siteinds(network(psi_bpc))
    gates = ITensor[]
    gate_vertices = Vector{Vector{Tuple{Int, Int}}}()
    for pos in 1:n_sites
        v = vertex_from_snake(pos, snake_to_2d, Ly)
        site = only(site_dict[v])
        push!(gates, local_gate(site, Float64(omega[pos]), Float64(delta[pos]), dt))
        push!(gate_vertices, [v])
    end
    return gates, gate_vertices
end

function pair_gate_layer(psi_bpc, lattice, dt::Float64)
    Ly = Int(lattice.Ly)
    snake_to_2d = Int.(collect(lattice.snake_to_2d))
    site_dict = siteinds(network(psi_bpc))
    gates = ITensor[]
    gate_vertices = Vector{Vector{Tuple{Int, Int}}}()
    for pair in lattice.vdw_pairs_1d
        i = Int(pair[1])
        j = Int(pair[2])
        strength = Float64(pair[3])
        if i == j || abs(strength) == 0
            continue
        end
        vi = vertex_from_snake(i, snake_to_2d, Ly)
        vj = vertex_from_snake(j, snake_to_2d, Ly)
        site_i = only(site_dict[vi])
        site_j = only(site_dict[vj])
        push!(gates, pair_gate(site_i, site_j, strength, dt))
        push!(gate_vertices, [vi, vj])
    end
    return gates, gate_vertices
end

function apply_gate_layer(psi_bpc, gates, gate_vertices, apply_kwargs, step_errors)
    if isempty(gates)
        return psi_bpc
    end
    psi_bpc, errors = apply_gates(
        gates,
        psi_bpc;
        gate_vertices = gate_vertices,
        apply_kwargs = apply_kwargs,
        verbose = false,
    )
    append!(step_errors, [Float64(abs(x)) for x in collect(errors)])
    return psi_bpc
end

function local_gate(site, omega::Float64, delta::Float64, dt::Float64)
    sx = ITensors.op("Sx", site)
    sz = ITensors.op("Sz", site)
    id = ITensors.op("Id", site)
    n_op = sz + 0.5 * id
    h = omega * sx - delta * n_op
    return exp(-1im * dt * h)
end

function pair_gate(site_i, site_j, strength::Float64, dt::Float64)
    n_i = ITensors.op("Sz", site_i) + 0.5 * ITensors.op("Id", site_i)
    n_j = ITensors.op("Sz", site_j) + 0.5 * ITensors.op("Id", site_j)
    h = strength * n_i * n_j
    return exp(-1im * dt * h)
end

function record_observables!(
    psi_bpc,
    lattice,
    t::Float64,
    observables,
    measurement_alg::String,
    measurement_bond_dim::Int,
    chi_2d_prime,
    cutoff::Float64,
    normalize_tensors::Bool,
    obs_sigma_z,
    obs_czz_centerline,
    recorded_times,
)
    push!(recorded_times, t)
    measure_bpc = measurement_cache(psi_bpc, chi_2d_prime, cutoff, normalize_tensors)
    need_sigma = ("sigma_z" in observables) || ("z_i" in observables) || ("czz_centerline" in observables)
    sigma = need_sigma ? sigma_z_2d(measure_bpc, lattice, measurement_alg, measurement_bond_dim) : Float64[]
    if "sigma_z" in observables || "z_i" in observables
        push!(obs_sigma_z, sigma)
    end
    if "czz_centerline" in observables
        push!(obs_czz_centerline, centerline_connected_zz(measure_bpc, lattice, sigma, measurement_alg, measurement_bond_dim))
    end
end

function measurement_cache(psi_bpc, chi_2d_prime, cutoff::Float64, normalize_tensors::Bool)
    isnothing(chi_2d_prime) && return psi_bpc
    isempty(collect(edges(psi_bpc))) && return psi_bpc
    return truncate(psi_bpc; maxdim = Int(chi_2d_prime), cutoff = cutoff, normalize_tensors = normalize_tensors)
end

function sigma_z_2d(psi_bpc, lattice, measurement_alg::String, measurement_bond_dim::Int)
    Ly = Int(lattice.Ly)
    n_sites = Int(lattice.N)
    values = zeros(Float64, n_sites)
    for i0 in 0:(n_sites - 1)
        v = vertex_from_row_major(i0, Ly)
        values[i0 + 1] = measure_scalar(psi_bpc, ("Z", v), measurement_alg, measurement_bond_dim)
    end
    return values
end

function centerline_connected_zz(psi_bpc, lattice, sigma_z, measurement_alg::String, measurement_bond_dim::Int)
    Ly = Int(lattice.Ly)
    pairs = centerline_pairs(Int(lattice.Lx), Ly)
    if isempty(pairs)
        return Float64[]
    end
    values = Float64[]
    for (i_2d, j_2d) in pairs
        vi = vertex_from_row_major(i_2d, Ly)
        vj = vertex_from_row_major(j_2d, Ly)
        zz = measure_scalar(psi_bpc, ("ZZ", [vi, vj]), measurement_alg, measurement_bond_dim)
        push!(values, zz - sigma_z[i_2d + 1] * sigma_z[j_2d + 1])
    end
    return values
end

function measure_scalar(psi_bpc, obs, measurement_alg::String, measurement_bond_dim::Int)
    if measurement_alg == "bp"
        return Float64(real(expect(psi_bpc, obs; alg = "bp")))
    elseif measurement_alg == "boundarymps"
        return Float64(real(expect(network(psi_bpc), obs; alg = "boundarymps", mps_bond_dimension = measurement_bond_dim)))
    elseif measurement_alg == "exact"
        return Float64(real(expect(network(psi_bpc), obs; alg = "exact")))
    end
    error("Unsupported measurement_alg: $measurement_alg")
end

function centerline_pairs(Lx::Int, Ly::Int)
    ix = div(Lx - 1, 2)
    iy0 = div(Ly - 1, 2)
    ref = ix * Ly + iy0
    pairs = Tuple{Int, Int}[]
    for iy in 0:(Ly - 1)
        site = ix * Ly + iy
        if site != ref
            push!(pairs, (ref, site))
        end
    end
    return pairs
end

function vertex_from_snake(pos1::Int, snake_to_2d, Ly::Int)
    return vertex_from_row_major(Int(snake_to_2d[pos1]), Ly)
end

function vertex_from_row_major(index0::Int, Ly::Int)
    return (div(index0, Ly) + 1, mod(index0, Ly) + 1)
end

function row_major_index0(v, Ly::Int)
    return (Int(v[1]) - 1) * Ly + (Int(v[2]) - 1)
end

function _stack_rows(rows::Vector{Vector{Float64}}, n_cols::Int)
    out = zeros(Float64, length(rows), n_cols)
    for i in eachindex(rows)
        out[i, :] = rows[i]
    end
    return out
end

function _property_or(obj, key::Symbol, default)
    return key in propertynames(obj) ? getproperty(obj, key) : default
end

function _optional_int(value)
    isnothing(value) && return nothing
    value === missing && return nothing
    return Int(value)
end

function _state_eltype(runtime, use_cuda::Bool)
    value = String(_property_or(runtime, :eltype, use_cuda ? "ComplexF32" : "ComplexF64"))
    value == "ComplexF32" && return ComplexF32
    value == "ComplexF64" && return ComplexF64
    value == "Float32" && return Float32
    value == "Float64" && return Float64
    error("Unsupported runtime.eltype: $value")
end

main()
