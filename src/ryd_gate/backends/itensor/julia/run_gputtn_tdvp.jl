#!/usr/bin/env julia

using CUDA
using ITensorNetworks
using ITensors
using JSON3
using LinearAlgebra
using NPZ
using NamedGraphs
using Statistics
using TensorOperations

function main()
    if length(ARGS) != 3
        error("Usage: run_gputtn_tdvp.jl payload.json result.npz result.json")
    end
    input_json, output_npz, output_json = ARGS
    payload = JSON3.read(read(input_json, String))
    result = run_gputtn_tdvp(payload)
    NPZ.npzwrite(output_npz, result["arrays"])
    open(output_json, "w") do io
        JSON3.write(io, result["metadata"])
    end
end

function run_gputtn_tdvp(payload)
    runtime = payload.runtime
    lattice = payload.lattice
    n_sites = Int(lattice.N)
    dt = Float64(runtime.dt)
    chi_max = Int(runtime.chi_max)
    cutoff = Float64(runtime.svd_min)
    use_cuda = Bool(runtime.use_cuda)
    require_gpu = hasproperty(runtime, :require_gpu) ? Bool(runtime.require_gpu) : use_cuda
    rk_order = hasproperty(runtime, :rk_order) ? Int(runtime.rk_order) : 4
    tdvp_nsites = hasproperty(runtime, :tdvp_nsites) ? Int(runtime.tdvp_nsites) : 2

    if use_cuda || require_gpu
        if !CUDA.functional()
            error("runtime.use_cuda=true or runtime.require_gpu=true but CUDA.functional() is false")
        end
    end

    graph = balanced_physical_tree(n_sites)
    root_vertex = n_sites == 1 ? 1 : 1
    sites = ITensorNetworks.siteinds("S=1/2", graph)
    initial = [Int(x) == 1 ? "Up" : "Dn" for x in payload.initial_occupations_1d]
    psi = ITensorNetworks.ttn(v -> initial[Int(v)], sites)

    if use_cuda
        psi = cu(psi)
    end

    schedule = payload.schedule
    record_steps = Set(Int(x) for x in payload.record_steps)
    observables = Set(String(x) for x in payload.observables)

    obs_sigma_z = Vector{Vector{Float64}}()
    obs_z_i = Vector{Vector{Float64}}()
    obs_n_i = Vector{Vector{Float64}}()
    obs_n_r = Vector{Vector{Float64}}()
    obs_n_mean = Float64[]
    obs_m_s = Float64[]
    obs_czz_centerline = Vector{Vector{Float64}}()
    recorded_times = Float64[]

    if 0 in record_steps
        record_observables!(
            psi,
            sites,
            lattice,
            0.0,
            observables,
            obs_sigma_z,
            obs_z_i,
            obs_n_i,
            obs_n_r,
            obs_n_mean,
            obs_m_s,
            obs_czz_centerline,
            recorded_times,
        )
    end

    gpu_storage_type = use_cuda ? string(typeof(storage(first(collect(values(psi)))))) : "none"

    for step_data in schedule
        H = hamiltonian_ttn(payload, sites, step_data, root_vertex)
        if use_cuda
            H = cu(H)
        end
        psi = ITensorNetworks.time_evolve(
            H,
            [0.0, dt],
            psi;
            cutoff = cutoff,
            maxdim = chi_max,
            order = rk_order,
            nsites = tdvp_nsites,
            sweep_callback = _ -> nothing,
        )
        psi = psi / norm(psi)

        step = Int(step_data.step)
        if step in record_steps
            record_observables!(
                psi,
                sites,
                lattice,
                step * dt,
                observables,
                obs_sigma_z,
                obs_z_i,
                obs_n_i,
                obs_n_r,
                obs_n_mean,
                obs_m_s,
                obs_czz_centerline,
                recorded_times,
            )
        end
    end

    psi_cpu = cpu_ttn(psi)
    final_sigma_z = sigma_z_2d(psi_cpu, sites, lattice)
    arrays = Dict{String, Any}(
        "times" => recorded_times,
        "final_sigma_z" => final_sigma_z,
    )
    add_obs_array!(arrays, "obs_sigma_z", obs_sigma_z, n_sites)
    add_obs_array!(arrays, "obs_z_i", obs_z_i, n_sites)
    add_obs_array!(arrays, "obs_n_i", obs_n_i, n_sites)
    add_obs_array!(arrays, "obs_n_r", obs_n_r, n_sites)
    if !isempty(obs_n_mean)
        arrays["obs_n_mean"] = obs_n_mean
    end
    if !isempty(obs_m_s)
        arrays["obs_m_s"] = obs_m_s
    end
    if !isempty(obs_czz_centerline)
        n_cols = length(obs_czz_centerline[1])
        arrays["obs_czz_centerline"] = _stack_rows(obs_czz_centerline, n_cols)
    end

    metadata = Dict{String, Any}(
        "backend" => "gputtn",
        "method" => "gputtn_tdvp",
        "engine_package" => "ITensorNetworks.jl",
        "n_sites" => n_sites,
        "chi_max" => chi_max,
        "dt" => dt,
        "n_steps" => length(schedule),
        "svd_min" => cutoff,
        "rk_order" => rk_order,
        "tdvp_nsites" => tdvp_nsites,
        "tree" => "balanced_physical",
        "use_cuda" => use_cuda,
        "gpu" => use_cuda,
        "accelerator" => use_cuda ? "cuda" : "cpu",
        "cuda_functional" => CUDA.functional(),
        "gpu_storage_type" => gpu_storage_type,
        "state_serialized" => false,
    )
    return Dict("arrays" => arrays, "metadata" => metadata)
end

function balanced_physical_tree(n_sites::Int)
    if n_sites < 1
        error("gputtn requires at least one site")
    end
    graph = NamedGraph(1:n_sites)
    if n_sites == 1
        return graph
    end

    function attach!(parent::Int, vertices::Vector{Int})
        isempty(vertices) && return
        mid = cld(length(vertices), 2)
        child = vertices[mid]
        NamedGraphs.add_edge!(graph, parent, child)
        attach!(child, vertices[1:(mid - 1)])
        attach!(child, vertices[(mid + 1):end])
    end

    attach!(1, collect(2:n_sites))
    return graph
end

function hamiltonian_ttn(payload, sites, step_data, root_vertex::Int)
    lattice = payload.lattice
    os = ITensors.OpSum()
    omega = Float64.(collect(step_data.omega_1d))
    delta = Float64.(collect(step_data.delta_1d))
    for i in 1:Int(lattice.N)
        if abs(omega[i]) > 0
            os += omega[i], "Sx", i
        end
        if abs(delta[i]) > 0
            os += -delta[i], "Sz", i
        end
    end
    for pair in lattice.vdw_pairs_1d
        i = Int(pair[1])
        j = Int(pair[2])
        strength = Float64(pair[3])
        if i != j && abs(strength) > 0
            os += strength, "Sz", i, "Sz", j
            os += 0.5 * strength, "Sz", i
            os += 0.5 * strength, "Sz", j
        end
    end
    return ITensorNetworks.ttn(ComplexF64, os, sites; root_vertex = root_vertex)
end

function cpu_ttn(psi)
    return ITensorNetworks.map_vertex_data_preserve_graph(ITensors.cpu, psi)
end

function record_observables!(
    psi,
    sites,
    lattice,
    t::Float64,
    observables,
    obs_sigma_z,
    obs_z_i,
    obs_n_i,
    obs_n_r,
    obs_n_mean,
    obs_m_s,
    obs_czz_centerline,
    recorded_times,
)
    push!(recorded_times, t)
    psi_cpu = cpu_ttn(psi)
    sigma_z = sigma_z_2d(psi_cpu, sites, lattice)
    n_i = 0.5 .* (sigma_z .+ 1.0)
    if "sigma_z" in observables
        push!(obs_sigma_z, sigma_z)
    end
    if "z_i" in observables
        push!(obs_z_i, sigma_z)
    end
    if "n_i" in observables
        push!(obs_n_i, n_i)
    end
    if "n_r" in observables
        push!(obs_n_r, n_i)
    end
    if "n_mean" in observables
        push!(obs_n_mean, mean(n_i))
    end
    if "m_s" in observables
        sublattice = Float64.(collect(lattice.sublattice))
        push!(obs_m_s, sum(sublattice .* sigma_z) / Float64(lattice.N))
    end
    if "czz_centerline" in observables
        push!(obs_czz_centerline, centerline_connected_zz(psi_cpu, sites, lattice, sigma_z))
    end
end

function sigma_z_2d(psi_cpu, sites, lattice)
    sz = local_sz_expectations(psi_cpu, sites, Int(lattice.N))
    sigma_2d = zeros(Float64, Int(lattice.N))
    snake_to_2d = Int.(collect(lattice.snake_to_2d))
    for pos in 1:Int(lattice.N)
        sigma_2d[snake_to_2d[pos] + 1] = 2.0 * sz[pos]
    end
    return sigma_2d
end

function local_sz_expectations(psi_cpu, sites, n_sites::Int)
    state = psi_cpu / norm(psi_cpu)
    out = zeros(Float64, n_sites)
    for v in 1:n_sites
        state = ITensorNetworks.orthogonalize(state, v)
        op_v = ITensors.op("Sz", only(sites[v]))
        out[v] = real((dag(state[v]) * ITensors.apply(op_v, state[v]))[])
    end
    return out
end

function centerline_connected_zz(psi_cpu, sites, lattice, sigma_z_2d)
    pairs = centerline_pairs(Int(lattice.Lx), Int(lattice.Ly))
    isempty(pairs) && return Float64[]
    inv_snake = Int.(collect(lattice.inv_snake))
    norm2 = inner(ITensorNetworks.Algorithm("exact"), psi_cpu, psi_cpu)
    values = Float64[]
    for (i_2d, j_2d) in pairs
        i = inv_snake[i_2d + 1] + 1
        j = inv_snake[j_2d + 1] + 1
        os = ITensors.OpSum()
        os += 1.0, "Sz", i, "Sz", j
        O = ITensorNetworks.ttn(ComplexF64, os, sites; root_vertex = 1)
        zz = real(inner(ITensorNetworks.Algorithm("exact"), psi_cpu, O, psi_cpu) / norm2)
        sz_i = 0.5 * sigma_z_2d[i_2d + 1]
        sz_j = 0.5 * sigma_z_2d[j_2d + 1]
        push!(values, 4.0 * (zz - sz_i * sz_j))
    end
    return values
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

function add_obs_array!(arrays, name::String, rows::Vector{Vector{Float64}}, n_cols::Int)
    if !isempty(rows)
        arrays[name] = _stack_rows(rows, n_cols)
    end
end

function _stack_rows(rows::Vector{Vector{Float64}}, n_cols::Int)
    out = zeros(Float64, length(rows), n_cols)
    for i in eachindex(rows)
        out[i, :] = rows[i]
    end
    return out
end

main()
