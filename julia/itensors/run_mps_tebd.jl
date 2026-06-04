#!/usr/bin/env julia

using ITensors
using ITensorMPS
using CUDA
using JSON3
using LinearAlgebra
using NPZ

function main()
    if length(ARGS) != 3
        error("Usage: run_mps_tebd.jl payload.json result.npz result.json")
    end
    input_json, output_npz, output_json = ARGS
    payload = JSON3.read(read(input_json, String))
    result = run_tebd(payload)
    NPZ.npzwrite(output_npz, result["arrays"])
    open(output_json, "w") do io
        JSON3.write(io, result["metadata"])
    end
end

function run_tebd(payload)
    runtime = payload.runtime
    lattice = payload.lattice
    n_sites = Int(lattice.N)
    dt = Float64(runtime.dt)
    chi_max = Int(runtime.chi_max)
    cutoff = Float64(runtime.svd_min)
    use_cuda = Bool(runtime.use_cuda)
    if use_cuda
        if !CUDA.functional()
            error("runtime.use_cuda=true but CUDA.functional() is false")
        end
        @warn "CUDA is available, but this ITensors bridge does not yet move all MPS tensors to GPU storage."
    end

    sites = siteinds("S=1/2", n_sites)
    initial = [Int(x) == 1 ? "Up" : "Dn" for x in payload.initial_occupations_1d]
    psi = productMPS(sites, initial)

    schedule = payload.schedule
    record_steps = Set(Int(x) for x in payload.record_steps)
    observables = Set(String(x) for x in payload.observables)

    obs_sigma_z = Vector{Vector{Float64}}()
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
            obs_czz_centerline,
            recorded_times,
        )
    end

    for step_data in schedule
        omega = Float64.(collect(step_data.omega_1d))
        delta = Float64.(collect(step_data.delta_1d))
        step = Int(step_data.step)

        local_half = [local_gate(sites[i], omega[i], delta[i], dt / 2) for i in 1:n_sites]
        psi = apply(local_half, psi; cutoff=cutoff, maxdim=chi_max)

        pair_gates = ITensor[]
        for pair in lattice.vdw_pairs_1d
            i = Int(pair[1])
            j = Int(pair[2])
            strength = Float64(pair[3])
            if i != j && abs(strength) > 0
                push!(pair_gates, pair_gate(sites[i], sites[j], strength, dt))
            end
        end
        if !isempty(pair_gates)
            psi = apply(pair_gates, psi; cutoff=cutoff, maxdim=chi_max)
        end
        psi = apply(local_half, psi; cutoff=cutoff, maxdim=chi_max)
        normalize!(psi)

        if step in record_steps
            record_observables!(
                psi,
                sites,
                lattice,
                step * dt,
                observables,
                obs_sigma_z,
                obs_czz_centerline,
                recorded_times,
            )
        end
    end

    final_sigma_z = sigma_z_2d(psi, lattice)
    arrays = Dict{String, Any}(
        "times" => recorded_times,
        "final_sigma_z" => final_sigma_z,
    )
    if !isempty(obs_sigma_z)
        arrays["obs_sigma_z"] = _stack_rows(obs_sigma_z, n_sites)
    end
    if !isempty(obs_czz_centerline)
        n_cols = length(obs_czz_centerline[1])
        arrays["obs_czz_centerline"] = _stack_rows(obs_czz_centerline, n_cols)
    end

    metadata = Dict{String, Any}(
        "backend" => "itensors",
        "method" => "itensors_tebd",
        "n_sites" => n_sites,
        "chi_max" => chi_max,
        "dt" => dt,
        "n_steps" => length(schedule),
        "svd_min" => cutoff,
        "use_cuda" => use_cuda,
        "state_serialized" => false,
    )
    return Dict("arrays" => arrays, "metadata" => metadata)
end

function local_gate(site, omega::Float64, delta::Float64, dt::Float64)
    sx = op("Sx", site)
    sz = op("Sz", site)
    id = op("Id", site)
    n_op = sz + 0.5 * id
    h = omega * sx - delta * n_op
    return exp(-1im * dt * h)
end

function pair_gate(site_i, site_j, strength::Float64, dt::Float64)
    n_i = op("Sz", site_i) + 0.5 * op("Id", site_i)
    n_j = op("Sz", site_j) + 0.5 * op("Id", site_j)
    h = strength * n_i * n_j
    return exp(-1im * dt * h)
end

function record_observables!(
    psi,
    sites,
    lattice,
    t::Float64,
    observables,
    obs_sigma_z,
    obs_czz_centerline,
    recorded_times,
)
    push!(recorded_times, t)
    if "sigma_z" in observables || "z_i" in observables
        push!(obs_sigma_z, sigma_z_2d(psi, lattice))
    end
    if "czz_centerline" in observables
        push!(obs_czz_centerline, centerline_connected_zz(psi, lattice))
    end
end

function sigma_z_2d(psi, lattice)
    sz_snake = expect(psi, "Sz")
    sigma_snake = 2.0 .* Float64.(sz_snake)
    sigma_2d = zeros(Float64, Int(lattice.N))
    snake_to_2d = Int.(collect(lattice.snake_to_2d))
    for pos in eachindex(sigma_snake)
        sigma_2d[snake_to_2d[pos] + 1] = sigma_snake[pos]
    end
    return sigma_2d
end

function centerline_connected_zz(psi, lattice)
    pairs = centerline_pairs(Int(lattice.Lx), Int(lattice.Ly))
    if isempty(pairs)
        return Float64[]
    end
    sz = Float64.(expect(psi, "Sz"))
    corr = correlation_matrix(psi, "Sz", "Sz")
    inv_snake = Int.(collect(lattice.inv_snake))
    values = Float64[]
    for (i_2d, j_2d) in pairs
        i = inv_snake[i_2d + 1] + 1
        j = inv_snake[j_2d + 1] + 1
        push!(values, 4.0 * (Float64(corr[i, j]) - sz[i] * sz[j]))
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

function _stack_rows(rows::Vector{Vector{Float64}}, n_cols::Int)
    out = zeros(Float64, length(rows), n_cols)
    for i in eachindex(rows)
        out[i, :] = rows[i]
    end
    return out
end

main()
