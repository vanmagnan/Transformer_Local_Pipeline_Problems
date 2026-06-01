include("constants.jl")

"""
Variable-n Ramsey colorings of Kneser graphs KG(n, R_KNESER).

KG(n,r) has vertices = all r-subsets of {1..n}, edges between disjoint pairs.
A p-clique in KG(n,r) is p pairwise-disjoint r-subsets (requires p*r ≤ n).

The search grows n by adding ground element n+1, which adds C(n, r-1) new Kneser
vertices — all mutually non-adjacent (they share n+1). SAT colors the new edges
independently per new vertex. Reward = n for a valid coloring of KG(n,r).

Default parameters: r=2, p=3, q=4.
"""

using Random
using Combinatorics
using PicoSAT

const R_KNESER = 2   # r-subset size
const P_KNESER = 3   # avoid monochromatic red K_P in KG
const Q_KNESER = 4   # avoid monochromatic blue K_Q in KG
const N_START  = 10  # starting ground set size
const MAX_N_KNESER = 40  # upper bound for valid_line precomputation

# Precomputed table: string length → n.
# String length for KG(n,r) = NE(n,r) + NV(n,r) - 1
# NV(n,r) = C(n,r),  NE(n,r) = C(n,2r)*C(2r,r)/2
const KNESER_VALID_LENGTHS = let
    d = Dict{Int,Int}()
    for n in (R_KNESER+1):MAX_N_KNESER
        nv = binomial(n, R_KNESER)
        ne = binomial(n, 2*R_KNESER) * binomial(2*R_KNESER, R_KNESER) ÷ 2
        d[ne + nv - 1] = n
    end
    d
end

# Override search_fc.jl default: accept any string length corresponding to a valid KG(n,r)
function valid_line(line::String)
    haskey(KNESER_VALID_LENGTHS, length(line)) || return false
    all(c in ('0', '1', '2') for c in line) || return false
    n  = KNESER_VALID_LENGTHS[length(line)]
    nv = binomial(n, R_KNESER)
    count(==('2'), line) == nv - 1 || return false
    return true
end

# ---------------------------------------------------------------------------
# Core graph construction
# Returns (vertices, edges, edge_index, forward_edges)
#   vertices[i]  : i-th r-subset of {1..n} in lex order (Vector{Int})
#   edges        : Vector of (i,j) pairs, i<j, with vertices[i]∩vertices[j]=∅
#   edge_index   : Dict (i,j) → edge position
#   fwd[i]       : edge indices of edges (i,j) with j>i (for encoding)
# ---------------------------------------------------------------------------
function build_kneser(n::Int, r::Int)
    vertices = [collect(c) for c in combinations(1:n, r)]
    nv = length(vertices)
    edges = Tuple{Int,Int}[]
    for i in 1:nv-1, j in i+1:nv
        isempty(intersect(vertices[i], vertices[j])) && push!(edges, (i, j))
    end
    eidx = Dict{Tuple{Int,Int},Int}(e => k for (k, e) in enumerate(edges))
    fwd  = [Int[] for _ in 1:nv]
    for (k, (i, j)) in enumerate(edges)
        push!(fwd[i], k)
    end
    return vertices, edges, eidx, fwd
end

# ---------------------------------------------------------------------------
# Encoding / decoding
# ---------------------------------------------------------------------------
function encode_kneser(coloring::Vector{Int}, fwd::Vector{Vector{Int}}, nv::Int)::String
    buf = IOBuffer()
    for i in 1:nv-1
        for ei in fwd[i]; write(buf, Char('0' + coloring[ei])); end
        write(buf, '2')
    end
    return String(take!(buf))
end

function decode_kneser(s::String)
    n       = KNESER_VALID_LENGTHS[length(s)]
    coloring = [Int(c) - Int('0') for c in s if c != '2']
    return coloring, n
end

# Alias required by search_fc.jl / tests
function convert_adjmat_to_string(coloring::Vector{Int}, n::Int)::String
    _, _, _, fwd = build_kneser(n, R_KNESER)
    return encode_kneser(coloring, fwd, binomial(n, R_KNESER))
end

function empty_starting_point()::OBJ_TYPE
    _, edges, _, fwd = build_kneser(N_START, R_KNESER)
    coloring = rand(0:1, length(edges))
    return encode_kneser(coloring, fwd, binomial(N_START, R_KNESER))
end

# ---------------------------------------------------------------------------
# Clique finding
# Find all monochromatic cliques of size k of given color among `subset`
# (a vector of vertex indices, sorted).
# ---------------------------------------------------------------------------
function find_color_cliques(subset::Vector{Int}, k::Int, color::Int,
                             coloring::Vector{Int},
                             eidx::Dict{Tuple{Int,Int},Int})::Vector{Vector{Int}}
    k <= 0 && return [Int[]]
    k > length(subset) && return Vector{Int}[]
    result = Vector{Int}[]
    clique  = Int[]
    function bt(start::Int)
        if length(clique) == k
            push!(result, copy(clique))
            return
        end
        remaining = k - length(clique)
        for i in start:length(subset)-(remaining-1)
            v = subset[i]
            ok = true
            for u in clique
                key = u < v ? (u, v) : (v, u)
                ei  = get(eidx, key, 0)
                if ei == 0 || coloring[ei] != color
                    ok = false; break
                end
            end
            ok || continue
            push!(clique, v); bt(i + 1); pop!(clique)
        end
    end
    bt(1)
    return result
end

# ---------------------------------------------------------------------------
# Violation checking
# Does vertex vi participate in any forbidden clique in the current subgraph?
# `deleted[e]` = true means ground element e has been removed.
# ---------------------------------------------------------------------------
function vertex_violates(vi::Int, verts, eidx, coloring::Vector{Int},
                          deleted::BitVector)::Bool
    red_nbrs  = Int[]
    blue_nbrs = Int[]
    for u in 1:length(verts)
        u == vi && continue
        any(e -> deleted[e], verts[u]) && continue
        isempty(intersect(verts[vi], verts[u])) || continue
        key = (min(vi, u), max(vi, u))
        ei  = get(eidx, key, 0); ei == 0 && continue
        coloring[ei] == 0 ? push!(red_nbrs, u) : push!(blue_nbrs, u)
    end
    !isempty(find_color_cliques(red_nbrs,  P_KNESER-1, 0, coloring, eidx)) && return true
    !isempty(find_color_cliques(blue_nbrs, Q_KNESER-1, 1, coloring, eidx)) && return true
    return false
end

# ---------------------------------------------------------------------------
# Deletion phase
# Single pass through vertices (shuffled). For each surviving violated vertex,
# delete a random ground element it contains. Rebuild coloring for new KG(m,r).
# ---------------------------------------------------------------------------
function deletion_phase(coloring::Vector{Int}, n::Int)
    verts, edges, eidx, _ = build_kneser(n, R_KNESER)
    nv      = length(verts)
    deleted = falses(n)

    for vi in shuffle(1:nv)
        any(e -> deleted[e], verts[vi]) && continue
        vertex_violates(vi, verts, eidx, coloring, deleted) || continue
        deleted[rand(verts[vi])] = true
    end

    !any(deleted) && return coloring, n

    # Rebuild for the surviving ground elements, relabeled 1..m
    surviving = [e for e in 1:n if !deleted[e]]
    m        = length(surviving)
    elem_map = Dict(old => new for (new, old) in enumerate(surviving))

    new_verts, new_edges, new_eidx, new_fwd = build_kneser(m, R_KNESER)

    # Map surviving old vertices → new vertex indices
    old_to_new = Dict{Int,Int}()
    for vi in 1:nv
        any(e -> deleted[e], verts[vi]) && continue
        relabeled = sort([elem_map[e] for e in verts[vi]])
        nvi = findfirst(==(relabeled), new_verts)
        old_to_new[vi] = nvi
    end
    new_to_old = Dict(nvi => vi for (vi, nvi) in old_to_new)

    new_coloring = zeros(Int, length(new_edges))
    for (new_ei, (i, j)) in enumerate(new_edges)
        old_i  = new_to_old[i]
        old_j  = new_to_old[j]
        old_ei = eidx[(min(old_i, old_j), max(old_i, old_j))]
        new_coloring[new_ei] = coloring[old_ei]
    end

    return new_coloring, m
end

# ---------------------------------------------------------------------------
# SAT-based coloring for a single new vertex v_{S'∪{n+1}} in KG(n+1,r).
# S_prime: the (r-1)-subset of {1..n} for this new vertex.
# Returns (x, nbrs) where x[k] = color of edge (nbrs[k], new_vertex),
# or (nothing, nbrs) if no valid coloring exists.
# ---------------------------------------------------------------------------
function color_new_vertex(S_prime::Vector{Int}, old_verts, old_eidx,
                           coloring::Vector{Int})
    nbrs = [u for u in 1:length(old_verts)
              if isempty(intersect(old_verts[u], S_prime))]
    isempty(nbrs) && return zeros(Int, 0), nbrs

    var_map = Dict(u => i for (i, u) in enumerate(nbrs))
    nvar    = length(nbrs)
    clauses = Vector{Int}[]

    # Each red (P-1)-clique in old graph within nbrs forces ≥1 blue edge to new vertex
    for clique in find_color_cliques(nbrs, P_KNESER-1, 0, coloring, old_eidx)
        push!(clauses, [var_map[u] for u in clique])   # positive literal = blue
    end
    # Each blue (Q-1)-clique forces ≥1 red edge
    for clique in find_color_cliques(nbrs, Q_KNESER-1, 1, coloring, old_eidx)
        push!(clauses, [-var_map[u] for u in clique])  # negative literal = red
    end

    # Try a random assignment first
    x = rand(0:1, nvar)
    rand_ok = all(clauses) do clause
        any(clause) do lit
            lit > 0 ? x[lit] == 1 : x[-lit] == 0
        end
    end
    rand_ok && return x, nbrs

    # Fall back to SAT with random phase hints
    hint = rand(Bool, nvar)
    hint_clauses = [[hint[i] ? i : -i] for i in 1:nvar]
    result = PicoSAT.solve(vcat(clauses, hint_clauses); vars=nvar)
    if result !== :unsatisfiable
        return [result[i] > 0 ? 1 : 0 for i in 1:nvar], nbrs
    end

    # Hints caused UNSAT — try without hints
    isempty(clauses) && return rand(0:1, nvar), nbrs
    result = PicoSAT.solve(clauses; vars=nvar)
    result === :unsatisfiable && return nothing, nbrs
    return [result[i] > 0 ? 1 : 0 for i in 1:nvar], nbrs
end

# ---------------------------------------------------------------------------
# Growth phase
# Repeatedly try to extend KG(n,r) → KG(n+1,r) by coloring the C(n,r-1) new
# vertices. New vertices are an independent set, so each is solved independently.
# Probe one random new vertex first; if UNSAT, stop immediately.
# ---------------------------------------------------------------------------
function growth_phase(coloring::Vector{Int}, n::Int, iters::Int)
    for _ in 1:iters
        old_verts, old_edges, old_eidx, _ = build_kneser(n, R_KNESER)
        old_nv   = length(old_verts)
        s_primes = [collect(c) for c in combinations(1:n, R_KNESER-1)]
        ns       = length(s_primes)

        probe = rand(1:ns)
        x_p, nbrs_p = color_new_vertex(s_primes[probe], old_verts, old_eidx, coloring)
        x_p === nothing && break

        results = Dict{Int,Tuple{Vector{Int},Vector{Int}}}(probe => (x_p, nbrs_p))
        failed  = false
        for idx in 1:ns
            idx == probe && continue
            x, nbrs = color_new_vertex(s_primes[idx], old_verts, old_eidx, coloring)
            if x === nothing; failed = true; break; end
            results[idx] = (x, nbrs)
        end
        failed && break

        new_verts, new_edges, new_eidx, _ = build_kneser(n+1, R_KNESER)
        new_coloring = zeros(Int, length(new_edges))
        old_to_new   = [findfirst(==(old_verts[i]), new_verts) for i in 1:old_nv]

        for (old_ei, (i, j)) in enumerate(old_edges)
            ni, nj = old_to_new[i], old_to_new[j]
            new_coloring[new_eidx[(min(ni,nj), max(ni,nj))]] = coloring[old_ei]
        end
        for (idx, (x, nbrs)) in results
            new_v = findfirst(==(sort([s_primes[idx]..., n+1])), new_verts)
            for (k, u) in enumerate(nbrs)
                nu  = old_to_new[u]
                new_coloring[new_eidx[(min(nu,new_v), max(nu,new_v))]] = x[k]
            end
        end

        coloring = new_coloring
        n += 1
    end
    return coloring, n
end

# ---------------------------------------------------------------------------
# Required interface for search_fc.jl
# ---------------------------------------------------------------------------
function reward_calc(obj::OBJ_TYPE)::REWARD_TYPE
    valid_line(obj) || return Float32(0)
    return Float32(KNESER_VALID_LENGTHS[length(obj)])
end

function greedy_search_from_startpoint(db, obj::OBJ_TYPE)::Vector{OBJ_TYPE}
    valid_line(obj) || return [empty_starting_point()]
    coloring, n = decode_kneser(obj)
    coloring, n = deletion_phase(coloring, n)
    coloring, n = growth_phase(coloring, n, max_search_iter)
    _, _, _, fwd = build_kneser(n, R_KNESER)
    return [encode_kneser(coloring, fwd, binomial(n, R_KNESER))]
end
