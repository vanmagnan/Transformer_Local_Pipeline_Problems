include("constants.jl")

"""
Find 2-colorings (red/blue) of edges of the Kneser graph K(n,r) avoiding a
red p-clique or a blue q-clique.

The Kneser graph K(n,r) has vertices = all r-subsets of {1,...,n}, with edges
between disjoint pairs. A p-clique in K(n,r) consists of p pairwise-disjoint
r-subsets (requires p*r ≤ n).

Default parameters: n=12, r=2, p=3, q=4.
  NV = C(12,2) = 66 vertices
  NE = 1485 edges
  String length = NE + NV - 1 = 1550

Object encoding: String of '0'/'1' with '2' row separators.
For vertex i = 1..NV-1: emit colors for edges (i,j) with j>i, then '2'.
0 = red, 1 = blue.
"""

using Random
using Combinatorics

const N_KNESER = 12
const R_KNESER = 2
const P_KNESER = 3
const Q_KNESER = 4

# --- Vertices: all r-subsets of {1..n} in lex order ---
const VERTICES = let
    verts = [collect(c) for c in combinations(1:N_KNESER, R_KNESER)]
    verts
end
const NV = length(VERTICES)  # C(12,2) = 66

# --- Edges: pairs (i,j) with i<j and VERTICES[i] ∩ VERTICES[j] = ∅ ---
const EDGES = let
    edges = Tuple{Int,Int}[]
    for i in 1:NV-1
        for j in i+1:NV
            if isempty(intersect(VERTICES[i], VERTICES[j]))
                push!(edges, (i, j))
            end
        end
    end
    edges
end
const NE = length(EDGES)  # 1485

# O(1) edge lookup by (i,j) vertex pair
const EDGE_INDEX = let
    d = Dict{Tuple{Int,Int},Int}()
    for (idx, e) in enumerate(EDGES)
        d[e] = idx
    end
    d
end

# For each vertex i: edge indices of edges (i,j) with j > i (forward edges used in encoding)
const VERTEX_FORWARD_EDGES = let
    fwd = [Int[] for _ in 1:NV]
    for (idx, (i, j)) in enumerate(EDGES)
        push!(fwd[i], idx)
    end
    fwd
end

# Expected string length: NE edge chars + (NV-1) row separators '2'
const STRING_LENGTH = NE + NV - 1  # 1550

# --- P-cliques: sets of P_KNESER pairwise-disjoint vertices ---
const P_CLIQUES = let
    cliques = Vector{Int}[]
    for verts in combinations(1:NV, P_KNESER)
        ok = true
        for a in 1:P_KNESER-1, b in a+1:P_KNESER
            if !isempty(intersect(VERTICES[verts[a]], VERTICES[verts[b]]))
                ok = false
                break
            end
        end
        ok && push!(cliques, verts)
    end
    cliques
end
const NPC = length(P_CLIQUES)

# For each p-clique: its C(P,2) edge indices
const P_CLIQUE_EDGES = let
    result = Vector{Int}[]
    for verts in P_CLIQUES
        edges_in_clique = Int[]
        for a in 1:P_KNESER-1
            for b in a+1:P_KNESER
                u, v = verts[a], verts[b]
                key = u < v ? (u, v) : (v, u)
                push!(edges_in_clique, EDGE_INDEX[key])
            end
        end
        push!(result, edges_in_clique)
    end
    result
end

# --- Q-cliques: sets of Q_KNESER pairwise-disjoint vertices ---
const Q_CLIQUES = let
    cliques = Vector{Int}[]
    for verts in combinations(1:NV, Q_KNESER)
        ok = true
        for a in 1:Q_KNESER-1, b in a+1:Q_KNESER
            if !isempty(intersect(VERTICES[verts[a]], VERTICES[verts[b]]))
                ok = false
                break
            end
        end
        ok && push!(cliques, verts)
    end
    cliques
end
const NQC = length(Q_CLIQUES)

# For each q-clique: its C(Q,2) edge indices
const Q_CLIQUE_EDGES = let
    result = Vector{Int}[]
    for verts in Q_CLIQUES
        edges_in_clique = Int[]
        for a in 1:Q_KNESER-1
            for b in a+1:Q_KNESER
                u, v = verts[a], verts[b]
                key = u < v ? (u, v) : (v, u)
                push!(edges_in_clique, EDGE_INDEX[key])
            end
        end
        push!(result, edges_in_clique)
    end
    result
end

# For each edge: p-clique indices containing it
const EDGE_P_CLIQUES = let
    lookup = [Int[] for _ in 1:NE]
    for (ci, edges) in enumerate(P_CLIQUE_EDGES)
        for ei in edges
            push!(lookup[ei], ci)
        end
    end
    lookup
end

# For each edge: q-clique indices containing it
const EDGE_Q_CLIQUES = let
    lookup = [Int[] for _ in 1:NE]
    for (ci, edges) in enumerate(Q_CLIQUE_EDGES)
        for ei in edges
            push!(lookup[ei], ci)
        end
    end
    lookup
end

const NP_EDGES = P_KNESER * (P_KNESER - 1) ÷ 2  # 3
const NQ_EDGES = Q_KNESER * (Q_KNESER - 1) ÷ 2  # 6

# --- String encoding ---
function convert_coloring_to_string(coloring::Vector{Int})::String
    buf = IOBuffer()
    for i in 1:NV-1
        for ei in VERTEX_FORWARD_EDGES[i]
            write(buf, Char('0' + coloring[ei]))
        end
        write(buf, '2')
    end
    return String(take!(buf))
end

# Alias required by search_fc.jl
function convert_adjmat_to_string(coloring::Vector{Int})::String
    return convert_coloring_to_string(coloring)
end

function empty_starting_point()::OBJ_TYPE
    coloring = rand(0:1, NE)
    return convert_coloring_to_string(coloring)
end

function reward_calc(obj::OBJ_TYPE)::REWARD_TYPE
    non_sep = [c for c in obj if c != '2']
    length(non_sep) != NE && return REWARD_TYPE(-1e9)
    coloring = [Int(c) - Int('0') for c in non_sep]

    total = 0
    for edges in P_CLIQUE_EDGES
        all(coloring[e] == 0 for e in edges) && (total += 1)
    end
    for edges in Q_CLIQUE_EDGES
        all(coloring[e] == 1 for e in edges) && (total += 1)
    end
    return -Float32(total)
end

function greedy_search_from_startpoint(db, obj::OBJ_TYPE)::Vector{OBJ_TYPE}
    # Validate: must have exactly NV-1 row separators
    num_twos = count(c -> c == '2', obj)
    if num_twos != NV - 1
        return greedy_search_from_startpoint(db, empty_starting_point())
    end

    non_sep = [c for c in obj if c != '2']
    if length(non_sep) != NE
        return greedy_search_from_startpoint(db, empty_starting_point())
    end

    coloring = [Int(c) - Int('0') for c in non_sep]
    if any(x -> x < 0 || x > 1, coloring)
        return greedy_search_from_startpoint(db, empty_starting_point())
    end

    # --- Incremental state initialization ---
    red_in_p  = zeros(Int, NPC)
    blue_in_q = zeros(Int, NQC)

    for (ci, edges) in enumerate(P_CLIQUE_EDGES)
        for ei in edges
            coloring[ei] == 0 && (red_in_p[ci] += 1)
        end
    end
    for (ci, edges) in enumerate(Q_CLIQUE_EDGES)
        for ei in edges
            coloring[ei] == 1 && (blue_in_q[ci] += 1)
        end
    end
    total_forbidden = count(==(NP_EDGES), red_in_p) + count(==(NQ_EDGES), blue_in_q)

    total_forbidden == 0 && return [convert_coloring_to_string(coloring)]

    # --- Simulated annealing ---
    T = 1.0
    cooling_rate = (1e-4)^(1.0 / max_search_iter)
    edge_order = shuffle!(collect(1:NE))

    pass = 0
    while total_forbidden > 0 && pass < max_search_iter
        pass += 1
        shuffle!(edge_order)

        for ei in edge_order
            old_color = coloring[ei]

            # Compute delta: positive = worse, negative = better
            delta = 0
            if old_color == 0  # red → blue: destroys red p-cliques, may create blue q-cliques
                for ci in EDGE_P_CLIQUES[ei]
                    red_in_p[ci] == NP_EDGES && (delta -= 1)
                end
                for ci in EDGE_Q_CLIQUES[ei]
                    blue_in_q[ci] == NQ_EDGES - 1 && (delta += 1)
                end
            else               # blue → red: destroys blue q-cliques, may create red p-cliques
                for ci in EDGE_Q_CLIQUES[ei]
                    blue_in_q[ci] == NQ_EDGES && (delta -= 1)
                end
                for ci in EDGE_P_CLIQUES[ei]
                    red_in_p[ci] == NP_EDGES - 1 && (delta += 1)
                end
            end

            if delta <= 0 || rand() < exp(-delta / T)
                # Apply flip
                coloring[ei] = 1 - old_color

                if old_color == 0  # red → blue
                    for ci in EDGE_P_CLIQUES[ei]
                        if red_in_p[ci] == NP_EDGES
                            total_forbidden -= 1
                        end
                        red_in_p[ci] -= 1
                    end
                    for ci in EDGE_Q_CLIQUES[ei]
                        blue_in_q[ci] += 1
                        if blue_in_q[ci] == NQ_EDGES
                            total_forbidden += 1
                        end
                    end
                else               # blue → red
                    for ci in EDGE_Q_CLIQUES[ei]
                        if blue_in_q[ci] == NQ_EDGES
                            total_forbidden -= 1
                        end
                        blue_in_q[ci] -= 1
                    end
                    for ci in EDGE_P_CLIQUES[ei]
                        red_in_p[ci] += 1
                        if red_in_p[ci] == NP_EDGES
                            total_forbidden += 1
                        end
                    end
                end

                total_forbidden == 0 && break
            end
        end

        T *= cooling_rate
    end

    return [convert_coloring_to_string(coloring)]
end
