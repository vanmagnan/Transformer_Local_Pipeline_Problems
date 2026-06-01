include("constants.jl")

"""
Find 2-colorings (red/blue) of edges of K_N avoiding a red K_P and a blue K_Q.

Targets the asymmetric Ramsey number R(P, Q): the smallest n such that every
2-coloring of K_n contains a red K_P or a blue K_Q. Primary target is R(4,6),
where the exact value is unknown and N=35 is a promising lower-bound frontier.

Uses precomputed clique membership tables for fast incremental updates after
each edge flip: only the C(N-2, P-2) P-cliques and C(N-2, Q-2) Q-cliques
containing the flipped edge need to be rechecked, rather than all C(N,P) and
C(N,Q) cliques.

Object encoding: String of '0'/'1' with '2' row separators.
Row i contains the colors of edges {i,j} for j = i+1,...,N, followed by '2'.
0 = red, 1 = blue.
"""

using Random
using Combinatorics

const N = 17   # number of vertices (R(4,6) lower bound frontier)
const P = 4    # red clique size to avoid
const Q = 4    # blue clique size to avoid
const NUM_EDGES = N * (N - 1) ÷ 2
const EDGES_KP  = P * (P - 1) ÷ 2   # 6 for K4
const EDGES_KQ  = Q * (Q - 1) ÷ 2   # 15 for K6

function edge_index(i::Int, j::Int)::Int
    # Requires i < j, 1-based indexing
    return (i - 1) * (2 * N - i) ÷ 2 + (j - i)
end

# All C(N,P) P-cliques, each stored as a vector of EDGES_KP edge indices.
# Computed once at load time.
const ALL_KPS = let
    kps = Vector{Vector{Int}}()
    for verts in combinations(1:N, P)
        push!(kps, [edge_index(verts[p], verts[q]) for p in 1:P-1 for q in p+1:P])
    end
    kps
end

# For each edge index, the indices of P-cliques (into ALL_KPS) that contain it.
# Computed once at load time.
const KPS_CONTAINING = let
    lookup = [Int[] for _ in 1:NUM_EDGES]
    for (ki, kp_edges) in enumerate(ALL_KPS)
        for ei in kp_edges
            push!(lookup[ei], ki)
        end
    end
    lookup
end

# All C(N,Q) Q-cliques, each stored as a vector of EDGES_KQ edge indices.
# Computed once at load time.
const ALL_KQS = let
    kqs = Vector{Vector{Int}}()
    for verts in combinations(1:N, Q)
        push!(kqs, [edge_index(verts[q1], verts[q2]) for q1 in 1:Q-1 for q2 in q1+1:Q])
    end
    kqs
end

# For each edge index, the indices of Q-cliques (into ALL_KQS) that contain it.
# Computed once at load time.
const KQS_CONTAINING = let
    lookup = [Int[] for _ in 1:NUM_EDGES]
    for (ki, kq_edges) in enumerate(ALL_KQS)
        for ei in kq_edges
            push!(lookup[ei], ki)
        end
    end
    lookup
end

# Row-separated encoding: edges in row i (i.e. {i,j} for j>i) followed by "2".
# Consistent with problem_monochromatic_clique_table.jl. Helps BPE learn row structure.
function convert_adjmat_to_string(coloring::Vector{Int})::String
    entries = []
    for i in 1:N-1
        for j in i+1:N
            push!(entries, string(coloring[edge_index(i, j)]))
        end
        push!(entries, "2")
    end
    return join(entries)
end

function empty_starting_point()::OBJ_TYPE
    # Random coloring so parallel searches start from diverse points
    coloring = [rand(0:1) for _ in 1:NUM_EDGES]
    return convert_adjmat_to_string(coloring)
end

function reward_calc(obj::OBJ_TYPE)::REWARD_TYPE
    coloring = [parse(Int, c) for c in obj if c != '2']
    total = 0
    for kp_edges in ALL_KPS
        if all(coloring[e] == 0 for e in kp_edges)   # all-red K_P
            total += 1
        end
    end
    for kq_edges in ALL_KQS
        if all(coloring[e] == 1 for e in kq_edges)   # all-blue K_Q
            total += 1
        end
    end
    return -Float32(total)
end

function greedy_search_from_startpoint(db, obj::OBJ_TYPE)::Vector{OBJ_TYPE}
    num_twos = count(c -> c == '2', obj)
    if num_twos != N - 1
        return greedy_search_from_startpoint(db, empty_starting_point())
    end

    coloring = [parse(Int, c) for c in obj if c != '2']
    if length(coloring) != NUM_EDGES
        return greedy_search_from_startpoint(db, empty_starting_point())
    end

    # --- Initialize incremental tracking ---
    red_count_p       = zeros(Int, length(ALL_KPS))  # red edges in each P-clique
    kp_forbidden_count = zeros(Int, NUM_EDGES)         # forbidden P-cliques per edge
    blue_count_q      = zeros(Int, length(ALL_KQS))  # blue edges in each Q-clique
    kq_forbidden_count = zeros(Int, NUM_EDGES)         # forbidden Q-cliques per edge
    total_forbidden   = 0

    for (ki, kp_edges) in enumerate(ALL_KPS)
        cnt = count(e -> coloring[e] == 0, kp_edges)
        red_count_p[ki] = cnt
        if cnt == EDGES_KP
            total_forbidden += 1
            for e in kp_edges; kp_forbidden_count[e] += 1; end
        end
    end
    for (ki, kq_edges) in enumerate(ALL_KQS)
        cnt = count(e -> coloring[e] == 1, kq_edges)
        blue_count_q[ki] = cnt
        if cnt == EDGES_KQ
            total_forbidden += 1
            for e in kq_edges; kq_forbidden_count[e] += 1; end
        end
    end

    # --- Greedy loop: first-improvement with random edge order ---
    # Each pass shuffles edge order and flips the first edge that strictly
    # improves total_forbidden (gain > loss). A full pass with no flip signals
    # local optimality.
    edge_order = collect(1:NUM_EDGES)
    pass = 0
    while total_forbidden > 0 && pass < max_search_iter
        pass += 1
        shuffle!(edge_order)
        found = false
        for ei in edge_order
            if coloring[ei] == 0  # would flip red → blue
                gain = kp_forbidden_count[ei]
                gain == 0 && continue
                loss = count(ki -> blue_count_q[ki] == EDGES_KQ - 1, KQS_CONTAINING[ei])
            else                  # would flip blue → red
                gain = kq_forbidden_count[ei]
                gain == 0 && continue
                loss = count(ki -> red_count_p[ki] == EDGES_KP - 1, KPS_CONTAINING[ei])
            end

            gain > loss || continue

            old_color = coloring[ei]
            coloring[ei] = 1 - old_color

            # --- Incremental update: only cliques containing ei can change ---
            if old_color == 0  # flipped red → blue
                for ki in KPS_CONTAINING[ei]
                    if red_count_p[ki] == EDGES_KP        # was forbidden, now not
                        total_forbidden -= 1
                        for e in ALL_KPS[ki]; kp_forbidden_count[e] -= 1; end
                    end
                    red_count_p[ki] -= 1
                end
                for ki in KQS_CONTAINING[ei]
                    blue_count_q[ki] += 1
                    if blue_count_q[ki] == EDGES_KQ        # newly forbidden
                        total_forbidden += 1
                        for e in ALL_KQS[ki]; kq_forbidden_count[e] += 1; end
                    end
                end
            else               # flipped blue → red
                for ki in KQS_CONTAINING[ei]
                    if blue_count_q[ki] == EDGES_KQ        # was forbidden, now not
                        total_forbidden -= 1
                        for e in ALL_KQS[ki]; kq_forbidden_count[e] -= 1; end
                    end
                    blue_count_q[ki] -= 1
                end
                for ki in KPS_CONTAINING[ei]
                    red_count_p[ki] += 1
                    if red_count_p[ki] == EDGES_KP         # newly forbidden
                        total_forbidden += 1
                        for e in ALL_KPS[ki]; kp_forbidden_count[e] += 1; end
                    end
                end
            end

            found = true
            break
        end
        !found && break  # full pass with no improvement → local optimum
    end

    return [convert_adjmat_to_string(coloring)]
end
