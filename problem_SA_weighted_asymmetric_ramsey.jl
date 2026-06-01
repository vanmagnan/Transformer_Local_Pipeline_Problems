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

Incorporates ideas from Exoo & Tatarevic (2015):
  - Weighted score: w_P * bad_KP + w_Q * bad_KQ, with w_P > w_Q for P < Q
    since small-clique violations are harder to eliminate (Algorithm 1).
  - Dynamic weight adjustment: weights track a moving average of bad subgraph
    counts so the search self-corrects when stuck on one color.

Object encoding: String of '0'/'1' with '2' row separators.
Row i contains the colors of edges {i,j} for j = i+1,...,N, followed by '2'.
0 = red, 1 = blue.
"""

using Random
using Combinatorics

const N = 36   # number of vertices (R(4,6) lower bound frontier)
const P = 4    # red clique size to avoid
const Q = 6    # blue clique size to avoid
const NUM_EDGES = N * (N - 1) ÷ 2
const EDGES_KP  = P * (P - 1) ÷ 2   # 6 for K4
const EDGES_KQ  = Q * (Q - 1) ÷ 2   # 15 for K6

# --- Weight schedule (Exoo & Tatarevic, Section 3) ---
# For off-diagonal R(P,Q) with P < Q, bad K_P's are harder to eliminate.
# Initial weight ratio w_P/w_Q should satisfy Q/P ≤ w_P/w_Q ≤ (Q/P)².
# These values set w_P/w_Q = Q/P as a conservative starting point.
const INIT_W_P = Float64(Q)   # weight for red K_P violations
const INIT_W_Q = Float64(P)   # weight for blue K_Q violations

# Moving-average smoothing constant K (paper recommends 10–100).
# Larger K = slower weight adaptation = less oscillation.
const WEIGHT_K = 50.0

# Tabu list length. Paper uses L=1000 for 150 < n < 200.
# For N=35 (much smaller), a shorter list suffices.
function edge_index(i::Int, j::Int)::Int
    # Requires i < j, 1-based indexing
    return (i - 1) * (2 * N - i) ÷ 2 + (j - i)
end

# All C(N,P) P-cliques, each stored as a vector of EDGES_KP edge indices.
const ALL_KPS = let
    kps = Vector{Vector{Int}}()
    for verts in combinations(1:N, P)
        push!(kps, [edge_index(verts[p], verts[q]) for p in 1:P-1 for q in p+1:P])
    end
    kps
end

# For each edge index, the indices of P-cliques (into ALL_KPS) that contain it.
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
const ALL_KQS = let
    kqs = Vector{Vector{Int}}()
    for verts in combinations(1:N, Q)
        push!(kqs, [edge_index(verts[q1], verts[q2]) for q1 in 1:Q-1 for q2 in q1+1:Q])
    end
    kqs
end

# For each edge index, the indices of Q-cliques (into ALL_KQS) that contain it.
const KQS_CONTAINING = let
    lookup = [Int[] for _ in 1:NUM_EDGES]
    for (ki, kq_edges) in enumerate(ALL_KQS)
        for ei in kq_edges
            push!(lookup[ei], ki)
        end
    end
    lookup
end

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
    coloring = [rand(0:1) for _ in 1:NUM_EDGES]
    return convert_adjmat_to_string(coloring)
end

function reward_calc(obj::OBJ_TYPE)::REWARD_TYPE
    coloring = [parse(Int, c) for c in obj if c != '2']
    bad_kp = 0
    bad_kq = 0
    for kp_edges in ALL_KPS
        if all(coloring[e] == 0 for e in kp_edges)
            bad_kp += 1
        end
    end
    for kq_edges in ALL_KQS
        if all(coloring[e] == 1 for e in kq_edges)
            bad_kq += 1
        end
    end
    # TODO: refactor greedy_search_from_startpoint to return (obj, bad_kp, bad_kq)
    # alongside the string, so local_search_on_object in search_fc.jl can call a
    # reward_from_counts(bad_kp, bad_kq) directly and avoid re-scanning 1.9M K6-cliques
    # per construction. Requires updating the return type and the reward(db, obj) call site.
    return -Float32(INIT_W_P * bad_kp + INIT_W_Q * bad_kq)
end

# ---------------------------------------------------------------------------
# Weighted score: w_P * bad_KP + w_Q * bad_KQ
# This is the central idea from Exoo & Tatarevic Algorithm 1.
# Using raw delta (loss - gain) ignores the asymmetry between P and Q cliques.
# ---------------------------------------------------------------------------
@inline function weighted_score(bad_kp::Int, bad_kq::Int,
                                 w_p::Float64, w_q::Float64)::Float64
    return w_p * bad_kp + w_q * bad_kq
end

# ---------------------------------------------------------------------------
# Dynamic weight update (Exoo & Tatarevic, Section 3)
# w_c ← (K * w_c + f_c) / ((K+1) * (f_P + f_Q))
# Keeps weights proportional to a moving average of bad subgraph counts.
# ---------------------------------------------------------------------------
function update_weights!(w_p::Float64, w_q::Float64,
                          bad_kp::Int, bad_kq::Int)::Tuple{Float64, Float64}
    total = bad_kp + bad_kq
    total == 0 && return (w_p, w_q)
    new_wp = (WEIGHT_K * w_p + bad_kp) / ((WEIGHT_K + 1) * total)
    new_wq = (WEIGHT_K * w_q + bad_kq) / ((WEIGHT_K + 1) * total)
    return (new_wp, new_wq)
end

# ---------------------------------------------------------------------------
# Compute the weighted delta for flipping edge ei.
# Returns (weighted_delta, kp_gain, kq_gain) where gains are counts of
# forbidden cliques that would be destroyed and losses created.
# ---------------------------------------------------------------------------
@inline function flip_delta(ei::Int, old_color::Int,
                             red_count_p::Vector{Int}, blue_count_q::Vector{Int},
                             kp_forbidden_count::Vector{Int}, kq_forbidden_count::Vector{Int},
                             w_p::Float64, w_q::Float64)::Float64
    kps = KPS_CONTAINING[ei]
    kqs = KQS_CONTAINING[ei]

    if old_color == 0  # red → blue: destroys red K_P's, may create blue K_Q's
        kp_gain   = kp_forbidden_count[ei]
        kq_loss   = count(ki -> blue_count_q[ki] == EDGES_KQ - 1, kqs)
        return w_q * kq_loss - w_p * kp_gain
    else               # blue → red: destroys blue K_Q's, may create red K_P's
        kq_gain   = kq_forbidden_count[ei]
        kp_loss   = count(ki -> red_count_p[ki] == EDGES_KP - 1, kps)
        return w_p * kp_loss - w_q * kq_gain
    end
end

# ---------------------------------------------------------------------------
# Apply a flip to edge ei and update all incremental counters in place.
# ---------------------------------------------------------------------------
function apply_flip!(ei::Int, coloring::Vector{Int},
                     red_count_p::Vector{Int}, blue_count_q::Vector{Int},
                     kp_forbidden_count::Vector{Int}, kq_forbidden_count::Vector{Int},
                     total_forbidden::Ref{Int})
    old_color = coloring[ei]
    coloring[ei] = 1 - old_color

    if old_color == 0  # red → blue
        for ki in KPS_CONTAINING[ei]
            if red_count_p[ki] == EDGES_KP
                total_forbidden[] -= 1
                for e in ALL_KPS[ki]; kp_forbidden_count[e] -= 1; end
            end
            red_count_p[ki] -= 1
        end
        for ki in KQS_CONTAINING[ei]
            blue_count_q[ki] += 1
            if blue_count_q[ki] == EDGES_KQ
                total_forbidden[] += 1
                for e in ALL_KQS[ki]; kq_forbidden_count[e] += 1; end
            end
        end
    else               # blue → red
        for ki in KQS_CONTAINING[ei]
            if blue_count_q[ki] == EDGES_KQ
                total_forbidden[] -= 1
                for e in ALL_KQS[ki]; kq_forbidden_count[e] -= 1; end
            end
            blue_count_q[ki] -= 1
        end
        for ki in KPS_CONTAINING[ei]
            red_count_p[ki] += 1
            if red_count_p[ki] == EDGES_KP
                total_forbidden[] += 1
                for e in ALL_KPS[ki]; kp_forbidden_count[e] += 1; end
            end
        end
    end
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
    red_count_p        = zeros(Int, length(ALL_KPS))
    kp_forbidden_count = zeros(Int, NUM_EDGES)
    blue_count_q       = zeros(Int, length(ALL_KQS))
    kq_forbidden_count = zeros(Int, NUM_EDGES)
    total_forbidden    = Ref(0)

    for (ki, kp_edges) in enumerate(ALL_KPS)
        for e in kp_edges
            red_count_p[ki] += (coloring[e] == 0)
        end
        if red_count_p[ki] == EDGES_KP
            total_forbidden[] += 1
            for e in kp_edges; kp_forbidden_count[e] += 1; end
        end
    end
    for (ki, kq_edges) in enumerate(ALL_KQS)
        for e in kq_edges
            blue_count_q[ki] += (coloring[e] == 1)
        end
        if blue_count_q[ki] == EDGES_KQ
            total_forbidden[] += 1
            for e in kq_edges; kq_forbidden_count[e] += 1; end
        end
    end

    # Current bad subgraph counts (tracked separately for weight updates)
    bad_kp = count(x -> x == EDGES_KP, red_count_p)
    bad_kq = count(x -> x == EDGES_KQ, blue_count_q)

    # --- Weights (Exoo & Tatarevic) ---
    w_p = INIT_W_P
    w_q = INIT_W_Q

    total_forbidden[] == 0 && return [convert_adjmat_to_string(coloring)]

    # Weighted simulated annealing: shuffles edge order each pass, accepts
    # worsening moves probabilistically. Weighted delta respects the P/Q asymmetry.
    T            = 1.0
    cooling_rate = (1e-4)^(1.0 / max_search_iter)
    edge_order   = collect(1:NUM_EDGES)

    pass = 0
    while total_forbidden[] > 0 && pass < max_search_iter
        pass += 1
        shuffle!(edge_order)

        for ei in edge_order
            d = flip_delta(ei, coloring[ei],
                           red_count_p, blue_count_q,
                           kp_forbidden_count, kq_forbidden_count,
                           w_p, w_q)

            if d <= 0.0 || rand() < exp(-d / T)
                apply_flip!(ei, coloring,
                            red_count_p, blue_count_q,
                            kp_forbidden_count, kq_forbidden_count,
                            total_forbidden)
            end

            total_forbidden[] == 0 && break
        end

        # Update weights once per pass
        bad_kp = count(x -> x == EDGES_KP, red_count_p)
        bad_kq = count(x -> x == EDGES_KQ, blue_count_q)
        w_p, w_q = update_weights!(w_p, w_q, bad_kp, bad_kq)

        T *= cooling_rate
    end

    return [convert_adjmat_to_string(coloring)]
end