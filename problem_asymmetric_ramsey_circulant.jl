include("constants.jl")

"""
Circulant search for 2-colorings of K_N avoiding a red K_P and a blue K_Q.

A circulant coloring of K_N is fully determined by a vector V of length
CIRC_LEN = ⌊N/2⌋, where V[k] ∈ {0,1} gives the color of every edge at
offset k (i.e. all edges {i, i+k mod N} for i = 1..N). When N is even,
offset N/2 connects antipodal pairs and each such edge appears once, so
V[N/2] represents N/2 edges rather than N.

This reduces the search space from C(N,2) edge dimensions down to ⌊N/2⌋
dimensions, making steepest-descent tabu search (Algorithm 1 of Exoo &
Tatarevic 2015) tractable: each pass evaluates only CIRC_LEN candidate
flips rather than NUM_EDGES.

The key complication vs. single-edge recoloring: flipping offset k recolors
N (or N/2) edges simultaneously, so multiple edges of the same clique may
change color at once. Gain/loss counts must be computed at the clique level
to avoid double-counting.

Search strategy: weighted tabu search (Algorithm 1 of Exoo & Tatarevic).
  - Score = w_P * bad_KP + w_Q * bad_KQ
  - Initial weights satisfy Q/P ≤ w_P/w_Q ≤ (Q/P)², per the paper's rule
  - Tabu list prevents re-flipping a recently changed offset
  - Dynamic weight update after each pass (Exoo & Tatarevic Section 3)

Bitmask representation: For CIRC_LEN ≤ 32 (N ≤ 65), each state V is mirrored
as a UInt32 V_uint where bit k-1 = V[k]. Each clique is precomputed as a
UInt32 bitmask of the offsets it spans (KP_MASK / KQ_MASK). This allows O(1)
forbidden-clique checks and eliminates VISITED deduplication overhead.

  All-red P-clique ki:  (V_uint & KP_MASK[ki]) == 0
  All-blue Q-clique ki: (V_uint & KQ_MASK[ki]) == KQ_MASK[ki]

Object encoding: compact CIRC_LEN-char offset vector.
Each character is '0' or '1' giving the color of circulant offset k=1..CIRC_LEN.
Example: "01101001011010110" (17 chars for N=35, no separators).
"""

using Random
using Combinatorics

const N = 35
const P = 4
const Q = 6
const NUM_EDGES = N * (N - 1) ÷ 2
const EDGES_KP  = P * (P - 1) ÷ 2
const EDGES_KQ  = Q * (Q - 1) ÷ 2
const CIRC_LEN  = N ÷ 2

# Initial weight ratio w_P/w_Q = Q/P (lower bound of paper's recommended range).
const INIT_W_P = Float64(Q)
const INIT_W_Q = Float64(P)

# Moving-average smoothing constant (paper recommends 10-100).
const WEIGHT_K = 50.0

# Tabu list length. Capped at CIRC_LEN ÷ 2 so at most half of offsets are
# ever forbidden simultaneously, keeping the search from getting stuck.
const CIRC_TABU_LENGTH = max(5, CIRC_LEN ÷ 2)

# ---------------------------------------------------------------------------
# Edge indexing
# ---------------------------------------------------------------------------

function edge_index(i::Int, j::Int)::Int
    return (i - 1) * (2 * N - i) ÷ 2 + (j - i)
end

# ---------------------------------------------------------------------------
# Clique tables
# ---------------------------------------------------------------------------

const ALL_KPS = let
    kps = Vector{Vector{Int}}()
    for verts in combinations(1:N, P)
        push!(kps, [edge_index(verts[a], verts[b]) for a in 1:P-1 for b in a+1:P])
    end
    kps
end

const ALL_KQS = let
    kqs = Vector{Vector{Int}}()
    for verts in combinations(1:N, Q)
        push!(kqs, [edge_index(verts[a], verts[b]) for a in 1:Q-1 for b in a+1:Q])
    end
    kqs
end

# ---------------------------------------------------------------------------
# Circulant structure
# ---------------------------------------------------------------------------

const OFFSET_EDGES = let
    oe = [Int[] for _ in 1:CIRC_LEN]
    for i in 1:N-1
        for j in i+1:N
            k = min(j - i, N - (j - i))
            push!(oe[k], edge_index(i, j))
        end
    end
    oe
end

# For each edge index, which circulant offset k it belongs to.
const EDGE_TO_OFFSET = let
    eto = zeros(Int, NUM_EDGES)
    for k in 1:CIRC_LEN
        for ei in OFFSET_EDGES[k]
            eto[ei] = k
        end
    end
    eto
end

# ---------------------------------------------------------------------------
# Bitmask tables (replaces KP_EDGES_IN_OFFSET, KQ_EDGES_IN_OFFSET,
#                 VISITED_KP, VISITED_KQ, red_count_p, blue_count_q)
# ---------------------------------------------------------------------------

# Precomputed bit k-1 for offset k. UInt32 supports CIRC_LEN ≤ 32 (N ≤ 65).
const OFFSET_BIT = UInt32[UInt32(1) << (k - 1) for k in 1:CIRC_LEN]

# KP_MASK[ki] = bitmask of circulant offsets used by P-clique ki.
# Bit k-1 is set iff the clique contains at least one edge at offset k.
# All-red check: (V_uint & KP_MASK[ki]) == 0
const KP_MASK = UInt32[
    reduce(|, OFFSET_BIT[EDGE_TO_OFFSET[e]] for e in kp_edges; init = UInt32(0))
    for kp_edges in ALL_KPS
]

# KQ_MASK[ki] = bitmask of circulant offsets used by Q-clique ki.
# All-blue check: (V_uint & KQ_MASK[ki]) == KQ_MASK[ki]
const KQ_MASK = UInt32[
    reduce(|, OFFSET_BIT[EDGE_TO_OFFSET[e]] for e in kq_edges; init = UInt32(0))
    for kq_edges in ALL_KQS
]

# KPS_AT_OFFSET[k] = indices of P-cliques that span circulant offset k.
# Precomputed so offset_flip_delta needs no VISITED deduplication.
const KPS_AT_OFFSET = let
    lists = [Int[] for _ in 1:CIRC_LEN]
    for (ki, mask) in enumerate(KP_MASK)
        for k in 1:CIRC_LEN
            (mask >> (k - 1)) & 1 == 1 && push!(lists[k], ki)
        end
    end
    lists
end

# KQS_AT_OFFSET[k] = indices of Q-cliques that span circulant offset k.
const KQS_AT_OFFSET = let
    lists = [Int[] for _ in 1:CIRC_LEN]
    for (ki, mask) in enumerate(KQ_MASK)
        for k in 1:CIRC_LEN
            (mask >> (k - 1)) & 1 == 1 && push!(lists[k], ki)
        end
    end
    lists
end

# ---------------------------------------------------------------------------
# String encoding
# ---------------------------------------------------------------------------

function convert_adjmat_to_string(V::Vector{Int})::String
    return join(string(v) for v in V)
end

# ---------------------------------------------------------------------------
# Required interface functions
# ---------------------------------------------------------------------------

function empty_starting_point()::OBJ_TYPE
    return join(string(rand(0:1)) for _ in 1:CIRC_LEN)
end

function reward_calc(obj::OBJ_TYPE)::REWARD_TYPE
    if length(obj) != CIRC_LEN
        return REWARD_TYPE(-1)
    end
    V_uint = UInt32(0)
    for (k, c) in enumerate(obj)
        c == '1' && (V_uint |= OFFSET_BIT[k])
    end
    total = 0
    for mask in KP_MASK
        (V_uint & mask) == 0    && (total += 1)
    end
    for mask in KQ_MASK
        (V_uint & mask) == mask && (total += 1)
    end
    return -Float32(total)
end

# ---------------------------------------------------------------------------
# Dynamic weight update (Exoo & Tatarevic Section 3)
# ---------------------------------------------------------------------------

function update_weights(w_p::Float64, w_q::Float64,
                        bad_kp::Int, bad_kq::Int)::Tuple{Float64, Float64}
    total = bad_kp + bad_kq
    total == 0 && return (w_p, w_q)
    new_wp = (WEIGHT_K * w_p + bad_kp) / ((WEIGHT_K + 1) * total)
    new_wq = (WEIGHT_K * w_q + bad_kq) / ((WEIGHT_K + 1) * total)
    return (new_wp, new_wq)
end

# ---------------------------------------------------------------------------
# Bitmask delta for flipping circulant offset k.
#
# No allocations, no VISITED deduplication — KPS_AT_OFFSET / KQS_AT_OFFSET
# contain unique clique indices per offset.
#
# Red → blue flip (V[k]: 0 → 1, bit k-1 gets set in V_uint):
#   gained_kp: P-cliques currently all-red that touch offset k.
#              Condition: (V_uint & KP_MASK[ki]) == 0
#              After flip, bit k-1 set → no longer all-red → P-violation destroyed.
#   lost_kq:   Q-cliques that will become all-blue after the flip.
#              Condition: (V_uint & KQ_MASK[ki]) == KQ_MASK[ki] ⊻ bit_k
#              i.e. all Q-clique offsets are already blue except offset k.
#
# Blue → red flip (V[k]: 1 → 0, bit k-1 gets cleared):
#   gained_kq: Q-cliques currently all-blue that touch offset k.
#   lost_kp:   P-cliques that will become all-red after the flip.
#              Condition: (V_uint & KP_MASK[ki]) == bit_k
#              i.e. offset k is the only blue offset in the P-clique.
# ---------------------------------------------------------------------------

function offset_flip_delta(k::Int, V_uint::UInt32, w_p::Float64, w_q::Float64)::Float64
    bit_k = OFFSET_BIT[k]
    if (V_uint >> (k - 1)) & 1 == 0  # red → blue
        gained_kp = count(ki -> (V_uint & KP_MASK[ki]) == 0,
                          KPS_AT_OFFSET[k])
        lost_kq   = count(ki -> (V_uint & KQ_MASK[ki]) == KQ_MASK[ki] ⊻ bit_k,
                          KQS_AT_OFFSET[k])
        return w_q * lost_kq - w_p * gained_kp
    else                               # blue → red
        gained_kq = count(ki -> (V_uint & KQ_MASK[ki]) == KQ_MASK[ki],
                          KQS_AT_OFFSET[k])
        lost_kp   = count(ki -> (V_uint & KP_MASK[ki]) == bit_k,
                          KPS_AT_OFFSET[k])
        return w_p * lost_kp - w_q * gained_kq
    end
end

# ---------------------------------------------------------------------------
# Apply a circulant offset flip and update V, V_uint, total_forbidden,
# and bad_kp (bad_kq = total_forbidden - bad_kp).
# ---------------------------------------------------------------------------

function apply_offset_flip!(k::Int, V::Vector{Int}, V_uint::Ref{UInt32},
                             total_forbidden::Ref{Int}, bad_kp::Ref{Int})
    bit_k     = OFFSET_BIT[k]
    old_color = V[k]
    old_uint  = V_uint[]
    V[k]      = 1 - old_color
    V_uint[]  = old_uint ⊻ bit_k
    new_uint  = V_uint[]

    if old_color == 0  # red → blue
        for ki in KPS_AT_OFFSET[k]
            if (old_uint & KP_MASK[ki]) == 0
                total_forbidden[] -= 1
                bad_kp[] -= 1
            end
        end
        for ki in KQS_AT_OFFSET[k]
            (new_uint & KQ_MASK[ki]) == KQ_MASK[ki] && (total_forbidden[] += 1)
        end
    else               # blue → red
        for ki in KQS_AT_OFFSET[k]
            (old_uint & KQ_MASK[ki]) == KQ_MASK[ki] && (total_forbidden[] -= 1)
        end
        for ki in KPS_AT_OFFSET[k]
            if (new_uint & KP_MASK[ki]) == 0
                total_forbidden[] += 1
                bad_kp[] += 1
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

function greedy_search_from_startpoint(db, obj::OBJ_TYPE)::Vector{OBJ_TYPE}
    if length(obj) != CIRC_LEN
        return greedy_search_from_startpoint(db, empty_starting_point())
    end
    V = [parse(Int, c) for c in obj]

    # Build V_uint from V.
    V_uint = Ref(UInt32(0))
    for k in 1:CIRC_LEN
        V[k] == 1 && (V_uint[] |= OFFSET_BIT[k])
    end

    # Initialize total_forbidden and bad_kp from scratch via bitmask scan.
    total_forbidden = Ref(0)
    bad_kp          = Ref(0)
    cur = V_uint[]
    for mask in KP_MASK
        if (cur & mask) == 0
            total_forbidden[] += 1
            bad_kp[] += 1
        end
    end
    for mask in KQ_MASK
        (cur & mask) == mask && (total_forbidden[] += 1)
    end

    w_p = INIT_W_P
    w_q = INIT_W_Q

    tabu      = zeros(Int, CIRC_TABU_LENGTH)
    tabu_head = 1
    tabu_set  = Set{Int}()

    pass = 0
    while total_forbidden[] > 0 && pass < max_search_iter
        pass += 1

        best_delta = Inf
        best_k     = -1

        cur = V_uint[]
        for k in 1:CIRC_LEN
            k in tabu_set && continue
            d = offset_flip_delta(k, cur, w_p, w_q)
            if d < best_delta
                best_delta = d
                best_k     = k
            end
        end

        # All offsets tabu: fall back to global best. Guarded by CIRC_TABU_LENGTH
        # <= CIRC_LEN ÷ 2, so this should be unreachable in practice.
        if best_k == -1
            best_k = argmin(k -> offset_flip_delta(k, V_uint[], w_p, w_q),
                            1:CIRC_LEN)
        end

        apply_offset_flip!(best_k, V, V_uint, total_forbidden, bad_kp)

        old_tabu = tabu[tabu_head]
        if old_tabu != 0; delete!(tabu_set, old_tabu); end
        tabu[tabu_head] = best_k
        push!(tabu_set, best_k)
        tabu_head = tabu_head % CIRC_TABU_LENGTH + 1

        kp = bad_kp[]
        kq = total_forbidden[] - kp
        w_p, w_q = update_weights(w_p, w_q, kp, kq)
    end

    return [convert_adjmat_to_string(V)]
end
