include("constants.jl")

"""
Circulant search for 2-colorings of K_N avoiding a red K_P and a blue K_Q.

V2: Near-violation cache optimization.

The original v1 called offset_flip_delta for each of CIRC_LEN offsets per
iteration, and each call scanned KQS_AT_OFFSET[k] (~1M entries at N=35, Q=6).
This made the inner loop O(CIRC_LEN × |KQS|) ≈ 17M ops/iter.

V2 maintains six incremental counters so that offset_flip_delta is O(1):
  n_pviol_at[k]  — # all-red P-cliques touching offset k   (gained_kp in red→blue)
  n_qnear_at[k]  — # Q-cliques near-blue with sole red = k (lost_kq  in red→blue)
  n_pnear_at[k]  — # P-cliques near-red  with sole blue= k (lost_kp  in blue→red)
  n_qviol_at[k]  — # all-blue Q-cliques touching offset k  (gained_kq in blue→red)

apply_offset_flip! still iterates KQS_AT_OFFSET[k] once (unavoidable), but
that is now the only expensive loop per iteration — reducing total work ~17×.

A circulant coloring of K_N is fully determined by a vector V of length
CIRC_LEN = ⌊N/2⌋, where V[k] ∈ {0,1} gives the color of every edge at
offset k (i.e. all edges {i, i+k mod N} for i = 1..N). When N is even,
offset N/2 connects antipodal pairs and each such edge appears once, so
V[N/2] represents N/2 edges rather than N.

Search strategy: weighted tabu search (Algorithm 1 of Exoo & Tatarevic 2015).
  - Score = w_P * bad_KP + w_Q * bad_KQ
  - Initial weights satisfy Q/P ≤ w_P/w_Q ≤ (Q/P)², per the paper's rule
  - Tabu list prevents re-flipping a recently changed offset
  - Dynamic weight update after each pass (Exoo & Tatarevic Section 3)

Bitmask representation: For CIRC_LEN ≤ 32 (N ≤ 65), each state V is mirrored
as a UInt32 V_uint where bit k-1 = V[k]. Each clique is precomputed as a
UInt32 bitmask of the offsets it spans (KP_MASK / KQ_MASK).

  All-red P-clique ki:  (V_uint & KP_MASK[ki]) == 0
  All-blue Q-clique ki: (V_uint & KQ_MASK[ki]) == KQ_MASK[ki]

Object encoding: compact CIRC_LEN-char offset vector.
Each character is '0' or '1' giving the color of circulant offset k=1..CIRC_LEN.
Example: "01101001011010110" (17 chars for N=35, no separators).
"""

using Random
using Combinatorics

const N = 36
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
# Bitmask tables
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
# O(1) delta using near-violation cache arrays.
#
# Red → blue (V[k] == 0):
#   gained_kp = n_pviol_at[k]  (all-red P-cliques at k, destroyed by flip)
#   lost_kq   = n_qnear_at[k]  (Q-cliques with sole red = k, created by flip)
#
# Blue → red (V[k] == 1):
#   gained_kq = n_qviol_at[k]  (all-blue Q-cliques at k, destroyed by flip)
#   lost_kp   = n_pnear_at[k]  (P-cliques with sole blue = k, created by flip)
# ---------------------------------------------------------------------------

@inline function offset_flip_delta_cached(k::Int, V_k::Int,
                                           w_p::Float64, w_q::Float64,
                                           n_pviol_at::Vector{Int},
                                           n_qviol_at::Vector{Int},
                                           n_pnear_at::Vector{Int},
                                           n_qnear_at::Vector{Int})::Float64
    if V_k == 0  # red → blue
        return w_q * n_qnear_at[k] - w_p * n_pviol_at[k]
    else          # blue → red
        return w_p * n_pnear_at[k] - w_q * n_qviol_at[k]
    end
end

# ---------------------------------------------------------------------------
# Apply flip and update V, V_uint, total_forbidden, bad_kp, and all six
# near-violation cache arrays.
#
# Transitions tracked per clique type:
#
#   P-cliques (kp_blue_count = # blue offsets in clique):
#     red→blue: kp_blue_count increases by 1 for cliques in KPS_AT_OFFSET[k]
#       0→1: P-violation destroyed; clique becomes near-red at k
#       1→2: was near-red at sole blue k′ ≠ k; no longer near-red
#       else: no viol/near transition
#     blue→red: kp_blue_count decreases by 1
#       1→0: P-violation created; clique was near-red at k
#       2→1: clique becomes near-red at remaining blue k′
#       else: no viol/near transition
#
#   Q-cliques (kq_red_count = # red offsets in clique):
#     red→blue: kq_red_count decreases by 1 for cliques in KQS_AT_OFFSET[k]
#       1→0: Q-violation created; clique was near-blue at k
#       2→1: clique becomes near-blue at remaining red k′
#       else: no viol/near transition
#       (0 is impossible: k is red and ki spans k, so kq_red_count ≥ 1)
#     blue→red: kq_red_count increases by 1
#       0→1: Q-violation destroyed; clique becomes near-blue at k
#       1→2: was near-blue at sole red k′ ≠ k; no longer near-blue
#       else: no viol/near transition
# ---------------------------------------------------------------------------

function apply_offset_flip!(k::Int, V::Vector{Int}, V_uint::Ref{UInt32},
                             total_forbidden::Ref{Int}, bad_kp::Ref{Int},
                             kp_blue_count::Vector{UInt8}, kq_red_count::Vector{UInt8},
                             n_pviol_at::Vector{Int}, n_qviol_at::Vector{Int},
                             n_pnear_at::Vector{Int}, n_qnear_at::Vector{Int})
    old_uint  = V_uint[]
    bit_k     = OFFSET_BIT[k]
    old_color = V[k]
    V[k]      = 1 - old_color
    V_uint[]  = old_uint ⊻ bit_k
    new_uint  = V_uint[]

    if old_color == 0  # red → blue

        # --- P-cliques: kp_blue_count increases ---
        for ki in KPS_AT_OFFSET[k]
            old_bc = Int(kp_blue_count[ki])
            kp_blue_count[ki] = old_bc + 1
            if old_bc == 0
                # P-violation destroyed
                total_forbidden[] -= 1
                bad_kp[] -= 1
                m = KP_MASK[ki]
                while m != UInt32(0)
                    n_pviol_at[trailing_zeros(m) + 1] -= 1
                    m &= m - UInt32(1)
                end
                # Clique now has sole blue = k → near-red at k
                n_pnear_at[k] += 1
            elseif old_bc == 1
                # Was near-red at sole blue k′ (k′ ≠ k since k was red)
                k′ = trailing_zeros(KP_MASK[ki] & old_uint) + 1
                n_pnear_at[k′] -= 1
                # Now has 2 blue offsets: no longer near-red anywhere
            end
        end

        # --- Q-cliques: kq_red_count decreases ---
        for ki in KQS_AT_OFFSET[k]
            old_rc = Int(kq_red_count[ki])
            kq_red_count[ki] = old_rc - 1
            if old_rc == 1
                # Was near-blue with sole red = k → now all-blue (Q-violation created)
                n_qnear_at[k] -= 1
                total_forbidden[] += 1
                m = KQ_MASK[ki]
                while m != UInt32(0)
                    n_qviol_at[trailing_zeros(m) + 1] += 1
                    m &= m - UInt32(1)
                end
            elseif old_rc == 2
                # Had 2 red offsets; after flip has 1 → near-blue at remaining red k′
                # remaining red = sole set bit in (KQ_MASK[ki] & ~new_uint)
                k′ = trailing_zeros(KQ_MASK[ki] & ~new_uint) + 1
                n_qnear_at[k′] += 1
            end
            # old_rc == 0 is impossible: k is red and ki spans k, so rc ≥ 1
        end

    else  # blue → red

        # --- P-cliques: kp_blue_count decreases ---
        for ki in KPS_AT_OFFSET[k]
            old_bc = Int(kp_blue_count[ki])
            kp_blue_count[ki] = old_bc - 1
            if old_bc == 1
                # Was near-red with sole blue = k → now all-red (P-violation created)
                n_pnear_at[k] -= 1
                total_forbidden[] += 1
                bad_kp[] += 1
                m = KP_MASK[ki]
                while m != UInt32(0)
                    n_pviol_at[trailing_zeros(m) + 1] += 1
                    m &= m - UInt32(1)
                end
            elseif old_bc == 2
                # Had 2 blue offsets; after flip has 1 → near-red at remaining blue k′
                # remaining blue = sole set bit in (KP_MASK[ki] & new_uint)
                k′ = trailing_zeros(KP_MASK[ki] & new_uint) + 1
                n_pnear_at[k′] += 1
            end
        end

        # --- Q-cliques: kq_red_count increases ---
        for ki in KQS_AT_OFFSET[k]
            old_rc = Int(kq_red_count[ki])
            kq_red_count[ki] = old_rc + 1
            if old_rc == 0
                # Was all-blue (Q-violation) → now near-blue at k (violation destroyed)
                m = KQ_MASK[ki]
                while m != UInt32(0)
                    n_qviol_at[trailing_zeros(m) + 1] -= 1
                    m &= m - UInt32(1)
                end
                total_forbidden[] -= 1
                n_qnear_at[k] += 1
            elseif old_rc == 1
                # Was near-blue at sole red k′ (k′ ≠ k since k was blue) → now 2 red
                # sole red before flip = set bit in (KQ_MASK[ki] & ~old_uint)
                k′ = trailing_zeros(KQ_MASK[ki] & ~old_uint) + 1
                n_qnear_at[k′] -= 1
                # No longer near-blue anywhere
            end
        end

    end
end

# ---------------------------------------------------------------------------
# Debug validation (disabled in production; enable by setting CIRC_DEBUG=true)
# Recomputes all six cache arrays from scratch and asserts they match.
# ---------------------------------------------------------------------------

const CIRC_DEBUG = false

function validate_cache(V_uint::UInt32,
                        kp_blue_count::Vector{UInt8}, kq_red_count::Vector{UInt8},
                        n_pviol_at::Vector{Int}, n_qviol_at::Vector{Int},
                        n_pnear_at::Vector{Int}, n_qnear_at::Vector{Int},
                        total_forbidden::Int, bad_kp::Int)
    ref_kp_blue  = UInt8[count_ones(KP_MASK[ki] &  V_uint) for ki in 1:length(KP_MASK)]
    ref_kq_red   = UInt8[count_ones(KQ_MASK[ki] & ~V_uint) for ki in 1:length(KQ_MASK)]
    ref_pviol    = zeros(Int, CIRC_LEN)
    ref_qviol    = zeros(Int, CIRC_LEN)
    ref_pnear    = zeros(Int, CIRC_LEN)
    ref_qnear    = zeros(Int, CIRC_LEN)
    ref_tf = 0; ref_bkp = 0

    for ki in 1:length(KP_MASK)
        bc = ref_kp_blue[ki]
        if bc == 0
            ref_tf += 1; ref_bkp += 1
            m = KP_MASK[ki]
            while m != UInt32(0); ref_pviol[trailing_zeros(m)+1] += 1; m &= m-UInt32(1); end
        elseif bc == 1
            ref_pnear[trailing_zeros(KP_MASK[ki] & V_uint)+1] += 1
        end
    end
    for ki in 1:length(KQ_MASK)
        rc = ref_kq_red[ki]
        if rc == 0
            ref_tf += 1
            m = KQ_MASK[ki]
            while m != UInt32(0); ref_qviol[trailing_zeros(m)+1] += 1; m &= m-UInt32(1); end
        elseif rc == 1
            ref_qnear[trailing_zeros(KQ_MASK[ki] & ~V_uint)+1] += 1
        end
    end

    @assert kp_blue_count == ref_kp_blue   "kp_blue_count mismatch"
    @assert kq_red_count  == ref_kq_red    "kq_red_count mismatch"
    @assert n_pviol_at    == ref_pviol     "n_pviol_at mismatch"
    @assert n_qviol_at    == ref_qviol     "n_qviol_at mismatch"
    @assert n_pnear_at    == ref_pnear     "n_pnear_at mismatch"
    @assert n_qnear_at    == ref_qnear     "n_qnear_at mismatch"
    @assert total_forbidden == ref_tf      "total_forbidden mismatch"
    @assert bad_kp          == ref_bkp     "bad_kp mismatch"
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

    # ---------------------------------------------------------------------------
    # Initialize all six near-violation cache arrays and total_forbidden/bad_kp.
    # Single pass over each clique table; replaces the old bitmask-scan init.
    # ---------------------------------------------------------------------------
    kp_blue_count = zeros(UInt8, length(KP_MASK))
    kq_red_count  = zeros(UInt8, length(KQ_MASK))
    n_pviol_at    = zeros(Int, CIRC_LEN)
    n_qviol_at    = zeros(Int, CIRC_LEN)
    n_pnear_at    = zeros(Int, CIRC_LEN)
    n_qnear_at    = zeros(Int, CIRC_LEN)
    total_forbidden = Ref(0)
    bad_kp          = Ref(0)
    cur = V_uint[]

    for ki in 1:length(KP_MASK)
        bc = count_ones(KP_MASK[ki] & cur)  # Int
        kp_blue_count[ki] = bc % UInt8
        if bc == 0
            total_forbidden[] += 1
            bad_kp[] += 1
            m = KP_MASK[ki]
            while m != UInt32(0)
                n_pviol_at[trailing_zeros(m) + 1] += 1
                m &= m - UInt32(1)
            end
        elseif bc == 1
            n_pnear_at[trailing_zeros(KP_MASK[ki] & cur) + 1] += 1
        end
    end

    for ki in 1:length(KQ_MASK)
        rc = count_ones(KQ_MASK[ki] & ~cur)  # Int
        kq_red_count[ki] = rc % UInt8
        if rc == 0
            total_forbidden[] += 1
            m = KQ_MASK[ki]
            while m != UInt32(0)
                n_qviol_at[trailing_zeros(m) + 1] += 1
                m &= m - UInt32(1)
            end
        elseif rc == 1
            n_qnear_at[trailing_zeros(KQ_MASK[ki] & ~cur) + 1] += 1
        end
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

        for k in 1:CIRC_LEN
            k in tabu_set && continue
            d = offset_flip_delta_cached(k, V[k], w_p, w_q,
                                         n_pviol_at, n_qviol_at,
                                         n_pnear_at, n_qnear_at)
            if d < best_delta
                best_delta = d
                best_k     = k
            end
        end

        # All offsets tabu: fall back to global best. Guarded by CIRC_TABU_LENGTH
        # <= CIRC_LEN ÷ 2, so this should be unreachable in practice.
        if best_k == -1
            best_k = argmin(k -> offset_flip_delta_cached(k, V[k], w_p, w_q,
                                                           n_pviol_at, n_qviol_at,
                                                           n_pnear_at, n_qnear_at),
                            1:CIRC_LEN)
        end

        apply_offset_flip!(best_k, V, V_uint, total_forbidden, bad_kp,
                           kp_blue_count, kq_red_count,
                           n_pviol_at, n_qviol_at, n_pnear_at, n_qnear_at)

        if CIRC_DEBUG && pass % 100 == 0
            validate_cache(V_uint[], kp_blue_count, kq_red_count,
                           n_pviol_at, n_qviol_at, n_pnear_at, n_qnear_at,
                           total_forbidden[], bad_kp[])
        end

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
