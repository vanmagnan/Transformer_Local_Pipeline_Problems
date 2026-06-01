include("constants.jl")

"""
Find 2-colorings (red/blue) of edges of K_N minimizing monochromatic K5 subgraphs.

This version precomputes K5 membership tables and tracks mono K5 counts
incrementally, avoiding a full O(C(N,5)) scan after every edge flip.

After flipping edge ei, only the C(N-2,3) K5s that contain ei can change
status, so the per-iteration work drops from O(C(N,5)) to O(C(N-2,3)).
For N=10 that is 252 → 56 (~4.5x); for N=42 it is 850668 → 9880 (~86x).

See problem_monochromatic_clique.jl for the simpler reference implementation.

Object encoding: String of '0'/'1' with '2' row separators.
Row i contains the colors of edges {i,j} for j = i+1,...,N, followed by '2'.
0 = red, 1 = blue.
"""

using Random

const N = 43
const NUM_EDGES = N * (N - 1) ÷ 2

function edge_index(i::Int, j::Int)::Int
    # Requires i < j, 1-based indexing
    return (i - 1) * (2 * N - i) ÷ 2 + (j - i)
end

# All C(N,5) five-cliques, each stored as a vector of their 10 edge indices.
# Computed once at load time so the inner loop never reconstructs them.
const ALL_K5S = let
    k5s = Vector{Vector{Int}}()
    for a in 1:N-4
        for b in a+1:N-3
            for c in b+1:N-2
                for d in c+1:N-1
                    for e in d+1:N
                        verts = (a, b, c, d, e)
                        push!(k5s, [edge_index(verts[p], verts[q]) for p in 1:4 for q in p+1:5])
                    end
                end
            end
        end
    end
    k5s
end

# For each edge index, the indices of K5s (into ALL_K5S) that contain it.
# Computed once at load time.
const K5S_CONTAINING = let
    lookup = [Int[] for _ in 1:NUM_EDGES]
    for (ki, k5_edges) in enumerate(ALL_K5S)
        for ei in k5_edges
            push!(lookup[ei], ki)
        end
    end
    lookup
end

# Row-separated encoding: edges in row i (i.e. {i,j} for j>i) followed by "2".
# Consistent with problem_4_cycle_free.jl. Gives BPE meaningful sub-sequences to learn.
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

function reward_calc(obj::OBJ_TYPE)::REWARD_TYPE
    coloring = [parse(Int, c) for c in obj if c != '2']
    total = 0
    for k5_edges in ALL_K5S
        c = coloring[k5_edges[1]]
        if all(coloring[e] == c for e in k5_edges)
            total += 1
        end
    end
    return -Float32(total)
end

function empty_starting_point()::OBJ_TYPE
    # Random coloring so parallel searches start from diverse points
    coloring = [rand(0:1) for _ in 1:NUM_EDGES]
    return convert_adjmat_to_string(coloring)
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

    # --- Initialise incremental tracking ---
    # same_color_count[ki] : number of red (color-0) edges in K5 ki
    # mono_count[ei]       : how many mono K5s currently contain edge ei
    # total_mono           : total number of mono K5s
    same_color_count = zeros(Int, length(ALL_K5S))
    mono_count       = zeros(Int, NUM_EDGES)
    total_mono       = 0

    for (ki, k5_edges) in enumerate(ALL_K5S)
        cnt = count(e -> coloring[e] == 0, k5_edges)
        same_color_count[ki] = cnt
        if cnt == 0 || cnt == 10
            total_mono += 1
            for e in k5_edges; mono_count[e] += 1; end
        end
    end

    iter = 0
    while total_mono > 0 && iter < max_search_iter
        iter += 1

        # Collect improving edges: those where gain > loss.
        #   gain[ei] = mono_count[ei]  (already tracked; free to read)
        #   loss[ei] = # of K5s in K5S_CONTAINING[ei] that would become
        #              monochromatic after the flip, i.e. all other 9 edges
        #              already have the new color.
        improving_edges = Int[]
        for ei in 1:NUM_EDGES
            gain = mono_count[ei]
            gain == 0 && continue

            target = coloring[ei] == 0 ? 1 : 9
            loss = 0
            for ki in K5S_CONTAINING[ei]
                loss += same_color_count[ki] == target
            end

            if gain > loss
                push!(improving_edges, ei)
            end
        end

        isempty(improving_edges) && break

        flip_ei = improving_edges[rand(1:length(improving_edges))]
        coloring[flip_ei] = 1 - coloring[flip_ei]

        # --- Incremental update ---
        # Only K5s that contain flip_ei can have changed mono status.
        delta_scc = coloring[flip_ei] == 0 ? 1 : -1
        for ki in K5S_CONTAINING[flip_ei]
            was_mono = same_color_count[ki] == 0 || same_color_count[ki] == 10
            same_color_count[ki] += delta_scc
            now_mono = same_color_count[ki] == 0 || same_color_count[ki] == 10
            if was_mono != now_mono
                delta = now_mono ? 1 : -1
                total_mono += delta
                for e in ALL_K5S[ki]; mono_count[e] += delta; end
            end
        end
    end

    return [convert_adjmat_to_string(coloring)]
end
