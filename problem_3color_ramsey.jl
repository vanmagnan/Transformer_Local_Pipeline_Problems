include("constants.jl")

"""
Find 3-colorings (red/blue/green) of the edges of the complete 3-uniform hypergraph
K_N^(3) avoiding monochromatic copies of K_4^(3)-e in every color.

K_4^(3)-e is K_4^(3) with one hyperedge removed (3 edges on 4 vertices).
For a 4-vertex set with k hyperedges of a given color, the number of
monochromatic K_4^(3)-e copies in that color is C(k, 3):

    k=0,1,2: 0 copies    k=3: C(3,3)=1    k=4: C(4,3)=4

Reward = -(total monochromatic K_4^(3)-e copies summed over all three colors).
Reward 0 = valid coloring.

Object encoding: String of '0'/'1'/'2' with '3' vertex-row separators.
For each i = 1,...,N-2, emit all hyperedge colors {i,j,k} with i<j<k in lex
order, then a '3'. Total length = C(N,3) + (N-2). 0=red, 1=blue, 2=green.

Target: R(K_4^(3)-e; 3) — the 3-color hypergraph Ramsey number. 
N=13 only unknown value
"""

using Random
using Combinatorics

const N = 13                                   # vertices; adjust to probe frontier
const NUM_HYPEREDGES = N * (N-1) * (N-2) ÷ 6  # = C(N,3)

# Precomputed 3D index table: HINDEX[i,j,k] = 1-based position of hyperedge
# {i,j,k} in the coloring vector (lex order, i < j < k only).
const HINDEX = let
    idx = zeros(Int, N, N, N)
    pos = 0
    for i in 1:N-2, j in i+1:N-1, k in j+1:N
        pos += 1
        idx[i, j, k] = pos
    end
    idx
end

# All C(N,4) four-vertex cliques, each stored as 4 hyperedge indices.
const ALL_K4S = let
    k4s = Vector{Vector{Int}}()
    for verts in combinations(1:N, 4)
        push!(k4s, [HINDEX[verts[p], verts[q], verts[r]]
                    for p in 1:2 for q in p+1:3 for r in q+1:4])
    end
    k4s
end

# For each hyperedge index, which 4-cliques (into ALL_K4S) contain it.
const K4S_CONTAINING = let
    lookup = [Int[] for _ in 1:NUM_HYPEREDGES]
    for (ki, k4_edges) in enumerate(ALL_K4S)
        for ei in k4_edges; push!(lookup[ei], ki); end
    end
    lookup
end

# Vertex-i row encoding: for each i = 1..N-2, emit all hyperedge colors
# {i,j,k} (j in i+1:N-1, k in j+1:N) in lex order, then a '3' separator.
function convert_adjmat_to_string(coloring::Vector{Int})::String
    entries = []
    for i in 1:N-2
        for j in i+1:N-1
            for k in j+1:N
                push!(entries, string(coloring[HINDEX[i, j, k]]))
            end
        end
        push!(entries, "3")
    end
    return join(entries)
end

function empty_starting_point()::OBJ_TYPE
    coloring = [rand(0:2) for _ in 1:NUM_HYPEREDGES]
    return convert_adjmat_to_string(coloring)
end

# Number of monochromatic K_4^(3)-e copies in a group of k same-colored edges = C(k,3).
@inline k4_copies(k::Int) = k >= 3 ? k * (k-1) * (k-2) ÷ 6 : 0

# Lookup tables for marginal costs: k is always in {0,1,2,3,4} (4 edges per K4),
# so table lookup is faster than branching + arithmetic.
#
# k4_marginal_loss[k+1] = C(k,2):   cost added when color-k group gains one edge.
# k4_marginal_gain[k+1] = C(k-1,2): cost removed when color-k group loses one edge.
const K4_MARGINAL_LOSS = (0, 0, 1, 3, 6)  # k = 0,1,2,3,4
const K4_MARGINAL_GAIN = (0, 0, 0, 1, 3)  # k = 0,1,2,3,4

function reward_calc(obj::OBJ_TYPE)::REWARD_TYPE
    coloring = [parse(Int, c) for c in obj if c != '3']
    total = 0
    for k4_edges in ALL_K4S
        for c in 0:2
            total += k4_copies(count(e -> coloring[e] == c, k4_edges))
        end
    end
    return -Float32(total)
end

function greedy_search_from_startpoint(db, obj::OBJ_TYPE)::Vector{OBJ_TYPE}
    if count(c -> c == '3', obj) != N - 2
        return greedy_search_from_startpoint(db, empty_starting_point())
    end
    coloring = [parse(Int, c) for c in obj if c != '3']
    if length(coloring) != NUM_HYPEREDGES
        return greedy_search_from_startpoint(db, empty_starting_point())
    end

    # --- Initialise incremental state ---
    # color_count_4[ki, c+1] = number of hyperedges of color c in 4-clique ki.
    # total_bad = sum over K4s, sum over colors c, C(color_count_4[ki,c+1], 3).
    color_count_4 = zeros(Int, length(ALL_K4S), 3)
    total_bad = 0

    for (ki, k4_edges) in enumerate(ALL_K4S)
        for e in k4_edges
            color_count_4[ki, coloring[e]+1] += 1
        end
        for c in 0:2
            total_bad += k4_copies(color_count_4[ki, c+1])
        end
    end

    # --- Simulated annealing with greedy hill-climbing ---
    #
    # Each pass shuffles edge order and evaluates every edge. For each edge, a
    # random alternative color is proposed. Improving moves (delta <= 0) are
    # always accepted; worsening moves are accepted with probability exp(-delta/T).
    # As T→0 this reduces to a greedy hill-climber.
    #
    # Cooling is adaptive: T decays from 1.0 to 1e-4 over exactly max_search_iter
    # passes. At T=1.0 a delta=1 worsening is accepted ~37% of the time.
    #
    # Delta when flipping ei from old_color to new_color across all K4s containing ei:
    #   each K4 ki contributes: +C(k_new, 2) - C(k_old-1, 2)
    #
    edge_order   = collect(1:NUM_HYPEREDGES)
    alt_colors   = [(1, 2), (0, 2), (0, 1)]  # alternatives for colors 0, 1, 2
    T            = 1.2
    cooling_rate = (1e-4)^(1.0 / max_search_iter)

    pass = 0
    while total_bad > 0 && pass < max_search_iter
        pass += 1
        shuffle!(edge_order)

        for ei in edge_order
            old_color = coloring[ei]
            new_color = rand(alt_colors[old_color+1])
            k4s = K4S_CONTAINING[ei]

            delta = 0
            for ki in k4s
                delta += K4_MARGINAL_LOSS[color_count_4[ki, new_color+1] + 1] -
                         K4_MARGINAL_GAIN[color_count_4[ki, old_color+1] + 1]
            end

            if delta <= 0 || rand() < exp(-delta / T)
                coloring[ei] = new_color
                for ki in k4s
                    k_old = color_count_4[ki, old_color+1]
                    total_bad -= K4_MARGINAL_GAIN[k_old + 1]
                    color_count_4[ki, old_color+1] -= 1
                    k_new = color_count_4[ki, new_color+1]
                    total_bad += K4_MARGINAL_LOSS[k_new + 1]
                    color_count_4[ki, new_color+1] += 1
                end
            end

            total_bad == 0 && break
        end

        T *= cooling_rate
    end

    return [convert_adjmat_to_string(coloring)]
end
