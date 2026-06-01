include("constants.jl")

"""
Find 2-colorings (red/blue) of the edges of the complete 3-uniform hypergraph
K_N^(3) avoiding red K_4^(3)-e and blue K_5^(3).

K_4^(3)-e is K_4^(3) with one hyperedge removed (3 edges on 4 vertices).
Any 3 of the 4 possible triples on a 4-vertex set form the star of one vertex,
which is isomorphic to K_4^(3)-e. So the number of red copies of K_4^(3)-e
on a 4-vertex set with k red hyperedges is C(k, 3):

    k=0,1,2: 0 copies    k=3: C(3,3)=1    k=4: C(4,3)=4

K_5^(3) is the complete 3-uniform hypergraph on 5 vertices (10 edges).
A 5-vertex set contributes 1 blue copy iff all 10 of its hyperedges are blue.

Reward = -(red K_4^(3)-e copies + blue K_5^(3) copies). Reward 0 = valid.

Object encoding: String of '0'/'1' with '2' vertex-row separators.
For each i = 1,...,N-2, emit all hyperedge colors {i,j,k} with i<j<k in lex
order, then a '2'. Total length = C(N,3) + (N-2). 0 = red, 1 = blue.

Target: R(K_4^(3)-e, K_5^(3)). Try N = 12..15 to probe the frontier.
"""

using Random
using Combinatorics

const N = 10                                   # vertices; change to 12/14/15
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

# All C(N,5) five-vertex cliques, each stored as 10 hyperedge indices.
const ALL_K5S = let
    k5s = Vector{Vector{Int}}()
    for verts in combinations(1:N, 5)
        push!(k5s, [HINDEX[verts[p], verts[q], verts[r]]
                    for p in 1:3 for q in p+1:4 for r in q+1:5])
    end
    k5s
end

# For each hyperedge index, which 5-cliques (into ALL_K5S) contain it.
const K5S_CONTAINING = let
    lookup = [Int[] for _ in 1:NUM_HYPEREDGES]
    for (ki, k5_edges) in enumerate(ALL_K5S)
        for ei in k5_edges; push!(lookup[ei], ki); end
    end
    lookup
end

# Vertex-i row encoding: for each i = 1..N-2, emit all hyperedge colors
# {i,j,k} (j in i+1:N-1, k in j+1:N) in lex order, then a '2' separator.
function convert_adjmat_to_string(coloring::Vector{Int})::String
    entries = []
    for i in 1:N-2
        for j in i+1:N-1
            for k in j+1:N
                push!(entries, string(coloring[HINDEX[i, j, k]]))
            end
        end
        push!(entries, "2")
    end
    return join(entries)
end

function empty_starting_point()::OBJ_TYPE
    coloring = [rand(0:1) for _ in 1:NUM_HYPEREDGES]
    return convert_adjmat_to_string(coloring)
end

# Number of red K_4^(3)-e copies in a 4-vertex set with k red hyperedges = C(k,3).
@inline k4_copies(k::Int) = k >= 3 ? k * (k-1) * (k-2) ÷ 6 : 0

# Marginal copies gained by adding one red edge to a K4 with k current reds:
# C(k+1,3) - C(k,3) = C(k,2).
@inline k4_marginal_red(k::Int) = k >= 2 ? k * (k-1) ÷ 2 : 0

# Marginal copies lost by removing one red edge from a K4 with k current reds:
# C(k,3) - C(k-1,3) = C(k-1,2).
@inline k4_marginal_blue(k::Int) = k >= 3 ? (k-1) * (k-2) ÷ 2 : 0

function reward_calc(obj::OBJ_TYPE)::REWARD_TYPE
    coloring = [parse(Int, c) for c in obj if c != '2']
    total = 0
    for k4_edges in ALL_K4S
        total += k4_copies(count(e -> coloring[e] == 0, k4_edges))
    end
    for k5_edges in ALL_K5S
        all(coloring[e] == 1 for e in k5_edges) && (total += 1)
    end
    return -Float32(total)
end

function greedy_search_from_startpoint(db, obj::OBJ_TYPE)::Vector{OBJ_TYPE}
    if count(c -> c == '2', obj) != N - 2
        return greedy_search_from_startpoint(db, empty_starting_point())
    end
    coloring = [parse(Int, c) for c in obj if c != '2']
    if length(coloring) != NUM_HYPEREDGES
        return greedy_search_from_startpoint(db, empty_starting_point())
    end

    # --- Initialise incremental state ---
    # red_count_4[ki]: red hyperedges in 4-clique ki.
    # blue_count_5[ki]: blue hyperedges in 5-clique ki.
    # total_bad: sum of C(red_count_4[ki],3) over K4s + [blue_count_5[ki]==10] over K5s.
    red_count_4  = zeros(Int, length(ALL_K4S))
    blue_count_5 = zeros(Int, length(ALL_K5S))
    total_bad    = 0

    for (ki, k4_edges) in enumerate(ALL_K4S)
        cnt = count(e -> coloring[e] == 0, k4_edges)
        red_count_4[ki] = cnt
        total_bad += k4_copies(cnt)
    end
    for (ki, k5_edges) in enumerate(ALL_K5S)
        cnt = count(e -> coloring[e] == 1, k5_edges)
        blue_count_5[ki] = cnt
        cnt == 10 && (total_bad += 1)
    end

    # --- Greedy loop: first-improvement with random hyperedge order ---
    #
    # Gain/loss when flipping ei:
    #
    #   flip red→blue: each K4 containing ei loses C(k-1,2) copies  (k = current reds)
    #                  each K5 containing ei gains 1 copy if blue_count reaches 10
    #
    #   flip blue→red: each K5 containing ei loses 1 copy if it was all-blue
    #                  each K4 containing ei gains C(k,2) copies  (k = current reds)
    #
    # Key asymmetry: for a K4 with k=4 reds, flipping one edge to blue reduces
    # copies from C(4,3)=4 to C(3,3)=1, a gain of C(3,2)=3, not 0.
    edge_order = collect(1:NUM_HYPEREDGES)
    pass = 0
    while total_bad > 0 && pass < max_search_iter
        pass += 1
        shuffle!(edge_order)
        found = false
        for ei in edge_order
            if coloring[ei] == 0  # candidate flip: red → blue
                # gain = sum of C(k-1,2) for K4s containing ei
                gain = 0
                for ki in K4S_CONTAINING[ei]
                    gain += k4_marginal_blue(red_count_4[ki])
                end
                gain == 0 && continue
                # loss = new blue K5^(3) copies created
                loss = count(ki -> blue_count_5[ki] == 9, K5S_CONTAINING[ei])
            else                  # candidate flip: blue → red
                # gain = existing all-blue K5s that would be broken
                gain = count(ki -> blue_count_5[ki] == 10, K5S_CONTAINING[ei])
                gain == 0 && continue
                # loss = sum of C(k,2) new K4^(3)-e copies created
                loss = 0
                for ki in K4S_CONTAINING[ei]
                    loss += k4_marginal_red(red_count_4[ki])
                end
            end
            gain > loss || continue

            old_color    = coloring[ei]
            coloring[ei] = 1 - old_color

            # --- Incremental update ---
            if old_color == 0   # flipped red → blue
                for ki in K4S_CONTAINING[ei]
                    k = red_count_4[ki]
                    total_bad     -= k4_marginal_blue(k)   # C(k-1,2) copies removed
                    red_count_4[ki] = k - 1
                end
                for ki in K5S_CONTAINING[ei]
                    blue_count_5[ki] += 1
                    blue_count_5[ki] == 10 && (total_bad += 1)
                end
            else                # flipped blue → red
                for ki in K5S_CONTAINING[ei]
                    old_cnt = blue_count_5[ki]
                    blue_count_5[ki] -= 1
                    old_cnt == 10 && (total_bad -= 1)
                end
                for ki in K4S_CONTAINING[ei]
                    k = red_count_4[ki]
                    total_bad      += k4_marginal_red(k)   # C(k,2) new copies
                    red_count_4[ki] = k + 1
                end
            end

            found = true
            break
        end
        !found && break
    end

    return [convert_adjmat_to_string(coloring)]
end
