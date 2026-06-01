include("constants.jl")

"""
Find 2-colorings (red/blue) of edges of K_N minimizing monochromatic K5 subgraphs.

Related to the Ramsey number R(5,5), known to be between 43 and 48.
For N ≤ 42, there exist 2-colorings with zero monochromatic K5s.

Object encoding: String of '0'/'1' with '2' row separators.
Row i contains the colors of edges {i,j} for j = i+1,...,N, followed by '2'.
0 = red, 1 = blue.
"""

using Random

const N = 10

function edge_index(i::Int, j::Int)::Int
    # Requires i < j, 1-based indexing
    return (i - 1) * (2 * N - i) ÷ 2 + (j - i)
end

function edge_pair(ei::Int)::Tuple{Int,Int}
    # Inverse of edge_index: given a linear index, return (i, j) with i < j
    i = 1
    remaining = ei
    while remaining > N - i
        remaining -= N - i
        i += 1
    end
    return (i, i + remaining)
end

function find_all_mono_k5s(coloring::Vector{Int})
    mono_k5s = Vector{NTuple{5,Int}}()
    for a in 1:N-4
        for b in a+1:N-3
            for c in b+1:N-2
                for d in c+1:N-1
                    for e in d+1:N
                        verts = (a, b, c, d, e)
                        edge_colors = [coloring[edge_index(verts[p], verts[q])] for p in 1:4 for q in p+1:5]
                        if all(col == 0 for col in edge_colors) || all(col == 1 for col in edge_colors)
                            push!(mono_k5s, verts)
                        end
                    end
                end
            end
        end
    end
    return mono_k5s
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
    mono_k5s = find_all_mono_k5s(coloring)
    return -Float32(length(mono_k5s))
end

function empty_starting_point()::OBJ_TYPE
    # Random coloring so parallel searches start from diverse points
    coloring = [rand(0:1) for _ in 1:N*(N-1)÷2]
    return convert_adjmat_to_string(coloring)
end

function greedy_search_from_startpoint(db, obj::OBJ_TYPE)::Vector{OBJ_TYPE}
    expected_len = N * (N - 1) ÷ 2
    num_twos = count(c -> c == '2', obj)
    if num_twos != N - 1
        return greedy_search_from_startpoint(db, empty_starting_point())
    end

    coloring = [parse(Int, c) for c in obj if c != '2']
    if length(coloring) != expected_len
        return greedy_search_from_startpoint(db, empty_starting_point())
    end

    mono_k5s = find_all_mono_k5s(coloring)

    max_iter = 10_000
    iter = 0
    while !isempty(mono_k5s) && iter < max_iter
        iter += 1
        edge_count = Dict{Int,Int}()
        for verts in mono_k5s
            for p in 1:4
                for q in p+1:5
                    ei = edge_index(verts[p], verts[q])
                    edge_count[ei] = get(edge_count, ei, 0) + 1
                end
            end
        end

        # For each candidate edge, compute:
        #   gain = existing mono K5s destroyed by this flip (= edge_count[ei])
        #   loss = new mono K5s created: K5s containing this edge where all 9
        #          other edges already have the new color
        # Keep only edges where gain > loss (strictly improving flips),
        # then select uniformly at random among them.
        improving_edges = Int[]
        for (ei, gain) in edge_count
            i, j = edge_pair(ei)
            new_color = 1 - coloring[ei]
            other_verts = [v for v in 1:N if v != i && v != j]
            loss = 0
            for ai in 1:length(other_verts)-2
                for bi in ai+1:length(other_verts)-1
                    for ci in bi+1:length(other_verts)
                        va, vb, vc = other_verts[ai], other_verts[bi], other_verts[ci]
                        if coloring[edge_index(min(i,va),  max(i,va))]  == new_color &&
                           coloring[edge_index(min(i,vb),  max(i,vb))]  == new_color &&
                           coloring[edge_index(min(i,vc),  max(i,vc))]  == new_color &&
                           coloring[edge_index(min(j,va),  max(j,va))]  == new_color &&
                           coloring[edge_index(min(j,vb),  max(j,vb))]  == new_color &&
                           coloring[edge_index(min(j,vc),  max(j,vc))]  == new_color &&
                           coloring[edge_index(min(va,vb), max(va,vb))] == new_color &&
                           coloring[edge_index(min(va,vc), max(va,vc))] == new_color &&
                           coloring[edge_index(min(vb,vc), max(vb,vc))] == new_color
                            loss += 1
                        end
                    end
                end
            end
            if gain > loss
                push!(improving_edges, ei)
            end
        end

        isempty(improving_edges) && break

        flip_ei = improving_edges[rand(1:length(improving_edges))]
        coloring[flip_ei] = 1 - coloring[flip_ei]
        mono_k5s = find_all_mono_k5s(coloring)
    end

    return [convert_adjmat_to_string(coloring)]
end
