include("constants.jl")
using Random, Combinatorics, PicoSAT

const P = 4   # red clique size to avoid
const Q = 6   # blue clique size to avoid, consider setting P<=Q

# Override valid_line: accept any valid variable-length graph encoding
function valid_line(line::String)
    len = length(line)
    len == 0 && return false
    k = 1
    while k*(k-1)÷2 + k-1 < len
        k += 1
    end
    k*(k-1)÷2 + k-1 == len || return false
    k >= 2 || return false
    all(c in ('0','1','2') for c in line) || return false
    count(==('2'), line) == k - 1 || return false
    return true
end

function edge_idx(i::Int, j::Int, k::Int)::Int
    # i < j, 1-based. Returns flat index into upper-triangular array of size k*(k-1)/2.
    return (i-1)*(2*k-i)÷2 + (j-i)
end

function parse_graph(obj::String)
    len = length(obj)
    k = 1
    while k*(k-1)÷2 + k-1 < len
        k += 1
    end
    colors = [parse(Int, c) for c in obj if c != '2']
    return colors, k
end

function encode_graph(coloring::Vector{Int}, k::Int)::String
    entries = String[]
    for i in 1:k-1
        for j in i+1:k
            push!(entries, string(coloring[edge_idx(i,j,k)]))
        end
        push!(entries, "2")
    end
    return join(entries)
end

function convert_adjmat_to_string(coloring::Vector{Int}, k::Int)::String
    return encode_graph(coloring, k)
end

function empty_starting_point()::OBJ_TYPE
    k0 = 8
    coloring = [rand(0:1) for _ in 1:k0*(k0-1)÷2]
    return encode_graph(coloring, k0)
end

function vertex_in_violation(v::Int, coloring::Vector{Int}, k::Int)::Bool
    others = [u for u in 1:k if u != v]
    red_nbrs = [u for u in others if coloring[edge_idx(min(v,u), max(v,u), k)] == 0]
    for S in combinations(red_nbrs, P-1)
        if all(coloring[edge_idx(S[a], S[b], k)] == 0
               for a in 1:P-2 for b in a+1:P-1)
            return true
        end
    end
    blue_nbrs = [u for u in others if coloring[edge_idx(min(v,u), max(v,u), k)] == 1]
    for S in combinations(blue_nbrs, Q-1)
        if all(coloring[edge_idx(S[a], S[b], k)] == 1
               for a in 1:Q-2 for b in a+1:Q-1)
            return true
        end
    end
    return false
end

function deletion_phase(coloring::Vector{Int}, k::Int)
    i = 1
    while i <= k
        if vertex_in_violation(i, coloring, k)
            new_coloring = Int[]
            for a in 1:k
                a == i && continue
                for b in a+1:k
                    b == i && continue
                    push!(new_coloring, coloring[edge_idx(a, b, k)])
                end
            end
            coloring = new_coloring
            k -= 1
            # Don't increment i: position i now holds the next vertex
        else
            i += 1
        end
    end
    return coloring, k
end

function reward_calc(obj::OBJ_TYPE)::REWARD_TYPE
    valid_line(obj) || return Float32(0)
    coloring, k = parse_graph(obj)
    coloring, k = deletion_phase(coloring, k)
    return Float32(k)
end


# Clique lists are 1-indexed: entry i holds all cliques of size i+1.
# red_cliques[1] = K2 edges, ..., red_cliques[P-2] = K_{P-1}
# blue_cliques[1] = K2 edges, ..., blue_cliques[Q-2] = K_{Q-1}
function build_clique_lists(coloring::Vector{Int}, k::Int)
    red_cliques  = [Vector{Int}[] for _ in 1:P-2]
    blue_cliques = [Vector{Int}[] for _ in 1:Q-2]

    for i in 1:k-1, j in i+1:k
        c = coloring[edge_idx(i, j, k)]
        c == 0 && push!(red_cliques[1],  [i, j])
        c == 1 && push!(blue_cliques[1], [i, j])
    end

    for i in 2:P-2
        for clique in red_cliques[i-1]
            for j in clique[end]+1:k
                if all(coloring[edge_idx(clique[a], j, k)] == 0 for a in 1:length(clique))
                    push!(red_cliques[i], push!(copy(clique), j))
                end
            end
        end
    end

    for i in 2:Q-2
        for clique in blue_cliques[i-1]
            for j in clique[end]+1:k
                if all(coloring[edge_idx(clique[a], j, k)] == 1 for a in 1:length(clique))
                    push!(blue_cliques[i], push!(copy(clique), j))
                end
            end
        end
    end

    return red_cliques, blue_cliques
end

# Incrementally extend clique lists after adding vertex k+1 with edge assignment x.
# x[u] = color of edge (u, k+1).  coloring/k still describe the OLD k-vertex graph.
# Process largest cliques first so newly appended K2 edges don't pollute K3 builds.
function update_clique_lists!(red_cliques, blue_cliques, k, x)
    for i in P-2:-1:2
        for clique in red_cliques[i-1]
            all(x[v] == 0 for v in clique) && push!(red_cliques[i], push!(copy(clique), k+1))
        end
    end
    for u in 1:k; x[u] == 0 && push!(red_cliques[1], [u, k+1]); end

    for i in Q-2:-1:2
        for clique in blue_cliques[i-1]
            all(x[v] == 1 for v in clique) && push!(blue_cliques[i], push!(copy(clique), k+1))
        end
    end
    for u in 1:k; x[u] == 1 && push!(blue_cliques[1], [u, k+1]); end
end

function find_valid_vertex_coloring(red_cliques, blue_cliques, k::Int)
    clauses = Vector{Int}[]
    for clique in red_cliques[P-2]
        push!(clauses, [u for u in clique])    # x[u]=1 (blue): positive literal
    end
    for clique in blue_cliques[Q-2]
        push!(clauses, [-u for u in clique])   # x[u]=0 (red): negative literal
    end
    result = PicoSAT.solve(clauses; vars=k)
    result === :unsatisfiable && return nothing
    return [result[u] > 0 ? 1 : 0 for u in 1:k]
end

function greedy_search_from_startpoint(db, obj::OBJ_TYPE)::Vector{OBJ_TYPE}
    valid_line(obj) || return [empty_starting_point()]
    coloring, k = parse_graph(obj)
    coloring, k = deletion_phase(coloring, k)

    red_cliques, blue_cliques = build_clique_lists(coloring, k)

    for _ in 1:max_search_iter
        x = find_valid_vertex_coloring(red_cliques, blue_cliques, k)
        x === nothing && break

        # Update clique lists before rebuilding coloring (needs old k).
        update_clique_lists!(red_cliques, blue_cliques, k, x)

        # Rebuild coloring with correct edge_idx layout for k+1 vertices.
        new_coloring = zeros(Int, (k + 1) * k ÷ 2)
        for i in 1:k-1
            for j in i+1:k
                new_coloring[edge_idx(i, j, k + 1)] = coloring[edge_idx(i, j, k)]
            end
        end
        for u in 1:k
            new_coloring[edge_idx(u, k + 1, k + 1)] = x[u]
        end
        coloring = new_coloring
        k += 1
    end

    return [encode_graph(coloring, k)]
end
