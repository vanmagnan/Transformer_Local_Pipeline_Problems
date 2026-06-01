max_search_iter = 50
include("../problem_vertex_ramsey.jl")

using Random
Random.seed!(42)

# ---------------------------------------------------------------------------
# Brute-force helpers (used as ground truth throughout)
# ---------------------------------------------------------------------------

# Return all monochromatic cliques of a given size and color as sorted Vector{Int}s.
function brute_cliques(coloring, k, size, color)
    result = Vector{Int}[]
    for S in combinations(1:k, size)
        if all(coloring[edge_idx(S[a], S[b], k)] == color for a in 1:size-1 for b in a+1:size)
            push!(result, collect(S))
        end
    end
    return result
end

# Build clauses the old brute-force way (ground truth for SAT tests).
function build_clauses_brute(coloring, k)
    clauses = Vector{Tuple{Int,Int}}[]
    for S in combinations(1:k, P-1)
        if all(coloring[edge_idx(S[a], S[b], k)] == 0 for a in 1:P-2 for b in a+1:P-1)
            push!(clauses, [(u, 1) for u in S])
        end
    end
    for S in combinations(1:k, Q-1)
        if all(coloring[edge_idx(S[a], S[b], k)] == 1 for a in 1:Q-2 for b in a+1:Q-1)
            push!(clauses, [(u, 0) for u in S])
        end
    end
    return clauses
end

function all_clauses_satisfied(clauses, x)
    for clause in clauses
        any(x[v] == val for (v, val) in clause) || return false
    end
    return true
end

function exhaustive_sat(clauses, k)
    for bits in 0:(2^k - 1)
        x = [(bits >> (i-1)) & 1 for i in 1:k]
        all_clauses_satisfied(clauses, x) && return x
    end
    return nothing
end

# Normalize a list of cliques (as Vector{Int}s) to a sorted set for comparison.
normalize_cliques(cs) = sort([sort(c) for c in cs])

# ---------------------------------------------------------------------------
# Test: valid_line and empty_starting_point
# ---------------------------------------------------------------------------

@assert valid_line(empty_starting_point())
@assert !valid_line("0010")  # not a valid encoding length

# empty_starting_point is now k0=8, not P-1
sp = empty_starting_point()
coloring0, k0 = parse_graph(sp)
@assert k0 == 8  "expected k0=8, got $k0"
@assert valid_line(sp)

# ---------------------------------------------------------------------------
# Test: build_clique_lists matches brute force
# ---------------------------------------------------------------------------

for trial in 1:10
    k = rand(4:10)
    n_edges = k*(k-1)÷2
    coloring = rand(0:1, n_edges)

    red_cliques, blue_cliques = build_clique_lists(coloring, k)

    # Check each stored clique size against brute force
    for s in 1:P-2
        expected = normalize_cliques(brute_cliques(coloring, k, s+1, 0))
        got      = normalize_cliques(red_cliques[s])
        @assert expected == got  "trial=$trial k=$k: red K$(s+1) mismatch\n  expected=$expected\n  got=$got"
    end

    for s in 1:Q-2
        expected = normalize_cliques(brute_cliques(coloring, k, s+1, 1))
        got      = normalize_cliques(blue_cliques[s])
        @assert expected == got  "trial=$trial k=$k: blue K$(s+1) mismatch\n  expected=$expected\n  got=$got"
    end
end

println("PASS: build_clique_lists matches brute force")

# ---------------------------------------------------------------------------
# Test: update_clique_lists! stays consistent with brute force after vertex additions
# ---------------------------------------------------------------------------

for trial in 1:10
    k = rand(3:8)
    n_edges = k*(k-1)÷2
    coloring = rand(0:1, n_edges)

    red_cliques, blue_cliques = build_clique_lists(coloring, k)

    # Add several vertices and verify clique lists match brute force each time
    for _ in 1:4
        x = rand(0:1, k)

        update_clique_lists!(red_cliques, blue_cliques, k, x)

        # Rebuild coloring for k+1 vertices
        new_coloring = zeros(Int, (k+1)*k÷2)
        for i in 1:k-1, j in i+1:k
            new_coloring[edge_idx(i, j, k+1)] = coloring[edge_idx(i, j, k)]
        end
        for u in 1:k
            new_coloring[edge_idx(u, k+1, k+1)] = x[u]
        end
        coloring = new_coloring
        k += 1

        for s in 1:P-2
            expected = normalize_cliques(brute_cliques(coloring, k, s+1, 0))
            got      = normalize_cliques(red_cliques[s])
            @assert expected == got  "trial=$trial after add: red K$(s+1) mismatch\n  expected=$expected\n  got=$got"
        end

        for s in 1:Q-2
            expected = normalize_cliques(brute_cliques(coloring, k, s+1, 1))
            got      = normalize_cliques(blue_cliques[s])
            @assert expected == got  "trial=$trial after add: blue K$(s+1) mismatch\n  expected=$expected\n  got=$got"
        end
    end
end

println("PASS: update_clique_lists! stays consistent after vertex additions")

# ---------------------------------------------------------------------------
# Integration smoke test: find_valid_vertex_coloring (clique lists → PicoSAT)
# agrees with brute-force clause build + exhaustive SAT.
# Clique correctness is already covered above; this catches wiring bugs.
# ---------------------------------------------------------------------------

for k in 3:12
    n_edges = k*(k-1)÷2
    coloring = rand(0:1, n_edges)

    red_cliques, blue_cliques = build_clique_lists(coloring, k)
    dpll_result = find_valid_vertex_coloring(red_cliques, blue_cliques, k)

    clauses_brute     = build_clauses_brute(coloring, k)
    exhaustive_result = exhaustive_sat(clauses_brute, k)

    dpll_sat = dpll_result !== nothing
    exh_sat  = exhaustive_result !== nothing
    @assert dpll_sat == exh_sat  "k=$k: DPLL=$dpll_sat but exhaustive=$exh_sat"

    if dpll_sat
        @assert all_clauses_satisfied(clauses_brute, dpll_result)  "k=$k: DPLL assignment violates a clause"
    end
end

println("PASS: find_valid_vertex_coloring end-to-end smoke test")

# ---------------------------------------------------------------------------
# Test: greedy_search_from_startpoint grows the graph and returns valid output
# ---------------------------------------------------------------------------

out = greedy_search_from_startpoint(nothing, empty_starting_point())
@assert length(out) == 1
@assert valid_line(out[1])

println("PASS: greedy_search_from_startpoint returns a valid encoding")

# ---------------------------------------------------------------------------
# Test: greedy output has no red K_P or blue K_Q (valid Ramsey coloring)
# ---------------------------------------------------------------------------

for trial in 1:5
    result = greedy_search_from_startpoint(nothing, empty_starting_point())
    obj = result[1]
    coloring_r, kr = parse_graph(obj)

    red_kp = brute_cliques(coloring_r, kr, P, 0)
    @assert isempty(red_kp)  "trial=$trial: found red K$P in output!"

    blue_kq = brute_cliques(coloring_r, kr, Q, 1)
    @assert isempty(blue_kq)  "trial=$trial: found blue K$Q in output!"
end

println("PASS: greedy output contains no red K$P or blue K$Q")

println("\nAll vertex_ramsey tests passed!")
