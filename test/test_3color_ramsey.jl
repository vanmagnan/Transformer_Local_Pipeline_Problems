max_search_iter = 1000

include("../problem_3color_ramsey.jl")

println("Testing problem_3color_ramsey.jl (N=$N)")

num_edges    = N * (N-1) * (N-2) ÷ 6   # = C(N,3)
expected_len = num_edges + N - 2        # hyperedges + (N-2) row separators '3'

# Test 1: correct string length
esp = empty_starting_point()
@assert length(esp) == expected_len "expected length $expected_len, got $(length(esp))"
println("PASS: length(empty_starting_point()) == $expected_len")

# Test 2: only valid characters
@assert all(c in ('0', '1', '2', '3') for c in esp)
println("PASS: only '0'/'1'/'2'/'3' characters")

# Test 3: reward ≤ 0
r = reward_calc(esp)
@assert r <= 0 "expected ≤ 0, got $r"
println("PASS: reward_calc(random) <= 0  (got $r)")

# Test 4: all-red coloring
# Each 4-vertex set has k=4 red edges → C(4,3)=4 copies. 0 blue/green copies.
# Total = 4 * C(N,4).
all_red          = convert_adjmat_to_string(zeros(Int, num_edges))
r_red            = reward_calc(all_red)
expected_monocolor = -Float32(4 * binomial(N, 4))
@assert r_red == expected_monocolor "all-red: expected $expected_monocolor, got $r_red"
println("PASS: reward_calc(all-red) == $expected_monocolor  (= -4·C($N,4))")

# Test 5: all-green coloring — symmetric to all-red by the 3-color symmetry.
all_green = convert_adjmat_to_string(fill(2, num_edges))
r_green   = reward_calc(all_green)
@assert r_green == expected_monocolor "all-green: expected $expected_monocolor, got $r_green"
println("PASS: reward_calc(all-green) == $expected_monocolor")

# Test 6: one blue edge in an otherwise all-red coloring.
# Edge {1,2,3} belongs to (N-3) four-vertex cliques {1,2,3,x}.
# For each such clique: red count drops 4→3 → copies drop C(4,3)→C(3,3), gain 3.
# Blue count rises 0→1 → k4_copies(1) = 0, no new copies added.
# Expected reward = expected_monocolor + 3*(N-3).
coloring_one_blue             = zeros(Int, num_edges)
coloring_one_blue[HINDEX[1, 2, 3]] = 1
r_one_blue   = reward_calc(convert_adjmat_to_string(coloring_one_blue))
expected_one_blue = expected_monocolor + Float32(3 * (N - 3))
@assert r_one_blue == expected_one_blue "one-blue-edge: expected $expected_one_blue, got $r_one_blue"
println("PASS: reward_calc with one blue edge matches expected  (got $r_one_blue)")

# Test 7: greedy_search returns Vector{String} of length 1
result = greedy_search_from_startpoint(nothing, esp)
@assert result isa Vector{String} && length(result) == 1
println("PASS: greedy_search returns Vector{String} of length 1")

# Test 8: greedy output has correct length and valid characters
g = result[1]
@assert length(g) == expected_len
@assert all(c in ('0', '1', '2', '3') for c in g)
println("PASS: greedy output has correct length and valid characters")

# Test 9: greedy output has non-positive reward
r_out = reward_calc(g)
@assert r_out <= 0
println("PASS: greedy output reward <= 0  (got $r_out)")

# Test 10: malformed input falls back gracefully
fallback = greedy_search_from_startpoint(nothing, "0"^num_edges)
@assert fallback isa Vector{String} && length(fallback) == 1
println("PASS: malformed input handled gracefully")

# Test 11: round-trip encoding
coloring_rt  = [rand(0:2) for _ in 1:num_edges]
str_rt       = convert_adjmat_to_string(coloring_rt)
coloring_out = [parse(Int, c) for c in str_rt if c != '3']
@assert coloring_rt == coloring_out "round-trip mismatch"
println("PASS: convert_adjmat_to_string round-trips correctly")

# Test 12: stress test — N=10 is well below the Ramsey frontier so valid
# 3-colorings (reward=0) exist. Run 50 searches and assert at least one finds one.
println("\nRunning stress test (300 searches from random starts)...")
best = maximum(reward_calc(greedy_search_from_startpoint(nothing, empty_starting_point())[1]) for _ in 1:300)
@assert best == 0 "stress test: expected at least one valid coloring (reward=0), best was $best"
println("PASS: found valid 3-coloring (reward=0) within 300 searches")

println("\nAll 3color_ramsey tests passed!")
