max_search_iter = 500

include("../problem_3uniform_ramsey.jl")

println("Testing problem_3uniform_ramsey.jl (N=$N)")

num_edges    = N * (N-1) * (N-2) ÷ 6   # = C(N,3)
expected_len = num_edges + N - 2        # hyperedges + (N-2) row separators

# Test 1: correct string length
esp = empty_starting_point()
@assert length(esp) == expected_len "expected length $expected_len, got $(length(esp))"
println("PASS: length(empty_starting_point()) == $expected_len")

# Test 2: only valid characters
@assert all(c in ('0', '1', '2') for c in esp)
println("PASS: only '0'/'1'/'2' characters")

# Test 3: reward ≤ 0
r = reward_calc(esp)
@assert r <= 0 "expected ≤ 0, got $r"
println("PASS: reward_calc(random) <= 0  (got $r)")

# Test 4: all-red coloring
# Each 4-vertex set has k=4 red edges → C(4,3)=4 copies of K_4^(3)-e.
# No all-blue 5-vertex sets.
# Total = 4 * C(N,4).
all_red = convert_adjmat_to_string(zeros(Int, num_edges))
r_red   = reward_calc(all_red)
expected_red = -Float32(4 * binomial(N, 4))
@assert r_red == expected_red "all-red: expected $expected_red, got $r_red"
println("PASS: reward_calc(all-red) == $expected_red  (= -4·C($N,4))")

# Test 5: all-blue coloring
# No red edges → 0 K_4^(3)-e copies.
# Each 5-vertex set is all-blue → C(N,5) blue K_5^(3) copies.
all_blue = convert_adjmat_to_string(ones(Int, num_edges))
r_blue   = reward_calc(all_blue)
expected_blue = -Float32(binomial(N, 5))
@assert r_blue == expected_blue "all-blue: expected $expected_blue, got $r_blue"
println("PASS: reward_calc(all-blue) == $expected_blue  (= -C($N,5))")

# Test 6: single K4 with 3 red edges → exactly 1 copy of K_4^(3)-e
coloring_k4 = zeros(Int, num_edges)   # start all-red
# Flip the last hyperedge of {1,2,3,4} to blue: {2,3,4} → HINDEX[2,3,4]
coloring_k4[HINDEX[2, 3, 4]] = 1
str_k4 = convert_adjmat_to_string(coloring_k4)
r_k4   = reward_calc(str_k4)
# Compared to all-red: we reduced {1,2,3,4}'s copies from 4 to 1 (gain 3).
# All other 4-vertex sets not containing edge {2,3,4}: still 4 red → unchanged.
# 4-vertex sets containing edge {2,3,4}: those are sets {2,3,4,x} for x≠1.
#   For these sets, k drops from 4 to 3 → copies drop from 4 to 1 (gain 3 each).
# Number of 4-vertex sets containing edge {2,3,4}: N-3 (add any one of the N-3
# remaining vertices). Each drops from 4 copies to 1 copy → gain of 3 per set.
# Total copy reduction: 3 * (N-3).
expected_k4 = expected_red + Float32(3 * (N - 3))
@assert r_k4 == expected_k4 "single-blue-edge: expected $expected_k4, got $r_k4"
println("PASS: reward_calc with one blue edge matches expected  (got $r_k4)")

# Test 7: greedy_search returns Vector{String} of length 1
result = greedy_search_from_startpoint(nothing, esp)
@assert result isa Vector{String} && length(result) == 1
println("PASS: greedy_search returns Vector{String} of length 1")

# Test 8: greedy output has correct length and valid characters
g = result[1]
@assert length(g) == expected_len
@assert all(c in ('0', '1', '2') for c in g)
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
coloring_rt  = [rand(0:1) for _ in 1:num_edges]
str_rt       = convert_adjmat_to_string(coloring_rt)
coloring_out = [parse(Int, c) for c in str_rt if c != '2']
@assert coloring_rt == coloring_out "round-trip mismatch"
println("PASS: convert_adjmat_to_string round-trips correctly")

println("\nAll 3uniform_ramsey tests passed!")
