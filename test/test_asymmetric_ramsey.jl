# max_search_iter is a free variable in greedy_search_from_startpoint;
# define it before include so it is available when the function is called.
max_search_iter = 200

include("../problem_asymmetric_ramsey_table.jl")

println("Testing problem_asymmetric_ramsey_table.jl (N=$N, P=$P, Q=$Q)")

num_edges    = N * (N - 1) ÷ 2
expected_len = num_edges + N - 1   # edges + (N-1) row separators '2'

# Test 1: empty_starting_point returns a string of the correct length
esp = empty_starting_point()
@assert length(esp) == expected_len "empty_starting_point() should have length $expected_len, got: $(length(esp))"
println("PASS: length(empty_starting_point()) == $expected_len")

# Test 2: empty_starting_point contains only '0', '1', '2' characters
@assert all(c in ('0', '1', '2') for c in esp) "empty_starting_point() contains unexpected characters"
println("PASS: empty_starting_point() contains only '0'/'1'/'2'")

# Test 3: reward_calc returns a non-positive value (violations are penalised)
r = reward_calc(esp)
@assert r <= 0 "reward_calc should return ≤ 0, got: $r"
println("PASS: reward_calc(empty_starting_point()) <= 0  (got $r)")

# Test 4: all-red coloring has reward == -C(N,P) (every P-clique is forbidden)
all_red = convert_adjmat_to_string(zeros(Int, num_edges))
r_red = reward_calc(all_red)
expected_red = -Float32(binomial(N, P))
@assert r_red == expected_red "reward_calc(all-red) should be $expected_red, got: $r_red"
println("PASS: reward_calc(all-red) == $expected_red")

# Test 5: all-blue coloring has reward == -C(N,Q) (every Q-clique is forbidden)
all_blue = convert_adjmat_to_string(ones(Int, num_edges))
r_blue = reward_calc(all_blue)
expected_blue = -Float32(binomial(N, Q))
@assert r_blue == expected_blue "reward_calc(all-blue) should be $expected_blue, got: $r_blue"
println("PASS: reward_calc(all-blue) == $expected_blue")

# Test 6: greedy_search returns a Vector{String} of length 1
result = greedy_search_from_startpoint(nothing, esp)
@assert result isa Vector{String} "greedy search should return Vector{String}"
@assert length(result) == 1 "greedy search should return vector of length 1, got: $(length(result))"
println("PASS: greedy_search returns Vector{String} of length 1")

# Test 7: greedy output has the correct length and valid characters
g = result[1]
@assert length(g) == expected_len "Output length should be $expected_len, got: $(length(g))"
@assert all(c in ('0', '1', '2') for c in g) "Output should only contain '0'/'1'/'2'"
println("PASS: greedy output has correct length and valid characters")

# Test 8: greedy output has non-positive reward
r_out = reward_calc(g)
@assert r_out <= 0 "Output reward should be <= 0, got: $r_out"
println("PASS: greedy output reward <= 0  (got $r_out)")

# Test 9: malformed inputs fall back gracefully (wrong number of '2' separators)
bad = "0" ^ num_edges   # no '2' separators at all
result_bad = greedy_search_from_startpoint(nothing, bad)
@assert result_bad isa Vector{String} && length(result_bad) == 1
println("PASS: malformed input handled gracefully")

println("\nAll asymmetric_ramsey tests passed!")
