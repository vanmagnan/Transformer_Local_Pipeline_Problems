include("../problem_monochromatic_clique.jl")

println("Testing problem_monochromatic_clique.jl (N=$N)")

num_edges = N * (N - 1) ÷ 2

# Test 1: empty_starting_point returns "0"^(N*(N-1)/2)
esp = empty_starting_point()
@assert esp == "0"^num_edges "empty_starting_point() should be \"0\"^$num_edges, got: $(repr(esp))"
println("PASS: empty_starting_point() == \"0\"^$num_edges")

# Test 2: reward_calc("0"^45) == -C(10,5) == -252 (all red → all K5s are monochromatic)
r_all_red = reward_calc("0"^num_edges)
expected = -Float32(binomial(N, 5))
@assert r_all_red == expected "reward_calc(all-red) should be $expected, got: $r_all_red"
println("PASS: reward_calc(all-red) == $expected")

# Test 3: reward_calc("1"^45) == -252 (all blue → same count)
r_all_blue = reward_calc("1"^num_edges)
@assert r_all_blue == expected "reward_calc(all-blue) should be $expected, got: $r_all_blue"
println("PASS: reward_calc(all-blue) == $expected")

# Test 4: greedy_search returns Vector{String} of length 1
result = greedy_search_from_startpoint(nothing, esp)
@assert result isa Vector{String} "greedy search should return Vector{String}"
@assert length(result) == 1 "greedy search should return vector of length 1, got: $(length(result))"
println("PASS: greedy_search returns Vector{String} of length 1")

# Test 5: Output has reward 0 (0 monochromatic K5s, achievable since R(5,5) > 10)
g = result[1]
r_out = reward_calc(g)
@assert r_out == 0 "Output should have 0 monochromatic K5s (reward=0), got: $r_out"
println("PASS: greedy output has reward 0 (no monochromatic K5s)")

# Test 6: Output has correct length and only '0'/'1' characters
@assert length(g) == num_edges "Output length should be $num_edges, got: $(length(g))"
@assert all(c in ('0', '1') for c in g) "Output should only contain '0'/'1' characters"
println("PASS: output has length $num_edges and only '0'/'1' characters")

println("\nAll monochromatic_clique tests passed!")
