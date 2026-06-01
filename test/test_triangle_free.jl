include("../problem_triangle_free.jl")

println("Testing problem_triangle_free.jl (N=$N)")

# Test 1: empty_starting_point returns "0"^(N*(N-1)/2)
esp = empty_starting_point()
@assert esp == "0"^45 "empty_starting_point() should be \"0\"^45, got: $(repr(esp))"
println("PASS: empty_starting_point() == \"0\"^45")

# Test 2: reward_calc of empty graph is 0
r = reward_calc(esp)
@assert r == 0 "reward_calc(empty) should be 0, got: $r"
println("PASS: reward_calc(empty) == 0")

# Test 3: K_{5,5} has 25 edges (maximum triangle-free graph on 10 vertices, Turán's theorem)
adjmat_k55 = zeros(Int, 10, 10)
for i in 1:5, j in 6:10
    adjmat_k55[i, j] = 1
    adjmat_k55[j, i] = 1
end
k55_str = convert_adjmat_to_string(adjmat_k55)
r3 = reward_calc(k55_str)
@assert r3 == 25 "reward_calc(K_{5,5}) should be 25, got: $r3"
println("PASS: reward_calc(K_{5,5}) == 25")

# Test 4: greedy_search returns Vector{String} of length 1
result = greedy_search_from_startpoint(nothing, esp)
@assert result isa Vector{String} "greedy search should return Vector{String}"
@assert length(result) == 1 "greedy search should return vector of length 1, got: $(length(result))"
println("PASS: greedy_search returns Vector{String} of length 1")

# Test 5: Result is stable (applying greedy again doesn't change the reward)
g1 = result[1]
r1 = reward_calc(g1)
g2 = greedy_search_from_startpoint(nothing, g1)[1]
r2 = reward_calc(g2)
@assert r1 == r2 "Greedy output should be stable under re-application: $r1 vs $r2"
println("PASS: greedy result is stable under re-application (reward=$r1)")

# Test 6: Starting from all-ones ("1"^45) produces a triangle-free graph with positive reward
result2 = greedy_search_from_startpoint(nothing, "1"^45)
r6 = reward_calc(result2[1])
@assert r6 > 0 "Greedy from all-ones should produce graph with positive reward, got: $r6"
println("PASS: greedy from all-ones produces positive reward ($r6)")

println("\nAll triangle_free tests passed!")
