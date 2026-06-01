include("../problem_4_cycle_free.jl")

println("Testing problem_4_cycle_free.jl (N=$N)")

# Test 1: empty_starting_point has N-1 row separators ('2' characters)
esp = empty_starting_point()
num_twos = count(c -> c == '2', esp)
@assert num_twos == N - 1 "empty_starting_point should have $(N-1) '2' separators, got: $num_twos"
println("PASS: empty_starting_point has $(N-1) '2' separators")

# Test 2: reward_calc of empty graph is 0
r = reward_calc(esp)
@assert r == 0 "reward_calc(empty) should be 0, got: $r"
println("PASS: reward_calc(empty) == 0")

# Test 3: greedy_search returns Vector{String} of length 1
println("Running greedy search from empty (N=$N, may take a moment)...")
result = greedy_search_from_startpoint(nothing, esp)
@assert result isa Vector{String} "greedy search should return Vector{String}"
@assert length(result) == 1 "greedy search should return vector of length 1, got: $(length(result))"
println("PASS: greedy_search returns Vector{String} of length 1")

# Test 4: Output has correct format (N-1 '2' separators)
g = result[1]
num_twos_out = count(c -> c == '2', g)
@assert num_twos_out == N - 1 "Output should have $(N-1) '2' separators, got: $num_twos_out"
println("PASS: greedy output has $(N-1) '2' separators")

# Test 5: Output has no 4-cycles (decode to adjmat and verify)
function decode_to_adjmat(obj::String)
    adjmat = zeros(Int, N, N)
    index = 1
    for i in 1:N-1
        for j in i+1:N
            while obj[index] == '2'
                index += 1
            end
            adjmat[i, j] = parse(Int, obj[index])
            adjmat[j, i] = adjmat[i, j]
            index += 1
        end
    end
    return adjmat
end

adjmat = decode_to_adjmat(g)
four_cycles = find_all_four_cycles(adjmat)
@assert isempty(four_cycles) "Output should have no 4-cycles, found $(length(four_cycles))"
println("PASS: greedy output has no 4-cycles")

println("\nAll 4_cycle_free tests passed!")
