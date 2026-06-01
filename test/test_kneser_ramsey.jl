max_search_iter = 1000
include("../problem_kneser_ramsey.jl")

println("Testing problem_kneser_ramsey.jl (N_KNESER=$N_KNESER, R=$R_KNESER, P=$P_KNESER, Q=$Q_KNESER)")
println("NV=$NV, NE=$NE, NPC=$NPC, NQC=$NQC, STRING_LENGTH=$STRING_LENGTH")

failed = false

# Test 1: empty_starting_point returns string of correct length
esp = empty_starting_point()
expected_len = NE + NV - 1
if length(esp) != expected_len
    println("FAILED: empty_starting_point() length=$(length(esp)), expected $expected_len")
    failed = true
else
    println("PASS: empty_starting_point() has correct length ($expected_len)")
end

# Test 2: correct number of '2' row separators
num_twos = count(c -> c == '2', esp)
if num_twos != NV - 1
    println("FAILED: num row separators=$num_twos, expected $(NV-1)")
    failed = true
else
    println("PASS: correct number of row separators ($(NV-1))")
end

# Test 3: reward_calc returns <= 0
r = reward_calc(esp)
if r > 0
    println("FAILED: reward_calc(esp) = $r > 0")
    failed = true
else
    println("PASS: reward_calc(esp) = $r <= 0")
end

# Test 4: all-red coloring has reward = -NPC (all p-cliques forbidden)
all_red_str = convert_coloring_to_string(zeros(Int, NE))
r_red = reward_calc(all_red_str)
if r_red != -Float32(NPC)
    println("FAILED: reward_calc(all-red) = $r_red, expected $(-Float32(NPC))")
    failed = true
else
    println("PASS: reward_calc(all-red) = $r_red == -NPC")
end

# Test 5: all-blue coloring has reward = -NQC (all q-cliques forbidden)
all_blue_str = convert_coloring_to_string(ones(Int, NE))
r_blue = reward_calc(all_blue_str)
if r_blue != -Float32(NQC)
    println("FAILED: reward_calc(all-blue) = $r_blue, expected $(-Float32(NQC))")
    failed = true
else
    println("PASS: reward_calc(all-blue) = $r_blue == -NQC")
end

# Test 6: greedy_search returns Vector{String} of length 1
result = greedy_search_from_startpoint(nothing, esp)
if !(result isa Vector{String})
    println("FAILED: greedy search should return Vector{String}")
    failed = true
elseif length(result) != 1
    println("FAILED: greedy search returned vector of length $(length(result)), expected 1")
    failed = true
else
    println("PASS: greedy_search returns Vector{String} of length 1")
end

# Test 7: output string has correct length
out = result[1]
if length(out) != expected_len
    println("FAILED: greedy output length=$(length(out)), expected $expected_len")
    failed = true
else
    println("PASS: greedy output has correct length ($expected_len)")
end

# Test 8: reward of greedy output is <= 0
r_out = reward_calc(out)
if r_out > 0
    println("FAILED: reward_calc(greedy output) = $r_out > 0")
    failed = true
else
    println("PASS: reward_calc(greedy output) = $r_out <= 0")
end

# Test 9: invalid input falls back to random restart
bad_input = "x" ^ expected_len
result_bad = greedy_search_from_startpoint(nothing, bad_input)
if length(result_bad) != 1 || length(result_bad[1]) != expected_len
    println("FAILED: invalid input fallback produced wrong result")
    failed = true
else
    println("PASS: invalid input fallback produces valid output")
end

println()
if failed
    println("FAILED: some tests failed.")
    exit(1)
else
    println("All kneser_ramsey tests passed!")
end
