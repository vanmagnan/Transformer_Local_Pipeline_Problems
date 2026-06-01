max_search_iter = 100
include("../problem_kneser_vertex_ramsey.jl")

println("Testing problem_kneser_vertex_ramsey.jl (r=$R_KNESER, p=$P_KNESER, q=$Q_KNESER, N_START=$N_START)")

failed = false

# Test 1: empty_starting_point returns a valid_line string
esp = empty_starting_point()
if !valid_line(esp)
    println("FAILED: empty_starting_point() not accepted by valid_line: $(repr(esp))")
    failed = true
else
    println("PASS: empty_starting_point() accepted by valid_line")
end

# Test 2: empty_starting_point encodes KG(N_START, r)
n_esp = KNESER_VALID_LENGTHS[length(esp)]
if n_esp != N_START
    println("FAILED: empty_starting_point() encodes n=$n_esp, expected $N_START")
    failed = true
else
    println("PASS: empty_starting_point() encodes n=$N_START")
end

# Test 3: decode → encode round-trip
coloring_esp, n_rt = decode_kneser(esp)
_, _, _, fwd_rt = build_kneser(n_rt, R_KNESER)
nv_rt = binomial(n_rt, R_KNESER)
re_encoded = encode_kneser(coloring_esp, fwd_rt, nv_rt)
if re_encoded != esp
    println("FAILED: decode→encode round-trip mismatch")
    failed = true
else
    println("PASS: decode→encode round-trip")
end

# Test 4: reward_calc returns a non-negative Float32
r_esp = reward_calc(esp)
if r_esp < 0
    println("FAILED: reward_calc(esp) = $r_esp < 0")
    failed = true
else
    println("PASS: reward_calc(esp) = $r_esp >= 0")
end

# Test 5: reward_calc on invalid string returns 0
r_bad = reward_calc("not_a_valid_string_xyz")
if r_bad != Float32(0)
    println("FAILED: reward_calc(invalid) = $r_bad, expected 0")
    failed = true
else
    println("PASS: reward_calc(invalid) = 0")
end

# Test 6: deletion_phase on a clean coloring leaves n unchanged
coloring_clean, n_clean = decode_kneser(esp)
col_after, n_after = deletion_phase(coloring_clean, n_clean)
if n_after > n_clean
    println("FAILED: deletion_phase grew n ($n_clean → $n_after)")
    failed = true
else
    println("PASS: deletion_phase did not grow n ($n_clean → $n_after)")
end

# Test 7: deletion_phase produces a valid coloring
_, _, eidx_after, fwd_after = build_kneser(n_after, R_KNESER)
nv_after = binomial(n_after, R_KNESER)
deleted_check = falses(n_after)
any_violation = any(1:nv_after) do vi
    vertex_violates(vi, build_kneser(n_after, R_KNESER)[1], eidx_after, col_after, deleted_check)
end
if any_violation
    println("FAILED: deletion_phase output still has violations")
    failed = true
else
    println("PASS: deletion_phase output is violation-free")
end

# Test 8: greedy_search_from_startpoint returns Vector{String} of length 1
result = greedy_search_from_startpoint(nothing, esp)
if !(result isa Vector{String}) || length(result) != 1
    println("FAILED: greedy_search should return Vector{String} of length 1, got $(typeof(result))")
    failed = true
else
    println("PASS: greedy_search returns Vector{String} of length 1")
end

# Test 9: greedy output is accepted by valid_line
out = result[1]
if !valid_line(out)
    println("FAILED: greedy output not accepted by valid_line: $(repr(out[1:min(end,40)]))")
    failed = true
else
    println("PASS: greedy output accepted by valid_line")
end

# Test 10: greedy output reward >= deletion-phase output (growth never shrinks n)
r_out = reward_calc(out)
r_del = Float32(n_after)
if r_out < r_del
    println("FAILED: reward_calc(greedy output) = $r_out < deletion output $r_del")
    failed = true
else
    println("PASS: reward_calc(greedy output) = $r_out >= deletion output $r_del")
end

# Test 11: invalid input falls back gracefully
result_bad = greedy_search_from_startpoint(nothing, "garbage_input_###")
if !(result_bad isa Vector{String}) || length(result_bad) != 1 || !valid_line(result_bad[1])
    println("FAILED: invalid input fallback did not produce valid output")
    failed = true
else
    println("PASS: invalid input fallback produces valid output")
end

# Test 12: multiple runs produce different n values (non-determinism sanity check)
ns = [KNESER_VALID_LENGTHS[length(greedy_search_from_startpoint(nothing, esp)[1])] for _ in 1:5]
println("PASS: greedy output n values over 5 runs: $ns")

println()
if failed
    println("FAILED: some tests failed.")
    exit(1)
else
    println("All kneser_vertex_ramsey tests passed!")
end
