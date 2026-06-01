"""
Tests for circulant_to_general.jl conversion logic.

Verifies that the circulant-to-general conversion is consistent:
  - all-red circulant → all-red general coloring
  - all-blue circulant → all-blue general coloring
  - edge offset mapping is correct for small N
  - output length is N*(N-1)/2 + N-1
  - row separators '2' appear in the right positions
"""

const N = 6    # small N for hand-checkable tests
const CIRC_LEN = N ÷ 2   # 3

function circulant_to_general(circ_str::String, n::Int)::String
    circ_len = n ÷ 2
    expected_out = n * (n - 1) ÷ 2 + n - 1
    entries = Vector{Char}(undef, expected_out)
    pos = 0
    for i in 1:n-1
        for j in i+1:n
            k = min(j - i, n - (j - i))
            pos += 1
            entries[pos] = circ_str[k]
        end
        pos += 1
        entries[pos] = '2'
    end
    return String(entries)
end

# --- Test 1: all-red circulant → all-red general ---
circ_red = "0" ^ CIRC_LEN
gen_red  = circulant_to_general(circ_red, N)
expected_len = N * (N - 1) ÷ 2 + N - 1
@assert length(gen_red) == expected_len "Wrong output length: $(length(gen_red)) != $expected_len"
@assert count(==('0'), gen_red) == N * (N - 1) ÷ 2  "All-red: wrong count of '0'"
@assert count(==('2'), gen_red) == N - 1              "Wrong number of row separators"
@assert !('1' in gen_red)                             "All-red: unexpected '1'"
println("PASS: all-red circulant → all-red general")

# --- Test 2: all-blue circulant → all-blue general ---
circ_blue = "1" ^ CIRC_LEN
gen_blue  = circulant_to_general(circ_blue, N)
@assert count(==('1'), gen_blue) == N * (N - 1) ÷ 2  "All-blue: wrong count of '1'"
@assert !('0' in gen_blue)                             "All-blue: unexpected '0'"
println("PASS: all-blue circulant → all-blue general")

# --- Test 3: row separator positions ---
# Row i (i=1..N-1) has (N-i) edge chars followed by one '2'.
# We verify the '2's appear exactly at the right positions.
let pos = 0
    for i in 1:N-1
        pos += (N - i)   # edge chars
        pos += 1         # separator
        @assert gen_red[pos] == '2' "Row $i separator at position $pos should be '2', got '$(gen_red[pos])'"
    end
    @assert pos == expected_len "Total length mismatch"
end
println("PASS: row separators at correct positions")

# --- Test 4: offset mapping for N=6 ---
# Edge {1,2}: offset min(1, 5) = 1 → color = V[1]
# Edge {1,4}: offset min(3, 3) = 3 → color = V[3]  (antipodal for even N=6)
# Edge {2,5}: offset min(3, 3) = 3 → color = V[3]
circ = "010"   # V[1]=0, V[2]=1, V[3]=0
gen  = circulant_to_general(circ, N)
# Row 1: edges {1,2},{1,3},{1,4},{1,5},{1,6} then '2'
# offsets:      1    2    3    2    1
# colors:       0    1    0    1    0    then 2
@assert gen[1] == '0' "Edge {1,2} (offset 1) should be '0'"
@assert gen[2] == '1' "Edge {1,3} (offset 2) should be '1'"
@assert gen[3] == '0' "Edge {1,4} (offset 3) should be '0'"
@assert gen[4] == '1' "Edge {1,5} (offset 2) should be '1'"
@assert gen[5] == '0' "Edge {1,6} (offset 1) should be '0'"
@assert gen[6] == '2' "Row separator after row 1 should be '2'"
println("PASS: offset mapping correct for N=6, circ='$circ'")

# --- Test 5: symmetry — a circulant coloring is vertex-transitive,
#     so in the general format, every edge at the same offset k has the same color. ---
circ5 = "101"
gen5  = circulant_to_general(circ5, N)
# Check every edge in general output has the color matching its offset in circ5.
let pos = 0
    for i in 1:N-1
        for j in i+1:N
            pos += 1
            k = min(j - i, N - (j - i))
            expected_c = circ5[k]
            @assert gen5[pos] == expected_c "Edge ($i,$j) offset $k: expected '$expected_c', got '$(gen5[pos])'"
        end
        pos += 1  # skip '2'
        @assert gen5[pos] == '2' "Row $i separator should be '2'"
    end
end
println("PASS: all edge offsets map correctly for N=6, circ='$circ5'")

# --- Test 6: N=5 (odd N, CIRC_LEN=2) ---
n5 = 5; circ_len5 = n5 ÷ 2  # 2
circ_n5 = "10"
gen_n5  = circulant_to_general(circ_n5, n5)
expected_len_n5 = n5 * (n5 - 1) ÷ 2 + n5 - 1  # 10 + 4 = 14
@assert length(gen_n5) == expected_len_n5 "N=5 output length should be $expected_len_n5, got $(length(gen_n5))"
# Edge {1,2}: offset min(1,4)=1 → V[1]='1'
# Edge {1,3}: offset min(2,3)=2 → V[2]='0'
@assert gen_n5[1] == '1' "N=5 edge {1,2} (offset 1) should be '1'"
@assert gen_n5[2] == '0' "N=5 edge {1,3} (offset 2) should be '0'"
println("PASS: odd N=5 handled correctly")

println("\nAll circulant_to_general tests passed!")
