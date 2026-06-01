"""
Convert circulant coloring strings to general upper-triangular adjacency strings.

Circulant format (input):  CIRC_LEN = N÷2 chars, each '0' or '1'.
  Character k gives the color of every edge {i, j} with min(j-i, N-(j-i)) == k.

General format (output):  N*(N-1)/2 + N-1 chars, '0'/'1' with '2' row separators.
  Row i contains colors of edges {i,j} for j = i+1,...,N, followed by '2'.

Usage:
  julia circulant_to_general.jl N input_file output_file
  julia circulant_to_general.jl N input_file        (writes to stdout)

N must be provided explicitly because CIRC_LEN = N÷2 is the same for N and N+1
when N is even (e.g. N=35 and N=34 both give CIRC_LEN=17).
"""

if length(ARGS) < 2
    println(stderr, "Usage: julia circulant_to_general.jl N input_file [output_file]")
    exit(1)
end

const N = parse(Int, ARGS[1])
const CIRC_LEN = N ÷ 2
const EXPECTED_IN_LEN  = CIRC_LEN
const EXPECTED_OUT_LEN = N * (N - 1) ÷ 2 + N - 1

function circulant_to_general(circ_str::String)::String
    entries = Vector{Char}(undef, EXPECTED_OUT_LEN)
    pos = 0
    for i in 1:N-1
        for j in i+1:N
            k = min(j - i, N - (j - i))
            pos += 1
            entries[pos] = circ_str[k]
        end
        pos += 1
        entries[pos] = '2'
    end
    return String(entries)
end

let
    input_file  = ARGS[2]
    output_file = length(ARGS) >= 3 ? ARGS[3] : nothing

    lines_in      = 0
    lines_out     = 0
    lines_skipped = 0

    out_io = output_file === nothing ? stdout : open(output_file, "w")

    try
        open(input_file, "r") do f
            for line in eachline(f)
                lines_in += 1
                if length(line) != EXPECTED_IN_LEN
                    println(stderr, "Warning: line $lines_in has length $(length(line)), expected $EXPECTED_IN_LEN — skipping")
                    lines_skipped += 1
                    continue
                end
                if !all(c in ('0', '1') for c in line)
                    println(stderr, "Warning: line $lines_in contains unexpected characters — skipping")
                    lines_skipped += 1
                    continue
                end
                println(out_io, circulant_to_general(line))
                lines_out += 1
            end
        end
    finally
        if output_file !== nothing
            close(out_io)
        end
    end

    println(stderr, "Done: $lines_in lines read, $lines_out converted, $lines_skipped skipped")
    println(stderr, "Input format:  CIRC_LEN=$CIRC_LEN chars per line")
    println(stderr, "Output format: $EXPECTED_OUT_LEN chars per line (N=$N, upper-triangular + row separators)")
end
