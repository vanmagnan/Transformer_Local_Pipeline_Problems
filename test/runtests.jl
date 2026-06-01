# Run each test file as a separate subprocess.
# Problem files define conflicting function/constant names so they cannot
# be included in the same Julia session.
#
# Usage:
#   julia test/runtests.jl          # run all tests
#   julia test/test_triangle_free.jl  # run a single test

function run_all_tests()
    test_dir = @__DIR__
    test_files = [
        "test_triangle_free.jl",
        "test_4_cycle_free.jl",
        "test_monochromatic_clique.jl",
        "test_asymmetric_ramsey.jl",
        "test_3uniform_ramsey.jl",
        "test_3color_ramsey.jl",
        "test_asymmetric_ramsey_circulant.jl",
        "test_vertex_ramsey.jl",
        "test_kneser_ramsey.jl",
    ]

    all_passed = true
    for file in test_files
        path = joinpath(test_dir, file)
        println("=" ^ 60)
        println("Running $file ...")
        println("=" ^ 60)
        try
            run(`julia $path`)
            println("PASSED: $file\n")
        catch e
            println("FAILED: $file")
            all_passed = false
        end
    end

    println("=" ^ 60)
    if all_passed
        println("All tests passed!")
    else
        println("Some tests failed!")
        exit(1)
    end
end

run_all_tests()
