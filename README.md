# PatternBoost: Constructions in Mathematics with a Little Help from AI

This is a fork of [zawagner22/transformers_math_experiments](https://github.com/zawagner22/transformers_math_experiments) with several contributions:

- **Apple Silicon (MPS) support**: the Python training and sampling loop has been retooled for macOS
- **Heavily extended Julia local search**: `search_fc.jl` has been significantly reworked
- **Many new problem files**: including Kneser graph Ramsey problems, variable-N vertex-add search via SAT, 3-color hypergraph Ramsey, asymmetric Ramsey variants, and more

## Overview

PatternBoost alternates between two phases to find new constructions in extremal combinatorics:

1. **Local phase** (Julia): a greedy/SA search optimizes mathematical constructions
2. **Global phase** (Python/PyTorch): a transformer trained on the best constructions generates new seeds for the next iteration

## Requirements

- macOS with Apple Silicon (MPS is used by default)
- Python 3.10+
- Julia 1.8+

**Other platforms:** `fc_loop.py` uses `torch.mps` APIs unconditionally. Running on Linux/CUDA requires patching the MPS calls in `fc_loop.py`.

## Installation

```bash
git clone <this-repo>
cd transformers_math_experiments
pip install torch numpy tokenizers tensorboard psutil
```

Julia packages (run once in the Julia REPL):
```julia
import Pkg
Pkg.add(["Dictionaries", "Plots", "Combinatorics", "Graphs",
         "GraphPlot", "SimpleGraphAlgorithms", "SimpleGraphs",
         "SimpleGraphConverter", "PicoSAT"])
```

## Usage

1. Choose a problem by editing the `include(...)` line near the top of `search_fc.jl`
2. Run PatternBoost:

```bash
python fc_loop.py --exp_name my_experiment --nb_threads 8
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--exp_name` | `debug` | Experiment name (output goes to `checkpoint/<name>/<id>/`) |
| `--nb_threads` | `1` | Julia threads for local search |
| `--nb_local_searches` | `1200` | Parallel searches per generation (divisible by thread count) |
| `--final_database_size` | `50000` | Training set size written after each generation |
| `--max_steps` | `20000` | Transformer training steps per generation |
| `--sample-only` | `500000` | Model samples generated per generation |
| `--temperature` | `1.0` | Sampling temperature |
| `--warmstart_file` | _(none)_ | Seed Julia's first generation from an existing constructions file |
| `--cpu` | `false` | Force CPU instead of MPS |

The script auto-resumes from the latest generation if interrupted.

## Available Problems

| File | Problem |
|------|---------|
| `problem_triangle_free.jl` | Maximize edges in a triangle-free graph |
| `problem_4_cycle_free.jl` | Maximize edges in a C4-free graph |
| `problem_monochromatic_clique.jl` | Edge coloring avoiding monochromatic K5 |
| `problem_monochromatic_clique_table.jl` | Same, with precomputed tables (~86x faster at N=43) |
| `problem_asymmetric_ramsey_table.jl` | Avoid red K_P, blue K_Q in K_N (targets R(4,6)) |
| `problem_3uniform_ramsey.jl` | 3-uniform hypergraph Ramsey coloring |
| `problem_3color_ramsey.jl` | 3-coloring of 3-uniform hyperedges avoiding monochromatic K_4^(3)-e |
| `problem_kneser_ramsey.jl` | Edge coloring of Kneser graph KG(n,r) avoiding monochromatic cliques |
| `problem_vertex_ramsey.jl` | Variable-N vertex-add Ramsey search via SAT |
| `problem_kneser_vertex_ramsey.jl` | Variable-N vertex-add Ramsey search on Kneser graphs via SAT |
| `problem_permanent_avoid_123.jl` | Pattern avoidance in permutations |
| `problem_longestrbpath.jl` | Minimize longest rainbow path |

## Adding a New Problem

1. Create `problem_myproblem.jl` implementing four functions: `empty_starting_point()`, `reward_calc()`, `greedy_search_from_startpoint()`, `convert_adjmat_to_string()`
2. Change the `include(...)` line in `search_fc.jl` to point to it
3. Add a test in `test/test_myproblem.jl` and register it in `test/runtests.jl`

## Testing

```bash
julia test/runtests.jl           # run all tests
julia test/test_triangle_free.jl # run a single test
```
