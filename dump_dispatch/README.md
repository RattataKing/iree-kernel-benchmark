Purpose: Find a smart sort function to evaluate tuner's benchmarking order

Plan:
1. Collect common GEMM info from existed [data](https://github.com/RattataKing/iree-kernel-benchmark/blob/main/iree_kernel_benchmark/gemmbench/problems.py#L7)
2. Run Tuner for all dispatches and collect candidates generation/compilation/benchmarking info
3. Do some data analysis and reversed engineering to find out what makes top candidates win
4. Design the heuristic sort function

Dir Hierarchy
--iree-build
--iree-kernel-benchmark

Run `dump_problem_mlir.py`:
```bash
source ~/iree-build/.env && export PYTHONPATH
export PATH="$(realpath ~/iree-build/tools):$PATH"
cd ~/iree-kernel-benchmark
python -m dump_dispatch.dump_problem_mlir
```


