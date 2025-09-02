Purpose: Find a smart sort function to evaluate tuner's benchmarking order

Plan:
1. Collect common GEMM info from existed [data](https://github.com/RattataKing/iree-kernel-benchmark/blob/main/iree_kernel_benchmark/gemmbench/problems.py#L7)
2. Run Tuner for all dispatches and collect candidates generation/compilation/benchmarking info
3. Do some data analysis and reversed engineering to find out what makes top candidates win
4. Design the heuristic sort function

Potential task: use transformer to learn/predict the optimal benchmark time, and check attention weights to see what factors correlated the most.

Notes:
Use `parquet` to store data.

`dispatches.parquet`:
```
dispatch_id        STRING
model_tag          STRING          -- e.g., llama-8b, sdxl
op_kind            STRING          -- matmul/conv/attention, ...
M                  INT32
N                  INT32
K                  INT32
dtype_a            STRING          -- f16/bf16/f32/int8/...
dtype_b            STRING
dtype_acc          STRING
trans_a            STRING          -- 'N'/'T' (logical transpose)
trans_b            STRING
source_mlir_digest STRING          -- hash of the generated MLIR text
notes              STRING          -- optional extra comments
```

`candidates.parquet`:
```
dispatch_id        STRING
candidate_id       STRING
op_kind            STRING          -- contraction/conv/attention
device             STRING          -- MI300X, RX7900XTX, ...
arch               STRING          -- gfx942, gfx1100, ...
cfg STRUCT<                        -- nested, extensible
  matmul_size      : LIST<INT32>,  -- [M,N,K] if relevant to the config
  types            : STRUCT<lhs:STRING, rhs:STRING, res:STRING>,
  vars             : STRUCT<
                      m_vars           : LIST<INT32>,
                      n_vars           : LIST<INT32>,
                      k_vars           : LIST<INT32>,
                      subgroup_m_vars  : LIST<INT32>,
                      subgroup_n_vars  : LIST<INT32>
                    >,
  num_subgroups    : INT32,
  subgroup_size    : INT32,
  intrinsic        : STRUCT<
                      family: STRING,     -- mfma, wmmar3, ...
                      name  : STRING,     -- v_mfma_f32_16x16x16_f16, ...
                      mn    : LIST<INT32>,-- [m,n]
                      k     : INT32
                    >,
  workgroup        : STRUCT<wg_x:INT32, wg_y:INT32, wg_z:INT32>,
  sg_m_cnt         : INT32,
  sg_n_cnt         : INT32,
  pipeline         : STRUCT<stages:INT32, lds_bytes_wg:INT32, prefetch:INT32>,
  memory           : STRUCT<vec_ld_bytes:INT32, vec_st_bytes:INT32>,
  regs             : STRUCT<vgpr_per_thr:INT32, agpr_per_thr:INT32, occupancy_waves_per_cu:INT32>,
  split_k          : INT32,
  mma_intrinsics   : LIST<STRING>,
  extras           : MAP<STRING, STRING>  -- open-ended metadata
>
is_baseline        BOOLEAN
tuner_commit       STRING
rocm_version       STRING
notes              STRING          -- optional extra comments
```

`tuning.parquet`:
```
dispatch_id                        STRING
candidate_id                       STRING
arch                               STRING          -- gfx942, gfx1100, ...
device                             STRING          -- MI300X, RX7900XTX, ...
dispatch_compile_order_in_list     INT32
dispatch_compile_status            STRING
dispatch_compile_error_class       STRING          -- optional: codegen_error, verifier, etc.
dispatch_benchmark_order_in_list   INT32
dispatch_benchmark_status          STRING
dispatch_benchmark_time_ms         FLOAT32
model_compile_order_in_list        INT32
model_compile_status               STRING          -- fail/pass
model_compile_error_class          STRING          -- error message
model_benchmark_order_in_list      INT32
model_benchmark_status             STRING          -- fail/pass
model_benchmark_time_ms            FLOAT32
model_baseline_ms                  FLOAT32
tuner_commit                       STRING
rocm_version                       STRING
notes                              STRING          -- optional extra comments
```
