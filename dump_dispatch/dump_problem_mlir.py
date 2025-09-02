from pathlib import Path
from iree.compiler import ir
from iree_kernel_benchmark.gemmbench.gemm_utils import generate_mlir, kDynamic
from iree_kernel_benchmark.gemmbench import problems
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone
from dataclasses import dataclass, field



# iree_kernel_benchmark.gemmbench.gemm_utils.generate_mlir()
# iree_kernel_benchmark.gemmbench.problems

DEFAULT_DTYPE = "f16"
DEFAULT_RAW_ACC_BOOL = True
KDYNAMIC = kDynamic
ALLOWED_TRANS = {"N", "T"}

@dataclass
class DispatchRecord:
    dispatch_id: str                    # stable UUID/ulid
    model_tag: str                      # e.g., llama-8b, sdxl
    op_kind: str                        # qkv_proj, attn_out, ffn_up, conv_im2col, etc.
    M: int
    N: int
    K: int
    dtype_a: str                        # f16/bf16/f32/int8...
    dtype_b: str
    dtype_acc: str
    trans_a: str                        # 'N'/'T'
    trans_b: str
    notes: str = ""                     # optional
    created_ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: int = 1             # start at 1

DISPATCH_SCHEMA = pa.schema([
    ("dispatch_id", pa.string()),
    ("model_tag", pa.string()),
    ("op_kind", pa.string()),
    ("M", pa.int32()),
    ("N", pa.int32()),
    ("K", pa.int32()),
    ("dtype_a", pa.string()),
    ("dtype_b", pa.string()),
    ("dtype_acc", pa.string()),
    ("trans_a", pa.string()),
    ("trans_b", pa.string()),
    ("notes", pa.string()),
    ("created_ts", pa.timestamp("us", tz="UTC")),
    ("schema_version", pa.int16()),
])

EXCLUDE_TAGS = (
    "llama8b_prefill",
    # "llama13bmatvec",
    # "llama70bmatvec",
    # "llama13bskinny",
    # "llama70bskinny",
    # "llama70bmemory",
)

def record_from(tag: str, cfg) -> DispatchRecord:
    # [WARNING]: Does not support kDynamic
    M, N, K = cfg.M, cfg.N, cfg.K

    # Basic validation
    if cfg.tA not in ALLOWED_TRANS or cfg.tB not in ALLOWED_TRANS:
        raise ValueError(f"Invalid transpose flags: tA={cfg.tA}, tB={cfg.tB}")

    return DispatchRecord(
        dispatch_id=f"{tag}_{cfg.get_name()}",
        model_tag=tag,
        op_kind="contraction",
        M=int(M),
        N=int(N),
        K=int(K),
        dtype_a=str(cfg.operand_element_type),
        dtype_b=str(cfg.operand_element_type),
        dtype_acc=str(cfg.accumulator_element_type),
        trans_a=str(cfg.tA),
        trans_b=str(cfg.tB),
        notes="",
    )

def write_dispatches_parquet(records: list[DispatchRecord], path: Path, compression: str = "zstd") -> None:
    rows = [{
        "dispatch_id": r.dispatch_id,
        "model_tag": r.model_tag,
        "op_kind": r.op_kind,
        "M": r.M, "N": r.N, "K": r.K,
        "dtype_a": r.dtype_a, "dtype_b": r.dtype_b, "dtype_acc": r.dtype_acc,
        "trans_a": r.trans_a, "trans_b": r.trans_b,
        "notes": r.notes,
        "created_ts": r.created_ts,            # tz-aware UTC datetime
        "schema_version": int(r.schema_version),
    } for r in records]
    table = pa.Table.from_pylist(rows, schema=DISPATCH_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression=compression)

def main():
    outdir = Path("dump_dispatch")
    outdir.mkdir(parents=True, exist_ok=True)
    mlir_outdir = Path("dump_dispatch/problem_mlir_dump")
    mlir_outdir.mkdir(parents=True, exist_ok=True)
    dtype = DEFAULT_DTYPE
    raw_accumulators = DEFAULT_RAW_ACC_BOOL


    problem_gemm_configs = problems.get_gemm_configs(dtype, raw_accumulators)
    gemm_configs = [(model, cfg) for model, cfg in problem_gemm_configs if model not in EXCLUDE_TAGS]

    # for tag, cfg in gemm_configs:
    #     print(cfg.get_name())

    # Convert to dispatch records
    records = [record_from(tag, cfg) for tag, cfg in gemm_configs]
    # # print(records)

    write_dispatches_parquet(records, outdir / "dispatches.parquet")
    print(f"Wrote {len(records)} rows -> {outdir/'dispatches.parquet'}")

    # Generate & write text MLIR
    with ir.Context():
        for tag, cfg in gemm_configs:
            mlir_text = generate_mlir(cfg)
            out_path = mlir_outdir / f"{tag}_{cfg.get_name()}.mlir"
            out_path.write_text(mlir_text)
            print("wrote", out_path)

main()
