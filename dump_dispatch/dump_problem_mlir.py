from pathlib import Path
from iree.compiler import ir
from iree_kernel_benchmark.gemmbench.gemm_utils import generate_mlir
from iree_kernel_benchmark.gemmbench import problems
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from dataclasses import dataclass
import hashlib

DEFAULT_DTYPE = "f16"
DEFAULT_RAW_ACC_BOOL = True
ALLOWED_TRANS = {"N", "T"}

@dataclass
class DispatchRecord:
    dispatch_id: str                    # stable UUID/ulid
    model: str                          # e.g., llama, GPT4
    op_tag: str                         # llama13bmatvec, llama70bmatvec, llama13bskinny, etc.
    M: int
    N: int
    K: int
    dtype_a: str                        # f16/bf16/f32/int8...
    dtype_b: str
    dtype_acc: str
    trans_a: str                        # 'N'/'T'
    trans_b: str
    source_mlir_hash: str
    notes: str = ""                     # optional

DISPATCH_SCHEMA = pa.schema([
    ("dispatch_id", pa.string()),
    ("model", pa.string()),
    ("op_tag", pa.string()),
    ("M", pa.int32()),
    ("N", pa.int32()),
    ("K", pa.int32()),
    ("dtype_a", pa.string()),
    ("dtype_b", pa.string()),
    ("dtype_acc", pa.string()),
    ("trans_a", pa.string()),
    ("trans_b", pa.string()),
    ("source_mlir_hash", pa.string()),
    ("notes", pa.string()),
])

EXCLUDE_TAGS = (
    # Tuner doesn't support prefill/skinny gemm
    "llama8b_prefill",
    "llama13bmatvec",
    "llama70bmatvec",
    "llama13bskinny",
    "llama70bskinny",
    "llama70bmemory",
)

def record_from(tag: str, cfg, notes:str="") -> DispatchRecord:
    # [WARNING]: Does not support kDynamic
    M, N, K = cfg.M, cfg.N, cfg.K

    # Basic validation
    if cfg.tA not in ALLOWED_TRANS or cfg.tB not in ALLOWED_TRANS:
        raise ValueError(f"Invalid transpose flags: tA={cfg.tA}, tB={cfg.tB}")

    return DispatchRecord(
        dispatch_id=f"{tag}_{cfg.get_name()}",
        model="", # Init val
        op_tag=tag,
        M=int(M),
        N=int(N),
        K=int(K),
        dtype_a=str(cfg.operand_element_type),
        dtype_b=str(cfg.operand_element_type),
        dtype_acc=str(cfg.accumulator_element_type),
        trans_a=str(cfg.tA),
        trans_b=str(cfg.tB),
        source_mlir_hash="", # Init val
        notes=notes,
    )

def write_dispatches_parquet(records: list[DispatchRecord], path: Path, compression: str = "zstd") -> None:
    rows = [{
        "dispatch_id": r.dispatch_id,
        "model": r.model,
        "op_tag": r.op_tag,
        "M": r.M, "N": r.N, "K": r.K,
        "dtype_a": r.dtype_a, "dtype_b": r.dtype_b, "dtype_acc": r.dtype_acc,
        "trans_a": r.trans_a, "trans_b": r.trans_b,
        "source_mlir_hash": r.source_mlir_hash,
        "notes": r.notes,
    } for r in records]
    table = pa.Table.from_pylist(rows, schema=DISPATCH_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression=compression)

def cal_mlir_hash(mlir_text: str) -> str:
    return hashlib.md5(mlir_text.encode("utf-8")).hexdigest()

def map_op_tag_to_model(op_tag: str) -> str:
    if "llama" in op_tag:
        return "llama"
    if "gpt4" in op_tag:
        return "gpt4"
    if "square" in op_tag:
        return "square"
    if "unet" in op_tag:
        return "unet"
    return ""

# Helper function to exclude some records
def filter_rules_failed(rec:DispatchRecord, verbose:bool=True) -> bool:
    rule = ""
    if rec.trans_b == "N":
        if verbose:
            rule = "rec.trans_b == N"
            print(f"Skipping {rec.dispatch_id} due to filter rule: {rule}.")
        return True
    return False

def main():
    outdir = Path("dump_dispatch")
    outdir.mkdir(parents=True, exist_ok=True)
    mlir_outdir = Path("dump_dispatch/problem_mlir_dump")
    mlir_outdir.mkdir(parents=True, exist_ok=True)
    dtype = DEFAULT_DTYPE
    raw_accumulators = DEFAULT_RAW_ACC_BOOL


    problem_gemm_configs = problems.get_gemm_configs(dtype, raw_accumulators)
    print(f"Excluded op_tags: {EXCLUDE_TAGS}")
    gemm_configs = [(tag, cfg) for tag, cfg in problem_gemm_configs if tag not in EXCLUDE_TAGS]

    # Convert to dispatch records
    # records = [record_from(tag, cfg) for tag, cfg in gemm_configs]

    records: list[DispatchRecord] = []
    # # Generate & write text MLIR
    with ir.Context():
        for tag, cfg in gemm_configs:
            rec = record_from(tag, cfg)
            if filter_rules_failed(rec):
                continue
            mlir_text = generate_mlir(cfg)
            mlir_hash = cal_mlir_hash(mlir_text)
            mlir_path = mlir_outdir / f"{tag}_{cfg.get_name()}.mlir"
            mlir_path.write_text(mlir_text)
            # print("wrote", mlir_path)
            rec.source_mlir_hash = mlir_hash
            rec.model = map_op_tag_to_model(rec.op_tag)
            records.append(rec)

    write_dispatches_parquet(records, outdir / "dispatches.parquet")
    print(f"Wrote {len(records)} rows -> {outdir/'dispatches.parquet'}")

main()
