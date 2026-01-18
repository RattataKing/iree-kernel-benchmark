from pathlib import Path
from iree.compiler import ir
from iree_kernel_benchmark.gemmbench.gemm_utils import generate_mlir
from iree_kernel_benchmark.gemmbench import problems
from datetime import datetime
from dataclasses import dataclass
import hashlib
import sys
import os

DTYPE_DUMP_LIST_CDNA3 = ["i8", "i32", "f8E4M3FNUZ", "f16", "f32"]
DTYPE_DUMP_LIST_UDNA4 = ["i8", "i32", "f8E4M3FN", "f16", "f32"]
SUPPORT_LIST = ["gfx942", "gfx1201", "gfx950"]
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
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m dump_dispatch.dump_problem_mlir <arch>\nExample: python compile_dump_exe.py gfx942")
    arch = sys.argv[1]
    assert arch in SUPPORT_LIST
    match arch:
        case "gfx942":
            dtype_dump_list = DTYPE_DUMP_LIST_CDNA3
        case "gfx950" | "gfx1201":
            dtype_dump_list = DTYPE_DUMP_LIST_UDNA4
        case _:
            assert False

    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    mlir_outdir = base_path / f"problem_mlir_dump_{arch}"
    mlir_outdir.mkdir(parents=True, exist_ok=True)

    raw_accumulators = DEFAULT_RAW_ACC_BOOL
    records: list[DispatchRecord] = []
    for dtype in dtype_dump_list:
        problem_gemm_configs = problems.get_gemm_configs(dtype, raw_accumulators)
        print(f"Excluded op_tags: {EXCLUDE_TAGS}")
        gemm_configs = [(tag, cfg) for tag, cfg in problem_gemm_configs if tag not in EXCLUDE_TAGS]

        # Convert to dispatch records
        # records = [record_from(tag, cfg) for tag, cfg in gemm_configs]
        # # Generate & write text MLIR
        with ir.Context():
            for tag, cfg in gemm_configs:
                rec = record_from(tag, cfg)
                if filter_rules_failed(rec):
                    continue
                mlir_text = generate_mlir(cfg)
                mlir_hash = cal_mlir_hash(mlir_text)
                filename = f"{tag}_{cfg.get_name()}.mlir" if DEFAULT_RAW_ACC_BOOL else f"{tag}_{cfg.get_name()}_acc{DEFAULT_DTYPE}.mlir"
                mlir_path = mlir_outdir / filename
                mlir_path.write_text(mlir_text)
                # print("wrote", mlir_path)
                rec.source_mlir_hash = mlir_hash
                rec.model = map_op_tag_to_model(rec.op_tag)
                records.append(rec)
        print(f"Dumped {filename}")

    print(f"Wrote {len(records)} mlir")

main()
