#!/usr/bin/env python3
"""Benchmark the March 20 branch-comparison targets against the current scorer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "scripts" / "multi_item" / "benchmark_score_matrix.py"


@dataclass(frozen=True)
class Target:
    name: str
    slug: str
    query_len: int
    num_items: int
    item_len: int
    target_rps: float
    page_sizes: tuple[int, ...]
    max_running_requests: tuple[int, ...]
    concurrency_levels: tuple[int, ...]


TARGETS: tuple[Target, ...] = (
    Target(
        name="Production S1 Mean",
        slug="prod_s1_mean",
        query_len=650,
        num_items=128,
        item_len=53,
        target_rps=8.01,
        page_sizes=(16, 32, 64),
        max_running_requests=(32, 64),
        concurrency_levels=(1, 2, 4, 8, 16, 32, 64),
    ),
    Target(
        name="Production S2 Mean",
        slug="prod_s2_mean",
        query_len=1265,
        num_items=128,
        item_len=50,
        target_rps=6.88,
        page_sizes=(16, 32, 64),
        max_running_requests=(32, 64),
        concurrency_levels=(1, 2, 4, 8, 16, 32, 64),
    ),
    Target(
        name="1000p / 100x26i",
        slug="p1000_i100_l26",
        query_len=1000,
        num_items=100,
        item_len=26,
        target_rps=9.77,
        page_sizes=(16, 32, 64),
        max_running_requests=(32, 64),
        concurrency_levels=(1, 2, 4, 8, 16, 32, 64),
    ),
    Target(
        name="W1 High Parallelism",
        slug="w1_high_parallelism",
        query_len=250,
        num_items=500,
        item_len=10,
        target_rps=7.71,
        page_sizes=(16, 32),
        max_running_requests=(64, 128),
        concurrency_levels=(1, 2, 4, 8, 16, 32),
    ),
    Target(
        name="W2 High Density",
        slug="w2_high_density",
        query_len=650,
        num_items=100,
        item_len=50,
        target_rps=16.16,
        page_sizes=(32, 64),
        max_running_requests=(24, 32, 64),
        concurrency_levels=(1, 2, 4, 8, 16, 32, 64),
    ),
    Target(
        name="W3 Long Context",
        slug="w3_long_context",
        query_len=1150,
        num_items=50,
        item_len=80,
        target_rps=16.54,
        page_sizes=(16, 32),
        max_running_requests=(32, 64),
        concurrency_levels=(1, 2, 4, 8, 16, 32, 64),
    ),
    Target(
        name="W4 Heavyweight",
        slug="w4_heavyweight",
        query_len=1150,
        num_items=500,
        item_len=10,
        target_rps=5.50,
        page_sizes=(16, 32),
        max_running_requests=(64, 128),
        concurrency_levels=(1, 2, 4, 8, 16, 32),
    ),
)


def _canonical_hot_shape(num_items: int, item_len: int) -> tuple[int, int]:
    if num_items >= 384:
        batch_size = 512
    elif num_items >= 96:
        batch_size = 128
    else:
        batch_size = 64

    total_tokens = max(1, num_items * item_len)
    for bucket in (4096, 8192, 10240, 12288, 16384):
        if total_tokens <= bucket:
            return batch_size, bucket
    rounded = ((total_tokens + 2047) // 2048) * 2048
    return batch_size, rounded


def _load_best_concurrency_row(path: Path) -> dict:
    payload = json.loads(path.read_text())
    rows = payload.get("concurrent", []) or payload.get("concurrency_results", [])
    if not rows:
        raise RuntimeError(f"No concurrent rows found in {path}")
    return max(rows, key=lambda row: float(row.get("qps", 0.0) or 0.0))


def _write_markdown(output_path: Path, rows: list[dict]) -> None:
    lines = [
        "# March 20 Branch Comparison Targets",
        "",
        "| Target | Report r/s | Best r/s | Margin | Best Config |",
        "| :--- | ---: | ---: | ---: | :--- |",
    ]
    for row in rows:
        cfg = row["best_config"]
        lines.append(
            "| {name} | {target:.2f} | {best:.2f} | {margin:+.2f} | PS={page_size}, MRR={mrr}, C={concurrency} |".format(
                name=row["name"],
                target=row["target_rps"],
                best=row["best_qps"],
                margin=row["best_qps"] - row["target_rps"],
                page_size=cfg["page_size"],
                mrr=cfg["max_running_requests"],
                concurrency=cfg["best_concurrency"],
            )
        )
    output_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="tpu")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--log-level", default="warning")
    parser.add_argument("--cache-timeout-s", type=float, default=60.0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=1)
    parser.add_argument("--requests-per-worker", type=int, default=1)
    parser.add_argument("--download-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--targets",
        default="all",
        help="Comma-separated target slugs to run, or 'all'.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to the next config after a failed benchmark subprocess.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "bench-results-march20-branch-targets" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = (
        {slug.strip() for slug in args.targets.split(",") if slug.strip()}
        if args.targets != "all"
        else {target.slug for target in TARGETS}
    )
    targets = [target for target in TARGETS if target.slug in selected]
    if not targets:
        raise SystemExit("No targets selected.")

    summary_rows: list[dict] = []
    for target in targets:
        hot_bs, hot_tokens = _canonical_hot_shape(target.num_items, target.item_len)
        best_row: dict | None = None
        best_config: dict | None = None
        failures: list[dict] = []

        for page_size in target.page_sizes:
            for max_running_requests in target.max_running_requests:
                result_path = output_dir / (
                    f"{target.slug}_ps{page_size}_mrr{max_running_requests}.json"
                )
                cmd = [
                    sys.executable,
                    str(BENCHMARK),
                    "--model-path",
                    args.model_path,
                    "--device",
                    args.device,
                    "--tp-size",
                    str(args.tp_size),
                    "--dp-size",
                    str(args.dp_size),
                    "--seed",
                    str(args.seed),
                    "--mem-fraction-static",
                    str(args.mem_fraction_static),
                    "--log-level",
                    args.log_level,
                    "--cache-timeout-s",
                    str(args.cache_timeout_s),
                    "--disable-precompile",
                    "--direct-label-only",
                    "--direct-warmup",
                    "--skip-path-probes",
                    "--page-size",
                    str(page_size),
                    "--max-running-requests",
                    str(max_running_requests),
                    "--query-len",
                    str(target.query_len),
                    "--num-items",
                    str(target.num_items),
                    "--item-len",
                    str(target.item_len),
                    "--warmup-runs",
                    str(args.warmup_runs),
                    "--timed-runs",
                    str(args.timed_runs),
                    "--requests-per-worker",
                    str(args.requests_per_worker),
                    "--direct-hot-shape-bs",
                    str(hot_bs),
                    "--direct-hot-shape-tokens",
                    str(hot_tokens),
                    "--concurrency-levels",
                    ",".join(str(x) for x in target.concurrency_levels),
                    "--output-json",
                    str(result_path),
                ]
                if args.download_dir:
                    cmd.extend(["--download-dir", args.download_dir])

                proc = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                )
                if proc.returncode != 0:
                    failures.append(
                        {
                            "page_size": page_size,
                            "max_running_requests": max_running_requests,
                            "returncode": proc.returncode,
                            "stderr": proc.stderr[-4000:],
                        }
                    )
                    if not args.keep_going:
                        raise RuntimeError(
                            f"Benchmark failed for {target.slug} ps={page_size} "
                            f"mrr={max_running_requests}:\n{proc.stderr}"
                        )
                    continue

                row = _load_best_concurrency_row(result_path)
                if best_row is None or float(row.get("qps", 0.0) or 0.0) > float(
                    best_row.get("qps", 0.0) or 0.0
                ):
                    best_row = row
                    best_config = {
                        "page_size": page_size,
                        "max_running_requests": max_running_requests,
                        "hot_shape_bs": hot_bs,
                        "hot_shape_tokens": hot_tokens,
                        "best_concurrency": int(row.get("concurrency", 0) or 0),
                        "result_path": str(result_path),
                    }

        if best_row is None or best_config is None:
            raise RuntimeError(f"No successful benchmarks completed for {target.slug}.")

        summary_rows.append(
            {
                "name": target.name,
                "slug": target.slug,
                "target_rps": target.target_rps,
                "best_qps": float(best_row.get("qps", 0.0) or 0.0),
                "best_ips": float(
                    best_row.get("throughput_items_s", 0.0)
                    or best_row.get("items_per_second", 0.0)
                    or 0.0
                ),
                "best_config": best_config,
                "failures": failures,
            }
        )

    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(json.dumps(summary_rows, indent=2) + "\n")
    _write_markdown(summary_md, summary_rows)

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
