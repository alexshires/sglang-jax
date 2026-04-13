#!/usr/bin/env python3
"""Summarize March 18 workload-family benchmark JSON artifacts.

This script expects a directory containing per-case JSON outputs from
`benchmark_score_matrix.py`, using case names like:

  - w3_proc.json
  - w3_sp.json
  - s1_split_proc.json
  - s1_split_sp.json
  - s2_split_proc.json
  - s2_split_sp.json
  - s1_fixed_proc.json
  - s1_fixed_sp.json
  - s2_fixed_proc.json
  - s2_fixed_sp.json

It emits a markdown report comparing the current v6e-8 rerun against the
March 18 report baselines and recommending the better execution lane per
workload bucket.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkloadSpec:
    key: str
    title: str
    shape: str
    march18_req_s: float


WORKLOADS: list[WorkloadSpec] = [
    WorkloadSpec("w3", "W3 Target", "1000p / 80x26i, c=100", 9.77),
    WorkloadSpec("s1_split", "S1 Split Proxy", "650p / 256x50i, c=1", 5.27),
    WorkloadSpec("s2_split", "S2 Split Proxy", "800p / 256x50i, c=1", 5.24),
    WorkloadSpec("s1_fixed", "S1 Fixed", "1900p / 500x10i, c=1", 3.56),
    WorkloadSpec("s2_fixed", "S2 Fixed", "2000p / 500x20i, c=1", 3.77),
]


def load_case(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    concurrent = data.get("concurrent", [])
    best_row = max(concurrent, key=lambda row: float(row.get("qps", 0.0)), default={})
    return {
        "path": path,
        "config": data.get("config", {}),
        "best_qps": float(best_row.get("qps", 0.0) or 0.0),
        "best_items_s": float(best_row.get("throughput_items_s", 0.0) or 0.0),
        "single_p50_ms": float(data["single_request"]["latency_s"]["p50_s"]) * 1000.0,
        "single_p99_ms": float(data["single_request"]["latency_s"]["p99_s"]) * 1000.0,
        "gate_pass": bool(data.get("gates", {}).get("pass", False)),
    }


def fmt_req_s(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_md = args.output_md.expanduser().resolve()

    rows: list[dict] = []
    notes: list[str] = []
    for spec in WORKLOADS:
        proc_path = input_dir / f"{spec.key}_proc.json"
        sp_path = input_dir / f"{spec.key}_sp.json"
        proc = load_case(proc_path) if proc_path.exists() else None
        sp = load_case(sp_path) if sp_path.exists() else None

        candidates = [
            (name, case) for name, case in (("proc", proc), ("sp", sp)) if case is not None
        ]
        winner_name = None
        winner = None
        if candidates:
            winner_name, winner = max(candidates, key=lambda item: item[1]["best_qps"])

        best_qps = winner["best_qps"] if winner is not None else None
        delta_vs_march18 = (
            ((best_qps - spec.march18_req_s) / spec.march18_req_s) * 100.0
            if best_qps is not None and spec.march18_req_s > 0
            else None
        )
        rows.append(
            {
                "spec": spec,
                "proc": proc,
                "sp": sp,
                "winner_name": winner_name,
                "winner": winner,
                "delta_vs_march18": delta_vs_march18,
            }
        )

        if proc is not None and sp is not None:
            faster = (
                "single-process appliance"
                if sp["best_qps"] > proc["best_qps"]
                else "process-mode lane"
            )
            notes.append(
                f"`{spec.key}`: {faster} won (`proc={proc['best_qps']:.2f} req/s`, `sp={sp['best_qps']:.2f} req/s`)."
            )

    lines: list[str] = []
    lines.append("# April 6, 2026: v6e-8 March 18 Workload Rerun")
    lines.append("")
    lines.append(
        "This report compares the current `sglang-jax-pr32-v6e8` scorer lanes against the March 18, 2026 "
        "Simplified Approach baselines for the key workload family."
    )
    lines.append("")
    lines.append(
        "| Workload | Shape | March 18 Req/s | Proc Req/s | SP Req/s | Recommended Lane | Best Req/s | Delta vs March 18 | Best P50 (ms) |"
    )
    lines.append("| :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |")
    for row in rows:
        spec = row["spec"]
        proc = row["proc"]
        sp = row["sp"]
        winner = row["winner"]
        winner_name = row["winner_name"]
        recommended = (
            "process-mode direct bulk"
            if winner_name == "proc"
            else "single-process score appliance" if winner_name == "sp" else "-"
        )
        best_p50_ms = winner["single_p50_ms"] if winner is not None else None
        lines.append(
            "| "
            + f"{spec.title} | {spec.shape} | {spec.march18_req_s:.2f} | "
            + f"{fmt_req_s(proc['best_qps'] if proc else None)} | "
            + f"{fmt_req_s(sp['best_qps'] if sp else None)} | "
            + f"{recommended} | "
            + f"{fmt_req_s(winner['best_qps'] if winner else None)} | "
            + f"{fmt_pct(row['delta_vs_march18'])} | "
            + f"{fmt_req_s(best_p50_ms)} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in notes:
        lines.append(f"- {note}")
    lines.append(
        "- The fixed-target single-request P50 values are inflated by end-to-end startup/prefill costs "
        "in this score-only benchmark mode. Use the concurrent Req/s column as the primary lane-selection signal."
    )
    lines.append("")
    lines.append("## Artifact Directory")
    lines.append("")
    lines.append(f"`{input_dir}`")
    lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote markdown report to {output_md}")


if __name__ == "__main__":
    main()
