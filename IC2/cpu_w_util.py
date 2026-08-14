from __future__ import annotations

import csv
import re
import subprocess
import time
from pathlib import Path

EXEC_DIR = Path(
    "/direct/sdcc+u/lchen6/workflow-mini-apps/examples/"
    "DeepDriveMD/Executables"
)
PYTHON = Path("/direct/sdcc+u/lchen6/venv/bin/python")
RESULTS_DIR = Path("/direct/sdcc+u/lchen6/cpu_results")
LOG_DIR = Path("/direct/sdcc+u/lchen6/cpu_logs")
NUM_TRIALS = 3

SCRIPTS = ["new_agent.py", "new_training.py"]
SAMPLES = [100]
#DENSE_DIMS = [2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11]
IO_SIZES = [(2**i) * 1024 * 1024 for i in range(2, 10)]
DENSE_DIMS = [2**i for i in range(2, 16)]
#IO_SIZES = [1024]

TOTAL_RE = re.compile(r"TOTAL_RUNTIME\s+([0-9.eE+-]+)")
TRANSFER_RE = re.compile(r"TOTAL_TRANSFER_H2D\s+([0-9.eE+-]+)")

ENERGY_RE = re.compile(
    r"^\s*([0-9.]+),Joules,power/energy-pkg/",
    re.MULTILINE,
)

CPU_UTILIZATION_RE = re.compile(
    r",([0-9.]+),CPUs utilized$",
    re.MULTILINE,
)
GHZ_RE = re.compile(r"([0-9.]+),GHz", re.IGNORECASE)


def append_row(results_file: Path, row: dict[str, object]) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    exists = results_file.exists()

    with results_file.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def completed_run_ids(results_file: Path) -> set[int]:
    if not results_file.exists():
        return set()

    with results_file.open(newline="") as handle:
        return {
            int(row["run_id"])
            for row in csv.DictReader(handle)
            if row.get("exit_code") == "0"
        }


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    phase_dir = EXEC_DIR / "phase0"
    backup_input = phase_dir / "data_0_0.input_backup.h5"

    
    run_id = 0
    for trial in range(1, NUM_TRIALS + 1):
        results_file = RESULTS_DIR / f"trial_{trial}.csv"
        completed = completed_run_ids(results_file)

        for script_name in SCRIPTS:
            for samples in SAMPLES:
                for dense_dim in DENSE_DIMS:
                    for io_size in IO_SIZES:
                        run_id += 1

                        if run_id in completed:
                            print(f"Skipping completed run {run_id}", flush=True)
                            continue

                        script_path = EXEC_DIR / script_name
                        log_file = LOG_DIR / (
                            f"{run_id:04d}_{script_path.stem}_cpu_"
                            f"samples{samples}_din{dense_dim}_dout{dense_dim}_"
                            f"read{io_size}_write{io_size}.log"
                        )

                        app_cmd = [
                            str(PYTHON),
                            str(script_path),
                            "--device",
                            "cpu",
                            "--num_sample",
                            str(samples),
                            "--instance_index",
                            str(run_id),
                            "--data_root_dir",
                            str(EXEC_DIR),
                            "--dense_dim_in",
                            str(dense_dim),
                            "--dense_dim_out",
                            str(dense_dim),
                            "--read_size",
                            str(io_size),
                            "--write_size",
                            str(io_size),
                            "--preprocess_time",
                            str(0),
                        ]

                        cmd = [
                            "perf",
                            "stat",
                            "-x",
                            ",",
                            "-e",
                            "task-clock,cycles,power/energy-pkg/",
                            "--",
                            *app_cmd,
                        ]

                        print(
                            f"Running {script_name}, samples={samples}",
                            flush=True,
                        )

                        start = time.perf_counter()

                        completed_process = subprocess.run(
                            cmd,
                            cwd=EXEC_DIR,
                            text=True,
                            capture_output=True,
                            check=False,
                        )

                        wall_runtime = time.perf_counter() - start

                        perf_output = completed_process.stderr
                        output = completed_process.stdout + "\n" + perf_output
                        if completed_process.returncode != 0:
                            log_file.write_text(output)

                        cpu_match = CPU_UTILIZATION_RE.search(perf_output)
                        cpu_utilization = (
                            float(cpu_match.group(1))
                            if cpu_match
                            else ""
                        )

                        ghz_match = GHZ_RE.search(perf_output)
                        cpu_ghz = (
                            float(ghz_match.group(1))
                            if ghz_match
                            else ""
                        )

                        runtime_match = TOTAL_RE.search(output)
                        reported_runtime = (
                            float(runtime_match.group(1))
                            if runtime_match
                            else ""
                        )

                        transfer_match = TRANSFER_RE.search(output)
                        transfer_total = (
                            float(transfer_match.group(1))
                            if transfer_match
                            else ""
                        )

                        append_row(
                            results_file,
                            {
                                "trial": trial,
                                "run_id": run_id,
                                "task_type": script_path.stem,
                                "device": "cpu",
                                "num_sample": samples,
                                "dense_dim_in": dense_dim,
                                "dense_dim_out": dense_dim,
                                "read_size_bytes": io_size,
                                "write_size_bytes": io_size,
                                "runtime_wall_s": wall_runtime,
                                "runtime_reported_s": reported_runtime,
                                "transfer_h2d_total_s": transfer_total,
                                "cpu_ghz": cpu_ghz,
                                "cpu_utilization": cpu_utilization,
                                "exit_code": completed_process.returncode,
                                "log_file": str(log_file),
                            }
                        )

                        cpu_text = (
                            f"{cpu_utilization:.3f} CPUs"
                            if isinstance(cpu_utilization, float)
                            else "unavailable"
                        )

                        print(
                            f"Finished with exit={completed_process.returncode}; "
                            f"runtime={wall_runtime:.3f}s; "
                            f"utilization={cpu_text}",
                            flush=True,
                        )


if __name__ == "__main__":
    main()