from __future__ import annotations
import csv
import re
import subprocess
import os
import time
from pathlib import Path

EXEC_DIR = Path("/direct/sdcc+u/lchen6/workflow-mini-apps/examples/DeepDriveMD/Executables")
PYTHON = Path("/direct/sdcc+u/lchen6/venv/bin/python")
RESULTS_DIR = Path("/direct/sdcc+u/lchen6/gpu_results")
LOG_DIR = Path("/direct/sdcc+u/lchen6/gpu_logs")

SCRIPTS = ["new_agent.py", "new_training.py"]
SAMPLES = [100]
#DENSE_DIMS = [2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11]
IO_SIZES = [(2**i) * 1024 * 1024 for i in range(2, 10)]
NUM_TRIALS = 3
DENSE_DIMS = [2**i for i in range(2, 16)]
#IO_SIZES = [1024]

TOTAL_RE = re.compile(r"TOTAL_RUNTIME\s+([0-9.eE+-]+)")
TRANSFER_H2D_RE = re.compile(r"TOTAL_TRANSFER_H2D\s+([0-9.eE+-]+)")
TRANSFER_D2H_RE = re.compile(r"TOTAL_TRANSFER_D2H\s+([0-9.eE+-]+)")
ENERGY_RE = re.compile(
    r"^\s*([0-9.,]+),Joules,power/energy-pkg/",
    re.MULTILINE,
)

def append_row(results_file: Path,row: dict[str, object],) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    exists = results_file.exists()

    with results_file.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(row.keys()),
        )
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

   

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    
    gpu_id = cuda_visible_devices.split(",")[0].strip()


    run_id = 0
    for trial in range(1, NUM_TRIALS+1):
        results_file = RESULTS_DIR / f"trial_{trial}.csv"
        completed_ids = completed_run_ids(results_file)
        for script_name in SCRIPTS:
            for samples in SAMPLES:
                for dense_dim in DENSE_DIMS:
                    for io_size in IO_SIZES:
                
                        run_id += 1

                        if run_id in completed_ids:
                            print(f"skipping {run_id}", flush=True)
                            continue
                        script_path = EXEC_DIR / script_name
                        log_file = LOG_DIR / (
                            f"{run_id:04d}_{script_path.stem}_gpu_"
                            f"samples{samples}_din{dense_dim}_dout{dense_dim}_"
                            f"read{io_size}_write{io_size}.log"
                        )

                        app_cmd = [
                            str(PYTHON),
                            str(script_path),
                            "--device", "gpu",
                            "--num_sample", str(samples),
                            "--instance_index", str(run_id),
                            "--data_root_dir", str(EXEC_DIR),
                            "--dense_dim_in", str(dense_dim),
                            "--dense_dim_out", str(dense_dim),
                            "--read_size", str(io_size),
                            "--write_size", str(io_size),
                            "--preprocess_time", str(0)
                        ]

                        cmd = [
                            "perf",
                            "stat",
                            "-x", ",",
                            "-e", "power/energy-pkg/",
                            "--",
                            *app_cmd,
                        ]

                        print(f"Running {script_name}, samples={samples}", flush=True)

                        start = time.perf_counter()
                        process = subprocess.Popen(
                            cmd,
                            cwd=EXEC_DIR,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )

                        gpu_util_samples: list[float] = []
                        gpu_clock_samples_mhz: list[float] = []


                        while process.poll() is None:
                            gpu_completed = subprocess.run(
                                [
                                    "nvidia-smi",
                                    "-i", gpu_id,
                                    "--query-gpu=utilization.gpu,clocks.current.sm",
                                    "--format=csv,noheader,nounits",
                                ],
                                text=True,
                                capture_output=True,
                                check=False,
                            )

                            if gpu_completed.returncode == 0:
                                try:
                                    line = gpu_completed.stdout.strip().splitlines()[0]
                                    util_text, clock_text = line.split(",", maxsplit=1)

                                    gpu_util_samples.append(float(util_text.strip()))
                                    gpu_clock_samples_mhz.append(float(clock_text.strip()))
                                except (ValueError, IndexError):
                                    pass

                            time.sleep(0.1)

                        stdout, stderr = process.communicate()
                        run_result = subprocess.CompletedProcess(
                            cmd,
                            process.returncode,
                            stdout,
                            stderr,
                        )
                        wall_runtime = time.perf_counter() - start
                        gpu_util_avg = (
                            sum(gpu_util_samples) / len(gpu_util_samples)
                            if gpu_util_samples else ""
                        )
                        gpu_util_max = (
                            max(gpu_util_samples)
                            if gpu_util_samples else ""
                        )
                        gpu_clock_avg_ghz = (
                            sum(gpu_clock_samples_mhz)
                            / len(gpu_clock_samples_mhz)
                            / 1000.0
                            if gpu_clock_samples_mhz
                            else ""
                        )

                        gpu_clock_max_ghz = (
                            max(gpu_clock_samples_mhz) / 1000.0
                            if gpu_clock_samples_mhz
                            else ""
                        )
                        energy_match = ENERGY_RE.search(run_result.stderr)
                        energy_pkg_j = (
                            float(energy_match.group(1).replace(",", ""))
                            if energy_match
                            else ""
                        )
                        output = run_result.stdout + "\n" + run_result.stderr
                        if run_result.returncode != 0:
                            log_file.write_text(output)

                        runtime_match = TOTAL_RE.search(output)
                        reported_runtime = (
                            float(runtime_match.group(1))
                            if runtime_match else ""
                        )

                        h2d_match = TRANSFER_H2D_RE.search(output)
                        transfer_h2d = (
                            float(h2d_match.group(1))
                            if h2d_match else ""
                        )

                        d2h_match = TRANSFER_D2H_RE.search(output)
                        transfer_d2h = (
                            float(d2h_match.group(1))
                            if d2h_match else ""
                        )

                        append_row(
                            results_file,
                            {
                            "trial": trial,
                            "run_id": run_id,
                            "monitored_gpu": gpu_id,
                            "task_type": script_path.stem,
                            "device": "gpu",
                            "num_sample": samples,
                            "dense_dim_in": dense_dim,
                            "dense_dim_out": dense_dim,
                            "read_size_bytes": io_size,
                            "write_size_bytes": io_size,
                            "runtime_wall_s": wall_runtime,
                            "runtime_reported_s": reported_runtime,
                            "transfer_h2d_s": transfer_h2d,   
                            "transfer_d2h_s": transfer_d2h,
                            "gpu_util_avg_pct": gpu_util_avg,
                            "gpu_util_max_pct": gpu_util_max,
                            "gpu_sm_clock_avg_ghz": gpu_clock_avg_ghz,
                            "gpu_sm_clock_max_ghz": gpu_clock_max_ghz,
                            "exit_code": run_result.returncode,
                            "log_file": str(log_file),
                        })

                        print(
                            f"Finished with exit={run_result.returncode}; "
                            f"runtime={wall_runtime:.3f}s",
                            flush=True,
                        )

if __name__ == "__main__":
    main()  
