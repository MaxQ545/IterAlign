import os
import csv
import subprocess
import json
import time
import threading
import psutil
import pynvml

# --- Configuration ---
# MODELS = [
#     "fIterAlign", "GradAlign", "iMMNC",
#     "IterAlign_rw", "MMNC", "MMNC_CENA", "oIterAlign", "REGAL",
#     "SLOTAlign", "WLAlign", "CENA", "CONE", "FINAL",
# ]
MODELS = ["CENA"]
DATASETS = ["Facebook-Twitter", "DBLP1-DBLP2", "Arxiv1-Arxiv2"]
TIMEOUT_SECONDS = 300
WORKER_SCRIPT = "main_worker.py"
LOG_FILE = "resource_usage_log.csv"


# --- End of Configuration ---

class ResourceMonitor:
    """
    Monitors a process from the outside.
    - Tracks the process's own peak RAM usage.
    - Tracks the peak INCREASE in total system VRAM usage.
    """

    def __init__(self, pid):
        self._pid = pid
        self._stop_event = threading.Event()
        self.peak_ram_mb = 0
        self.peak_vram_increase_mb = 0
        self._initial_vram_used = 0
        self._gpu_handle = None
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)

    def start(self):
        try:
            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            self._initial_vram_used = mem_info.used
        except (pynvml.NVMLError, FileNotFoundError):
            print("Warning: NVML not found or GPU not available. VRAM will not be monitored.")
            self._gpu_handle = None

        self._monitor_thread.start()

    def stop(self):
        self._stop_event.set()
        self._monitor_thread.join(timeout=2)  # Wait briefly for thread to exit
        if self._gpu_handle:
            pynvml.nvmlShutdown()

    def _monitor_loop(self):
        try:
            process = psutil.Process(self._pid)
            # Baseline RAM is taken as 0, we measure the process's total RSS.
        except psutil.NoSuchProcess:
            return  # Process already ended

        while not self._stop_event.is_set():
            try:
                # Get process RAM usage (Resident Set Size)
                ram_usage_bytes = process.memory_info().rss
                self.peak_ram_mb = max(self.peak_ram_mb, ram_usage_bytes / (1024 * 1024))

                # Get increase in total VRAM usage
                if self._gpu_handle:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                    vram_increase_bytes = mem_info.used - self._initial_vram_used
                    self.peak_vram_increase_mb = max(self.peak_vram_increase_mb, vram_increase_bytes / (1024 * 1024))

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process has terminated, so we stop monitoring
                break
            except pynvml.NVMLError:
                # GPU error during monitoring
                break

            time.sleep(0.1)  # Poll interval


def setup_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["model", "dataset", "peak_ram_mb", "peak_vram_mb", "execution_time_sec", "status"])


def log_result(model, dataset, peak_ram, peak_vram, exec_time, status):
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([model, dataset, f"{peak_ram:.2f}", f"{peak_vram:.2f}", exec_time, status])


if __name__ == "__main__":
    setup_log_file()
    print("Starting experiments with active monitoring...")
    print("-" * 40)

    for model_name in MODELS:
        for dataset_name in DATASETS:
            if model_name == "CONE" and dataset_name == "Facebook-Twitter":
                print(f"Skipping {model_name} on {dataset_name} due to known issues.")
                continue

            print(f"🚀 Running Model: {model_name} on Dataset: {dataset_name}")

            command = ["python", WORKER_SCRIPT, "--model", model_name, "--dataset", dataset_name]

            exec_time = "N/A"
            monitor = None

            try:
                # Use Popen for non-blocking process creation
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Start monitoring the new process immediately
                monitor = ResourceMonitor(process.pid)
                monitor.start()

                # Wait for process to finish, with a timeout
                stdout, stderr = process.communicate(timeout=TIMEOUT_SECONDS)

                if process.returncode == 0:
                    status = "completed"
                    # Get precise execution time from worker's output
                    result_data = json.loads(stdout)
                    exec_time = f"{result_data['time']:.2f}"
                    print(f"✅ Success! Time: {exec_time}s")
                else:
                    status = "error"
                    print(f"❌ ERROR! Worker script failed. Stderr:")
                    print(stderr)

            except subprocess.TimeoutExpired:
                status = "timeout"
                print(f"❌ TIMEOUT! The run exceeded {TIMEOUT_SECONDS} seconds. Terminating...")
                # process.communicate() already sent SIGKILL, but we ensure it's gone
                process.kill()
                process.wait()  # Clean up zombie process
                exec_time = f">{TIMEOUT_SECONDS}"

            except FileNotFoundError:
                print(f"❌ CRITICAL ERROR! Could not find '{WORKER_SCRIPT}' or python executable.")
                exit()

            except Exception as e:
                status = "manager_error"
                print(f"An unexpected error occurred in the manager: {e}")

            finally:
                if monitor:
                    monitor.stop()
                    peak_ram = monitor.peak_ram_mb
                    peak_vram = monitor.peak_vram_increase_mb
                    print(f"    Peak Net RAM: {peak_ram:.2f} MB")
                    print(f"    Peak Net VRAM: {peak_vram:.2f} MB")
                    log_result(model_name, dataset_name, peak_ram, peak_vram, exec_time, status)

            print("-" * 40)

    print("All experiments finished. Results are saved in 'resource_usage_log.csv'.")