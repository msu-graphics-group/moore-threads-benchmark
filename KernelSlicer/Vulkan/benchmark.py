#!/usr/bin/env python3
"""
Benchmark runner script that:
1. Builds samples using cmake/make
2. Runs benchmarks with multiple argument sets
3. Parses execution output for timing metrics
4. Aggregates results and exports to CSV

Features:
- Handles segmentation faults and crashes
- Detects missing output and invalid measurements
- Records "-" for failed measurements in CSV
- Continues execution even if some results fail
- Optional CMake cache cleanup
"""

import os
import subprocess
import re
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from statistics import mean, stdev
import sys
import signal
import shutil

# ============================================================================
# CONFIGURATION
# ============================================================================

# Number of runs per sample/argument combination
N_RUNS = 3
GPU_ID = 0

LOG_ENABLE = True
if LOG_ENABLE:
  LOG_FILE = open('z_log.txt', 'w', encoding='utf-8')

# Clean CMake cache before building (removes CMakeCache.txt and CMakeFiles/)
CLEAN_CMAKE_CACHE = False

# List of samples to benchmark
SAMPLES = [
    
    {
        "sampleName": "base01_mandelbrot",
        "exeName": "testapp",
        "args": [
            ["-size", "512",  "--gpu"],
            ["-size", "1024", "--gpu"],
            ["-size", "2048", "--gpu"],
            ["-size", "4096", "--gpu"]
        ]
    },
    {
        "sampleName": "base02_spheres_pt",
        "exeName": "testapp",
        "args": [
            ["-size", "512",  "--gpu"],
            ["-size", "1024", "--gpu"],
            ["-size", "2048", "--gpu"],
            ["-size", "4096", "--gpu"]
        ]
    },
    {
        "sampleName": "base03_reduction1",
        "exeName": "testapp",
        "args": [
            ["-size", "512",  "--gpu"],
            ["-size", "1024", "--gpu"],
            ["-size", "2048", "--gpu"],
            ["-size", "4096", "--gpu"]
        ]
    },

    {
        "sampleName": "base04_nbody_smpl",
        "exeName": "testapp",
        "args": [
            ["-size", "256",  "--gpu"],
            ["-size", "512",  "--gpu"],
            #["-size", "1024", "--gpu"],
            #["-size", "2048", "--gpu"]
        ]
    },
    {
        "sampleName": "base05_append_buf",
        "exeName": "testapp",
        "args": [
            ["-size", "512",  "--gpu"],
            ["-size", "1024", "--gpu"],
            ["-size", "2048", "--gpu"],
            ["-size", "4096", "--gpu"]
        ]
    },
    {
        "sampleName": "base06_prefixsum1",
        "exeName": "testapp",
        "args": [
            ["-size", "512",  "--gpu"],
            ["-size", "1024", "--gpu"],
            ["-size", "2048", "--gpu"],
            ["-size", "4096", "--gpu"]
        ]
    },
    
    {
        "sampleName": "base07_sort_uint2",
        "exeName": "testapp",
        "args": [
            ["-size", "512",  "--gpu"],
            ["-size", "1024", "--gpu"],
            ["-size", "2048", "--gpu"],
            ["-size", "4096", "--gpu"]
        ]
    },

    {
        "sampleName": "base08_proctex_v1",
        "exeName": "testapp",
        "args": [
            ["-size", "512",  "--gpu", "-branching", "1"],
            ["-size", "1024", "--gpu", "-branching", "1"],
            ["-size", "2048", "--gpu", "-branching", "1"],
            ["-size", "4096", "--gpu", "-branching", "1"]
        ]
    },

    {
        "sampleName": "base08_proctex_v1",
        "exeName": "testapp",
        "args": [
            ["-size", "512",  "--gpu", "-branching", "2"],
            ["-size", "1024", "--gpu", "-branching", "2"],
            ["-size", "2048", "--gpu", "-branching", "2"],
            ["-size", "4096", "--gpu", "-branching", "2"]
        ]
    },    
]

# Output CSV file
OUTPUT_CSV = "benchmark_results.csv"

# Pattern to extract timing values
# Matches: "SomeName(exec) = 1.486 ms" or "SomeName(copy) = 2.604 ms"
TIMING_PATTERN = r"(\w+)\((exec|copy)\)\s*=\s*([\d.]+)\s*ms"

# Minimum valid timing value (in ms)
MIN_VALID_TIMING = 0.001

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def run_command(cmd: List[str], cwd: str = None) -> Tuple[int, str, str]:
    """
    Execute a shell command and return exit code, stdout, stderr.
    Handles timeouts and crashes gracefully.
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def clean_cmake_cache(sample_dir: str) -> bool:
    """
    Remove CMake cache files and directories.
    Returns True if successful, False otherwise.
    """
    try:
        # Remove CMakeCache.txt
        cache_file = os.path.join(sample_dir, "CMakeCache.txt")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        
        # Remove CMakeFiles directory
        cmake_files_dir = os.path.join(sample_dir, "CMakeFiles")
        if os.path.exists(cmake_files_dir):
            shutil.rmtree(cmake_files_dir)
        
        # Remove cmake_install.cmake and Makefile (generated by CMake)
        for filename in ["cmake_install.cmake", "Makefile"]:
            filepath = os.path.join(sample_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
        
        return True
    except Exception as e:
        print(f"    Warning: Failed to clean CMake cache: {e}")
        return False


def detect_crash_reason(exit_code: int, stderr: str) -> str:
    """
    Detect the reason for crash/failure and return a descriptive string.
    """
    if exit_code == -11 or "segmentation" in stderr.lower():
        return "Segmentation fault (SIGSEGV)"
    elif exit_code == -6 or "abort" in stderr.lower():
        return "Abort signal (SIGABRT)"
    elif exit_code == -9 or "killed" in stderr.lower():
        return "Process killed (SIGKILL)"
    elif exit_code == -15:
        return "Terminated (SIGTERM)"
    elif exit_code == 139:  # Standard segfault exit code
        return "Segmentation fault (exit code 139)"
    elif exit_code == 134:  # Standard abort exit code
        return "Abort signal (exit code 134)"
    elif exit_code < 0:
        return f"Signal received (exit code {exit_code})"
    else:
        return f"Exit code {exit_code}"


def build_sample(sample_dir: str) -> bool:
    """
    Build a sample using cmake and make.
    Returns True if successful, False otherwise.
    """
    print(f"  Building {sample_dir}...", end=" ", flush=True)
    
    # Clean CMake cache if requested
    if CLEAN_CMAKE_CACHE:
        clean_cmake_cache(sample_dir)
    
    # CMake configuration
    code, out, err = run_command(
        ["cmake", "-DCMAKE_BUILD_TYPE=Release", "."],
        cwd=sample_dir
    )
    if code != 0:
        print(f"FAILED (cmake)")
        print(f"    Error: {err}")
        return False
    
    # Make build
    code, out, err = run_command(
        ["make", "-j", "4"],
        cwd=sample_dir
    )
    if code != 0:
        print(f"FAILED (make)")
        print(f"    Error: {err}")
        return False
    
    print("OK")
    return True


def extract_timings(output: str) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Parse benchmark output and extract timing values.
    
    Returns dict: {
        "name": {
            "exec": value,
            "copy": value
        }
    }
    
    Returns None if output is empty or contains no valid timings.
    """
    if not output or output.strip() == "":
        return None
    
    timings = {}
    
    matches = re.finditer(TIMING_PATTERN, output)
    for match in matches:
        name = match.group(1)
        metric = match.group(2)  # "exec" or "copy"
        value = float(match.group(3))
        
        # Validate timing value (must be positive and non-zero)
        if value <= 0 or value < MIN_VALID_TIMING:
            return None  # Invalid measurement (zero or negative)
        
        if name not in timings:
            timings[name] = {}
        timings[name][metric] = value
    
    # Return None if no valid timings were found
    if not timings or all(not metrics for metrics in timings.values()):
        return None
    
    return timings


def run_benchmark(sample_dir: str, exe_name: str, args: List[str], 
                  run_num: int) -> Tuple[Optional[Dict[str, Dict[str, float]]], Optional[str]]:
    """
    Run a single benchmark execution from the sample directory.
    
    Returns tuple: (timings_dict, error_message)
    - timings_dict: parsed timings or None on failure
    - error_message: None if successful, error description if failed
    """
    exe_path = os.path.join(sample_dir, exe_name)
    
    if not os.path.exists(exe_path):
        return None, f"Executable not found: {exe_path}"
    
    # Run executable from its own directory (so it can find shaders)
    code, stdout, stderr = run_command([f"./{exe_name}"] + args + ["-gpu_id", str(GPU_ID)], cwd=sample_dir)
    
    # Check for crashes and abnormal exits
    if code != 0:
        crash_reason = detect_crash_reason(code, stderr)
        return None, crash_reason
    
    # Combine stdout and stderr for parsing (timing info might be in either)
    combined_output = stdout + stderr
    
    if LOG_ENABLE:
        LOG_FILE.write(combined_output)
        LOG_FILE.write("\n===========================================================\n")

    # Check if output is empty
    if not combined_output or combined_output.strip() == "":
        return None, "No output from benchmark"
    
    # Try to parse timings
    timings = extract_timings(combined_output)
    
    if timings is None:
        return None, "No valid timing measurements found in output"
    
    return timings, None


def extract_size_from_args(args: List[str]) -> int:
    """
    Extract -size argument value from args list.
    Returns the size value, or 0 if not found.
    """
    try:
        if "-size" in args:
            idx = args.index("-size")
            if idx + 1 < len(args):
                return int(args[idx + 1])
    except (ValueError, IndexError):
        pass
    return 0


def aggregate_runs(all_runs: List[Dict[str, Dict[str, float]]]) -> Dict[str, Optional[float]]:
    """
    Aggregate multiple runs by computing average and minimum values.
    
    Input: List of dicts from extract_timings()
    Output: {
        "exec_avg": value or None,
        "exec_min": value or None,
        "copy_avg": value or None,
        "copy_min": value or None
    }
    """
    exec_values = []
    copy_values = []
    
    for run_timings in all_runs:
        # Find first (any) timing entry that has exec/copy values
        for name, metrics in run_timings.items():
            if "exec" in metrics:
                exec_values.append(metrics["exec"])
            if "copy" in metrics:
                copy_values.append(metrics["copy"])
    
    result = {}
    
    if exec_values:
        result["exec_avg"] = mean(exec_values)
        result["exec_min"] = min(exec_values)
    else:
        result["exec_avg"] = None
        result["exec_min"] = None
    
    if copy_values:
        result["copy_avg"] = mean(copy_values)
        result["copy_min"] = min(copy_values)
    else:
        result["copy_avg"] = None
        result["copy_min"] = None
    
    return result


def format_csv_value(value: Optional[float]) -> str:
    """
    Format a value for CSV output.
    Returns formatted number or "-" for None/invalid values.
    """
    if value is None:
        return "-"
    try:
        return f"{value:.3f}"
    except (TypeError, ValueError):
        return "-"


# ============================================================================
# MAIN BENCHMARK LOGIC
# ============================================================================

def main():
    print("=" * 70)
    print("BENCHMARK RUNNER (with error handling)")
    print("=" * 70)
    print(f"Number of runs per sample: {N_RUNS}")
    print(f"Clean CMake cache: {CLEAN_CMAKE_CACHE}")
    print(f"Output file: {OUTPUT_CSV}")
    print()
    
    # Validate samples configuration
    if not SAMPLES:
        print("ERROR: No samples configured in SAMPLES list")
        sys.exit(1)
    
    # Store results for CSV output
    results = []
    failed_runs_log = []  # Track failed runs for summary
    
    # Process each sample
    for sample in SAMPLES:
        sample_name = sample["sampleName"]
        exe_name = sample["exeName"]
        args_list = sample["args"]
        
        sample_dir = sample_name
        
        # Check if sample directory exists
        if not os.path.isdir(sample_dir):
            print(f"WARNING: Sample directory not found: {sample_dir}")
            print()
            continue
        
        print(f"Processing sample: {sample_name}")
        
        # Build the sample
        if not build_sample(sample_dir):
            print(f"Skipping {sample_name} due to build failure")
            print()
            continue
        
        # Run benchmarks for each argument set
        for arg_set in args_list:
            size = extract_size_from_args(arg_set)
            args_str = " ".join(arg_set)
            
            print(f"  Running {N_RUNS} iterations with args: {args_str}")
            
            all_runs = []
            successful_runs = 0
            failed_runs = []
            
            for run_num in range(1, N_RUNS + 1):
                print(f"    Run {run_num}/{N_RUNS}...", end=" ", flush=True)
                
                timings, error_msg = run_benchmark(sample_dir, exe_name, arg_set, run_num)
                
                if timings is not None:
                    all_runs.append(timings)
                    successful_runs += 1
                    print("OK")
                else:
                    failed_runs.append(error_msg)
                    failed_runs_log.append({
                        "sample": sample_name,
                        "args": args_str,
                        "run": run_num,
                        "error": error_msg
                    })
                    print(f"FAILED ({error_msg})")
            
            # Aggregate results - continue even if some runs failed
            if successful_runs > 0:
                aggregated = aggregate_runs(all_runs)
                
                result_row = {
                    "name": sample_name,
                    "size": size,
                    "exec_avg": aggregated.get("exec_avg"),
                    "exec_min": aggregated.get("exec_min"),
                    "copy_avg": aggregated.get("copy_avg"),
                    "copy_min": aggregated.get("copy_min"),
                    "runs_ok": successful_runs,
                    "runs_total": N_RUNS
                }
                
                results.append(result_row)
                
                # Print result summary
                exec_avg_str = format_csv_value(aggregated.get("exec_avg"))
                exec_min_str = format_csv_value(aggregated.get("exec_min"))
                copy_avg_str = format_csv_value(aggregated.get("copy_avg"))
                copy_min_str = format_csv_value(aggregated.get("copy_min"))
                
                print(f"    Results: exec_avg={exec_avg_str}ms, exec_min={exec_min_str}ms, "
                      f"copy_avg={copy_avg_str}ms, copy_min={copy_min_str}ms "
                      f"({successful_runs}/{N_RUNS} successful)")
            else:
                # All runs failed - still record a row with "-" values
                result_row = {
                    "name": sample_name,
                    "size": size,
                    "exec_avg": None,
                    "exec_min": None,
                    "copy_avg": None,
                    "copy_min": None,
                    "runs_ok": 0,
                    "runs_total": N_RUNS
                }
                
                results.append(result_row)
                print(f"    WARNING: All {N_RUNS} runs failed for {args_str}")
                for error in failed_runs:
                    print(f"      - {error}")
            
            print()
    
    # Write results to CSV
    print("=" * 70)
    print("Writing results to CSV...")
    
    if results:
        try:
            with open(OUTPUT_CSV, 'w', newline='') as csvfile:
                fieldnames = ["name", "size", "exec_avg", "exec_min", "copy_avg", "copy_min", "runs_ok", "runs_total"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for row in results:
                    # Convert None values to "-" for display
                    row_for_csv = {k: format_csv_value(v) if k in ["exec_avg", "exec_min", "copy_avg", "copy_min"] else v 
                                   for k, v in row.items()}
                    writer.writerow(row_for_csv)
            
            print(f"Successfully wrote {len(results)} result(s) to {OUTPUT_CSV}")
            print()
            print("Results summary:")
            print("-" * 70)
            for row in results:
                name = row['name']
                size = row['size']
                exec_avg = format_csv_value(row['exec_avg'])
                exec_min = format_csv_value(row['exec_min'])
                copy_avg = format_csv_value(row['copy_avg'])
                copy_min = format_csv_value(row['copy_min'])
                runs = f"{row['runs_ok']}/{row['runs_total']}"
                print(f"{name:20} size={size:6} "
                      f"exec_avg={exec_avg:>8} exec_min={exec_min:>8} "
                      f"copy_avg={copy_avg:>8} copy_min={copy_min:>8} "
                      f"[{runs}]")
        
        except Exception as e:
            print(f"ERROR: Failed to write CSV file: {e}")
            sys.exit(1)
    else:
        print("WARNING: No results to write (no samples processed)")
        sys.exit(1)
    
    # Print summary of failed runs
    if failed_runs_log:
        print()
        print("=" * 70)
        print(f"Failed runs summary ({len(failed_runs_log)} total):")
        print("-" * 70)
        for entry in failed_runs_log:
            print(f"{entry['sample']:20} args={entry['args']:30} run={entry['run']} -> {entry['error']}")
    
    print("=" * 70)
    print("Benchmark completed!")


if __name__ == "__main__":
    main()
