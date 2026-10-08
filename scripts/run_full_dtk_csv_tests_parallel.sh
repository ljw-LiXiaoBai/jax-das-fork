#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# 环境变量：
#   GPU_IDS        逗号分隔的 HIP 设备号。默认使用已设置的 HIP_VISIBLE_DEVICES，
#                  否则用 rocminfo 探测本机全部 HCU。
#   PROCS_PER_GPU  每张卡同时运行的测试进程数，默认 1。
#   TIMING_FILE    历史 summary.tsv，按其中的耗时从长到短调度。默认取
#                  test_logs/ 下最近一次的结果；都没有时按测试文件大小排序。

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DTK_DIR="${DTK_DIR:-/opt/dtk}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
TEST_ROOT="${TEST_ROOT:-${ROOT_DIR}/tests}"
TEST_PATTERN="${TEST_PATTERN:-*_test.py}"
RUN_NAME="${RUN_NAME:-dtk_full_csv_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/test_logs/${RUN_NAME}}"
TEST_TIMEOUT="${TEST_TIMEOUT:-1800}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"

SUMMARY_FILE="${LOG_ROOT}/summary.tsv"
PROGRESS_FILE="${LOG_ROOT}/progress.txt"
CSV_FILE="${LOG_ROOT}/subcase_results.csv"
CLAIM_DIR="${LOG_ROOT}/.claims"
DONE_DIR="${LOG_ROOT}/.done"

rm -rf "${CLAIM_DIR}" "${DONE_DIR}"
mkdir -p "${LOG_ROOT}/collect" "${LOG_ROOT}/logs" "${LOG_ROOT}/xml" "${CLAIM_DIR}" "${DONE_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${PROGRESS_FILE}"
}

if [[ ! -f "${DTK_DIR}/env.sh" ]]; then
  log "DTK env not found: ${DTK_DIR}/env.sh"
  exit 2
fi
if [[ ! -f "${ROOT_DIR}/pyproject.toml" ]]; then
  log "pyproject.toml not found: ${ROOT_DIR}/pyproject.toml"
  exit 2
fi
if [[ ! "${PROCS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  log "PROCS_PER_GPU must be a positive integer, got: ${PROCS_PER_GPU}"
  exit 2
fi

set +u
source "${DTK_DIR}/env.sh"
set -u
unset PYTHONPATH
export PYTHONPATH="${ROOT_DIR}"

export PY_COLORS="${PY_COLORS:-1}"
export JAX_SKIP_SLOW_TESTS="${JAX_SKIP_SLOW_TESTS:-true}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-0}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_command_buffer=}"

gpu_source="GPU_IDS"
if [[ -z "${GPU_IDS:-}" && -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
  GPU_IDS="${HIP_VISIBLE_DEVICES}"
  gpu_source="HIP_VISIBLE_DEVICES"
fi
if [[ -z "${GPU_IDS:-}" ]]; then
  gpu_count="$("${DTK_DIR}/bin/rocminfo" 2>/dev/null | grep -cE '^[[:space:]]*Device Type:[[:space:]]*GPU' || true)"
  if [[ -z "${gpu_count}" || "${gpu_count}" == "0" ]]; then
    log "Cannot detect any HCU with ${DTK_DIR}/bin/rocminfo; set GPU_IDS=0,1,..."
    exit 2
  fi
  GPU_IDS="$(seq -s, 0 $((gpu_count - 1)))"
  gpu_source="rocminfo"
fi
GPU_IDS="${GPU_IDS// /}"
if [[ ! "${GPU_IDS}" =~ ^[^,]+(,[^,]+)*$ ]]; then
  log "Invalid GPU_IDS='${GPU_IDS}', expected a comma-separated list such as 0,1,2,3"
  exit 2
fi
IFS=',' read -r -a gpu_ids <<<"${GPU_IDS}"
job_count=$((${#gpu_ids[@]} * PROCS_PER_GPU))

{
  echo "date: $(date -Is)"
  echo "root: ${ROOT_DIR}"
  echo "log_root: ${LOG_ROOT}"
  echo -n "python: "
  "${PYTHON_BIN}" -c 'import sys; print(sys.executable)' || true
  echo "test_timeout: ${TEST_TIMEOUT}"
  echo "gpu_ids: ${GPU_IDS} (from ${gpu_source})"
  echo "procs_per_gpu: ${PROCS_PER_GPU}"
  echo "jobs: ${job_count}"
  "${PYTHON_BIN}" - <<'PY' || true
import importlib.metadata as md
import jax
import jaxlib

print("jax_file:", jax.__file__)
print("jax:", md.version("jax"))
print("jaxlib:", jaxlib.__version__)
for dist in ("jax-rocm6-plugin", "jax-rocm6-pjrt"):
  try:
    print(f"{dist}:", md.version(dist))
  except md.PackageNotFoundError:
    print(f"{dist}: not installed")
PY
  true
} >"${LOG_ROOT}/environment.log" 2>&1

declare -a tests
if (($# > 0)); then
  for test_path in "$@"; do
    if [[ -d "${test_path}" ]]; then
      while IFS= read -r file; do
        tests+=("${file}")
      done < <(find "${test_path}" -type f -name "${TEST_PATTERN}" | sort)
    elif [[ -f "${test_path}" ]]; then
      tests+=("$(cd "$(dirname "${test_path}")" && pwd)/$(basename "${test_path}")")
    elif [[ -f "${ROOT_DIR}/${test_path}" ]]; then
      tests+=("${ROOT_DIR}/${test_path}")
    else
      log "Test path not found: ${test_path}"
      exit 2
    fi
  done
else
  mapfile -t tests < <(find "${TEST_ROOT}" -type f -name "${TEST_PATTERN}" | sort)
fi

if [[ -z "${TIMING_FILE:-}" ]]; then
  TIMING_FILE="$(ls -1t "${ROOT_DIR}"/test_logs/*/summary.tsv 2>/dev/null | grep -vxF "${SUMMARY_FILE}" | head -n1 || true)"
fi
order_tests() {
  "${PYTHON_BIN}" - "$@" <<'PY'
import csv
import math
import os
import sys

root, timing_file, *tests = sys.argv[1:]
seconds = {}
if timing_file and os.path.isfile(timing_file):
  with open(timing_file, encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
      try:
        seconds[row["test"]] = float(row["seconds"])
      except (KeyError, TypeError, ValueError):
        pass

def key(path):
  rel = path[len(root) + 1:] if path.startswith(root + "/") else path
  # 没有历史耗时的文件（新增测试）排在最前，避免最后才开始跑长测试。
  return (-seconds.get(rel, math.inf), -os.path.getsize(path))

print("\n".join(sorted(tests, key=key)))
PY
}
if ((${#tests[@]} > 0)); then
  mapfile -t ordered < <(order_tests "${ROOT_DIR}" "${TIMING_FILE}" "${tests[@]}")
  if ((${#ordered[@]} == ${#tests[@]})); then
    tests=("${ordered[@]}")
  else
    log "WARNING: failed to order tests by duration, keeping path order"
  fi
fi
printf '%s\n' "${tests[@]}" >"${LOG_ROOT}/test_files.txt"
printf 'status\texit_code\tseconds\ttest\tlog\txml\tgpu\n' >"${SUMMARY_FILE}"

total="${#tests[@]}"
log "Found ${total} test files"
log "Running on HCU ${GPU_IDS} (from ${gpu_source}), ${PROCS_PER_GPU} process(es) per HCU, ${job_count} jobs"
log "Scheduling by $([[ -f "${TIMING_FILE}" ]] && echo "durations in ${TIMING_FILE}" || echo "test file size")"

run_test() {
  local index="$1" slot="$2" gpu="$3"
  local test_file="${tests[index]}"
  local rel="${test_file#${ROOT_DIR}/}"
  local safe="${rel//\//__}"
  safe="${safe%.py}"
  local collect_file="${LOG_ROOT}/collect/${safe}.txt"
  local collect_log="${LOG_ROOT}/collect/${safe}.log"
  local log_file="${LOG_ROOT}/logs/${safe}.log"
  local xml_file="${LOG_ROOT}/xml/${safe}.xml"
  local position="$((index + 1))/${total}"

  log "COLLECT [${slot}] ${position} ${rel}"
  set +e
  (cd /tmp && "${PYTHON_BIN}" -m pytest -p no:cacheprovider \
    --rootdir="${ROOT_DIR}" -c "${ROOT_DIR}/pyproject.toml" \
    --collect-only -q "${ROOT_DIR}/${rel}") \
    >"${collect_file}" 2>"${collect_log}"
  local collect_code=$?
  set -e
  if [[ "${collect_code}" != "0" ]]; then
    log "COLLECT_FAIL [${slot}] ${rel} exit=${collect_code}"
  fi
  printf '# REL:%s\n' "${rel}" >>"${collect_file}"

  local start_ts exit_code
  start_ts="$(date +%s)"
  log "START [${slot}] ${position} ${rel}"
  set +e
  if [[ "${TEST_TIMEOUT}" == "0" ]]; then
    (cd /tmp && "${PYTHON_BIN}" -m pytest -p no:cacheprovider \
      --rootdir="${ROOT_DIR}" -c "${ROOT_DIR}/pyproject.toml" --tb=short -q \
      --junitxml="${xml_file}" "${ROOT_DIR}/${rel}") >"${log_file}" 2>&1
    exit_code=$?
  else
    (cd /tmp && timeout "${TEST_TIMEOUT}" "${PYTHON_BIN}" -m pytest -p no:cacheprovider \
      --rootdir="${ROOT_DIR}" -c "${ROOT_DIR}/pyproject.toml" --tb=short -q \
      --junitxml="${xml_file}" "${ROOT_DIR}/${rel}") >"${log_file}" 2>&1
    exit_code=$?
  fi
  set -e
  local elapsed
  elapsed=$(( $(date +%s) - start_ts )) || true

  local status="PASS"
  if [[ "${exit_code}" != "0" ]]; then
    status="FAIL"
  fi
  if [[ "${exit_code}" == "124" ]]; then
    status="TIMEOUT"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${status}" "${exit_code}" "${elapsed}" "${rel}" "${log_file}" "${xml_file}" "${gpu}" >>"${SUMMARY_FILE}"
  mkdir "${DONE_DIR}/${index}"
  local done_count
  done_count="$(find "${DONE_DIR}" -mindepth 1 -maxdepth 1 | wc -l)"
  log "DONE [${slot}] (${done_count}/${total}) ${rel}: ${status} exit=${exit_code} seconds=${elapsed}"
}

worker() {
  local slot="$1" gpu="$2" index
  export HIP_VISIBLE_DEVICES="${gpu}"
  unset CUDA_VISIBLE_DEVICES
  for index in "${!tests[@]}"; do
    # mkdir 是原子操作：每个测试文件只会被一个 worker 领取。
    mkdir "${CLAIM_DIR}/${index}" 2>/dev/null || continue
    run_test "${index}" "${slot}" "${gpu}"
  done
}

descendants() {
  local child
  for child in $(pgrep -P "$1" 2>/dev/null); do
    echo "${child}"
    descendants "${child}"
  done
}

# 后台 worker 会忽略 SIGINT，timeout 又自成进程组，Ctrl+C 或 CI 取消时
# 必须主动结束整棵进程树，否则各卡上的 pytest 会继续运行。
on_interrupt() {
  trap - INT TERM
  log "Interrupted, stopping ${#worker_pids[@]} workers"
  local pids=("${worker_pids[@]}") pid
  for pid in "${worker_pids[@]}"; do
    pids+=($(descendants "${pid}"))
  done
  kill -TERM "${pids[@]}" 2>/dev/null || true
  wait || true
  exit 130
}

worker_pids=()
trap on_interrupt INT TERM
start_all="$(date +%s)"
for ((proc = 1; proc <= PROCS_PER_GPU; proc++)); do
  for gpu in "${gpu_ids[@]}"; do
    worker "gpu${gpu}.${proc}" "${gpu}" &
    worker_pids+=("$!")
  done
done
wait "${worker_pids[@]}" || true
trap - INT TERM
log "All ${job_count} jobs finished in $(($(date +%s) - start_all)) seconds"

for index in "${!tests[@]}"; do
  if [[ ! -d "${DONE_DIR}/${index}" ]]; then
    rel="${tests[index]#${ROOT_DIR}/}"
    log "NOT_RUN ${rel}"
    printf 'NOT_RUN\t-\t0\t%s\t-\t-\t-\n' "${rel}" >>"${SUMMARY_FILE}"
  fi
done
rm -rf "${CLAIM_DIR}" "${DONE_DIR}"
{
  head -n1 "${SUMMARY_FILE}"
  tail -n +2 "${SUMMARY_FILE}" | sort -t $'\t' -k4,4
} >"${SUMMARY_FILE}.tmp"
mv "${SUMMARY_FILE}.tmp" "${SUMMARY_FILE}"

log "Generating CSV ${CSV_FILE}"
"${PYTHON_BIN}" - "${LOG_ROOT}" "${CSV_FILE}" <<'PY'
from __future__ import annotations

import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

run_dir = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
ansi_re = re.compile(r"\x1b\[[0-9;]*m")
ws_re = re.compile(r"\s+")

def clean_reason(text: str | None) -> str:
  if not text:
    return ""
  text = ansi_re.sub("", text)
  text = ws_re.sub(" ", text).strip()
  if len(text) > 500:
    text = text[:497].rstrip() + "..."
  return text

def rel_from_safe(stem: str) -> str:
  # fallback：仅在 collect 文件缺少 # REL: 头时使用
  return stem.replace("__", "/") + ".py"

def read_rel_from_collect(collect_path: Path) -> str:
  try:
    with collect_path.open(encoding="utf-8", errors="replace") as f:
      for line in f:
        if line.startswith("# REL:"):
          return line.split(":", 1)[1].strip()
  except OSError:
    pass
  return rel_from_safe(collect_path.stem)

def extract_subcase(line: str) -> str | None:
  """从 pytest --collect-only -q 的一行输出里提取子用例名。

  输出可能是 tests/foo_test.py::test_a，也可能是
  /abs/root/tests/foo_test.py::test_a，或 tests/foo_test.py::TestClass::test_a。
  因为每个 collect 文件只对应一个测试文件，这里只取第一个 '::' 之后的部分，
  不依赖路径前缀，兼容任何 cwd / rootdir 组合。
  """
  line = ansi_re.sub("", line).strip()
  if "::" not in line:
    return None
  return line.split("::", 1)[1]

def xml_nodeid(rel: str, classname: str, name: str) -> str:
  module = Path(rel).stem
  cls = classname.split(".")[-1] if classname else ""
  if cls and cls != module:
    return f"{rel}::{cls}::{name}"
  return f"{rel}::{name}"

def subcase_from_nodeid(nodeid: str, rel: str) -> str:
  prefix = f"{rel}::"
  return nodeid[len(prefix):] if nodeid.startswith(prefix) else nodeid

def segfault_reason(text: str) -> str | None:
  marker = "Fatal Python error: Segmentation fault"
  if marker not in text:
    return None

  stack = text.split(marker, 1)[1]
  test_frame = re.search(
      r'File "([^"]*/tests/[^"]+)", line (\d+) in ([^\n]+)', stack)
  if test_frame:
    test_path, line, function = test_frame.groups()
    trigger = f"{Path(test_path).name}:{line}::{function.strip()}"
  else:
    trigger = "backend_compile_and_load（日志中未提取到具体测试函数）"

  ignored_frames = {
      "backend_compile_and_load", "wrapper", "_compile_and_write_cache",
      "compile_or_get_cached", "_cached_compilation", "from_hlo", "compile",
      "_pjit_call_impl_python", "_run_python_pjit", "cache_miss",
      "reraise_with_filtered_traceback", "apply_primitive", "process_primitive",
      "bind_with_trace", "bind", "call_wrapped",
  }
  operation = ""
  for path, function in re.findall(
      r'File "([^"]*/site-packages/jax/[^"]+)", line \d+ in ([^\n]+)', stack):
    function = function.strip()
    if function not in ignored_frames:
      operation = f"，相关 JAX 算子栈：{Path(path).name}::{function}"
      break

  return clean_reason(
      "error: Segmentation fault；文件级崩溃触发点：XLA "
      f"backend_compile_and_load 编译 {trigger}{operation} 时发生 native SIGSEGV；"
      "推测根因：gfx936 GPU lowering 与 Triton/AILLVM 适配尚不完整，"
      "需通过 core/gdb native backtrace 最终确认；说明：pytest 若未写出 JUnit XML，"
      "文件内无结果的子测例会批量继承本错误，并非每条子测例都单独发生 SIGSEGV")

def reason_from_log(log_path: Path, fallback: str) -> str:
  if not log_path.exists():
    return fallback
  text = ansi_re.sub("", log_path.read_text(errors="replace"))
  segfault = segfault_reason(text)
  if segfault:
    return segfault
  patterns = [
      r"Fatal Python error: Aborted.*",
      r"LLVM ERROR:.*",
      r"jax\.errors\.JaxRuntimeError:.*",
      r"E\s+jax\.errors\.JaxRuntimeError:.*",
      r"INTERNAL:.*",
      r"No FFI handler registered.*",
      r"HCU clang.*",
      r"Autotuner failed.*",
      r"error:.*",
  ]
  for pattern in patterns:
    m = re.search(pattern, text)
    if m:
      return clean_reason(m.group(0))
  return fallback

summary = {}
summary_path = run_dir / "summary.tsv"
if summary_path.exists():
  with summary_path.open(encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
      summary[row["test"]] = row

rows = []
for collect_path in sorted((run_dir / "collect").glob("*.txt")):
  rel = read_rel_from_collect(collect_path)
  collected = []
  for line in collect_path.read_text(errors="replace").splitlines():
    subcase = extract_subcase(line)
    if subcase is not None:
      collected.append(f"{rel}::{subcase}")
  collected = sorted(dict.fromkeys(collected))

  results: dict[str, tuple[str, str]] = {}
  xml_path = run_dir / "xml" / f"{collect_path.stem}.xml"
  if xml_path.exists():
    try:
      root = ET.parse(xml_path).getroot()
      for case in root.iter("testcase"):
        nodeid = xml_nodeid(rel, case.attrib.get("classname", ""), case.attrib.get("name", ""))
        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")
        if failure is None and error is None and skipped is None:
          results[nodeid] = ("pass", "")
        else:
          elem = failure if failure is not None else error if error is not None else skipped
          reason = clean_reason(elem.attrib.get("message") or elem.text)
          if skipped is not None:
            results[nodeid] = ("skip", reason)
          else:
            results[nodeid] = ("fail", reason)
    except ET.ParseError as exc:
      fallback = f"JUnit XML parse failed: {exc}"
      for nodeid in collected:
        results.setdefault(nodeid, ("fail", fallback))

  row = summary.get(rel, {})
  log_path = Path(row.get("log", "")) if row.get("log") else run_dir / "logs" / f"{collect_path.stem}.log"
  exit_code = row.get("exit_code", "")
  fallback = "not executed or no JUnit result"
  if exit_code == "124":
    fallback = "timeout"
  elif exit_code == "134":
    fallback = "process aborted"
  elif exit_code and exit_code != "0":
    fallback = f"process failed with exit code {exit_code}"
  fallback = reason_from_log(log_path, fallback)

  all_nodeids = collected or sorted(results)
  for nodeid in all_nodeids:
    status, reason = results.get(nodeid, ("fail", fallback))
    sort_status = {"pass": 0, "skip": 1, "fail": 2}.get(status, 3)
    rows.append((sort_status, rel, subcase_from_nodeid(nodeid, rel), status, reason))

rows.sort(key=lambda item: (item[0], item[1], item[2]))
with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
  writer = csv.writer(f)
  writer.writerow(["测例", "子测例", "是否通过", "报错原因"])
  for _, rel, subcase, status, reason in rows:
    writer.writerow([rel, subcase, status, reason])

pass_count = sum(1 for row in rows if row[3] == "pass")
skip_count = sum(1 for row in rows if row[3] == "skip")
fail_count = sum(1 for row in rows if row[3] == "fail")
print(f"csv={out_csv}")
print(f"total={len(rows)} pass={pass_count} skip={skip_count} fail={fail_count}")
PY

log "CSV complete: ${CSV_FILE}"
touch "${LOG_ROOT}/DONE"
