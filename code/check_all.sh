#!/usr/bin/env bash
# 一键四重核验 —— 任何实验改动/补实验后，交付前必跑。
#
# 用法：
#   bash check_all.sh            # 核验（数据/数字/超参数/算法-实现对照/图形摘要）
#   bash check_all.sh --compile  # 额外重编译论文并检查 0 undefined
#
# 四套核验各自覆盖一类漂移，缺一不可：
#   [1] 数据内部一致性  [2] 表格数字<->JSON
#   [3] 超参数<->运行时代码  [4] 伪代码机制<->实现逻辑
#   [5] 图形摘要几何/字号/数字溯源（独立于论文表格，独立于算法伪代码）
# 前三类是"数字/参数"漂移，第四类是"机制描述"漂移（最隐蔽，曾漏检 5 处），
# 第五类保护 Elsevier 规范的图形摘要（13×5 cm / 无文字重叠 / 无中文 / 数据可溯源）。
#
# 退出码：0 = 全部通过；非 0 = 有失败项（此时不得交付）
set -u
cd "$(dirname "$0")"

# Interpreters are resolved, never hard-coded: an absolute path bakes in one
# developer's machine, and this script ships inside the submission package.
# Set PY / GPU in the environment to pin a specific interpreter; otherwise the
# first python on PATH is used.
if [ -z "${PY:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then PY=python3
  elif command -v python >/dev/null 2>&1; then PY=python
  else
    echo "FAIL: no python on PATH; run as: PY=/path/to/python bash check_all.sh"
    exit 1
  fi
fi
GPU="${GPU:-$PY}"
TEX="ManifoldValuedSpikingNeurons_Neurocomputing_v38.tex"

# Fail loudly on the wrong interpreter. Without this, a python that merely
# lacks numpy produces a ModuleNotFoundError that looks like a broken
# experiment rather than a mis-set PY.
"$PY" -c "import numpy" >/dev/null 2>&1 || {
  echo "FAIL: '$PY' cannot import numpy."
  echo "      Point PY at the project environment, e.g.:"
  echo "        PY=/path/to/spikergnn/python bash check_all.sh"
  exit 1
}

fail=0
hr() { printf '%s\n' "------------------------------------------------------------"; }

echo "============================================================"
echo " 机器核验 (data / numbers / hyperparams / algorithm / GA / math) @ $(date '+%F %T')"
echo "============================================================"

echo
echo "[1/6] 实验数据内部一致性 —— audit_v38_data.py"
hr
"$PY" audit_v38_data.py 2>&1 | tail -3
[ "${PIPESTATUS[0]}" -eq 0 ] || fail=1

echo
echo "[2/6] 论文表格数字 <-> 结果 JSON —— verify_v38_numbers.py"
hr
"$PY" verify_v38_numbers.py 2>&1 | grep -E "^RESULT|\[!!\]" | head -20
"$PY" verify_v38_numbers.py >/dev/null 2>&1 || fail=1

echo
echo "[3/6] 论文超参数 <-> 运行时代码 —— verify_hyperparams.py"
hr
CUDA_VISIBLE_DEVICES=0 "$GPU" verify_hyperparams.py 2>&1 | grep -E "DRIFT|ALL HYPERPARAM|- " | head -20
CUDA_VISIBLE_DEVICES=0 "$GPU" verify_hyperparams.py >/dev/null 2>&1 || fail=1

echo
echo "[4/6] Algorithm 1 机制 <-> 实现逻辑 —— verify_algorithm_mapping.py"
hr
"$PY" verify_algorithm_mapping.py 2>&1 | grep -E "FAIL|ALL [0-9]+ ALGORITHM|MISMATCH" | head -10
"$PY" verify_algorithm_mapping.py >/dev/null 2>&1 || fail=1

echo
echo "[5/6] 图形摘要几何/字号/数据溯源 —— check_graphical_abstract.py"
hr
"$PY" check_graphical_abstract.py 2>&1 | grep -E "^RESULT|\[!!\]" | head -12
"$PY" check_graphical_abstract.py >/dev/null 2>&1 || fail=1

echo
echo "[6/6] 数学主张 <-> 可执行代码 —— verify_math.py"
hr
"$PY" verify_math.py 2>&1 | grep -E "^RESULTS|FAIL" | head -6
"$PY" verify_math.py >/dev/null 2>&1 || fail=1

if [ "$#" -gt 0 ] && [ "$1" = "--compile" ]; then
  echo
  echo "[+1] 编译论文并检查 undefined"
  hr
  pdflatex -interaction=nonstopmode "$TEX" >/dev/null 2>&1
  pdflatex -interaction=nonstopmode "$TEX" 2>&1 | grep -E "Output written"
  # grep -c may emit multiple lines on some builds (binary log); take the first
  n=$(grep -ac "undefined" "${TEX%.tex}.log" 2>/dev/null | head -1 | tr -d '[:space:]')
  n=${n:-0}
  echo "undefined refs: $n"
  [ "$n" -eq 0 ] || fail=1
fi

echo
echo "============================================================"
if [ "$fail" -eq 0 ]; then
  echo " ✅ ALL CHECKS PASSED —— 可以交付"
  echo
  echo " ⚠ 提醒（改过核心代码必读）：四重核验只防『已有条目漂移』，"
  echo "   不防『新增内容漏记』。若本次改动引入了对照表中没有的新机制"
  echo "   （新算子 / 新层 / 新损失项），核验不会报警 —— 必须手工补："
  echo "     verify_algorithm_mapping.py 的 CHECKS 条目"
  echo "     + 论文附录 C 对照表行 + Algorithm 1 伪代码步骤"
  exit 0
else
  echo " ❌ CHECKS FAILED —— 修复后再交付（不要改核验脚本迁就）"
  exit 1
fi
