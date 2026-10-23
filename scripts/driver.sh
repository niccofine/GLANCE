#!/bin/bash
# Orchestrates the full pipeline in order. Resumable.
# Start:  nohup bash scripts/driver.sh > driver.log 2>&1 &
set -eu
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONDONTWRITEBYTECODE=1
PY="python"
STEPS=results/driver/steps; mkdir -p "$STEPS"

step() {
  local name=$1; shift
  if [ -f "$STEPS/$name.done" ]; then echo "skip $name (done)"; return 0; fi
  echo "start $name"
  if "$@"; then
    touch "$STEPS/$name.done"; echo "done $name"
  else
    echo "FAILED $name - driver stopped"; exit 1
  fi
}

workers() {
  local name=$1 module=$2
  if [ -f "$STEPS/$name.done" ]; then echo "skip $name (done)"; return 0; fi
  echo "start $name (2 GPU workers)"
  $PY -m "$module" worker --gpu 1 &
  $PY -m "$module" worker --gpu 0 &
  wait
  if $PY -m "$module" check; then
    touch "$STEPS/$name.done"; echo "done $name"
  else
    echo "FAILED $name: runs missing after 2 attempts - driver stopped"; exit 1
  fi
}

step   dataset_md5scan      $PY -m scripts.dataset md5scan
step   dataset_exclusions   $PY -m scripts.dataset exclusions
step   dataset_features     $PY -m scripts.dataset features
step   dataset_final        $PY -m scripts.dataset final

step    baselines_cv    $PY -m scripts.finaltraining baselines cv
workers stage1          scripts.crossvalidation
step    select          $PY -m scripts.lossselection select
workers stage2          scripts.lossselection
step    epochs          $PY -m scripts.finaltraining epochs
step    cv_eval         $PY -m scripts.evaluation cv
workers stage3          scripts.finaltraining
step    baselines_final $PY -m scripts.finaltraining baselines final
step    test_eval       env CUDA_VISIBLE_DEVICES=1 $PY -m scripts.evaluation test
step    explain         env CUDA_VISIBLE_DEVICES=1 $PY -m scripts.explainability
step    figures         $PY -m scripts.figures median

echo "pipeline done. Post-pipeline analyses (run manually, in this order):"
echo "  python -m scripts.posthoc explain59   (needs scripts.explainability.part_C via glance)"
echo "  python -m scripts.evaluation properties"
echo "  python -m scripts.posthoc onset"
echo "  python -m scripts.posthoc figs"
echo "  python -m scripts.posthoc bygrade"
echo "  python -m scripts.figures trajectories"
echo "  python -m scripts.figures example_patients"
echo "  python -m glance.irregular worker --gpu 0   (then --gpu 1; or train one run at a time)"
echo "  python -m scripts.posthoc irregular"
echo "  env CUDA_VISIBLE_DEVICES=1 python -m scripts.posthoc querystep"
