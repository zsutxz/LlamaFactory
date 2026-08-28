@echo off
cd /d E:\AI\LLaMA-Factory
set PYTHONUTF8=1
python -u train_test/run_eval_v5_dpo_ab.py >> train_test\logs\eval_v5dpo_ab.log 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> train_test\logs\eval_v5dpo_ab.log
