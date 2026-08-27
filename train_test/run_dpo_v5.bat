@echo off
cd /d E:\AI\LLaMA-Factory
set LF_ALLOW_TORCH29_CONV3D=1
set PYTHONUTF8=1
set HF_HOME=E:\AI\LLaMA-Factory\hf_cache
set HF_HUB_OFFLINE=1
call "C:\Users\skype\.conda\envs\llama-factory\Scripts\llamafactory-cli.exe" train train_test/examples/train_lora/qwen3_5_9b_domain_pt_sft_v5_then_dpo.yaml >> train_test\logs\dpo_v5_train_restart.log 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> train_test\logs\dpo_v5_train_restart.log
