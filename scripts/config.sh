export PYTHONPATH=".":$PYTHONPATH
wandb_online="True"
phase="D4RL"
entity="lamda-rl"

if [ ${wandb_online} == "False" ]; then
    export WANDB_API_KEY="2402c2667aaa839651d079fa5f022ed059984521"
    export WANDB_MODE="offline"
fi