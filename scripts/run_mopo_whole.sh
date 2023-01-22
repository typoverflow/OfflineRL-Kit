source scripts/config.sh

all_args=("$@")
task=$1
quality=$2
data_limit=$3
rest_args=("${all_args[@]:3}")
project="MOPO-D4RL-datalimit"

only_pretrain=0
dynamics_path_root="mopo_dynamics_datalimit/"

if [ $task == "all" ]; then
    tasks=( "halfcheetah" "hopper" "walker2d")
else
    tasks=( $task )
fi

if [ $quality == "all" ]; then
    qualities=( "medium" "medium-replay" "medium-expert" "random" )
else
    qualities=( $quality )
fi

dynamics_name=$data_limit

declare -A penalty
penalty=(
    ["halfcheetah-random-v2"]="0.5"
    ["hopper-random-v2"]="1"
    ["walker2d-random-v2"]="1"
    ["halfcheetah-medium-v2"]="1"
    ["hopper-medium-v2"]="5"
    ["walker2d-medium-v2"]="5"
    ["halfcheetah-medium-replay-v2"]="1"
    ["hopper-medium-replay-v2"]="1"
    ["walker2d-medium-replay-v2"]="1"
    ["halfcheetah-medium-expert-v2"]="1"
    ["hopper-medium-expert-v2"]="1"
    ["walker2d-medium-expert-v2"]="2"
)

declare -A rollout
rollout=(
    ["halfcheetah-random-v2"]="5"
    ["hopper-random-v2"]="5"
    ["walker2d-random-v2"]="1"
    ["halfcheetah-medium-v2"]="1"
    ["hopper-medium-v2"]="5"
    ["walker2d-medium-v2"]="5"
    ["halfcheetah-medium-replay-v2"]="5"
    ["hopper-medium-replay-v2"]="5"
    ["walker2d-medium-replay-v2"]="1"
    ["halfcheetah-medium-expert-v2"]="5"
    ["hopper-medium-expert-v2"]="5"
    ["walker2d-medium-expert-v2"]="1"
)

for task in ${tasks[@]}; do
    for q in ${qualities[@]}; do
        dataset=${task}-${q}-v2
        dynamics_path=$dynamics_path_root/$task-$q-v2/$dynamics_name
        echo python3 run_example/run_mopo.py \
            --task ${dataset} --penalty-coef ${penalty[${dataset}]} \
            --rollout-length ${rollout[${dataset}]}\
            --project $project --entity ${entity} \
            --save-dynamics-path ${dynamics_path} \
            --log-path "./log_mopo_datalimit" \
            ${rest_args[@]}

        python3 run_example/run_mopo.py \
            --task ${dataset} --penalty-coef ${penalty[${dataset}]} \
            --rollout-length ${rollout[${dataset}]}\
            --project $project --entity ${entity} \
            --save-dynamics-path ${dynamics_path} \
            --log-path "./log_mopo_datalimit" \
            ${rest_args[@]}

    done
done

