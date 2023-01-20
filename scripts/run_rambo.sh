source scripts/config.sh

all_args=("$@")
task=$1
quality=$2
pretrain_name=$3
rest_args=("${all_args[@]:3}")
project="RAMBO-D4RL"

pretrain_only=0
pretrain_path_root="rambo_pretrain/"

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

if [ -z "$pretrain_name" ]; then
    pretrain_name="default"
fi

declare -A adv_weight
adv_weight=(
    ["halfcheetah-random-v2"]="0"
    ["hopper-random-v2"]="0.0003"
    ["walker2d-random-v2"]="0"
    ["halfcheetah-medium-v2"]="0.0003"
    ["hopper-medium-v2"]="0.0003"
    ["walker2d-medium-v2"]="0.0003"
    ["halfcheetah-medium-replay-v2"]="0.0003"
    ["hopper-medium-replay-v2"]="0.0003"
    ["walker2d-medium-replay-v2"]="0"
    ["halfcheetah-medium-expert-v2"]="0.0003"
    ["hopper-medium-expert-v2"]="0.0003"
    ["walker2d-medium-expert-v2"]="0.0003"
)

declare -A rollout
rollout=(
    ["halfcheetah-random-v2"]="5"
    ["hopper-random-v2"]="2"
    ["walker2d-random-v2"]="5"
    ["halfcheetah-medium-v2"]="5"
    ["hopper-medium-v2"]="5"
    ["walker2d-medium-v2"]="5"
    ["halfcheetah-medium-replay-v2"]="5"
    ["hopper-medium-replay-v2"]="2"
    ["walker2d-medium-replay-v2"]="5"
    ["halfcheetah-medium-expert-v2"]="5"
    ["hopper-medium-expert-v2"]="5"
    ["walker2d-medium-expert-v2"]="2"
)

for task in ${tasks[@]}; do
    for q in ${qualities[@]}; do
        dataset=${task}-${q}-v2
        pretrain_path=$pretrain_path_root/$task-$q-v2/$pretrain_name
        while true
        do
            if [ -d ${pretrain_path} ]; then
                break
            else
                echo sleep 600 for ${pretrain_path}
                sleep 600
            fi
        done
        echo python3 run_example/run_rambo.py \
            --task ${dataset} --adv-weight ${adv_weight[${dataset}]} \
            --rollout-length ${rollout[${dataset}]}\
            --project $project --entity ${entity} \
            --pretrain_only ${pretrain_only} \
            --load-pretrain-path ${pretrain_path} \
            ${rest_args[@]}

        python3 run_example/run_rambo.py \
            --task ${dataset} --adv-weight ${adv_weight[${dataset}]} \
            --rollout-length ${rollout[${dataset}]}\
            --project $project --entity ${entity} \
            --pretrain_only ${pretrain_only} \
            --load-pretrain-path ${pretrain_path} \
            ${rest_args[@]}
    done
done

