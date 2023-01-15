export PYTHONPATH=".":${PATHONPATH}
all_domain=(
    # "halfcheetah"
    # "hopper"
    "walker2d"
)

all_quality=(
    "medium"
    "medium-replay"
    "medium-expert"
    "random"
)

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

seed=$1

name=${cql_weight}-${lagrange}

for task in ${all_domain[@]}; do
    for q in ${all_quality[@]}; do
        dataset=${task}-${q}-v2
        echo python3 run_example/run_mopo.py --task ${dataset} --penalty-coef ${penalty[${dataset}]} --rollout-length ${rollout[${dataset}]} --seed ${seed}
        python3 run_example/run_mopo.py --task ${dataset} --penalty-coef ${penalty[${dataset}]} --rollout-length ${rollout[${dataset}]} --seed ${seed}
    done
done

