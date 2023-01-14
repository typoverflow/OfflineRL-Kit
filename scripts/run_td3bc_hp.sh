all_domain=(
    # "halfcheetah"
    "hopper"
    # "walker2d"
)

all_quality=(
    "medium"
    "medium-replay"
    "medium-expert"
    "random"
)

seed=$1

name=${cql_weight}-${lagrange}

for task in ${all_domain[@]}; do
    for q in ${all_quality[@]}; do
        echo python3 run_example/run_td3bc.py --task ${task}-${q}-v2 --seed ${seed}
        python3 run_example/run_td3bc.py --task ${task}-${q}-v2 --seed ${seed}
    done
done

