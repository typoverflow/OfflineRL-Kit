source scripts/config.sh

all_args=("$@")
task=$1
quality=$2
rest_args=("${all_args[@]:2}")
project="D2MG-transfer"

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

for task in ${tasks[@]}; do
    for quality in ${qualities[@]}; do
        echo python3 test_example/run_mopo.py --task $task-$quality-v2 ${rest_args[@]}
        python3 test_example/run_mopo.py --task $task-$quality-v2 ${rest_args[@]}
    done
done