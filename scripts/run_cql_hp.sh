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

cql_weight=$1
lagrange=$2

if [[ $lagrange == "0.0" ]]; then
    use_lagrange="False"
else
    use_lagrange="True"
fi

name=${cql_weight}-${lagrange}

for task in ${all_domain[@]}; do
    for q in ${all_quality[@]}; do
        echo python3 run_example/run_cql.py --task ${task}-${q}-v2 --cql-weight ${cql_weight} --with-lagrange ${use_lagrange} --lagrange-threshold ${lagrange} --exp_name ${name}
        python3 run_example/run_cql.py --task ${task}-${q}-v2 --cql-weight ${cql_weight} --with-lagrange ${use_lagrange} --lagrange-threshold ${lagrange} --exp_name ${name} --seed 1
    done
done

