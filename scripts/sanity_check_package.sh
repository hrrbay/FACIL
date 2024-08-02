#!/bin/sh


: '
    This script sanity-checks two commits (`original_commit` and `package_commit`) against each other. by default, `package_commit` is the **latest** commit on `master`.

    This is done by comparing results for all approaches. LUCIR and IL2M are expected to crash as both lead to errors on the original version itself.
    
    You can change the parameters used for experiments or commits to compare below.
'

# commits to compare to each other
original_commit=e09d2c83320a1aa945a6157d4875437515824dc9
package_commit=master

# experiment arguments
nepochs=3
num_tasks=3
gridsearch_tasks=3
num_exemplars=20
network=LeNet
dataset=mnist

num_errors=0

# do everything in /tmp
out_path=/tmp/facil_check
mkdir -p $out_path
cd $out_path

# remove all old checks
rm -rf original
rm -rf facil
rm -rf package

# clone repo
git clone https://github.com/hrrbay/facil.git $out_path/facil
cd facil/src  
echo "PWD: $(pwd)"


# this environment variable is used as data-path in the updated version. Set it to the previous default.
export DATA_PATH="../data"

check_equal() {
    # Compute diff of two files ($1, $2)
    diff $1 $2
    diff_ret=$?
    if [ $diff_ret -ne 0 ]; then
        echo "I observed different results for this approach. This is not good. Check the logs in $out_path."
        echo -e "\t$1"
        echo -e "\t$2"
        return 1
    fi

    return 0
}


print_suc_err() {
    # Used to print checkmark or X depending on last exit status (passed in $1)
    if [ $1 -ne 0 ]; then
        echo -e "\u2715"
    else
        echo -e "\u2713"
    fi
}


print_stderr() {
    # Check if last process failed (exit-code passed in $1). If it did, print corresponding logged stderr
    ret_val=$1
    if [ $ret_val -eq 0 ]; then
        return
    fi
    appr=$2
    version=$3

    stderr_file=$(ls -t ${out_path}/${version}/${dataset}_${appr}/stderr* | head -1)
    echo "Error running $appr on $version version. If no output follows, stderr does not exist."
    if [ ! -z $stderr_file ]; then
        cat $stderr_file
    fi
    echo ""

}

run_exp() {
    # Run an experiment for either original or packaged version. Then check exit-code. Return 0 if successful, 1 otherwise
    commit=$1
    approach=$2
    args=$3
    version=
    if [ $commit == $original_commit ]; then
        version="original"
    elif [ $commit == $package_commit ]; then
        version="package"
    else
        echo "Invalid commit $commit to run experiment."
        return 1
    fi


    echo -n "running $approach $version... "
    git checkout $commit > /dev/null 2>&1
    results_path=${out_path}/${version}
    python3 main_incremental.py --results-path $results_path --approach $approach $args > /dev/null 2>&1
    exit_code=$?
    print_suc_err $exit_code
    print_stderr $exit_code $approach $version
    return $exit_code
}

failed_approaches=()
test_approach() {
    : '
    Train an approach ($1) with both, the original facil, and the packaged version and then run diff on results.
    Remaining parameters are given as a string in $2.
    '
    num_wrong=0
    approach=$1
    args=$2

    # train original facil
    run_exp $original_commit $approach "$args"
    num_wrong=$((num_wrong+$?))

    # train package-facils
    run_exp $package_commit $approach "$args"
    num_wrong=$((num_wrong+$?))

    if [ $num_wrong -eq 0 ]; then
        # if no approach failed, compare their results
        echo -n "checking task-aware... "

        # check equal results taw
        original=$(ls -t ${out_path}/original/mnist_${1}/results/acc_taw* | head -1)
        package=$(ls -t ${out_path}/package/mnist_${1}/results/acc_taw* | head -1)

        check_equal $original $package 
        # increase number of errors if not equal
        ret_val=$?
        num_wrong=$((num_wrong+ret_val))
        print_suc_err $ret_val


        echo -n "checking task-agnostic... "

        # check equal results tag
        original=$(ls -t ${out_path}/original/mnist_${1}/results/acc_tag* | head -1)
        package=$(ls -t ${out_path}/package/mnist_${1}/results/acc_tag* | head -1)

        check_equal $original $package
        # increase number of errors if not equal
        ret_val=$?
        num_wrong=$((num_wrong+ret_val))
        print_suc_err $ret_val
    fi

    # only count errors once for approach.
    if [ $num_wrong -gt 0 ]; then
        num_errors=$((num_errors+1))
        failed_approaches+=($1)
    fi;

    echo "errors: $num_errors"
    echo "-------------------------------------------------------------------"
}

# NOTE: we have to run joint and dmc separately as they do not take exemplars but bic needs them
test_approach dmc "--network $network --num-tasks $num_tasks --nepochs $nepochs --gridsearch-tasks $gridsearch_tasks  --aux-dataset mnist --num-exemplars 0 --datasets mnist"
test_approach joint  "--network $network --num-tasks $num_tasks --nepochs $nepochs --gridsearch-tasks $gridsearch_tasks  --num-exemplars 0 --datasets mnist" 

for approach in bic eeil ewc finetuning freezing icarl il2m lucir lwf mas path_integral r_walk; do
    test_approach $approach "--network $network --num-tasks $num_tasks --nepochs $nepochs --gridsearch-tasks $gridsearch_tasks --num-exemplars 20 --datasets mnist"
done

num_errors=1
if [ $num_errors -eq 0 ]; then
    echo "NICE! All good."
else
    echo "$num_errors approaches failed: ${failed_approaches[@]}"
fi