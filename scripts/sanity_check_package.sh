#!/bin/sh


: '
    This script sanity-checks the last original facil commit (see below) against the packaged version.

    This is done by comparing results for all approaches. Note that DMC and LUCIR are expected to crash.
        - DMC requires imagenet
        - LUCIR does not run on original version itself (at least not with the arguments used here)
    
'

# the commit before converting facil to package-like
original_commit=66d94117a4b2b2bd5752278639d7ef6d385c73d8
# the final commit of package-like facil
# package_commit=176377bea8d980db89b9e14b4c64c5e56f5109c8

# TODO: update after adding readme
package_commit=0cf6503522d0534ab8cdb530dc546722e43413e3
# args
nepochs=3
num_tasks=3
gridsearch_tasks=3
num_exemplars=20
network=LeNet
dataset=mnist

num_errors=0

out_path=/tmp/facil_check
rm -rf $out_path

check_equal() {
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
    if [ $1 -ne 0 ]; then
        echo -e "\u2715"
    else
        echo -e "\u2713"
    fi
}

test_approach() {
    : '
    Train an approach ($1) with both, the original facil, and the packaged version and then run diff on results.
    Most parameters are fixed above, with $2 being --num-exemplars.
    '
    echo -n "running $1 original... "
    num_wrong=0
    
    # train original facil
    python3 main_incremental.py --approach $1 --datasets $dataset --network $network --num-tasks $num_tasks --nepochs $nepochs --gridsearch-tasks $gridsearch_tasks --results-path $out_path/original --num-exemplars $2 --aux-dataset mnist --gradcam-layer conv1 2> /dev/null
    # ret_val=$?
    # num_wrong=$((num_wrong+ret_val))
    # print_suc_err $ret_val
    git checkout $original_commit > /dev/null 2>&1

    # train package-facils
    echo -n "running $1 pacakaged... "
    python3 main_incremental.py --approach $1 --datasets $dataset --network $network --num-tasks $num_tasks --nepochs $nepochs --gridsearch-tasks $gridsearch_tasks --results-path $out_path/package --num-exemplars $2 --aux-dataset mnist --gradcam-layer conv1 2> /dev/null
    # ret_val=$?
    # num_wrong=$((num_wrong+ret_val))
    # print_suc_err $ret_val
    git checkout $package_commit > /dev/null 2>&1

    if [ $num_wrong -eq 0 ]; then
        # count number of errors for approach
        echo -n "checking task-aware... "
        # check equal results taw
        original=$(ls -t ${out_path}/original/mnist_${1}/results/acc_taw* | head -1)
        package=$(ls -t ${out_path}/package/mnist_${1}/results/acc_taw* | head -1)

        check_equal $original $package
        
        ret_val=$?
        num_wrong=$((num_wrong+ret_val))
        print_suc_err $ret_val


        echo -n "checking task-agnostic... "
        # check equal results tag
        original=$(ls -t ${out_path}/original/mnist_${1}/results/acc_tag* | head -1)
        package=$(ls -t ${out_path}/package/mnist_${1}/results/acc_tag* | head -1)

        check_equal $original $package
        ret_val=$?
        num_wrong=$((num_wrong+ret_val))
        print_suc_err $ret_val
    fi
    # only count errors once for approach.
    if [ $num_wrong -gt 0 ]; then
        num_errors=$((num_errors+1))
    fi;

    echo "errors: $num_errors"
    echo "-------------------------------------------------------------------"
}

# NOTE: we have to run joint and dmc separately as they do not take exemplars but bic needs them
test_approach dmc 0 
test_approach joint 0 

for approach in bic dmc eeil ewc finetuning freezing icarl il2m lucir lwf lwm mas path_integral r_walk; do
    test_approach $approach 20 
done

if [ $num_errors -eq 0 ]; then
    echo "NICE! All good."
else
    echo "$num_errors approaches failed."
fi