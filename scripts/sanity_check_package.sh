#!/bin/sh


: '
    This script sanity-checks the last original facil commit (see below) against the packaged version.

    This is done by comparing results for all approaches. 
    Note that LUCIR is expected to crash as it does not run on original version itself (at least not with arguments used here).
    
'

# the commit before converting facil to package-like
original_commit=66d94117a4b2b2bd5752278639d7ef6d385c73d8
# the final commit of package-like facil
# package_commit=176377bea8d980db89b9e14b4c64c5e56f5109c8

# TODO: update after adding readme
# package_commit=389070bf1e016c60d13aff91718114b7df787acb
package_commit=package
# args
nepochs=1
num_tasks=2
gridsearch_tasks=2
num_exemplars=20
network=LeNet
dataset=mnist

num_errors=0

out_path=/tmp/facil_check
rm -rf $out_path
mkdir -p $out_path
cd $out_path
rm -rf facil
git clone https://github.com/hrrbay/facil.git $out_path/facil
cd facil/src  
echo "PWD: $(pwd)"
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

compare_stdout() {
    appr=$1
    original=$(ls -t ${out_path}/original/mnist_${appr}/stdout* | head -1)
    package=$(ls -t ${out_path}/package/mnist_${appr}/stdout* | head -1)

    echo "diff of stdout (original to package):"
    diff -y $original $package
}

echo ""

print_suc_err() {
    if [ $1 -ne 0 ]; then
        echo -e "\u2715"
    else
        echo -e "\u2713"
    fi
}


print_stderr() {
    ret_val=$1
    if [ $ret_val -eq 0 ]; then
        return
    fi
    appr=$2
    version=$3

    stderr_file=$(ls -t ${out_path}/$version/${dataset}_${appr}/stderr* | head -1)

    echo "Error running $appr on $version version:"
    cat $stderr_file
    echo ""
}

failed_approaches=()
test_approach() {
    : '
    Train an approach ($1) with both, the original facil, and the packaged version and then run diff on results.
    Most parameters are fixed above, with $2 being --num-exemplars.
    '
    num_wrong=0
    
    # train original facil
    echo -n "running $1 original... "
    git checkout $original_commit > /dev/null 2>&1
    python3 main_incremental.py --results-path $out_path/original --approach $1 $2 > /dev/null 2>&1
    ret_val=$?
    num_wrong=$((num_wrong+ret_val))
    print_suc_err $ret_val
    print_stderr $ret_val $1 "original"

    # train package-facils
    git checkout $package_commit > /dev/null 2>&1
    echo -n "running $1 pacakaged... "
    python3 main_incremental.py --results-path $out_path/package --approach $1 $2 > /dev/null 2>&1
    ret_val=$?
    num_wrong=$((num_wrong+ret_val))
    print_suc_err $ret_val
    print_stderr $ret_val $1 "package"

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

        compare_stdout $1
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