#!/bin/bash
set -ex

kernel=$1
if [ -z "$kernel" ]; then
    kernel=cybersim
fi

script_dir=$(dirname "$0")

pushd "$script_dir/.."

output_dir=notebooks/output/baseline_rulebased
output_plot_dir=$output_dir/plots

run () {
    base=$1
    suffix=$2
    cat notebook notebooks/$base.py \
        | jupytext --to ipynb  - \
        | papermill --kernel $kernel $output_dir/$base$suffix.ipynb  "${@:3}"
}

jupyter kernelspec list

mkdir $output_dir -p
mkdir $output_plot_dir -p

run baseline_rulebased '-chain' -y "
    gymid: 'CyberBattleChain-v0'
    iteration_count: 200
    training_episode_count: 20
    eval_episode_count: 3
    maximum_node_count: 20
    maximum_total_credentials: 20
    env_size: 10
    plots_dir: $output_plot_dir
"
