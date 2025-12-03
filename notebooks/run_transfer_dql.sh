#!/bin/bash
set -ex

kernel=$1
if [ -z "$kernel" ]; then
    kernel=cybersim
fi

script_dir=$(dirname "$0")

pushd "$script_dir/.."

output_dir=notebooks/output/transfer_dql
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

run transfer_dql '' -y "
    iteration_count: 200
    training_episode_count: 5
    eval_episode_count: 2
    plots_dir: $output_plot_dir
"