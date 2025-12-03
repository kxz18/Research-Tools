#!/bin/bash
if [ $# -lt 4 ];then
    echo ""
    echo "USAGE: `basename $0` [parameter]"
    echo "parameter:"
    echo "        --json_path                       json_path  [required]"
    echo "        --output_dir                      output_dir  [required]"
    echo "        --flash_attention_implementation  implementation(triton, xla, cudnn)"
    echo "        --model_dir                       dir_of_model_parameters"
    echo "        --db_dir                          dir_of_database"
    echo "        --num_diffusion_samples           number_of_diffusion_samples"
    echo "        --run_inference                   msa_generate_only [True]"
    echo ""
    exit
fi

set -e

############################################################
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
# <<< conda initialize <<<
############################################################

REPO_DIR=/absolute/path/to/alphafold3

prog=$REPO_DIR/run_alphafold.py
env_root=$REPO_DIR/env
conda activate $env_root

source $REPO_DIR/source.sh
model_dir=$REPO_DIR/model_para
db_dir=$REPO_DIR/database
flash_attention_implementation=triton
#flash_attention_implementation=xla	# for V100
run_inference=True
num_diffusion_samples=5

while [ $# -gt 0 ];do
    case $1 in 
        --json_path)
            shift
            json_path=$1;;
        --output_dir)
            shift
            output_dir=$1;;
        --flash_attention_implementation)
            shift
            flash_attention_implementation=$1;;
        --model_dir)
            shift
            model_dir=$1;;
        --db_dir)
            shift
            db_dirx=$1;;
        --num_diffusion_samples)
            shift
            num_diffusion_samples=$1;;
        --run_inference)
            shift
            run_inference=$1;;
        *)
            echo ""
            echo "Error: wrong command argument \"$1\" "
            echo ""
            exit 2;;
    esac
    shift
done

if [ ! -s $output_dir ];then
    mkdir $output_dir
fi

python $prog \
     --json_path=$json_path \
     --model_dir=$model_dir \
     --db_dir=$db_dir \
     --output_dir=$output_dir \
     --flash_attention_implementation=$flash_attention_implementation \
     --num_diffusion_samples=$num_diffusion_samples