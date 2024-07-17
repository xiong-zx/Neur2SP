#!/bin/bash
#SBATCH --job-name=tune_pp
#SBATCH --output=slurm/output_pp_%A_%a.txt
#SBATCH --error=slurm/error_pp_%A_%a.txt
#SBATCH --time=1-00:00:00
#SBATCH --partition=savio3
#SBATCH --account=fc_2023sp
#SBATCH --cpus-per-task=1   # 1 CPU per task
#SBATCH --ntasks-per-node=8 # Adjust this based on your node's CPU capacity
#SBATCH --array=0-80        # 81 tasks in the array

module load anaconda3/2024.02-1

# Initialize Conda
source /global/software/rocky-8.x86_64/manual/modules/langs/anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate sp

# Directly configure wandb with the API key
export WANDB_API_KEY=6108e5e435854619d56cf1aea29029c1ff25a7c0
wandb login $WANDB_API_KEY

# Define hyperparameter ranges
embed_hidden_dims=(32 64 128)
embed_dim1s=(64 128 256)
embed_dim2s=(128 256 512)
relu_hidden_dims=(256 512 1024)

# Function to calculate the combination based on SLURM_ARRAY_TASK_ID
get_combination() {
    local id=$1
    local total=${#embed_hidden_dims[@]}*${#embed_dim1s[@]}*${#embed_dim2s[@]}*${#relu_hidden_dims[@]}
    local count=0

    for embed_hidden_dim in "${embed_hidden_dims[@]}"; do
        for embed_dim1 in "${embed_dim1s[@]}"; do
            for embed_dim2 in "${embed_dim2s[@]}"; do
                for relu_hidden_dim in "${relu_hidden_dims[@]}"; do
                    if [ $count -eq $id ]; then
                        echo "$embed_hidden_dim $embed_dim1 $embed_dim2 $relu_hidden_dim"
                        return
                    fi
                    count=$((count + 1))
                done
            done
        done
    done
}

# Get the hyperparameters for the current task
params=$(get_combination $SLURM_ARRAY_TASK_ID)
set -- $params
embed_hidden_dim=$1
embed_dim1=$2
embed_dim2=$3
relu_hidden_dim=$4

# Define the problem (replace with actual problems if needed)
problem="pp"  # This can also be an array if multiple problems need to be run

# Run the training script
python -m nsp.scripts.train_model --model_type nn_e --problem $problem \
--embed_hidden_dim $embed_hidden_dim \
--embed_dim1 $embed_dim1 \
--embed_dim2 $embed_dim2 \
--relu_hidden_dim $relu_hidden_dim \
--agg_type mean \
--lr 1e-3 \
--dropout 0.02 \
--optimizer Adam \
--batch_size 128 \
--loss_fn MSELoss \
--wt_lasso 0.0 \
--wt_ridge 0.0 \
--log_freq 10 \
--n_epochs 1000 \
--use_wandb 1
