#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --mem=8g
##########SBATCH --gres=gpu:1
#SBATCH -p qTRD
#SBATCH -t 24:00:00
#SBATCH -J generator
#SBATCH -A ...
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yxiao11@student.gsu.edu
#SBATCH --output=/data/users2/yxiao11/model/satellite_project/generator.out
#SBATCH --error=/data/users2/yxiao11/model/satellite_project/generator.err
#SBATCH --array=0-6  # <-- This runs 10 jobs (each on a separate node if available)

# sleep 5s

source /data/users2/yxiao11/.bashrc
source activate p38

rm -r /data/users2/yxiao11/model/satellite_project/data/*/*/*.npy
rm -r /data/users2/yxiao11/model/satellite_project/database/*/*/*.npy


# export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

python -u /data/users2/yxiao11/model/satellite_project/generator_realtime.py --run_forever False

python -u /data/users2/yxiao11/model/satellite_project/generator_realtime.py --run_forever True


