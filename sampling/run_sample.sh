#!/bin/bash
#SBATCH --job-name=register_sample
#SBATCH --account=project_462000449  # for resource and queue efficiency
#SBATCH --partition=small
#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=64
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

module load LUMI
module load parallel


START=$1
END=$2
lang="eng_Latn"

text_file_template=/scratch/project_462000353/HPLT-REGISTERS/splits/deduplicated/${lang}/{{split}}/0{{num}}.jsonl
label_file_template=/scratch/project_462000353/HPLT-REGISTERS/predictions_xlmrl/deduplicated/${lang}/{{split}}/0{{num}}.jsonl

echo "Start: $(date)"
# loop over folders given by params
# modify template to point to correct files
for d in $(seq $START $END); do
    text_file_dir=${text_file_template/"{{split}}"/$d}
    label_file_dir=${label_file_template/"{{split}}"/$d}
    for i in $(seq 0 7); do   # 8 files always
        text_file=${text_file_dir/"{{num}}"/$i}
        label_file=${label_file_dir/"{{num}}"/$i}
        echo "In split ${d} shard ${i}: $(date)"
        paste $text_file $label_file | parallel --pipe -j64 --block 10M python3 sample.py --exclude_hybrids --file_suffix="_${d}_0${i}" --lang=$lang
    done
done

echo "end: $(date)"

mv logs/$SLURM_JOBID.out logs/sample-${START}-${END}.out
mv logs/$SLURM_JOBID.err logs/sample-${START}-${END}.err