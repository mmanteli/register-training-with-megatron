#!/bin/bash


module purge
module load LUMI
module use /appl/local/csc/modulefiles; module load pytorch

for register in dtp HI HI-IN ID IN IP LY MT NA ne OP SP; do

    path="/scratch/project_462000353/HPLT-REGISTERS/samples-150B-by-register-xlmrl/tokenized/${register}/eng_Latn_text_document"

    python decode_bin.py $path "gpt2" > ${register}_tokens.txt
done
