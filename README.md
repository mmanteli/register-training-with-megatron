# register-training-with-megatron
Training 1.71B parameter Llama models with register labelled data

## Setup

``requirements.txt`` is a good start but not a promise it will work. Download needed packages to a pythonuserbase, as LUMI does not like venv.

## Sampling

Using HPLT v2 dedup data and associated labels to divide the dataset based on labels.

``sbatch run_sample.sh`` runs ``sample.py``, where the label assignment is done, and for larger register classes, a random threshold for selection is given to limit the final dataset size.

``run_concatenate_and_check.sh`` runs ``concatenate_and_check.py`` which checks the jsonl output, combines the shards, and moves the files to target location.

``resample-HI-and-get-register-distribution.py`` does exactly what it says, a wrong threshold was used for HI in the beginning so resampling was needed, it was also easy to calculate distribution at the same time.

## Tokenisation

Tokenisation was done using ``gpt-neox``, previously before moving to Megatron-LM, so here is just an example script.

## Training

Multiple training scripts given:

- ``pretrain.sh`` takes in register (HI, IN, OP, etc.) and fetches the corresponding dataset and trains with it
- ``pretrain-[register combination].sh`` has the data selection inside the script, e.g. use 0.34 HI, 0.33 dtp, 0.33 OP
- ``test-pretrain-w-1N.sh`` tests pretraining on one node and gpu-test partition, to see if the environment works properly

## Conversion

Conversion is needed as ``LightEval`` needs the models in different format as what Megatron-LM produces.

``convert_checkpoints.sh`` takes the register as a parameter, finds the checkpoints for that register, and converts them all. Inside the script you can define a limit for the checkpoints, e.g. you can stop the conversion to iteration 20k if you want.

## Evaluation

You can run any eval you want, pre-made options given in ``evaluate.sh``.

- ``run_evaluate.sh`` launches ``evaluate.sh`` with different model-checkpoint combinations
- corresponding scripts for baselines ``run_baselines.sh``
- ``eval_all_ckpts.sh`` not used but can be helpful

Everything with ``.ipynb`` is used for visualisation. 

