#!/bin/bash

python main.py --log_dir runs/exp2_happo_seed10 --algo happo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 10
python main.py --log_dir runs/exp2_happo_seed11 --algo happo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 11
python main.py --log_dir runs/exp2_happo_seed20 --algo happo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 20
python main.py --log_dir runs/exp2_happo_seed21 --algo happo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 21
python main.py --log_dir runs/exp2_happo_seed30 --algo happo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 30
python main.py --log_dir runs/exp2_happo_seed31 --algo happo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 31
python main.py --log_dir runs/exp2_happo_seed40 --algo happo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 40
python main.py --log_dir runs/exp2_happo_seed41 --algo happo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 41