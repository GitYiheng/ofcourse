#!/bin/bash

python main.py --log_dir runs/exp2_ippo_seed10 --algo ippo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 10
python main.py --log_dir runs/exp2_ippo_seed11 --algo ippo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 11
python main.py --log_dir runs/exp2_ippo_seed20 --algo ippo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 20
python main.py --log_dir runs/exp2_ippo_seed21 --algo ippo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 21
python main.py --log_dir runs/exp2_ippo_seed30 --algo ippo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 30
python main.py --log_dir runs/exp2_ippo_seed31 --algo ippo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 31
python main.py --log_dir runs/exp2_ippo_seed40 --algo ippo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 40
python main.py --log_dir runs/exp2_ippo_seed41 --algo ippo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 41