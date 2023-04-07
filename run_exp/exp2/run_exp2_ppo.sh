#!/bin/bash

python main.py --log_dir runs/exp2_ppo_seed10 --algo ppo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 10
python main.py --log_dir runs/exp2_ppo_seed11 --algo ppo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 11
python main.py --log_dir runs/exp2_ppo_seed20 --algo ppo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 20
python main.py --log_dir runs/exp2_ppo_seed21 --algo ppo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 21
python main.py --log_dir runs/exp2_ppo_seed30 --algo ppo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 30
python main.py --log_dir runs/exp2_ppo_seed31 --algo ppo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 31
python main.py --log_dir runs/exp2_ppo_seed40 --algo ppo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 40
python main.py --log_dir runs/exp2_ppo_seed41 --algo ppo --env exp2 --undelivered_penalty 120 --r_baseline 400 --seed 41