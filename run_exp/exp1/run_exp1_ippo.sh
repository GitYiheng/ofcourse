#!/bin/bash

python main.py --log_dir runs/exp1_ippo_seed20 --algo ippo --env exp1 --seed 20
python main.py --log_dir runs/exp1_ippo_seed21 --algo ippo --env exp1 --seed 21
python main.py --log_dir runs/exp1_ippo_seed30 --algo ippo --env exp1 --seed 30
python main.py --log_dir runs/exp1_ippo_seed31 --algo ippo --env exp1 --seed 31
python main.py --log_dir runs/exp1_ippo_seed40 --algo ippo --env exp1 --seed 40
python main.py --log_dir runs/exp1_ippo_seed41 --algo ippo --env exp1 --seed 41