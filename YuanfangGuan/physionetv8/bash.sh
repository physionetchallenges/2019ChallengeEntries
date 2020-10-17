#!/bin/bash

perl split.pl ${1}
python train.py
rm -rf result
python driver.py training_data/ result
perl create_gs.pl
python evaluate_sepsis_score.py gs/ result/ >> evaluation.txt
