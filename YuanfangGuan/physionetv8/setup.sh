#! /bin/bash
#
# file: setup.sh
#
# This bash script performs any setup necessary in order to test your
# entry.  It is run only once, before running any other code belonging
# to your entry.

set -e
set -o pipefail
python -m venv --system-site-packages ~/my-venv
. ~/my-venv/bin/activate
pip install wheel
pip install lightgbm

chmod a+x ./get_sepsis_score.py
