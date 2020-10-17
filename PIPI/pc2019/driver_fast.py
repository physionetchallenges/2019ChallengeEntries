import sys
from classifier import *
from full_seq import build_model

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')

    classify(sys.argv[1], sys.argv[2], build_model)
