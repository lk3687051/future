#!/usr/bin/env python
import argparse
from future.features import gen_train_dataset
parser = argparse.ArgumentParser()
parser.add_argument('name', choices=['history'], help='The name of train feature')
args = parser.parse_args()
print(args.name)
gen_train_dataset(args.name)
