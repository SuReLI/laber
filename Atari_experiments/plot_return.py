#!/usr/bin/env python

import os
import pickle
import json
import argparse
from collections import defaultdict

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('path', help="Tf path to event files from which to extract variables")
parser.add_argument('-w', '--write', default=None, type=str, dest='write_dir')
parser.add_argument('-j', '--json', action='store_true')
args = parser.parse_args()


def extract_file(file, values):
    for event in tf.compat.v1.train.summary_iterator(file):
        for value in event.summary.value:
            values[value.tag].append(value.simple_value)

    return values


if not os.path.exists(args.path):
    print("No such file or directory")
    exit()

if os.path.isfile(args.path):
    files = [args.path]

elif os.path.isdir(args.path):
    files = []
    for directory, _, file_list in os.walk(args.path):
        for file in file_list:
            if file.startswith('events.out.tfevents.'):
                files.append(os.path.join(directory, file))

    if not files:
        print("No event file found")
        exit()

else:
    print("Invalid file type")
    exit()


if args.write_dir and not os.path.exists(args.write_dir):
    os.makedirs(args.write_dir)

values = defaultdict(list)
for file in files:
    values = extract_file(file, values)
    
    if args.write_dir:
        save_file_nb, computer_name = file.split('.')[-2:]
        extension = '.json' if args.json else '.pkl'
        save_file_name = 'Events_' + str(save_file_nb) + '_' + computer_name + extension
        mode = 'w' if args.json else 'wb'
        with open(os.path.join(args.write_dir, save_file_name), mode) as save_file:
            if args.json:
                json.dump(values, save_file)
            else:
                pickle.dump(values, save_file)

for k, v in values.items():
    if 'Returns' in k:
        plt.plot(v, label=k)

plt.legend()
plt.show()
