#!/usr/bin/env python3
""" Dump layer names from lora """

import argparse
import torch

from pathlib import Path

from model_police import ModelPolice

model_police_officer = ModelPolice()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    is_lora, model_class, model_layer_names_and_shapes = model_police_officer.inspect(args.model)

    for k in model_layer_names_and_shapes:
        print(k)
