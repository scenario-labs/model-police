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

    is_lora, model_classes, layer_names_with_shapes, error = model_police_officer.inspect(args.model)

    if error is not None:
        raise Exception(error)

    for k in layer_names_with_shapes:
        print(k)
