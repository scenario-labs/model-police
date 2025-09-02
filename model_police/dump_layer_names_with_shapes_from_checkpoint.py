#!/usr/bin/env python3
""" Dump layer names from lora """

import argparse
import torch

from pathlib import Path

from model_police import ModelPolice

def main():

    model_police_officer = ModelPolice()

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("model", help="Flux, SDXL, SD15...", default="")
    parser.add_argument("framework", help="Diffusers, Kohya, Unknown", default="")
    args = parser.parse_args()

    is_lora, model_classes, layer_names_with_shapes, error = model_police_officer.inspect(args.checkpoint)

    if error is not None:
        raise Exception(error)

    if args.model and args.framework:

        for k in layer_names_with_shapes:
            print(k)

if __name__ == "__main__":
    main()
