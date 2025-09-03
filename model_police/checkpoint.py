#!/usr/bin/env python3
""" Dump layer names from lora """

import argparse
import torch

from pathlib import Path

from model_police import ModelPolice

here = Path(__file__).parent

def main():

    model_police_officer = ModelPolice()

    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="dump, keys, classify")
    parser.add_argument("checkpoint")
    parser.add_argument("model", help="Flux, SDXL, SD15... (only for command 'dump')", default="", nargs='?')
    parser.add_argument("framework", help="Diffusers, Kohya, Unknown.. (only for command 'dump')", default="", nargs='?')
    args = parser.parse_args()

    is_lora, model_classes, layer_names_with_shapes, error = model_police_officer.inspect(args.checkpoint)
    
    if error is not None:
        print(error)
        exit()

    if args.command == "dump":
        if not (args.model and args.framework):
            print("missing model and framework in the command dump")
        model_dict = here / "model_dictionaries" / f"{args.model}_{'lora' if is_lora else 'full'}_{args.framework}.csv".lower()
        if model_dict.exists() and input(f"File {model_dict} already exists, override it ? [y/N]") != "y":
            exit()
        with open(model_dict, "w") as f:
            for k in layer_names_with_shapes:
                f.write(k + "\n")

    elif args.command == "keys":
        for k in layer_names_with_shapes:
            print(k)

    elif args.command == "classify":
        print(f"is_lora: {is_lora}")
        if model_classes:
            print("Classes:")
            for model_class in model_classes:
                print(f"{model_class}({len(model_classes[model_class])})")
                if len(model_classes[model_class]) < 500:
                    for k in list(model_classes[model_class].keys())[:10]:
                        print("    ", k)
        else:
            print("No class found")

    else:
        print("Unknown command {args.command}")

if __name__ == "__main__":
    main()
