#!/usr/bin/env python3
""" Extract layer names and classify checkpoints """

import argparse
import torch

from pathlib import Path

from model_police import ModelPolice

here = Path(__file__).parent

def clean(name):
    name = name.replace(".", "-")
    name = name.replace("_", "-")
    name = name.replace(" ", "-")
    name = name.lower()
    return name


def main():

    model_police_officer = ModelPolice()

    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="'keys' or 'classify'")
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    full_models, checkpoint_list, error = model_police_officer.inspect(args.checkpoint)
    
    if error is not None:
        raise Exception(error)
 
    if args.command == "keys":
        subfolders = set(str(c["subfolder"]) for c in checkpoint_list)
        add_prefix = len(subfolders) > 1

        keys = []
        for checkpoint in checkpoint_list:
            if add_prefix:
                prefix = model_police_officer.get_prefix_from_subfolder(checkpoint["subfolder"])
            else:
                prefix = ""
            for k in checkpoint["layer_names_with_shapes"]:
                keys.append(f"{prefix}{k}")

        for k in sorted(list(set(keys))):
            print(k)

    elif args.command == "classify":
        print("Found full models:", full_models)
        print("Found checkpoints:")
        for checkpoint in checkpoint_list:
            print("   - Files:")
            for f in checkpoint["files"]:
                print(f"     - {f}")
            
            print(f"     is_lora: {checkpoint['is_lora']}")
            if checkpoint["is_lora"]:
                print("     Lora compatibilities")
                for model_class in checkpoint["model_classes"]:
                    print(f"     - {model_class}({len(checkpoint['model_classes'][model_class])})")
                    if len(checkpoint["model_classes"][model_class]) < 500:
                        for k in list(checkpoint["model_classes"][model_class].keys())[:10]:
                            print("            ", k)
            else:
                print("     Model components:", checkpoint["model_components"])
            print()
    else:
        print("Unknown command {args.command}")

if __name__ == "__main__":
    main()
