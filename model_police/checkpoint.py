#!/usr/bin/env python3
""" Extract layer names and classify checkpoints """

import argparse
import shutil
import sys 
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
    parser.add_argument("command", choices=['keys', 'classify'])
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    full_models, checkpoint_list, error = model_police_officer.inspect(args.checkpoint)
    
    sys.stderr.write("\n")  # clearing diffusers TQDM ;)

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
            print(" "*3, "- Files:")
            for f in checkpoint["files"]:
                print(" "*7, f"- {f}")
            
            print(" "*5, f"Lora: {checkpoint['is_lora']}")
            if checkpoint["is_lora"]:
                print(" "*5, "Lora model compatibility:")
                for family in checkpoint["lora_model_family"]:
                    print(" "*7, f"- {family}")
                    coverage = checkpoint['lora_model_family'][family]['coverage']
                    print(" "*9, f"Coverage: {coverage:.2f} ({int(coverage*100)}%)" )
                    print(" "*9, f"Dictnames:" )
                    family_dictnames = checkpoint["lora_model_family"][family]["matched_dictnames"]
                    for dictname in family_dictnames:
                        print(" "*9, f"- {dictname} ({len(family_dictnames[dictname])})")
                        if dictname == "unknown" or dictname == "flux_lora_kohya ":
                            print(" "*11, f"Missing keys:")
                            for k in list(family_dictnames[dictname].keys())[:10] + ["..."]:
                                print(" "*15, k)


                        if  dictname == "sdxl_lora_unknown":
                            for k in list(family_dictnames[dictname].keys()):
                                print(" "*15, k)
            else:
                print(" "*5, "Model components:", checkpoint["model_components"])
            print()


if __name__ == "__main__":
    main()
