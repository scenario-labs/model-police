#!/usr/bin/env python3
""" Extract layer names and classify checkpoints """

import argparse
import shutil
import sys 
import torch

from pathlib import Path

from .model_police import ModelPolice

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
    parser.add_argument("command", choices=['keys', 'classify', 'diff', 'match'])
    parser.add_argument("params", help="Flux, SDXL, SD15...", default="", nargs='+')
    args = parser.parse_args()
    
    checkpoint_path = args.params[0]
    full_models, checkpoint_list, error = model_police_officer.inspect(checkpoint_path)
    
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
            print("- Files:")
            for f in checkpoint["files"]:
                print(" "*3, f"- {f}")
            
            print(" ", f"Lora: {checkpoint['is_lora']}")
            print(" ", f"Num keys: {checkpoint['num_keys']}")
            if checkpoint["is_lora"]:
                print(" ", "Lora model compatibility:")
                for family in checkpoint["lora_model_family"]:
                    print(" ", f"- {family}")
                    coverage = checkpoint['lora_model_family'][family]['coverage']
                    print(" "*5, f"Coverage: {coverage:.2f} ({int(coverage*100)}%)" )  
                    print(" "*5, f"Dictnames:" )
                    family_dictnames = checkpoint["lora_model_family"][family]["matched_dictnames"]
                    family_dictnames_recall = checkpoint["lora_model_family"][family]["matched_dictnames_recall"]
                    for dictname in family_dictnames:
                        recall = f" - recall: {int(family_dictnames_recall[dictname]*100)}%" if dictname in family_dictnames_recall else ""
                        print(" "*5, f"- {dictname} ({len(family_dictnames[dictname])} keys{recall})")
                        if dictname == "unknown":
                            print(" "*8, f"To check missing keys, run: `checkpoint diff {checkpoint_path} {family}`")

            else:
                print(" ", "Model components:", checkpoint["model_components"])
            print()


    elif args.command in ["diff", "match"]:
        if len(args.params) == 1:
            raise ValueError("Requires a second arg: checkpoint diff checkpoint model_family\n")
        family = args.params[1]

        if len(full_models):
            raise ValueError(f"Found full models: {full_models}, works only with lora\n")

        if len(checkpoint_list) > 1:
            raise ValueError(f"Found multiple checkpoints {[c['files'][0] for c in checkpoint_list]}, works only with lora\n")

        checkpoint = checkpoint_list[0]

        if not checkpoint["is_lora"]:
            raise ValueError("Works only with loras")

        if family not in checkpoint["lora_model_family"]:
            raise ValueError(f"Family not found")
        family_dictnames = checkpoint["lora_model_family"][family]["matched_dictnames"]

        if args.command == "diff":
            if "unknown" not in family_dictnames:
                exit()

            for k in list(family_dictnames["unknown"].keys()):
                sys.stdout.write(k + "\n")

        else: # match
            all_keys = set()
            for d in family_dictnames:
                if d == "unknown":
                    continue
                all_keys.update(family_dictnames[d].keys())
            for k in sorted(all_keys):
                sys.stdout.write(k + "\n")

if __name__ == "__main__":
    main()
