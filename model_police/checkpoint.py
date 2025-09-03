#!/usr/bin/env python3
""" Dump layer names from lora """

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
    parser.add_argument("command", help="dump, keys, classify")
    parser.add_argument("checkpoint")
    parser.add_argument("model", help="Flux, SDXL, SD15... (only for command 'dump')", default="", nargs='?')
    parser.add_argument("framework", help="Diffusers, Kohya, Unknown.. (only for command 'dump')", default="", nargs='?')
    args = parser.parse_args()

    full_models, checkpoint_list, error = model_police_officer.inspect(args.checkpoint)
    
    if error is not None:
        print(error)
        exit()

    if args.command == "dump":
        if not (args.model and args.framework):
            print("missing model and framework in the command dump")
            exit()

        full_model_dict = None
        if len(checkpoint_list) > 1:
            full_model_dict = here / "model_dictionaries" / f"{clean(args.model)}_full_{clean(args.framework)}.csv"
            if full_model_dict.exists() and input(f"File {full_model_dict} already exists, override it ? [y/N]") != "y":
                full_model_dict = None
            if full_model_dict is not None:
                full_model_dict = open(full_model_dict, "w")


        for checkpoint in checkpoint_list:
            prefix = checkpoint["prefix"]
            is_lora = checkpoint["is_lora"]
            part_or_lora_model_dict = here / "model_dictionaries" / f"{clean(args.model)}_{'lora' if is_lora else (prefix if prefix else 'part')}_{clean(args.framework)}.csv"
            if part_or_lora_model_dict.exists() and input(f"File {part_or_lora_model_dict} already exists, override it ? [y/N]") != "y":
                part_or_lora_model_dict = None
            if part_or_lora_model_dict is not None:
                part_or_lora_model_dict = open(part_or_lora_model_dict, "w")


            if prefix:
                prefix += "."
        
            for k in checkpoint["layer_names_with_shapes"]:
                if part_or_lora_model_dict is not None:
                    part_or_lora_model_dict.write(f"{k}\n")
                if full_model_dict is not None:
                    full_model_dict.write(f"{prefix}{k}\n")
                
            if part_or_lora_model_dict is not None:
                part_or_lora_model_dict.close()

        if full_model_dict is not None:
            full_model_dict.close()
            

    elif args.command == "keys":
        for checkpoint in checkpoint_list:
            prefix = checkpoint["prefix"]
            if prefix:
                prefix += "."
        
            for k in checkpoint["layer_names_with_shapes"]:
                print(f"{prefix}{k}")

    elif args.command == "classify":
        print("Found full models:", full_models)
        print("Found checkpoints:")
        for checkpoint in checkpoint_list:
            print("   Files:")
            for f in checkpoint["files"]:
                print(f"   - {f}")
            
            print(f"is_lora: {checkpoint['is_lora']}")
            print("Model parts:", checkpoint["model_parts"])
            
            # if model_classes:
            #     print("Classes:")
            #     for model_class in model_classes:
            #         print(f"{model_class}({len(model_classes[model_class])})")
            #         if len(model_classes[model_class]) < 500:
            #             for k in list(model_classes[model_class].keys())[:10]:
            #                 print("    ", k)

    else:
        print("Unknown command {args.command}")

if __name__ == "__main__":
    main()
