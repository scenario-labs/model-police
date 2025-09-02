#!/usr/bin/env python3
""" Dump layer names from pipeline """

import argparse
import torch

from diffusers import AutoPipelineForText2Image
from pathlib import Path

here = Path(__file__).parent

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("model", help="Flux, SDXL, SD15...", default="", nargs='?')
    parser.add_argument("framework", help="Diffusers, Kohya, Unknown", default="", nargs='?')
    args = parser.parse_args()

    # load model
    pipe = AutoPipelineForText2Image.from_pretrained(args.repo_id, torch_dtype=torch.bfloat16)

    if args.model and args.framework:
        full_model_dict = here / "model_dictionaries" / f"{args.model}_{args.framework}.csv".lower()
        if full_model_dict.exists() and input(f"File {full_model_dict} already exists, override it ? [y/N]") != "y":
            exit()
        full_model_file = open(full_model_dict, "w")

    # print layer names
    for component_name, component in list(pipe.components.items()):
        if component is None or not hasattr(component, "named_modules"):
            continue
    
        try:
            component_keys = []
            for module_name, module in component.named_modules():    
                if isinstance(module, torch.nn.Linear):
                    key_without_prefix = f"{module_name},{module.in_features},{module.out_features}"
                    component_keys.append(key_without_prefix)

                    key_with_prefix = f"{component_name}.{module_name},{module.in_features},{module.out_features}"
                    if args.model and args.framework:
                        full_model_file.write(key_with_prefix + "\n")
                    else:
                        print(key_with_prefix) 
            if component_keys and args.model and args.framework:
                component_model_dict = here / "model_dictionaries" / f"{args.model}_{component_name}_{args.framework}.csv".lower()
                if not component_model_dict.exists() or input(f"File {component_model_dict} already exists, override it ? [y/N]") == "y":
                    with open(component_model_dict, "w") as f:
                        for k in component_keys:
                            f.write(k + "\n")
                    print(f"{component_name} keys with shapes dumped to {component_model_dict}")   
        
        except Exception as e:
            print(str(e))

    if args.model and args.framework:
        full_model_file.close()
        print(f"Full model keys with shapes dumped to {full_model_dict}")


if __name__ == "__main__":
    main()