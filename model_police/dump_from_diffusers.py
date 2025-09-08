#!/usr/bin/env python3
""" Dump layer names from pipeline """

import argparse
import torch

from diffusers import DiffusionPipeline
from pathlib import Path

here = Path(__file__).parent

force = True

def clean(name):
    name = name.replace(".", "-")
    name = name.replace("_", "-")
    name = name.replace(" ", "-")
    name = name.lower()
    return name


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("model", help="Flux, SDXL, SD15...", default="", nargs='?')
    args = parser.parse_args()

    # load model
    pipe = DiffusionPipeline.from_pretrained(args.repo_id, torch_dtype=torch.bfloat16)

    full_model_keys = []
    for component_name, component in list(pipe.components.items()):
        if component is None or not hasattr(component, "named_parameters"):
            continue
    
        try:
            component_keys = []

            for module_name, module in component.named_modules():
                for weight_suffix in [
                    "weight", "bias", "concept_embeds", "concept_embeds", "concept_embeds_weights",
                    "special_care_embeds", "special_care_embeds_weights", "class_embedding",
                ]: # "position_ids",
                    if (w:= getattr(module, weight_suffix, None)) is not None:
                        if isinstance(w, torch.Tensor):
                            shape_to_list = ','.join(map(str, list(w.shape)))
                        elif isinstance(w, float):
                            shape_to_list = ','

                        key_without_prefix = f"{module_name + '.' if module_name else '' }{weight_suffix},{shape_to_list}"
                        component_keys.append(key_without_prefix)

                        key_with_prefix = f"{component_name}.{module_name + '.' if module_name else ''}{weight_suffix},{shape_to_list}"
                        if args.model:
                            full_model_keys.append(key_with_prefix)
                        else:
                            print(key_with_prefix) 

            if component_keys and args.model:
                component_model_dict = here / "model_dictionaries" / f"{clean(args.model)}_{clean(component_name)}_diffusers.csv".lower()
                if not component_model_dict.exists() or force or input(f"File {component_model_dict} already exists, override it ? [y/N]") == "y":
                    with open(component_model_dict, "w") as f:
                        for k in sorted(list(set(component_keys))):
                            f.write(k + "\n")
                    print(f"{component_name} keys with shapes dumped to {component_model_dict}")   
        
        except Exception as e:
            print(str(e))
            raise e

    if args.model:
        full_model_dict = here / "model_dictionaries" / f"{clean(args.model)}_full_diffusers.csv"
        if full_model_dict.exists() and not force and input(f"File {full_model_dict} already exists, override it ? [y/N]") != "y":
            exit()
        with open(full_model_dict, "w") as full_model_file:
            for key in sorted(list(set(full_model_keys))):
                full_model_file.write(key + "\n")

        print(f"Full model keys with shapes dumped to {full_model_dict}")


if __name__ == "__main__":
    main()