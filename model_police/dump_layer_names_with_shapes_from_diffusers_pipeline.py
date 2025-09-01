#!/usr/bin/env python3
""" Dump layer names from pipeline """

import argparse
import torch

from diffusers import AutoPipelineForText2Image

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    # load model
    pipe = AutoPipelineForText2Image.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    # print layer names
    for component_name, component in list(pipe.components.items()):
        try:
            for module_name, module in component.named_modules():    
                if isinstance(module, torch.nn.Linear):
                    print(f"{component_name}.{module_name},{module.in_features},{module.out_features}") 

        except:
            pass

if __name__ == "__main__":
    main()