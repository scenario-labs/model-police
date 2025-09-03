<p align="center">
    <br>
    <img src="https://github.com/scenario-labs/model-police/blob/main/docs/sources/modelpolice.png?raw=true" width="400"/>
    <br>
</p>

`model_police` library aims to help developers to deal with model and framework jungle. It classifies model checkpoints against model dictionaries, in order to check their class compatibility for safe loading. With following rule in mind: better add more model dictionaries than add more rules to a code base that mixes with data and ends as a manually coded random forest.

The model police inspector takes as input either a model `state_dict`, a checkpoint path, or a folder path containing full models such as Flux.1 Dev.

The model police inspector requires a list of dictionaries in the folder `model_dictionaries` in the following form: `(model)_(type)_(framework)(_*).csv` where `model` is a model class identifier ('sdxl', 'flux', 'sd1-5', ...), `type` is checkpoint type ('full', 'lora', or components such as 'transformer', 'unet', 'vae', 'text-encoder', 'text-encoder-2', 'safety-checker') and `framework` is the model code format ('diffusers', 'bfl', 'kohya', ...). Anything after in the model dictionary name is ignored. The scripts `./checkpoinit dump model.safetensors` and `./dump_from_diffusers repo_id` enable to easily dump new model dictionary either from a checkpoint file or folder, or for any diffusers' library pipeline. 

It's possible to use it as a command line:

```bash
checkpoint classify model.safetensors 

checkpoint classify FLux.1-Dev/ 
```

or as a Python lib:

```python
from model_police import ModelPolice

model_police_officer = ModelPolice()

full_models, checkpoint_list, error = model_police_officer.inspect(state_dict_or_checkpoint_path)
```

`full_models` is a list containing the list of full model classes that have been detected (`model_dictionaries` that have `full` type in their name). That means that all of the required keys have been matched.

In the checkpoint list, you'll find all checkpoints with their `layer_names_with_shapes` and `is_lora` attributes. If the checkpoint is a lora, you will find class model compatibility list. If the checkpoint is not a lora, you'll find the list of part models (['sdxl_vae*', 'sd1-5_vae*'] or ['sdxl_text-encoder', 'sd1-5_text-encoder'] for example since they are the same)


The `inspect()` method is safe and in case of production failure to classify a model, the list of layer names with shapes can be found in `layer_names_with_shapes` in order to add it to a new model dictionary for further classifications.


### create new model dictionaries


From a checkpoint:

```bash
checkpoint keys model_bqx1GeNju51BZVJfxJ4Wm59b.safetensors 

# or save the dictionary in the lib
checkpoint dump model_bqx1GeNju51BZVJfxJ4Wm59b.safetensors flux kohya 
```

For a dump from a diffusers pipeline, you can use the script `dump_from_diffusers` which takes as argument the repo_id, as well as full model name and framework name. It will create multiple model dictionaries and dump them in the `model_dictionaries` folder. Type will be automatically added to model dictionary names. If model and framework not given, it will dump the keys to the standard output.

```bash
pip install -r requirements-diffusers.txt

dump_from_diffusers stable-diffusion-v1-5/stable-diffusion-v1-5 sd1-5 diffusers
dump_from_diffusers stabilityai/stable-diffusion-xl-base-1.0 sdxl diffusers
dump_from_diffusers black-forest-labs/FLUX.1-dev flux diffusers
```


### dictionaries

3 lora suffixes dictionaries: 
- `lora_up_suffixes.txt`
- `lora_down_suffixes.txt`
- `lora_ignore_suffixes.txt`

Model dictionaries are in `model_dictionaries` folder. Naming convention is `model_type_framework(*).csv`. For example: `flux_full_diffusers.csv`, `flux_lora_kohya.csv`, or `flux_lora_kohya_1.csv`


### loras

<!-- if len(model_classes) > 1:
    raise ValueError("It's a mixture of models")

model_class = list(model_classes.keys())[0]
state_dict = model_classes[model_class]

if model_class == "unknown":
    raise ValueError("Unknown model")
else:
    print(f"Model class: {model_class}")

if model_class == "flux_kohya":
    from conversions import kohya
    state_dict = kohya.convert_sd_scripts_to_ai_toolkit(state_dict)


if (not is_lora and "diffusers" in model_class):
    components = model_police_officer.get_diffusers_components(layer_names_with_shapes)
    print(components)  # ["vae", "safety_checker", "unet", "text_encoder"] -->



### limitations

- the lib only reads gguf and safetensors. In particular, it does not read binary and onnx files yet.
- when multiple version of the same model in different precision are present in the folder, the lib overrides the key if already present. Should not be a problem, except if there is a mix of different models or framework, the lib creates a dictionary with all keys together.