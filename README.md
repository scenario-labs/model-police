<p align="center">
    <br>
    <img src="https://github.com/scenario-labs/model-police/blob/main/docs/sources/modelpolice.png?raw=true" width="400"/>
    <br>
</p>

`model_police` library aims to help deal with model and framework jungle. It classifies model checkpoints against model dictionaries, in order to check their class compatibility for safe loading. With the following rule in mind: better add more model dictionaries than add more rules to a code base that mixes with data and ends as a manually coded random forest :-)

The model police inspector takes as input either a model `state_dict`, a checkpoint path, or a folder path containing full models such as Flux.1 Dev.

The model police inspector requires a list of dictionaries in the folder `model_dictionaries` in the following form: `(model)_(type)_(framework)(_*).csv` where `model` is a model class identifier ('sdxl', 'flux', 'sd1-5', ...), `type` is checkpoint type ('full', 'lora', or components such as 'transformer', 'unet', 'vae', 'text-encoder', 'text-encoder-2', 'safety-checker') and `framework` is the model code format ('diffusers', 'bfl', 'kohya', ...). Anything else after in the model dictionary name is ignored. The scripts `./checkpoint keys model.safetensors` and `./dump_from_diffusers repo_id` enable to easily dump new model dictionary either from a checkpoint file or folder, or for any diffusers' library pipeline. 

It's possible to use it as a command line:

```bash
# local files
checkpoint classify model.safetensors 
checkpoint classify FLux.1-Dev/ 

# remote files (weights are downloaded to ~/.cache/modelpolice if env var CACHE_MODEL_POLICE not set)
checkpoint classify ByteDance/Hyper-SD
checkpoint classify ByteDance/Hyper-SD:Hyper-SD15-12steps-CFG-lora.safetensors
checkpoint classify "https://civitai.com/api/download/models/1264088?type=Model&format=SafeTensor&token=$CIVITAI_TOKEN" 
```

or as a Python lib:

```python
from model_police import ModelPolice

model_police_officer = ModelPolice()

full_models, checkpoint_list, error = model_police_officer.inspect(state_dict_or_checkpoint_path)
```

`full_models` is a list containing the list of full model classes that have been detected (`model_dictionaries` that have `full` type in their name). That means that all of the required keys have been matched.

In the checkpoint list, you'll find all checkpoints with their `files`, `layer_names_with_shapes` and `is_lora` attributes. If the checkpoint is a lora, you will find class model compatibility list. If the checkpoint is not a lora, you'll find the list of part models (['sdxl_vae*', 'sd1-5_vae*'] or ['sdxl_text-encoder', 'sd1-5_text-encoder'] for example since they are the same)


The `inspect()` method is safe and in case of production failure to classify a model, the list of layer names with shapes can be found in `layer_names_with_shapes` in order to add it to a new model dictionary for further classifications.


### create new model dictionaries


From a checkpoint:

```bash
checkpoint keys model_bqx1GeNju51BZVJfxJ4Wm59b.safetensors 

# or save the dictionary in the lib
checkpoint keys model_bqx1GeNju51BZVJfxJ4Wm59b.safetensors > flux_lora_kohya.csv 
```

For a dump from a diffusers pipeline, you can use the script `dump_from_diffusers` which takes as argument the repo_id, and optionally the model name. In case the model name is not provided, it will dump the keys to standard output. Otherwise when model name is provide, it will create multiple model dictionaries in the `model_dictionaries` folder. Type will be automatically added to model dictionary names.

```bash
pip install -r requirements-diffusers.txt

dump_from_diffusers stable-diffusion-v1-5/stable-diffusion-v1-5 sd1-5
dump_from_diffusers stabilityai/stable-diffusion-xl-base-1.0 sdxl
dump_from_diffusers black-forest-labs/FLUX.1-dev flux
dump_from_diffusers stabilityai/stable-diffusion-3-medium-diffusers sd3
```


### dictionaries

3 lora suffixes dictionaries: 
- `lora_up_suffixes.txt`
- `lora_down_suffixes.txt`
- `lora_ignore_suffixes.txt`

Model dictionaries are in `model_dictionaries` folder. Naming convention is `model_type_framework(*).csv`. For example: `flux_full_diffusers.csv`, `flux_lora_kohya.csv`, or `flux_lora_kohya_1.csv`



#### a note about Wan dictionaries:

- **wan2-14b_lora_wanai**: lora models loadable on any Wan 14B transformer (either Wan 2.1 transformer, or wan 2.2 transformers) - all tasks (T2V, I2V, TI2V)
- **wan2-low-i2v-14b_lora_wanai** : lora models loadable only on Low Noise Wan 2.2 14B transformer or Wan 2.1 14B transformer for I2V task  
- **wan2-high-i2v-14b_lora_wanai**: lora models loadable only on the Wan 2.2 high noise 14B transformer for I2V task
- **wan2-5b_lora_wanai**: lora models loadable on any Wan 5B transformer


### limitations

- the lib only reads gguf and safetensors. In particular, it does not read binary and onnx files yet.
- full model detection requires to give the correct root as input to the inspector, since it will prefix keys with subfolder names
- if multiple full models are present in a folder, there is no mean to separate these full models. Only loras and part models are separable.