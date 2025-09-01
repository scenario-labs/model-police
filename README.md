<p align="center">
    <br>
    <img src="https://github.com/scenario-labs/model-police/blob/main/docs/sources/modelpolice.png?raw=true" width="400"/>
    <br>
</p>

`model_police` Python library aims to analyse the model checkpoints and their compatibility.

```python
from model_police import ModelPolice

model_police_officer = ModelPolice()

try:
    is_lora, model_class, layer_names_and_shapes = model_police_officer.inspect(state_dict_or_checkpoint_path)
except Exception as e:
    logger.error(e)
```

### create new model dictionaries


From diffusers:

```bash
./dump_layer_names_with_shapes_from_diffusers_pipeline.py "black-forest-labs/FLUX.1-dev" > model_dictionaries/flux_diffusers.csv
```

From a checkpoint:

```bash
./dump_layer_names_with_shapes_from_checkpoint.py model_bqx1GeNju51BZVJfxJ4Wm59b.safetensors > model_dictionaries/flux_kohya.csv
```

### dictionaries

3 lora suffixes dictionaries: 
- `lora_up_suffixes.txt`
- `lora_down_suffixes.txt`
- `lora_alpha_suffixes.txt`

Model dictionaries are in `model_dictionaries` folder. Naming convention is `modelclass_framework(*).csv`. For example: `flux_diffusers.csv`, `flux_kohya.csv`, or `flux_kohya_1.csv`