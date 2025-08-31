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