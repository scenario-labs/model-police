<p align="center">
    <br>
    <img src="https://github.com/scenario-labs/model-police/blob/main/docs/sources/modelpolice.png?raw=true" width="400"/>
    <br>
</p>

`model_police` Python library aims to analyse the model checkpoints and their compatibility.

## Model safeguarding

```python
from model_police import PoliceOfficer

police_officer = PoliceOfficer()

try:
    model_class, model_type, framework, components = police_officer.inspect(checkpoint_path)
except Exception as e:
    logger.error(e)
```

Model class: flux.1, sd-1_5, sd-xl

Model type: full, lora

Model components: unet, vae, transformer, text_encoder ...

Model framework: diffusers, kohya, bfl, xlabs


## Model conversion


```python
from model_police import PoliceOfficer

police_officer = PoliceOfficer()

try:
    state_dict = police_officer.convert(
        state_dict, type="lora", from="kohya", to="diffusers"
    )
except Exception as e:
    logger.error(e)
```