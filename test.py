from safetensors import safe_open
from model_police import ModelPolice

validator = ModelPolice()

lora_path = "/workspace/reforge/models/kohya_training/model_bqx1GeNju51BZVJfxJ4Wm59b.safetensors"
lora_path = "/workspace/reforge/models/kohya_training/model_bqx1GeNju51BZVJfxJ4Wm59b_converted.safetensors"

print("Loading state dict")
state_dict = {}
with safe_open(lora_path, framework="pt", device=0) as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)

print(f"Initial state dict length: {len(state_dict)}")
is_lora, model_classes, layer_names_with_shapes, error = validator.inspect(state_dict)

if len(model_classes) > 1:
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
    print(state_dict.keys())

if (not is_lora and "diffusers" in model_class):
    components = validator.get_diffusers_components(layer_names_with_shapes)
    print(components)  # ["vae", "safety_checker", "unet", "text_encoder"]
    