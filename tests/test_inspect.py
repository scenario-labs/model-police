import pytest
import torch

from safetensors.torch import save_file

from model_police import ModelPolice


@pytest.fixture(scope="module")
def police():
    return ModelPolice()


def make_checkpoint(tmp_path, state_dict):
    checkpoint_path = tmp_path / "model.safetensors"
    save_file(state_dict, str(checkpoint_path))
    return checkpoint_path


def test_inspect_keeps_layer_names_on_mixed_keys_error(police, tmp_path):
    """When is_lora() fails with mixed lora/full keys, the checkpoint must still
    expose layer_names_with_shapes so consumers (e.g. Sentry attachments) can
    report which keys caused the failure."""
    lora_down = police._lora_down_suffixes[0]
    lora_up = police._lora_up_suffixes[0]

    checkpoint_path = make_checkpoint(tmp_path, {
        f"transformer.blocks.0.attn.to_q{lora_down}": torch.zeros(4, 64),
        f"transformer.blocks.0.attn.to_q{lora_up}": torch.zeros(64, 4),
        "transformer.blocks.0.attn.to_q.weight": torch.zeros(64, 64),
    })

    full_models, checkpoint_list, error = police.inspect(checkpoint_path)

    assert error == "Mixed lora keys and full keys"
    assert len(checkpoint_list) == 1

    layer_names = checkpoint_list[0]["layer_names_with_shapes"]
    assert len(layer_names) == 3
    assert f"transformer.blocks.0.attn.to_q{lora_down},4,64" in layer_names
    assert "transformer.blocks.0.attn.to_q.weight,64,64" in layer_names


def test_inspect_lora_layer_names_use_lora_format(police, tmp_path):
    """For a valid LoRA, layer_names_with_shapes must keep the merged
    lora format (key,in_features,out_features), not the raw key list."""
    lora_down = police._lora_down_suffixes[0]
    lora_up = police._lora_up_suffixes[0]

    checkpoint_path = make_checkpoint(tmp_path, {
        f"transformer.blocks.0.attn.to_q{lora_down}": torch.zeros(4, 64),
        f"transformer.blocks.0.attn.to_q{lora_up}": torch.zeros(64, 4),
    })

    full_models, checkpoint_list, error = police.inspect(checkpoint_path)

    assert error is None
    assert len(checkpoint_list) == 1
    assert checkpoint_list[0]["is_lora"] is True
    assert checkpoint_list[0]["layer_names_with_shapes"] == [
        "transformer.blocks.0.attn.to_q,64,64",
    ]
