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


def test_inspect_lokr_layer_names_use_kronecker_dims(police, tmp_path):
    """For a LoKr, the layer dimensions are the products of the two factor
    dimensions (delta = w1 kron w2), so a (4,4) x (960,960) pair maps to a
    3840x3840 layer."""
    checkpoint_path = make_checkpoint(tmp_path, {
        "diffusion_model.layers.0.attention.to_q.lokr_w1": torch.zeros(4, 4),
        "diffusion_model.layers.0.attention.to_q.lokr_w2": torch.zeros(960, 960),
        "diffusion_model.layers.0.attention.to_q.alpha": torch.tensor(4.0),
    })

    full_models, checkpoint_list, error = police.inspect(checkpoint_path)

    assert error is None
    assert len(checkpoint_list) == 1
    assert checkpoint_list[0]["is_lora"] is True
    assert checkpoint_list[0]["layer_names_with_shapes"] == [
        "diffusion_model.layers.0.attention.to_q,3840,3840",
    ]


def test_inspect_mixed_lokr_and_lora_modules(police, tmp_path):
    """A single adapter may mix lokr modules and plain lora modules
    (e.g. ai-toolkit Z-Image lokr): both must contribute layer names and no
    'Mixed lora keys and full keys' error must be raised."""
    lora_down = police._lora_down_suffixes[0]
    lora_up = police._lora_up_suffixes[0]

    checkpoint_path = make_checkpoint(tmp_path, {
        "diffusion_model.layers.0.attention.to_q.lokr_w1": torch.zeros(4, 4),
        "diffusion_model.layers.0.attention.to_q.lokr_w2": torch.zeros(960, 960),
        "diffusion_model.layers.0.attention.to_q.alpha": torch.tensor(4.0),
        f"diffusion_model.layers.0.feed_forward.w1{lora_down}": torch.zeros(32, 3840),
        f"diffusion_model.layers.0.feed_forward.w1{lora_up}": torch.zeros(10240, 32),
    })

    full_models, checkpoint_list, error = police.inspect(checkpoint_path)

    assert error is None
    assert checkpoint_list[0]["is_lora"] is True
    assert checkpoint_list[0]["layer_names_with_shapes"] == [
        "diffusion_model.layers.0.attention.to_q,3840,3840",
        "diffusion_model.layers.0.feed_forward.w1,3840,10240",
    ]


def test_inspect_zimage_lokr_classifies_as_zimage(police, tmp_path):
    """A Z-Image lokr (ComfyUI/ai-toolkit key format) must classify against the
    existing zimage lora dictionaries with full coverage."""
    state_dict = {}
    for module, w2_shape in (
        ("attention.to_q", (960, 960)),
        ("attention.to_k", (960, 960)),
        ("attention.to_v", (960, 960)),
        ("adaLN_modulation.0", (3840, 64)),
    ):
        state_dict[f"diffusion_model.layers.0.{module}.lokr_w1"] = torch.zeros(4, 4)
        state_dict[f"diffusion_model.layers.0.{module}.lokr_w2"] = torch.zeros(*w2_shape)
        state_dict[f"diffusion_model.layers.0.{module}.alpha"] = torch.tensor(4.0)
    for module, in_features, out_features in (
        ("feed_forward.w1", 3840, 10240),
        ("feed_forward.w2", 10240, 3840),
        ("feed_forward.w3", 3840, 10240),
        ("attention.to_out.0", 3840, 3840),
    ):
        state_dict[f"diffusion_model.layers.0.{module}.lora_A.weight"] = torch.zeros(32, in_features)
        state_dict[f"diffusion_model.layers.0.{module}.lora_B.weight"] = torch.zeros(out_features, 32)

    checkpoint_path = make_checkpoint(tmp_path, state_dict)

    full_models, checkpoint_list, error = police.inspect(checkpoint_path)

    assert error is None
    checkpoint = checkpoint_list[0]
    assert checkpoint["is_lora"] is True
    assert "zimage" in checkpoint["lora_model_family"]
    zimage = checkpoint["lora_model_family"]["zimage"]
    assert zimage["coverage"] == 1.0
    assert zimage["num_missing"] == 0
    # all lokr and lora weights are routed to a matched dictionary,
    # alphas are reported as ignored
    matched = zimage["matched_dictnames"]
    assert "unknown" not in matched
    assert sorted(matched.pop("ignored")) == sorted(
        k for k in state_dict if k.endswith(".alpha")
    )
    matched_weight_keys = [k for d in matched.values() for k in d]
    assert sorted(matched_weight_keys) == sorted(
        k for k in state_dict if not k.endswith(".alpha")
    )
