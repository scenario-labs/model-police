# from kohya to diffusers
import torch
import logging

logger = logging.getLogger(__name__)


def convert_to_ai_toolkit(sds_sd, ait_sd, sds_key, ait_key, debug=False):
    if sds_key + ".lora_down.weight" not in sds_sd:
        return
    down_weight = sds_sd.pop(sds_key + ".lora_down.weight")

    # scale weight by alpha and dim
    rank = down_weight.shape[0]
    alpha = sds_sd.pop(sds_key + ".alpha").item()  # alpha is scalar
    scale = alpha / rank  # LoRA is scaled by 'alpha / rank' in forward pass, so we need to scale it back here
    if debug: print(f"rank: {rank}, alpha: {alpha}, scale: {scale}")

    # calculate scale_down and scale_up to keep the same value. if scale is 4, scale_down is 2 and scale_up is 2
    scale_down = scale
    scale_up = 1.0
    while scale_down * 2 < scale_up:
        scale_down *= 2
        scale_up /= 2
    if debug: print(f"scale: {scale}, scale_down: {scale_down}, scale_up: {scale_up}")

    ait_sd[ait_key + ".lora_A.weight"] = down_weight * scale_down
    ait_sd[ait_key + ".lora_B.weight"] = sds_sd.pop(sds_key + ".lora_up.weight") * scale_up


def convert_to_ai_toolkit_cat(sds_sd, ait_sd, sds_key, ait_keys, dims=None):
    if sds_key + ".lora_down.weight" not in sds_sd:
        return
    down_weight = sds_sd.pop(sds_key + ".lora_down.weight")
    up_weight = sds_sd.pop(sds_key + ".lora_up.weight")
    sd_lora_rank = down_weight.shape[0]

    # scale weight by alpha and dim
    alpha = sds_sd.pop(sds_key + ".alpha")
    scale = alpha / sd_lora_rank

    # calculate scale_down and scale_up
    scale_down = scale
    scale_up = 1.0
    while scale_down * 2 < scale_up:
        scale_down *= 2
        scale_up /= 2

    down_weight = down_weight * scale_down
    up_weight = up_weight * scale_up

    # calculate dims if not provided
    num_splits = len(ait_keys)
    if dims is None:
        dims = [up_weight.shape[0] // num_splits] * num_splits
    else:
        assert sum(dims) == up_weight.shape[0]

    # check upweight is sparse or not
    is_sparse = False
    if sd_lora_rank % num_splits == 0:
        ait_rank = sd_lora_rank // num_splits
        is_sparse = True
        i = 0
        for j in range(len(dims)):
            for k in range(len(dims)):
                if j == k:
                    continue
                is_sparse = is_sparse and torch.all(up_weight[i : i + dims[j], k * ait_rank : (k + 1) * ait_rank] == 0)
            i += dims[j]
        if is_sparse:
            logger.info(f"weight is sparse: {sds_key}")

    # make ai-toolkit weight
    ait_down_keys = [k + ".lora_A.weight" for k in ait_keys]
    ait_up_keys = [k + ".lora_B.weight" for k in ait_keys]
    if not is_sparse:
        # down_weight is copied to each split
        ait_sd.update({k: down_weight for k in ait_down_keys})

        # up_weight is split to each split  
        ait_sd.update({k: v for k, v in zip(ait_up_keys, torch.split(up_weight, dims, dim=0))})
    else:
        # down_weight is chunked to each split
        ait_sd.update({k: v for k, v in zip(ait_down_keys, torch.chunk(down_weight, num_splits, dim=0))})

        # up_weight is sparse: only non-zero values are copied to each split
        i = 0
        for j in range(len(dims)):
            ait_sd[ait_up_keys[j]] = up_weight[i : i + dims[j], j * ait_rank : (j + 1) * ait_rank].contiguous()
            i += dims[j]


def convert_sd_scripts_to_ai_toolkit(sds_sd):
    ait_sd = {}
    for i in range(19):
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_img_attn_proj", f"transformer.transformer_blocks.{i}.attn.to_out.0"
        )
        convert_to_ai_toolkit_cat(
            sds_sd,
            ait_sd,
            f"lora_unet_double_blocks_{i}_img_attn_qkv",
            [
                f"transformer.transformer_blocks.{i}.attn.to_q",
                f"transformer.transformer_blocks.{i}.attn.to_k",
                f"transformer.transformer_blocks.{i}.attn.to_v",
            ],
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_img_mlp_0", f"transformer.transformer_blocks.{i}.ff.net.0.proj"
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_img_mlp_2", f"transformer.transformer_blocks.{i}.ff.net.2"
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_img_mod_lin", f"transformer.transformer_blocks.{i}.norm1.linear"
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_txt_attn_proj", f"transformer.transformer_blocks.{i}.attn.to_add_out"
        )
        convert_to_ai_toolkit_cat(
            sds_sd,
            ait_sd,
            f"lora_unet_double_blocks_{i}_txt_attn_qkv",
            [
                f"transformer.transformer_blocks.{i}.attn.add_q_proj",
                f"transformer.transformer_blocks.{i}.attn.add_k_proj",
                f"transformer.transformer_blocks.{i}.attn.add_v_proj",
            ],
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_txt_mlp_0", f"transformer.transformer_blocks.{i}.ff_context.net.0.proj"
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_txt_mlp_2", f"transformer.transformer_blocks.{i}.ff_context.net.2"
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_txt_mod_lin", f"transformer.transformer_blocks.{i}.norm1_context.linear"
        )

    for i in range(38):
        convert_to_ai_toolkit_cat(
            sds_sd,
            ait_sd,
            f"lora_unet_single_blocks_{i}_linear1",
            [
                f"transformer.single_transformer_blocks.{i}.attn.to_q",
                f"transformer.single_transformer_blocks.{i}.attn.to_k",
                f"transformer.single_transformer_blocks.{i}.attn.to_v",
                f"transformer.single_transformer_blocks.{i}.proj_mlp",
            ],
            dims=[3072, 3072, 3072, 12288],
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_single_blocks_{i}_linear2", f"transformer.single_transformer_blocks.{i}.proj_out"
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_single_blocks_{i}_modulation_lin", f"transformer.single_transformer_blocks.{i}.norm.linear"
        )

    if any(
        k.startswith("lora_te1_text_model_encoder_layers") for k in sds_sd
    ):
        logger.info("Converting text encoder layers")

        for i in range(12):
            convert_to_ai_toolkit(
                sds_sd, ait_sd, f"lora_te1_text_model_encoder_layers_{i}_mlp_fc1", f"text_encoder.text_model.encoder.layers.{i}.mlp.fc1"
            )

            convert_to_ai_toolkit(
                sds_sd, ait_sd, f"lora_te1_text_model_encoder_layers_{i}_mlp_fc2", f"text_encoder.text_model.encoder.layers.{i}.mlp.fc2"
            )

            convert_to_ai_toolkit(
                sds_sd, ait_sd, f"lora_te1_text_model_encoder_layers_{i}_self_attn_k_proj", f"text_encoder.text_model.encoder.layers.{i}.self_attn.k_proj"
            )

            convert_to_ai_toolkit(
                sds_sd, ait_sd, f"lora_te1_text_model_encoder_layers_{i}_self_attn_out_proj", f"text_encoder.text_model.encoder.layers.{i}.self_attn.out_proj"
            )

            convert_to_ai_toolkit(
                sds_sd, ait_sd, f"lora_te1_text_model_encoder_layers_{i}_self_attn_q_proj", f"text_encoder.text_model.encoder.layers.{i}.self_attn.q_proj"
            )

            convert_to_ai_toolkit(
                sds_sd, ait_sd, f"lora_te1_text_model_encoder_layers_{i}_self_attn_v_proj", f"text_encoder.text_model.encoder.layers.{i}.self_attn.v_proj"
            )

    if len(sds_sd) > 0:
        logger.warning(f"Unsuppored keys for ai-toolkit: {sds_sd.keys()}")
    return ait_sd
