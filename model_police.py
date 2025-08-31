import logging
import torch

from pathlib import Path
from gguf.gguf_reader import GGUFReader
from safetensors import safe_open

from kohya import convert_sd_scripts_to_ai_toolkit

logger = logging.getLogger(__name__)


here = Path(__file__).parent

class ModelPolice:
    _layername_and_shape_to_dictname: dict[str, str] = {}
    _lora_down_suffixes: list[str] = []
    _lora_up_suffixes: list[str] = []
    _lora_alpha_suffixes: list[str] = []

    def __init__(self):
        
        logging.info("Reading model validator dictionaries")
        with open(here / "lora_up_suffixes.txt") as f:
            self._lora_up_suffixes = [s.strip() for s in f.readlines()]
        with open(here / "lora_down_suffixes.txt") as f:
            self._lora_down_suffixes = [s.strip() for s in f.readlines()]
        with open(here / "lora_alpha_suffixes.txt") as f:
            self._lora_alpha_suffixes = [s.strip() for s in f.readlines()]
        
        for dict_filepath in (here / "model_dictionaries").iterdir():
            dict_name = dict_filepath.name.removesuffix(".csv")
            ## dictname is in the form "modelclass_framework"
            # model_class, framework, *_ = dict_name.split("_")

            with open(dict_filepath, "r") as f:                
                for line in f:
                    line = line.strip()
                    if line not in self._layername_and_shape_to_dictname:
                        self._layername_and_shape_to_dictname[line] = []
                    self._layername_and_shape_to_dictname[line].append(dict_name)
                    ## line is in the form "key,in_features,out_features"
                    # layer_name, in_features, out_features = line.split(",")

    @staticmethod
    def get_state_dict_shapes(state_dict_or_checkpoint_path):
        
        if isinstance(state_dict_or_checkpoint_path, dict):
            # input is a state dict
            state_dict_shape = {k: t.shape for k, v in state_dict_or_checkpoint_path.items()}

        elif isinstance(state_dict_or_checkpoint_path, str):
            # input is a safetensors or gguf file
            checkpoint_suffix = Path(state_dict_or_checkpoint_path).suffix
            state_dict_shape = {}
            if checkpoint_suffix == ".safetensors":
                with safe_open(state_dict_or_checkpoint_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict_shape[key] = list(f.get_tensor(key).shape)
            elif checkpoint_suffix == ".gguf":
                reader = GGUFReader(state_dict_or_checkpoint_path)
                for tensor in reader.tensors:
                    state_dict_shape[tensor.name] = tensor.shape.tolist()
            else: 
                raise ValueError(f"Unknown checkpoint suffix: {checkpoint_suffix}")
        else:
            raise ValueError(
                f"Unknown type for checkpoint input: {type(state_dict_or_checkpoint_path)}"
            )

        return state_dict_shape


    @staticmethod
    def state_dict_shapes_to_list(state_dict_shapes):
        return [
            f"{k},{','.join(map(str, v))}" for k, v in state_dict_shapes.items()
        ]


    def is_lora_key(self, key):
        for suffix in self._lora_down_suffixes + self._lora_up_suffixes + self._lora_alpha_suffixes:
            if key.endswith(suffix):
                return True
        return False


    def is_lora(self, state_dict, first_key_only=False):
        is_lora = None
        for key in state_dict:
            _is_lora_key = self.is_lora_key(key)
            if is_lora is None:
                is_lora = _is_lora_key 
            elif is_lora != _is_lora_key:
                raise ValueError(f"Mixed lora keys and full keys")
            if first_key_only:
                break
        return is_lora


    def remove_lora_suffix(self, key):
        for s in self._lora_down_suffixes + self._lora_up_suffixes + self._lora_alpha_suffixes:
            if key.endswith(s):
                return key.removesuffix(s)
        raise ValueError(f"{key} is not a lora key")


    def split_key_and_lora_suffix(self, key):
        for s in self._lora_down_suffixes + self._lora_up_suffixes + self._lora_alpha_suffixes:
            if key.endswith(s):
                return key.removesuffix(s), s
        raise ValueError(f"{key} is not a lora key")


    def get_layer_names_and_shapes_from_lora(self, state_dict_or_state_dict_shapes):
        in_features = {}  # dict : key => in_featues
        out_features = {}  # dict : key => out_features

        for k, t in state_dict_or_state_dict_shapes.items():

            if isinstance(t, torch.Tensor):
                t = list(t.shape)

            layer_key, lora_suffix = self.split_key_and_lora_suffix(k)

            if lora_suffix in self._lora_down_suffixes:
                _rank, _in_features = t
                in_features[layer_key] = _in_features

            if lora_suffix in self._lora_up_suffixes:
                _out_features, _rank = t
                out_features[layer_key] = _out_features

        assert len(out_features) == len(in_features), "Number of up and down keys do not match in the lora"

        final_keys = [f"{k},{in_features[k]},{out_features[k]}" for k in in_features]
        
        return final_keys


    def classify_and_convert_if_possible(self, layer_names_and_shapes):
        # check if lora suffix have been 
        if self.is_lora_key(layer_names_and_shapes[0]):
            raise ValueError("Classification requires layer names and shapes. Use 'get_layer_names_and_shapes_from_loras()'")

        # vote for dictname
        dictname_votes = {}
        for k in keys:
            if k in self._layername_and_shape_to_dictname:
                for d in self._layername_and_shape_to_dictname[k]:
                    if d not in dictname_votes:
                        dictname_votes[d] = 1
                    else:
                        dictname_votes[d] += 1

        new_state_dict = {}
        model_class = None
        for matched_dictname in sorted(dictname_votes, key=dictname_votes.get):
            # find keys 
            matched_keys = [
                k.split(",")[0]
                for k in keys 
                if matched_dictname in self._layername_and_shape_to_dictname[k]
            ]

            # extract state_dict that match
            matched_state_dict = {
                k: state_dict.pop(k) for k in list(state_dict.keys()) 
                if self._remove_lora_suffix(k) in matched_keys 
            }

            _model_class, framework, *_ = matched_dictname.split("_")

            if model_class is None:
                model_class = _model_class
            elif _model_class != model_class:
                raise ValueError(
                    f"Matched lora keys are from different model class: {model_class} != {_model_class}"
                )

            # if diffusers, do nothing
            if framework == "diffusers":
                logger.info("Lora diffusers")
                new_state_dict.update(matched_state_dict)

            elif framework == "kohya":
                logger.info("Kohya conversion")
                new_state_dict.update(
                    convert_sd_scripts_to_ai_toolkit(matched_state_dict)
                )

            else:
                raise ValueError("Unknown framework")

            # temporary: we don't go for mixture of loras in this version
            # but in the future we'll remove this so that the for loop will 
            # continue for the remaining keys matched on other frameworks
            break

        if len(state_dict) > 0:
            raise ValueError(f"{len(state_dict)} remaining keys: {state_dict.keys()}")

        return new_state_dict, model_class


    def inspect(self, state_dict_or_checkpoint_path):
        state_dict_shapes = self.get_state_dict_shapes(state_dict_or_checkpoint_path)
        is_lora = self.is_lora(state_dict_shapes)

        if is_lora:
            layer_names_and_shapes = self.get_layer_names_and_shapes_from_lora(state_dict_shapes)
        else:
            layer_names_and_shapes = self.state_dict_shapes_to_list(state_dict_shapes)

        model_class = None 
        diffusers_state_dict = {}
        # diffusers_state_dict, model_class = self.classify_and_convert_if_possible(layer_names_and_shapes)

        return is_lora, model_class, diffusers_state_dict, layer_names_and_shapes
