import glob
import logging
import torch

from pathlib import Path
from gguf.gguf_reader import GGUFReader
from safetensors import safe_open

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
    def read_state_dict_from_checkpoint(checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.isdir():
            checkpoint_list = []
            checkpoints = glob.glob("**/*.safetensors", root_dir=checkpoint_path, recursive=True)
            for f in checkpoints:
                prefix = str(Path(f).parent)
                if prefix == ".":  # .parent returns "." when the file is at the root of folder
                    prefix = ""
                prefix = prefix.replace("/", ".")  # diffusers prefix layer naming convention
                if prefix:
                    prefix = f"{prefix}."  # adding a point
                
                name = Path(f).name
                checkpoint_list.append((prefix, name))
        else:
            checkpoint_list = [("", checkpoint_path)]


        for prefix, checkpoint_path in checkpoint_list:
            # input is a safetensors or gguf file
            state_dict = {}
            if checkpoint_path.suffix == ".safetensors":
                with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[f"{prefix}{key}"] = f.get_tensor(key)
            elif checkpoint_path.suffix == ".gguf":
                reader = GGUFReader(checkpoint_path)
                for tensor in reader.tensors:
                    state_dict[f"{prefix}{tensor.name}"] = tensor.data
            else: 
                raise ValueError(f"Unknown checkpoint suffix: {checkpoint_path.suffix}")

        return state_dict


    @staticmethod
    def get_state_dict_shapes(state_dict):
        return {
            k: list(v.shape) 
            for k, v in state_dict.items()
        }


    @staticmethod
    def state_dict_shapes_to_list(state_dict_shapes):
        return sorted([
            f"{k},{','.join(map(str, v))}" for k, v in state_dict_shapes.items()
        ])


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


    def get_layer_names_with_shapes_from_lora(self, state_dict_or_state_dict_shapes):
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

        final_keys = sorted([f"{k},{in_features[k]},{out_features[k]}" for k in in_features])
        
        return final_keys


    def classify_keys(self, layer_names_with_shapes, check_shapes=True):
        if not layer_names_with_shapes:
            return {}

        if check_shapes:
            input_keys = layer_names_with_shapes.copy()
            _layername_and_shape_to_dictname = self._layername_and_shape_to_dictname
        else:
            input_keys = [ k.split(",")[0] for k in layer_names_with_shapes]
            _layername_and_shape_to_dictname = { k.split(",")[0]: v for k, v in self._layername_and_shape_to_dictname }

        # check if it's not lora keys
        if self.is_lora_key(input_keys[0]):
            raise ValueError(
                "Classification requires layer names and shapes. Use 'get_layer_names_with_shapes_from_lora()'"
            )

        # vote for dictname
        dictname_votes = {}
        for k in input_keys:
            if k in _layername_and_shape_to_dictname:
                for d in _layername_and_shape_to_dictname[k]:
                    if d not in dictname_votes:
                        dictname_votes[d] = 1
                    else:
                        dictname_votes[d] += 1

        model_classes = {}
        for matched_dictname in sorted(dictname_votes, key=dictname_votes.get):
            # find keys 
            matched_keys = []
            remaining_keys = []
            for k in input_keys:
                if matched_dictname in _layername_and_shape_to_dictname[k]:
                    matched_keys.append(k.split(",")[0])
                else:
                    remaining_keys.append(k)

            model_classes[matched_dictname] = matched_keys
            input_keys = remaining_keys

        if len(input_keys) > 0:
            model_classes["unknown"] = [k.split(",")[0] for k in input_keys]

        return model_classes


    def inspect(self, state_dict_or_checkpoint_path):
        # the inspect method does not raise error
        is_lora = None 
        model_classes = None 
        layer_names_with_shapes = None
        error = None

        try:
            if isinstance(state_dict_or_checkpoint_path, dict):
                # input is a state dict
                state_dict = state_dict_or_checkpoint_path

            elif isinstance(state_dict_or_checkpoint_path, str):
                state_dict = self.read_state_dict_from_checkpoint(state_dict_or_checkpoint_path)
            
            else:
                raise ValueError(
                    f"Unknown type for checkpoint input: {type(state_dict_or_checkpoint_path)}"
                )

            state_dict_shapes = self.get_state_dict_shapes(state_dict)
            is_lora = self.is_lora(state_dict_shapes)

            if is_lora:
                layer_names_with_shapes = self.get_layer_names_with_shapes_from_lora(state_dict_shapes)
            else:
                layer_names_with_shapes = self.state_dict_shapes_to_list(state_dict_shapes)

            model_classes = self.classify_keys(layer_names_with_shapes, check_shapes=is_lora)

            for model_class in list(model_classes.keys()):            
                matched_keys = model_classes[model_class]

                # extract state_dict that match
                matched_state_dict = {
                    k: state_dict.pop(k) for k in list(state_dict.keys()) 
                    if self.remove_lora_suffix(k) in matched_keys 
                }
                assert len(matched_state_dict) > 0
                model_classes[model_class] = matched_state_dict

            return is_lora, model_classes, layer_names_with_shapes, error

        except Exception as e:
            error = str(e)
            return is_lora, model_classes, layer_names_with_shapes, error
