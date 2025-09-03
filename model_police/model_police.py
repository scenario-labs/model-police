import glob
import logging
import re
import torch
import traceback

from pathlib import Path
from gguf.gguf_reader import GGUFReader
from safetensors import safe_open

logger = logging.getLogger(__name__)


here = Path(__file__).parent


class ModelPolice:
    _model_dictionaries: dict[str, list[str]] = {} 
    _layername_and_shape_to_dictname: dict[str, str] = {}
    _lora_down_suffixes: list[str] = []
    _lora_up_suffixes: list[str] = []
    _lora_ignore_suffixes: list[str] = []
    _full_ignore_suffixes: list[str] = []


    def __init__(self):
        
        logging.info("Reading model validator dictionaries")
        with open(here / "lora_up_suffixes.txt") as f:
            self._lora_up_suffixes = [s.strip() for s in f.readlines()]
        with open(here / "lora_down_suffixes.txt") as f:
            self._lora_down_suffixes = [s.strip() for s in f.readlines()]
        with open(here / "lora_ignore_suffixes.txt") as f:
            self._lora_ignore_suffixes = [s.strip() for s in f.readlines()]
        with open(here / "full_ignore_suffixes.txt") as f:
            self._full_ignore_suffixes = [s.strip() for s in f.readlines()]        

        for dict_filepath in (here / "model_dictionaries").iterdir():
            dict_name = dict_filepath.name.removesuffix(".csv")
            self._model_dictionaries[dict_name] = []

            with open(dict_filepath, "r") as f:                
                for line in f:
                    line = line.strip()
                    self._model_dictionaries[dict_name].append(line)
                    if line not in self._layername_and_shape_to_dictname:
                        self._layername_and_shape_to_dictname[line] = []
                    self._layername_and_shape_to_dictname[line].append(dict_name)


    @staticmethod
    def read_state_dict_from_checkpoint(checkpoint_path, prefix=""):
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise ValueError(f"File path {checkpoint_path} does not exist")

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
    def get_checkpoint_list_with_prefixes(checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)

        if not checkpoint_dir.exists():
            raise ValueError(f"Path {checkpoint_dir} does not exist")

        if not checkpoint_dir.is_dir():
            raise ValueError(f"Path {checkpoint_dir} is not a folder")

        checkpoint_dir_with_prefixes = []
        checkpoints = (
            glob.glob("**/*.safetensors", root_dir=checkpoint_dir, recursive=True) +
            glob.glob("**/*.gguf", root_dir=checkpoint_dir, recursive=True)
        )
        for f in checkpoints:
            prefix = str(Path(f).parent)
            if prefix == ".":  # .parent returns "." when the file is at the root of folder
                prefix = ""
            prefix = prefix.replace("/", ".")  # diffusers prefix layer naming convention
            if prefix:
                prefix = f"{prefix}."  # adding a point
            
            checkpoint_dir_with_prefixes.append((prefix, checkpoint_dir / f))

        return checkpoint_dir_with_prefixes


    @staticmethod
    def get_state_dict_shapes(state_dict):
        return {
            k: list(v.shape) 
            for k, v in state_dict.items()
        }


    def state_dict_shapes_to_list(self, state_dict_shapes):
        return sorted([
            f"{k},{','.join(map(str, v))}" for k, v in state_dict_shapes.items()
        ])


    def is_lora(self, state_dict, first_key_only=False):
        is_lora = None
        for key in state_dict:

            _next = False
            for suffix in self._lora_ignore_suffixes:
                if key.endswith(suffix):
                    _next = True 
            if _next: continue

            _is_lora_key = False
            for suffix in self._lora_down_suffixes + self._lora_up_suffixes:
                if key.endswith(suffix):
                    _is_lora_key = True
                    break

            if is_lora is None:
                is_lora = _is_lora_key 
            elif is_lora != _is_lora_key:
                raise ValueError(f"Mixed lora keys and full keys")
            if first_key_only:
                break
        return is_lora


    def remove_lora_suffix(self, key):
        for s in self._lora_down_suffixes + self._lora_up_suffixes + self._lora_ignore_suffixes:
            if key.endswith(s):
                return key.removesuffix(s)
        return key


    def split_key_and_lora_suffix(self, key):
        for s in self._lora_down_suffixes + self._lora_up_suffixes + self._lora_ignore_suffixes:
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


    @staticmethod
    def replace_key_numbers_with_zero(key):
        """ this method is useful when the number of layer or blocks might vary"""
        key = key.split(",")
        key[0] = re.sub('\.\d+\.', '.0.', key[0])
        return ','.join(key)


    def classify_keys(self, layer_names_with_shapes, is_lora=True):
        if not layer_names_with_shapes:
            return {}

        if is_lora:
            input_keys = layer_names_with_shapes.copy()
            _layername_and_shape_to_dictname = self._layername_and_shape_to_dictname
        else:
            input_keys = [ k.split(",")[0] for k in layer_names_with_shapes ]  # removing shape
            
            _layername_and_shape_to_dictname = {}
            for k, v in self._layername_and_shape_to_dictname.items(): 
                k = k.split(",")[0]
                if k in _layername_and_shape_to_dictname: # avoid overrides
                    _layername_and_shape_to_dictname[k].extend(v)
                else:
                    _layername_and_shape_to_dictname[k] = v.copy()

        # vote for dictname
        dictname_votes = {}
        for k in input_keys:
            if k in _layername_and_shape_to_dictname:
                for d in _layername_and_shape_to_dictname[k]:
                    if d not in dictname_votes:
                        dictname_votes[d] = 1
                    else:
                        dictname_votes[d] += 1

        # add the recall
        for d in list(dictname_votes.keys()):
            dictname_votes[d] = (dictname_votes[d], dictname_votes[d] / self._dictname_num_keys[d])

        model_classes = {}
        for matched_dictname, (num_matched_keys, model_recall) in sorted(dictname_votes.items(), key=lambda x:x[1], reverse=True):
            # find keys 
            matched_keys = []
            remaining_keys = []
            for k in input_keys:
                if k in _layername_and_shape_to_dictname and matched_dictname in _layername_and_shape_to_dictname[k]:
                    matched_keys.append(k.split(",")[0])
                else:
                    remaining_keys.append(k)

            if len(matched_keys):
                model_classes[matched_dictname] = matched_keys

            input_keys = remaining_keys
            if not len(remaining_keys):
                break

        if len(input_keys) > 0:
            model_classes["unknown"] = [k.split(",")[0] for k in input_keys]

        return model_classes


    @staticmethod
    def is_fully_covered(dict_keys, all_keys):
        for key in dict_keys:
            if key not in all_keys:
                teturn False
        return True


    def inspect(self, state_dict_or_checkpoint_path):
        # the inspect method does not raise error
        full_models = None 
        checkpoint_list = None
        error = None

        try:
            if isinstance(state_dict_or_checkpoint_path, dict):  # input is a state dict
                checkpoint_list = [{
                    "prefix": "",
                    "files": [],
                    "state_dict": state_dict_or_checkpoint_path,
                }]

            elif isinstance(state_dict_or_checkpoint_path, str) or isinstance(state_dict_or_checkpoint_path, Path):                

                if Path(state_dict_or_checkpoint_path).is_dir():
                    checkpoint_list_with_prefixes = self.get_checkpoint_list_with_prefixes(state_dict_or_checkpoint_path)

                else:
                    checkpoint_list_with_prefixes = [("", checkpoint_or_folder_path)]

                # read state dict and merge in the form "-0000X-of-0000Y"
                checkpoint_list = []
                for prefix, checkpoint_file in checkpoint_list_with_prefixes:
                    key = re.sub('-\d{5}-of-\d{5}', '', str(checkpoint_file)) # remove "-0000X-of-0000Y" in order to merge
                    state_dict = self.read_state_dict_from_checkpoint(checkpoint_file)
                    if key in checkpoint_list: 
                        checkpoint_list[key]["files"].append(checkpoint_file)
                        checkpoint_list[key]["state_dict"] |= state_dict
                    else:
                        checkpoint_list[key] = {
                            "prefix": prefix,
                            "files": [checkpoint_file],
                            "state_dict": state_dict,
                        }

                    checkpoint_list = checkpoint_list.values() # we don't need keys after deduplication
             
            else:
                raise ValueError(
                    f"Unknown type for checkpoint input: {type(state_dict_or_checkpoint_path)}"
                )

            # lora detection
            for checkpoint in checkpoint_list:
                state_dict = checkpoint["state_dict"]

                state_dict_shapes = self.get_state_dict_shapes(state_dict)
                is_lora = self.is_lora(state_dict_shapes)

                # layer_names_with_shapes creation
                if is_lora:
                    layer_names_with_shapes = self.get_layer_names_with_shapes_from_lora(state_dict_shapes)
                else:
                    layer_names_with_shapes = self.state_dict_shapes_to_list(state_dict_shapes)

                checkpoint.update({
                    "is_lora": is_lora,
                    "layer_names_with_shapes": layer_names_with_shapes,
                    "model_parts": []
                })

                if is_lora:
                    # lora classification
                # model_classes = self.classify_keys(layer_names_with_shapes, is_lora=is_lora)
                

                # for model_class in list(model_classes.keys()):     

                #     matched_keys = model_classes[model_class]

                #     # extract state_dict that match
                #     matched_state_dict = {
                #         k: state_dict.pop(k) for k in list(state_dict.keys()) 
                #         if self.remove_lora_suffix(k) in matched_keys
                #     }
                #     assert len(matched_state_dict) > 0
                #     model_classes[model_class] = matched_state_dict

                    pass

                else:
                    # find model parts
                    for dict_name, dict_keys in self._model_dictionaries.items():
                        if self.is_fully_covered(dict_keys, layer_names_with_shapes):
                            checkpoint["model_parts"].append(dict_name)

            # full and part model detection:
            # we consider there is a full or part model if all of his keys are present
            all_keys = set()
            for checkpoint in checkpoint_list:
                all_keys.update(checkpoint["layer_names_with_shapes"])
            
            full_models = []
            for dict_name, dict_keys in self._model_dictionaries.items():
                if "full" in dict_name and self.is_fully_covered(dict_keys, all_keys):
                    full_models.append(dict_name)

            return full_models, checkpoint_list, error

        except Exception as e:
            error = str(e)
            print(traceback.format_exc())
            return full_models, checkpoint_list, error

    @staticmethod
    def get_diffusers_components(layer_names_with_or_without_shapes):
        if not layer_names_with_or_without_shapes:
            return []

        first_key = layer_names_with_or_without_shapes[0]
        if "." not in first_key:
            raise ValueError("Requires diffusers model format")

        prefixes = [k.split(".")[0] for k in layer_names_with_or_without_shapes]
        
        return sorted(list(set(prefixes)))
    