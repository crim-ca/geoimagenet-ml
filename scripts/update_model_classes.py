#!/usr/bin/env python
"""
Utility script that updates the class mapping of the model's task.
This model can then be employed with process :class:`geoimagenet_ml.processes.runners.ProcessRunnerModelTester`
for inference of dataset patches using corresponding taxonomy class indices.
"""
import argparse
import os
from typing import TYPE_CHECKING

import thelper  # noqa
import torch    # noqa

if TYPE_CHECKING:
    from geoimagenet_ml.typedefs import List, Optional, Tuple, Union


def update_model_class_mapping(class_mappings, model_path, model_output=None):
    # type: (List[Tuple[Union[str,int], Union[str,int]]], str, Optional[str]) -> None
    """Updates the model task using provided class mapping."""
    model = thelper.utils.load_checkpoint(model_path)
    model_task = model.get("config", {}).get("task", model["task"])  # backward compatibility
    if not isinstance(model_task, dict):
        model_task = thelper.tasks.utils.create_task(model_task)
    if len(model_task.class_names) != len(class_mappings):
        raise ValueError(f"Task classes and class mapping size do not match "
                         f"({len(model_task.class_names)} != {len(class_mappings)}) :\n"
                         f"  {model_task.class_names}\n  {class_mappings} ")
    class_mapped = {}
    class_mappings = dict(class_mappings)
    for class_name in model_task.class_names:
        if class_name not in class_mappings:
            raise ValueError(f"Missing mapping for class '{class_name}'.")
        new_class = class_mappings[class_name]
        idx_class = model_task.class_indices[class_name]
        class_mapped[new_class] = idx_class
    setattr(model_task, "_class_indices", class_mapped)
    model_outputs_sorted_by_index = list(sorted(class_mapped.items(), key=lambda _map: _map[1]))
    setattr(model_task, "_class_names", [_map[0] for _map in model_outputs_sorted_by_index])
    model["task"] = model_task
    model["config"]["task"] = model_task
    model_params = model.pop("model_params", None)
    # backward compatibility of model parameters that can contain num_classes info
    if model_params and "params" not in model["config"]:
        model["config"]["params"] = model_params
    if "params" in model["config"] and "num_classes" in model["config"]["params"]:
        model["config"]["params"]["num_classes"] = len(model_task.class_names)
    if not model_output:
        model_file, model_ext = os.path.splitext(model_path)
        model_output = model_file + ".updated" + model_ext
    torch.save(model, model_output)


def ClassMapping(s):
    try:
        original_name, new_class = s.split(",")
        if new_class != "dontcare":
            new_class = int(new_class)
        return str(original_name), new_class
    except Exception:
        raise argparse.ArgumentTypeError("Class mapping must be sets of (<old-class-name>, <taxonomy-class>).")


def main():
    ap = argparse.ArgumentParser(prog=__file__, description=__doc__, add_help=True, # noqa
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("model_path", help="Path to the model to employ.")
    ap.add_argument("class_mappings", nargs="+", type=ClassMapping,
                    help="""
One or more mapping set of old class name to new class index from taxonomy. Spaces separate
different sets while commas separate the from/to items of that mapping set.
(e.g.: car,109 plane,120). 

First item of each set must be a string or integer that matches exactly an existing class name
from the model's task. The second item must be an integer that will be mapped to some taxonomy 
class index. Alternatively, second item can be 'dontcare' which will mark it as to be ignored
in the updated model task during inference. 

Note: if your class name has spaces, use quotes around the set (e.g.: 'Small plane,120').""")
    ap.add_argument("-o", "--output", dest="model_output",
                    help="Path to save the updated model checkpoint (default: <input-checkpoint-path>.updated.pth).""")
    args = ap.parse_args()
    update_model_class_mapping(args.class_mappings, args.model_path, args.model_output)


if __name__ == "__main__":
    main()
