"""
This module contains a parsing interface for the tree bark classification dataset of Carpentier et al.
('Tree Species Identification from Bark Images Using Convolutional Neural Networks', arXiv: 1803.00949)

Refer to https://github.com/ulaval-damas/tree-bark-classification for download links and information.
"""

import copy
import glob
import logging
import os

import PIL.Image

import thelper.data

logger = logging.getLogger(__name__)


class TreeBarkDataset(thelper.data.ClassificationDataset):
    """Parser interface for loading tree bark classification images.

    This interface derives from :class:`thelper.data.ClassificationDataset`, and must thus have a
    compatible constructor (name/root/config/transforms). It should also provide a ``__getitem__``
    member function.

    Attributes:
        samples: list of sample dictionaries that will be returned one by one via __getitem__(...).
    """

    def __init__(self, name, root, config, transforms=None):
        """Parses the dataset configuration and initializes the sample list.

        Args:
            name: unique name of the dataset used to refer to this particular instance. Should be
                forwarded to the super class initializer.
            root: path to the dataset root folder (may be ``None``). If it exists and if that folder
                contains a 'tree-bark' sub-folder, it is used as the dataset root.
            config: dictionary containing all configuration parameters for this dataset. The parameters
                themselves may contain strings, scalars, dicts, or lists. See the code for more info.
            transforms: optional sample transformation operations to apply to samples when loading them.
        """
        thelper.data.logger.debug("pre-initializing barknet dataset")
        target_class = config["target_class"] if "target_class" in config else None
        if target_class:
            if not isinstance(target_class, (str, list)):
                raise AssertionError("unexpected target class type")
            if isinstance(target_class, str):
                target_class = target_class.split(",")
            elif not isinstance(target_class[0], str):
                raise AssertionError("unexpected target class type")
        ignored_class = config["ignored_class"] if "ignored_class" in config else None
        if ignored_class:
            if not isinstance(ignored_class, (str, list)):
                raise AssertionError("unexpected ignored class type")
            if isinstance(ignored_class, str):
                ignored_class = ignored_class.split(",")
            elif not isinstance(ignored_class[0], str):
                raise AssertionError("unexpected ignored class type")
        if root is not None and os.path.exists(root) and os.path.exists(os.path.join(root, "tree-bark")):
            image_folder_root = os.path.join(root, "tree-bark")
            if "root" in config and config["root"]:
                logger.warning("overriding dataset 'root' field value with '%s'" % image_folder_root)
        else:
            if "root" not in config or not config["root"]:
                raise AssertionError("missing 'root' field in barknet dataset config")
            image_folder_root = config["root"]
            if not os.path.exists(image_folder_root):
                raise AssertionError("invalid root folder '%s'" % image_folder_root)
        image_folder_names = [folder_name for folder_name in os.listdir(image_folder_root)
                              if os.path.isdir(os.path.join(image_folder_root, folder_name))]
        if not image_folder_names:
            raise AssertionError("could not find any tree species folder in '%s'" % image_folder_root)
        if target_class:
            for target_name in target_class:
                if target_name not in image_folder_names:
                    raise AssertionError("missing target '%s' in list of all known classes" % target_name)
        if ignored_class:
            for ignored_name in ignored_class:
                if ignored_name not in image_folder_names:
                    raise AssertionError("missing ignored '%s' in list of all known classes" % ignored_name)
        raw_sample_map = {}
        for image_folder_name in image_folder_names:
            if (target_class and image_folder_name not in target_class) or \
               (ignored_class and image_folder_name in ignored_class):
                image_list = []
            else:
                image_list = glob.glob(os.path.join(image_folder_root, image_folder_name, "*.jpg"))
            raw_sample_map[image_folder_name] = image_list
        self.image_key = thelper.utils.get_key_def("image_key", config, "image")
        self.label_key = thelper.utils.get_key_def("label_key", config, "label")
        self.path_key = thelper.utils.get_key_def("path_key", config, "path")
        self.idx_key = thelper.utils.get_key_def("idx_key", config, "idx")
        super(TreeBarkDataset, self).__init__(name, image_folder_names, self.image_key, self.label_key,
                                              meta_keys=[self.path_key, self.idx_key], config=config,
                                              transforms=transforms, bypass_deepcopy=True)
        thelper.data.logger.debug("initializing barknet dataset with %d classes" % len(image_folder_names))
        self.samples = []
        for class_name, samples in raw_sample_map.items():
            for sample_idx, sample_path in enumerate(samples):
                self.samples.append({
                    self.image_key: None,  # loaded when needed only
                    self.label_key: class_name,
                    self.path_key: sample_path,
                    self.idx_key: sample_idx
                })
        thelper.data.logger.debug("barknet dataset: parsed %d samples" % len(self.samples))

    def __getitem__(self, idx):
        """Returns the `idx`-th sample loaded by the dataset as a dictionary.

        Args:
            idx: the (0-based) index of the sample to load.

        Returns:
            A dictionary containing all of the sample's loaded data.
        """
        if idx < 0 or idx > len(self.samples):
            raise AssertionError("sample index (%d) is out-of-range" % idx)
        sample = copy.copy(self.samples[idx])  # we keep the unloaded version in the list
        sample[self.image_key] = PIL.Image.open(sample[self.path_key])
        sample[self.image_key].load()
        if self.transforms:
            sample[self.image_key] = self.transforms(sample[self.image_key])
        return sample
