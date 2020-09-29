from geoimagenet_ml.utils import get_sane_name, fully_qualified_name, isclass
from geoimagenet_ml.ml.utils import parse_rasters, parse_geojson, parse_coordinate_system, process_feature_crop
from google_drive_downloader import GoogleDriveDownloader as gdd
from six.moves.urllib.parse import urlparse
from six.moves.urllib.request import urlopen
from tempfile import NamedTemporaryFile
from copy import deepcopy
from io import BytesIO
from osgeo import gdal
from PIL import Image
import requests
import logging
import json
import numpy as np
import random
import shutil
import six
import ssl
import ast
import re
import os
import cv2
from typing import TYPE_CHECKING

import thelper  # noqa

if TYPE_CHECKING:
    from geoimagenet_ml.store.datatypes import Job, Model, Dataset  # noqa: F401
    from geoimagenet_ml.store.interfaces import DatasetStore  # noqa: F401
    from geoimagenet_ml.typedefs import (  # noqa: F401
        Any, AnyStr, Callable, Dict, List, Tuple, Union, JSON, SettingsType, Number,
        Optional, FeatureType, RasterDataType, ParamsType
    )
    from geoimagenet_ml.processes.runners import ProcessRunnerModelTester
    from geoimagenet_ml.utils import ClassCounter  # noqa: F401
    # FIXME: add other task definitions as needed + see MODEL_TASK_MAPPING
    AnyTask = Union[thelper.tasks.classif.Classification]  # noqa: F401
    CkptData = thelper.typedefs.CheckpointContentType  # noqa: F401
    ClassMap = Dict[int, Optional[Union[int, AnyStr]]]  # noqa: F401

# enforce GDAL exceptions (otherwise functions return None)
gdal.UseExceptions()

LOGGER = logging.getLogger(__name__)

# keys used across methods to find matching configs, must be unique and non-conflicting with other sample keys
IMAGE_DATA_KEY = "data"     # key used to store temporarily the loaded image data
IMAGE_LABEL_KEY = "label"   # key used to store the class label used by the model
TEST_DATASET_KEY = "dataset"

DATASET_FILES_KEY = "files"             # list of all files in the dataset batch
DATASET_DATA_KEY = "data"               # dict of data below
DATASET_DATA_TAXO_KEY = "taxonomy"
DATASET_DATA_MAPPING_KEY = "taxonomy_model_map"     # taxonomy ID -> model labels
DATASET_DATA_ORDERING_KEY = "model_class_order"     # model output classes (same indices)
DATASET_DATA_MODEL_MAPPING = "model_output_mapping"    # model output classes (same indices)
DATASET_DATA_PATCH_KEY = "patches"
DATASET_DATA_PATCH_CLASS_KEY = "class"       # class id associated to the patch
DATASET_DATA_PATCH_SPLIT_KEY = "split"       # group train/test of the patch
DATASET_DATA_PATCH_CROPS_KEY = "crops"       # extra data such as coordinates
DATASET_DATA_PATCH_IMAGE_KEY = "image"       # original image path that was used to generate the patch
DATASET_DATA_PATCH_PATH_KEY = "path"         # crop image path of the generated patch
DATASET_DATA_PATCH_MASK_KEY = "mask"         # mask image path of the generated patch
DATASET_DATA_PATCH_INDEX_KEY = "index"       # data loader getter index reference
DATASET_DATA_PATCH_FEATURE_KEY = "feature"   # annotation reference id
DATASET_BACKGROUND_ID = 999                  # background class id
DATASET_DATA_PATCH_DONTCARE = 255            # dontcare value in the test set
DATASET_DATA_CHANNELS = "channels"           # channels information
DATASET_CROP_MODES = {
    -1: -1,
    "reduce": -1,
    0: 0,
    "raw": 0,
    1: 1,
    "extend": 1,
}
DATASET_CROP_MODE_NAMES = {v: k for k, v in DATASET_CROP_MODES.items() if isinstance(k, str)}

# see bottom for mapping definition
MAPPING_TASK = "task"
MAPPING_LOADER = "loader"
MAPPING_RESULT = "result"
MAPPING_TESTER = "tester"


class ConfigurationError(Exception):
    """Error related to ``thelper`` configuration."""


class ConfigurationWarning(Warning):
    """Warning related to ``thelper`` configuration."""


class ConfigurationSecurityWarning(ConfigurationWarning):
    """Warning related to ``thelper`` configuration specifically for security issues."""


def load_model(model_file):
    # type: (Union[Any, AnyStr]) -> Tuple[bool, CkptData, Optional[BytesIO], Optional[Exception]]
    """
    Tries to load a model checkpoint file from the file-like object, file path or URL.

    :return: tuple of (success, data, buffer, exception) accordingly.
    :raises: None (nothrow)
    """
    try:
        model_buffer = model_file
        if isinstance(model_file, six.string_types):
            url_spec = urlparse(model_file)
            if url_spec.scheme:
                if "drive.google" in model_file:
                    with NamedTemporaryFile() as tmp_model:
                        file_id = url_spec.path
                        if any(file_id.endswith(path) for path in ["/view", "/edit"]):
                            file_id = file_id.split("/")[-2]
                        else:
                            file_id = file_id.split("/")[-1]
                        model_file = tmp_model.name
                        gdd.download_file_from_google_drive(file_id=file_id, dest_path=model_file, overwrite=True)
                        with open(model_file, 'rb') as f:
                            model_buffer = BytesIO(f.read())
                else:
                    no_ssl = ssl.create_default_context()
                    no_ssl.check_hostname = False
                    no_ssl.verify_mode = ssl.CERT_NONE
                    url_buffer = urlopen(model_file, context=no_ssl)
                    model_buffer = BytesIO(url_buffer.read())
            else:
                with open(model_file, 'rb') as f:
                    model_buffer = BytesIO(f.read())
        thelper.utils.bypass_queries = True     # avoid blocking ui query
        model_checkpoint_info = thelper.utils.load_checkpoint(model_buffer)
    except Exception as ex:
        return False, {}, None, ex
    if model_checkpoint_info:
        return True, model_checkpoint_info, model_buffer, None
    return False, {}, None, None


def validate_model(model_data):
    # type: (CkptData) -> Tuple[bool, Optional[Exception]]
    """
    Accomplishes required model checkpoint validation to restrict unexpected behaviour during other function calls.

    All security checks or alternative behaviours allowed by native :mod:`thelper` library but that should be forbidden
    within this API for process execution should be done here.

    :param model_data: model checkpoint data with configuration parameters (typically loaded by :func:`load_model`)
    :return: tuple of (success, exception) accordingly
    :raises: None (nothrow)
    """
    model_task = model_data.get("task")
    if isinstance(model_task, dict):
        model_type = model_task.get("type")
        if not isinstance(model_type, six.string_types):
            LOGGER.debug(f"Model task: [{model_type!s}]")
            return False, ConfigurationError(
                f"Forbidden model checkpoint task defines unknown operation: [{model_type!s}]"
            )
        model_params = model_task.get("params")
        if not isinstance(model_params, dict):
            LOGGER.debug(f"Model task: [{model_params!s}]")
            return False, ConfigurationError(
                "Forbidden model checkpoint task missing JSON definition of parameter section."
            )
        model_classes = model_params.get("class_names")
        if not (isinstance(model_classes, list) and all([isinstance(c, (int, str)) for c in model_classes])):
            LOGGER.debug(f"Model task: [{model_classes!s}]")
            return False, ConfigurationError(
                "Forbidden model checkpoint task contains invalid JSON class names parameter section."
            )
    elif isinstance(model_task, thelper.tasks.Task):
        model_type = fully_qualified_name(model_task)
        if model_type not in MODEL_TASK_MAPPING:
            LOGGER.debug(f"Model task: [{model_type!s}]")
            return False, ConfigurationError(
                f"Forbidden model checkpoint task defines unknown operation: [{model_type!s}]"
            )
    else:
        # thelper security risk, refuse literal string definition of task loaded by eval() unless it can be validated
        LOGGER.warning(f"Model task not defined as dictionary nor `thelper.task.Task` class: [{model_task!s}]")
        if not (isinstance(model_task, str) and model_task.startswith("thelper.task")):
            return False, ConfigurationSecurityWarning(
                "Forbidden model checkpoint task definition as string doesn't refer to a `thelper.task`."
            )
        model_task_cls = model_task.split("(")[0]
        LOGGER.debug(f"Verifying model task as string: {model_task_cls!s}")
        model_task_cls = thelper.utils.import_class(model_task_cls)
        if not (isclass(model_task_cls) and issubclass(model_task_cls, thelper.tasks.Task)):
            return False, ConfigurationSecurityWarning(
                "Forbidden model checkpoint task definition as string is not a known `thelper.task`."
            )
        if model_task.count("(") != 1 or model_task.count(")") != 1:
            return False, ConfigurationSecurityWarning(
                "Forbidden model checkpoint task definition as string has unexpected syntax."
            )
        LOGGER.warning("Model task defined as string allowed after basic validation.")
        try:
            fix_str_model_task(model_task)  # attempt update but don't actually apply it
        except ValueError:
            return False, ConfigurationError(
                "Forbidden model checkpoint task defined as string doesn't respect expected syntax."
            )
        LOGGER.debug("Model task as string validated with successful parameter conversion")
    return True, None


def fix_str_model_task(model_task):
    # type: (str) -> ParamsType
    """
    Attempts to convert the input model task definition as literal string to the equivalent dictionary of task
    input parameters.

    For example, a model with classification task is expected to have the following format::

        "thelper.tasks.classif.Classification(class_names=['cls1', 'cls2'], input_key='0', label_key='1', meta_keys=[])"

    And will be converted to::

        {'class_names': ['cls1', 'cls2'], 'input_key': '0', 'label_key': '1', 'meta_keys': []}

    :return: dictionary of task input parameters converted from the literal string definition
    :raises ValueError: if the literal string cannot be parsed as a task input parameters definition
    """
    try:
        if not isinstance(model_task, str):
            raise ValueError(f"Invalid input is not a literal string for model task parsing, got '{type(model_task)}'")
        if not model_task.startswith("thelper.tasks."):
            raise ValueError(f"Unknown task 'type' in task config not defined by thelper: '{model_task!s}'")
        params = model_task.split("(", 1)[-1].split(")", 1)[0]
        params = re.sub(r"(\w+)\s*=", r"'\1': ", params)  # replace <key>=<val> by <key>: <val> for dict-like format
        return ast.literal_eval(f"{{{params}}}")
    except ValueError:
        raise   # failing ast converting raises ValueError
    except Exception as exc:
        raise ValueError(f"Failed literal string parsing for model task, exception: [{exc!s}]")


class CallbackIterator(thelper.session.base.SessionRunner):
    def __init__(self, runner, loader, start_percent, final_percent):
        # type: (ProcessRunnerModelTester, thelper.data.loader.DataLoader, Number, Number) -> None
        self._runner = runner
        self._loader = loader
        self._start_percent = float(start_percent)
        self._final_percent = float(final_percent)

    def _iter_logger_callback(self,  # see `thelper.typedefs.IterCallbackParams` for more info
                              task,  # type: thelper.tasks.utils.Task
                              input,  # type: thelper.typedefs.InputType
                              pred,  # type: thelper.typedefs.AnyPredictionType
                              target,  # type: thelper.typedefs.AnyTargetType
                              sample,  # type: thelper.typedefs.SampleType
                              loss,  # type: Optional[float]
                              iter_idx,  # type: int
                              max_iters,  # type: int
                              epoch_idx,  # type: int
                              max_epochs,  # type: int
                              output_path,  # type: AnyStr
                              # note: kwargs must contain two args here: 'set_name' and 'writers'
                              **kwargs,  # type: Any
                              ):  # type: (...) -> None
        """
        Callback called on each batch iteration of the evaluation process.
        Must have same signature as original.

        Update job log using available progress metadata for progressive feedback of job execution.
        """

        total_sample_count = self._loader.sample_count
        batch_size = self._loader.batch_size
        batch_index = iter_idx + 1
        eval_sample_count = min(batch_size * batch_index, total_sample_count)  # if last batch is partial
        progress = np.interp(float(batch_index) / float(max_iters), [0, 1], [self._start_percent, self._final_percent])
        msg = "evaluating... [samples: {}/{}, batches: {}/{}]" \
            .format(eval_sample_count, total_sample_count, batch_index, max_iters)
        self._runner.update_job_status(self._runner.job.status, msg, progress)
        self._runner.db.jobs_store.update_job(self._runner.job)


def get_test_data_runner(process_runner, model_checkpoint_config, model, dataset, start_percent, final_percent):
    # type: (ProcessRunnerModelTester, CkptData, Model, Dataset, Number, Number) -> thelper.train.Trainer
    """
    Obtains a trainer specialized for testing data predictions using the provided model checkpoint and dataset loader.
    """
    settings = process_runner.registry.settings
    test_config = test_loader_from_configs(model_checkpoint_config, model, dataset, settings)
    config = test_config["config"]

    job_uuid = process_runner.job.uuid
    jobs_path = settings.get("geoimagenet_ml.ml.jobs_path")
    save_dir = os.path.join(jobs_path, job_uuid)
    _, _, _, test_loader = thelper.data.utils.create_loaders(config, save_dir=save_dir)
    model = thelper.nn.create_model(config, None, save_dir=save_dir, ckptdata=test_config)
    task = model.task
    loaders = None, None, test_loader   # type: thelper.typedefs.MultiLoaderType

    # link the batch iteration with a callback for progress tracking
    loader_batch_size = config["loaders"]["test_batch_size"]
    logger_callback = CallbackIterator(process_runner, test_loader, start_percent, final_percent)
    config["trainer"]["test_logger"] = logger_callback._iter_logger_callback  # noqa

    # session name as Job UUID will write data under '<geoimagenet_ml.ml.models_path>/<model-UUID>/output/<job-UUID>/'
    trainer = thelper.train.create_trainer(job_uuid, save_dir, config, model, task, loaders, model_checkpoint_config)
    return trainer



class ImageFolderSegDataset(thelper.data.SegmentationDataset):
    """Image folder dataset specialization interface for segmentation tasks.

    This specialization is used to parse simple image subfolders, and it essentially replaces the very
    basic ``torchvision.datasets.ImageFolder`` interface with similar functionalities. It it used to provide
    a proper task interface as well as path metadata in each loaded packet for metrics/logging output.

    .. seealso::
        | :class:`thelper.data.parsers.ImageDataset`
        | :class:`thelper.data.parsers.SegmentationDataset`
    """

    def __init__(self, root, transforms=None, channels= None,
                 image_key="image", label_key="label", mask_key="mask", path_key="path", idx_key="idx"):
        """Image folder dataset parser constructor."""
        self.root = root
        if self.root is None or not os.path.isdir(self.root):
            raise AssertionError("invalid input data root '%s'" % self.root)
        class_map = {}
        for child in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, child)):
                class_map[child] = []
        if not class_map:
            raise AssertionError("could not find any image folders at '%s'" % self.root)
        image_exts = [".jpg", ".jpeg", ".bmp", ".png", ".ppm", ".pgm", ".tif"]
        self.image_key = image_key
        self.path_key = path_key
        self.idx_key = idx_key
        self.label_key = label_key
        self.mask_key = mask_key
        self.channels = channels if channels else [1, 2, 3]
        samples = []
        for class_name in class_map:
            class_folder = os.path.join(self.root, class_name)
            for folder, subfolder, files in os.walk(class_folder):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in image_exts:
                        class_map[class_name].append(len(samples))
                        samples.append({
                            self.path_key: os.path.join(folder, file),
                            self.label_key: class_name
                        })
        old_unsorted_class_names = list(class_map.keys())
        class_map = {k: class_map[k] for k in sorted(class_map.keys()) if len(class_map[k]) > 0}
        if old_unsorted_class_names != list(class_map.keys()):
            # new as of v0.4.4; this may only be an issue for old models trained on windows and ported to linux
            # (this is caused by the way os.walk returns folders in an arbitrary order on some platforms)
            logger.warning("class name ordering changed due to folder name sorting; this may impact the "
                           "behavior of previously-trained models as task class indices may be swapped!")
        if not class_map:
            raise AssertionError("could not locate any subdir in '%s' with images to load" % self.root)
        meta_keys = [self.path_key, self.idx_key]
        super(ImageFolderSegDataset, self).__init__(class_names=list(class_map.keys()), input_key=self.image_key,
                                                 label_key=self.label_key, meta_keys=meta_keys, transforms=transforms)
        self.samples = samples

    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        if isinstance(idx, slice):
            return self._getitems(idx)
        if idx >= len(self.samples):
            raise AssertionError("sample index is out-of-range")
        if idx < 0:
            idx = len(self.samples) + idx
        sample = self.samples[idx]
        image_path = sample[self.path_key]
        rasterfile = gdal.Open(image_path, gdal.GA_ReadOnly)
        # image = cv2.imread(image_path)
        image = []
        for raster_band_idx in self.channels:
            curr_band = rasterfile.GetRasterBand(raster_band_idx)  # offset, starts at 1
            band_array = curr_band.ReadAsArray()
            band_nodataval = curr_band.GetNoDataValue()
            # band_ma = np.ma.array(band_array.astype(np.float32))
            image.append(band_array)
        image = np.dstack(image)
        rasterfile = None  # close input fd
        mask_path = getattr(sample, self.mask_key, None)
        mask = None
        if mask_path is not None:
            mask = cv2.imread(mask_path)
            mask = mask if mask.ndim == 2 else mask[:, :, 0]  # masks saved with PIL have three bands
        if image is None:
            raise AssertionError("invalid image at '%s'" % image_path)
        sample = {
            self.image_key: np.array(image.data, copy=True, dtype='float32'),
            self.mask_key: mask,
            self.label_key: sample[self.label_key],
            self.idx_key: idx,
            # **sample
        }
        # FIXME: not clear how to handle transformations on the image as well as on the mask
        #  in particular for geometric transformations
        if self.transforms:
            sample = self.transforms(sample)
        return sample


class BatchTestPatchesBaseDatasetLoader(ImageFolderSegDataset):
    """
    Batch dataset parser that loads only patches from 'test' split and matching
    class IDs (or their parents) known by the model as defined in its ``task``.

    .. note::

        Uses :class:`thelper.data.ImageFolderDataset` ``__getitem__`` implementation to load image
        from a folder, but overrides the ``__init__`` to adapt the configuration to batch format.
    """

    # noinspection PyMissingConstructor
    def __init__(self, dataset=None, transforms=None):
        if not (isinstance(dataset, dict) and len(dataset)):
            raise ValueError("Expected dataset parameters as configuration input.")
        thelper.data.Dataset.__init__(self, transforms=transforms, deepcopy=False)
        self.root = dataset["path"]
        # keys matching dataset config for easy loading and referencing to same fields
        self.image_key = IMAGE_DATA_KEY     # key employed by loader to extract image data (pixel values)
        self.label_key = IMAGE_LABEL_KEY    # class id from API mapped to match model task
        self.path_key = DATASET_DATA_PATCH_PATH_KEY  # actual file path of the patch
        self.idx_key = DATASET_DATA_PATCH_INDEX_KEY  # increment for __getitem__
        self.mask_key = DATASET_DATA_PATCH_MASK_KEY  # actual mask path of the patch
        self.meta_keys = [self.path_key, self.idx_key, DATASET_DATA_PATCH_CROPS_KEY,
                          DATASET_DATA_PATCH_IMAGE_KEY, DATASET_DATA_PATCH_FEATURE_KEY]
        model_class_map = dataset[DATASET_DATA_KEY][DATASET_DATA_MAPPING_KEY]
        model_class_to_id = dataset[DATASET_DATA_KEY][DATASET_DATA_MODEL_MAPPING]
        sample_class_ids = set()
        samples = []
        channels = dataset.get(DATASET_DATA_CHANNELS, None) #FIXME: the user needs to specified the channels used by the model
        self.channels = channels if channels else [1, 2, 3] # by default we take the first 3 channels
        for patch_path, patch_info in zip(dataset[DATASET_FILES_KEY],
                                          dataset[DATASET_DATA_KEY][DATASET_DATA_PATCH_KEY]):
            if patch_info[DATASET_DATA_PATCH_SPLIT_KEY] == "test":
                # convert the dataset class ID into the model class ID using mapping, drop sample if not found
                class_name = model_class_map.get(patch_info[DATASET_DATA_PATCH_CLASS_KEY])
                class_model_id = model_class_to_id.get(patch_info[DATASET_DATA_PATCH_CLASS_KEY])
                if class_name is not None:
                    sample_class_ids.add(class_name)
                    samples.append(deepcopy(patch_info))
                    samples[-1][self.path_key] = os.path.join(self.root, patch_path)
                    samples[-1][self.label_key] = class_model_id

        if not len(sample_class_ids):
            raise ValueError("No patch/class could be retrieved from batch loading for specific model task.")
        self.samples = samples
        self.sample_class_ids = sample_class_ids


class BatchTestPatchesBaseSegDatasetLoader(ImageFolderSegDataset):
    """
    Batch dataset parser that loads only patches from 'test' split and matching
    class IDs (or their parents) known by the model as defined in its ``task``.

    .. note::

        Uses :class:`thelper.data.SegmentationDataset` ``__getitem__`` implementation to load image
        from a folder, but overrides the ``__init__`` to adapt the configuration to batch format.
    """

    # noinspection PyMissingConstructor
    def __init__(self, dataset=None, transforms=None):
        if not (isinstance(dataset, dict) and len(dataset)):
            raise ValueError("Expected dataset parameters as configuration input.")
        thelper.data.Dataset.__init__(self, transforms=transforms, deepcopy=False)
        self.root = dataset["path"]
        # keys matching dataset config for easy loading and referencing to same fields
        self.image_key = IMAGE_DATA_KEY     # key employed by loader to extract image data (pixel values)
        self.label_key = IMAGE_LABEL_KEY    # class id from API mapped to match model task
        self.path_key = DATASET_DATA_PATCH_PATH_KEY  # actual file path of the patch
        self.idx_key = DATASET_DATA_PATCH_INDEX_KEY  # increment for __getitem__
        self.mask_key = DATASET_DATA_PATCH_MASK_KEY  # actual mask path of the patch
        self.meta_keys = [self.path_key, self.idx_key, self.mask_key, DATASET_DATA_PATCH_CROPS_KEY,
                          DATASET_DATA_PATCH_IMAGE_KEY, DATASET_DATA_PATCH_FEATURE_KEY]
        model_class_map = dataset[DATASET_DATA_KEY][DATASET_DATA_MAPPING_KEY]
        sample_class_ids = set()
        samples = []
        channels = dataset.get(DATASET_DATA_CHANNELS, None)  # FIXME: the user needs to specified the channels used by the model
        self.channels = channels if channels else [1, 2, 3]  # by default we take the first 3 channels
        for patch_path, patch_info in zip(dataset[DATASET_FILES_KEY],
                                          dataset[DATASET_DATA_KEY][DATASET_DATA_PATCH_KEY]):
            if patch_info[DATASET_DATA_PATCH_SPLIT_KEY] == "test":
                # convert the dataset class ID into the model class ID using mapping, drop sample if not found
                class_name = model_class_map.get(patch_info[DATASET_DATA_PATCH_CLASS_KEY])
                if class_name is not None:
                    sample_class_ids.add(class_name)
                    samples.append(deepcopy(patch_info))
                    samples[-1][self.path_key] = os.path.join(self.root, patch_path)
                    samples[-1][self.label_key] = class_name
                    mask_name = patch_info.get(DATASET_DATA_PATCH_CROPS_KEY)[0].get(self.mask_key, None)
                    if mask_name is not None:
                        samples[-1][self.mask_key] = os.path.join(self.root, mask_name)
        if not len(sample_class_ids):
            raise ValueError("No patch/class could be retrieved from batch loading for specific model task.")
        self.samples = samples
        self.sample_class_ids = sample_class_ids


class BatchTestPatchesClassificationDatasetLoader(BatchTestPatchesBaseDatasetLoader):
    def __init__(self, dataset=None, transforms=None):
        super(BatchTestPatchesClassificationDatasetLoader, self).__init__(dataset, transforms)
        self.task = thelper.tasks.Classification(
            class_names=list(self.sample_class_ids),
            input_key=self.image_key,
            label_key=self.label_key,
            meta_keys=self.meta_keys,
        )


class BatchTestPatchesSegmentationDatasetLoader(BatchTestPatchesBaseSegDatasetLoader):
    def __init__(self, dataset=None, transforms=None):
        super(BatchTestPatchesSegmentationDatasetLoader, self).__init__(dataset, transforms)
        class_names = [str(c) for c in dataset.get('data').get('model_class_order', list(self.sample_class_ids))]
        self.task = thelper.tasks.Segmentation(
            class_names=class_names,
            input_key=self.image_key,
            label_map_key=self.label_key,
            meta_keys=self.meta_keys,
        )


def adapt_dataset_for_model_task(model_task, dataset):
    # type: (AnyTask, Dataset) -> JSON
    """
    Generates dataset parameter definition for loading from checkpoint configuration with ``thelper``.

    Retrieves available classes from the loaded dataset parameters and preserves only matching classes with the task
    defined by the original model task. Furthermore, parent/child class IDs are resolved recursively in a bottom-top
    manner to adapt specific classes into corresponding `categories` in the case the model uses them as more generic
    classes.

    .. seealso::
        - :class:`BatchTestPatchesBaseDatasetLoader` for dataset parameters used for loading filtered patches.

    :param model_task: original task defined by the model training which specifies known classes.
    :param dataset: batch of patches from which to extract matching classes known to the model.
    :return: configuration that allows ``thelper`` to generate a data loader for testing the model on desired patches.
    """
    try:
        dataset_params = dataset.json()     # json required because thelper dumps config during logs
        all_classes_mapping = dict()        # child->parent taxonomy class ID mapping
        all_model_ordering = list()         # class ID order as defined by the model
        all_model_mapping = dict()          # taxonomy->model class ID mapping
        all_child_classes = set()           # only taxonomy child classes IDs
        all_test_patch_files = list()       # list of the test patch files

        def find_class_mapping(taxonomy_class, parent=None):
            """Finds existing mappings defined by taxonomy."""
            children = taxonomy_class.get("children")
            class_id = taxonomy_class.get("id")
            if children:
                for child in children:
                    find_class_mapping(child, taxonomy_class)
            else:
                all_child_classes.add(class_id)
            all_classes_mapping[class_id] = None if not parent else parent.get("id")

        # Some models will use a generic background class so we add it systematically in case the model needs it
        for taxo in dataset_params[DATASET_DATA_KEY][DATASET_DATA_TAXO_KEY]:
            taxo.get("children").insert(0, {"id": DATASET_BACKGROUND_ID,
                                            "name_fr": "Classe autre",
                                            "taxonomy_id": taxo.get("taxonomy_id"),
                                            "code": "BACK",
                                            "name_en": "Background",
                                            "children": []})

        for taxo in dataset_params[DATASET_DATA_KEY][DATASET_DATA_TAXO_KEY]:
            find_class_mapping(taxo)
        LOGGER.debug("Taxonomy class mapping:  {}".format(all_classes_mapping))
        LOGGER.debug("Taxonomy class children: {}".format(all_child_classes))

        # find model mapping using taxonomy hierarchy
        def get_children_class_ids(parent_id):
            children_ids = set()
            filtered_ids = set([c for c, p in all_classes_mapping.items() if p == parent_id])
            for c in filtered_ids:
                if c not in all_child_classes:
                    children_ids = children_ids | get_children_class_ids(c)
            return children_ids | filtered_ids

        for model_class_id in model_task.class_names:
            # attempt str->int conversion of model string, they should match taxonomy class IDs
            try:
                model_class_id = int(model_class_id)
            except ValueError:
                raise ValueError("Unknown class ID '{}' cannot be matched with taxonomy classes".format(model_class_id))
            if model_class_id not in all_classes_mapping:
                raise ValueError("Unknown class ID '{}' cannot be found in taxonomy".format(model_class_id))
            # while looking for parent/child mapping, also convert IDs as thelper requires string labels
            if model_class_id in all_child_classes:
                LOGGER.debug("Class {0}: found direct child ID ({0}->{0})".format(model_class_id))
                all_model_mapping[model_class_id] = str(model_class_id)
            else:
                categorized_classes = get_children_class_ids(model_class_id)
                for cat_id in categorized_classes:
                    all_model_mapping[cat_id] = str(model_class_id)
                LOGGER.debug("Class {0}: found category class IDs ({0}->AnyOf{1})"
                             .format(model_class_id, list(categorized_classes)))
            all_model_ordering.append(model_class_id)
        all_model_mapping = {c: all_model_mapping[c] for c in sorted(all_model_mapping)}
        LOGGER.debug("Model class mapping (only supported classes): {}".format(all_model_mapping))
        LOGGER.debug("Model class ordering (indexed class outputs): {}".format(all_model_ordering))

        # add missing classes mapping
        all_model_mapping.update({c: None for c in sorted(set(all_classes_mapping) - set(all_model_mapping))})
        LOGGER.debug("Model class mapping (added missing classes): {}".format(all_model_mapping))

        # update obtained mapping with dataset parameters for loader
        dataset_params[DATASET_DATA_KEY][DATASET_DATA_MAPPING_KEY] = all_model_mapping
        dataset_params[DATASET_DATA_KEY][DATASET_DATA_ORDERING_KEY] = all_model_ordering
        dataset_params[DATASET_DATA_KEY][DATASET_DATA_MODEL_MAPPING] = model_task.class_indices
        dataset_params[DATASET_FILES_KEY] = all_test_patch_files

        # update patch info for classes of interest
        # this is necessary for BatchTestPatchesClassificationDatasetLoader
        class_mapped = [c for c, m in all_model_mapping.items() if m is not None]
        samples_all = dataset_params[DATASET_DATA_KEY][DATASET_DATA_PATCH_KEY]  # type: JSON
        all_model_classes = set(class_mapped + all_model_ordering)
        samples_mapped = [s for s in samples_all if s[DATASET_DATA_PATCH_CLASS_KEY] in all_model_classes]
        # retain class Ids with test patches
        classes_with_files = sorted(set([s["class"] for s in samples_mapped if s["split"] == "test"]))
        if len(classes_with_files) == 0:
            raise ValueError("No test patches for the classes of interest!")
        all_test_patch_files = [s[DATASET_DATA_PATCH_CROPS_KEY][0]["path"] for s in samples_mapped]
        dataset_params[DATASET_FILES_KEY] = all_test_patch_files

        # test_samples = [
        #    {"class_id": s[DATASET_DATA_PATCH_CLASS_KEY],
        #     "sample_id": s[DATASET_DATA_PATCH_FEATURE_KEY]} for s in samples_all
        # ]

        model_task_name = fully_qualified_name(model_task)
        return {
            "type": MODEL_TASK_MAPPING[model_task_name][MAPPING_LOADER],
            "params": {TEST_DATASET_KEY: dataset_params},
        }
    except Exception as exc:
        raise RuntimeError("Failed dataset adaptation to model task classes for evaluation. [{!r}]".format(exc))


def test_loader_from_configs(model_checkpoint_config, model_config_override, dataset_config_override, settings):
    # type: (CkptData, Model, Dataset, SettingsType) -> JSON
    """
    Obtains a simplified version of the configuration for 'test' task corresponding to the model and dataset.
    Removes parameters from the original file that would require additional unnecessary operations other than testing.
    Overrides checkpoint training configurations, model name and datasets to enforce with the ones passed.
    """

    # transfer required parts, omit training specific values or error-prone configurations
    test_config = deepcopy(model_checkpoint_config)
    test_config["name"] = model_config_override["name"]
    for key in ["epoch", "iter", "sha1", "outputs", "optimizer"]:
        test_config.pop(key, None)

    # override deployed model and dataset references
    #   - override model task defined as string to equivalent parameter dictionary representation
    #   - override model task input/label keys that could be modified by user to match definitions of dataset loader
    #   - override model task classes instead of full dataset so we don't specialize the model
    #     (test only on classes known by the model, or on any class nested under a more generic category)
    test_dataset_name = dataset_config_override["name"]

    if isinstance(test_config["task"], str):
        task_str = test_config["task"]
        task_class_str = task_str.split("(", 1)[0]
        LOGGER.debug("Task type in the config:\n"
                     f"  task type: [{task_class_str}]")
        task_params = fix_str_model_task(task_str)
        LOGGER.debug("Overriding model task string definition by parameter dictionary representation:\n"
                     f"  original task: [{task_str}]\n"
                     f"  modified task: [{task_params}]")
        test_config["task"] = {"params": task_params, "type": task_class_str}
        if "Detection" in task_class_str:
            test_config["task"]["params"].update({"input_key": IMAGE_DATA_KEY})
        elif "Segmentation" in task_class_str:
            test_config["task"]["params"].update({"input_key": IMAGE_DATA_KEY, "label_map_key": IMAGE_LABEL_KEY})
        else:
            test_config["task"]["params"].update({"input_key": IMAGE_DATA_KEY, "label_key": IMAGE_LABEL_KEY})
        test_model_task = thelper.tasks.create_task(test_config["task"])
    else:
        task_class_str = fully_qualified_name(test_config["task"])
        test_model_task = test_config["task"]

    test_model_task_name = fully_qualified_name(test_model_task)
    test_config["config"]["name"] = model_config_override["name"]
    test_config["config"]["datasets"] = {
        test_dataset_name: adapt_dataset_for_model_task(test_model_task, dataset_config_override)
    }

    # back-compatibility replacement
    test_config["config"]["loaders"] = test_config["config"].pop("data_config", test_config["config"].get("loaders"))
    if "loaders" not in test_config["config"]:
        raise ValueError("Missing 'loaders' configuration from model checkpoint.")

    loaders = test_config["config"]["loaders"]  # type: JSON
    trainer = test_config["config"]["trainer"]  # type: JSON

    # remove additional unnecessary sub-parts or error-prone configurations
    for key in ["sampler", "train_augments", "train_split", "valid_split", "eval_augments"]:
        loaders.pop(key, None)

    # override image key to match loaded test data
    for transform in loaders.get("base_transforms", []):
        transform["target_key"] = IMAGE_DATA_KEY

    # override required values with modified parameters and remove error-prone configurations
    loaders["test_split"] = {
        test_dataset_name: 1.0    # use every single dataset patch found by the test loader
    }
    loaders["shuffle"] = False  # easier to compare inference samples when order matches + cannot merge lists otherwise
    trainer["use_tbx"] = False
    trainer["type"] = MODEL_TASK_MAPPING[test_model_task_name][MAPPING_TESTER]
    for key in ["device", "train_device", "optimization", "monitor"]:
        trainer.pop(key, None)

    # enforce multiprocessing workers count and batch size according to settings
    # note:
    #   job worker process must be non-daemonic to allow data loader workers spawning
    # see:
    #   ``geoimagenet_ml.api.routes.processes.utils.process_ml_job_runner`` for worker setup
    loaders["workers"] = int(settings.get("geoimagenet_ml.ml.data_loader_workers", 0))
    loaders["test_batch_size"] = int(settings.get("geoimagenet_ml.ml.data_loader_batch_size", 10))
    loaders.pop("batch_size", None)  # raises if both generic and test-specific exist at the same time

    # override metrics to retrieve the ones required for result output
    if "Classification" in task_class_str or thelper.concepts.supports(test_model_task, "classification"):
        class_count = len(test_model_task.class_names)
        trainer["metrics"] = {
            "predictions": {
                "type": "thelper.train.utils.ClassifLogger",
                "params": {
                    "format": "json",
                    "top_k": class_count,  # obtain the prediction of every class from the model
                }
            },
            "report": {
                "type": "thelper.train.utils.ClassifReport",
                "params": {
                    "format": "json"
                }
            },
            "top_1_accuracy": {
                "type": "thelper.optim.metrics.Accuracy",
                "params": {
                    "top_k": 1,
                }
            },
            "ConfusionMatrix": {
                "type": "thelper.train.utils.ConfusionMatrix",
            }
        }
    elif "Segmentation" in task_class_str or thelper.concepts.supports(test_model_task, "segmentation"):
        # FIXME: Not sure this the right mechanism but we need to enforce the right keys
        #  that are expected by the data loader, those keys should not be specified by the user
        test_model_task.input_key = IMAGE_DATA_KEY
        test_model_task.gt_key = DATASET_DATA_PATCH_MASK_KEY
        test_model_task.dontcare = DATASET_DATA_PATCH_DONTCARE
        # Metrics more appropriate for segmentation
        trainer["metrics"] = {
            # FIXME: segmentation results must be extracted specifically for this task type
            #        see 'classification_test_results_finder' and 'MODEL_TASK_MAPPING'
            #        without correct extraction of results, job output will have nothing
            # "AveragePrecision": {
            #    "type": "thelper.optim.metrics.AveragePrecision",
            # },
            "mIoU": {
                "type": "thelper.optim.metrics.IntersectionOverUnion",
            },
            # "top_5_accuracy": {
            #    "type": "thelper.optim.metrics.CategoryAccuracy",
            #    "params": {
            #        "top_k": 5,
            #    }
            # }
        }
    # need to copy the model parameters because this field is expected by 'thelper.nn.create_model'
    if "model_params" not in test_config:
        test_config["model_params"] = test_config["config"]["params"]
    return test_config


def classification_test_results_finder(test_dataset, test_results, test_output_path):
    # type: (Dataset, JSON, AnyStr) -> JSON
    """Specialized result finder for classification task.

    Format of ``predictions`` sub-field for this task type is:

    .. code-blocK:: json

        {
            "target": {
                "class_id": "<class_id>", "score": <score>
            },
            "outputs": [
                {"class_id": "<class-id>", "score": <score>}
                <...>
            ]
        }

    Where ``outputs`` are the Top-K output predictions (inference) of every class generated by the classification model,
    and ``target`` contains the prediction score of the defined ground-truth sample class.

    Output conforms to schema definition of :func:`get_test_results`.

    .. seealso::
        - :func:`get_test_results` and ``MODEL_TASK_MAPPING`` for automatic resolution by task type.
        - :func:`test_loader_from_configs` for configured metrics available for the given task type.
    """
    class_mapping = test_dataset[DATASET_DATA_KEY][DATASET_DATA_MAPPING_KEY]   # type: ClassMap
    class_outputs = test_dataset[DATASET_DATA_KEY][DATASET_DATA_ORDERING_KEY]
    class_mapped = [c for c, m in class_mapping.items() if m is not None]
    samples_all = test_dataset[DATASET_DATA_KEY][DATASET_DATA_PATCH_KEY]  # type: JSON
    samples_mapped = [s for s in samples_all if s[DATASET_DATA_PATCH_CLASS_KEY] in class_mapped]
    test_samples = [
        {"class_id": s[DATASET_DATA_PATCH_CLASS_KEY],
         "sample_id": s[DATASET_DATA_PATCH_FEATURE_KEY]} for s in samples_all
    ]
    test_summary = {
        "classes_total": len(class_mapping),
        "classes_mapped": len(class_mapped),
        "classes_dropped": len(class_mapping) - len(class_mapped),
        "samples_total": len(test_samples),
        "samples_mapped": len(samples_mapped),
        "samples_dropped": len(test_samples) - len(samples_mapped),
    }
    test_classes = [
        {"model_index": i, "class_id": c} for i, c in enumerate(class_outputs)
    ]
    test_predictions = test_results.get("predictions", None)
    if not test_predictions:
        prediction_path = os.path.join(test_output_path, "predictions.json")
        if os.path.isfile(prediction_path):
            with open(prediction_path, "r") as f:
                test_predictions = json.load(f)
    if test_predictions:
        test_predictions = [
            {
                "target": {"class_id": p["target_name"], "score": p["target_score"]},
                "outputs": [{"class_id": p[f"pred_{i}_name"], "score": p[f"pred_{i}_score"]}
                            for i in range(1, len(class_outputs) + 1)]
            }
            for p in test_predictions
        ]
    else:
        test_predictions = []
    test_metrics = {m: s for m, s in test_results.items() if m != "predictions"}
    # since storage doesn't support int keys and classes could be defined as int, convert to dict
    test_mapping = [{"class_id": c, "model_id": m} for c, m in class_mapping.items()]
    classification_results = {
        "summary": test_summary,
        "metrics": test_metrics,
        "samples": test_samples,
        "classes": test_classes,
        "mapping": test_mapping,
        "predictions": test_predictions,
    }

    return classification_results


def get_test_results(test_runner, test_results):
    # type: (thelper.train.Trainer, JSON) -> JSON
    """
    Obtains a JSON result representation matching the ``test_runner`` specialized task
    and ``test_results`` retrieved from the model evaluation with ``metrics`` defined in the task.

    Typically, results would be obtained from calling::

        test_runner.eval()

    The function ensures the backward mapping of predictions to class IDs.
    It also cleans up the results as they would otherwise be too verbose for output.

    The resulting JSON is in the form:

    .. code-block:: json

        {
            "summary": {"<stat>": <value>},
            "metrics": {"<name>": <value>},
            "samples": [{"class_id": <id>, "sample_id": <id>}],
            "classes": [{"class_id": <id>, "model_index": <idx>}],
            "mapping": [{"class_id": <model-id/None>}]
            "predictions": [{<prediction-format-according-to-task-type>}]
        }

    .. seealso::
        - :func:`test_loader_from_configs` for configuration that leads to produced results by the task runner.
    """
    test_task_name = fully_qualified_name(test_runner.task)
    test_result_finder = MODEL_TASK_MAPPING[test_task_name][MAPPING_RESULT]

    # only one test dataset submitted for evaluation (ie: unique 'test_split' in 'test_loader_from_configs')
    test_results = test_results[0].get("test/metrics")
    test_dataset = list(test_runner.config["datasets"].values())[0]["params"][TEST_DATASET_KEY]
    test_outputs = test_runner.output_paths["test"]  # where additional metric loggers are dumped to file

    return test_result_finder(test_dataset, test_results, test_outputs)


def retrieve_annotations(geojson_urls):
    # type: (List[AnyStr]) -> List[JSON]
    """Fetches GeoJSON structured feature(s) data for each of the provided URL."""
    annotations = list()
    for url in geojson_urls:
        resp = requests.get(url, headers={"Accept": "application/json"})
        code = resp.status_code
        if code != 200:
            raise RuntimeError("Could not retrieve GeoJSON from [{}], server response was [{}]".format(url, code))
        annotations.append(resp.json())
    if not annotations:
        raise RuntimeError("Could not find any annotation from URL(s): {}".format(geojson_urls))
    return annotations


def retrieve_taxonomy(taxonomy_url):
    # type: (AnyStr) -> JSON
    """Fetches JSON structured taxonomy classes hierarchy from the provided URL."""
    resp = requests.get(taxonomy_url, headers={"Accept": "application/json"})
    code = resp.status_code
    if code != 200:
        raise RuntimeError("Could not retrieve taxonomy JSON from [{}], server response was [{}]"
                           .format(taxonomy_url, code))
    taxo = resp.json()
    if not taxo:
        raise RuntimeError("Could not find any taxonomy detail from URL: {}".format(taxonomy_url))
    taxo_multi = [taxo] if isinstance(taxo, dict) else taxo  # support both specific or 'all taxonomies' responses
    validate_taxonomy_format(taxo_multi)
    return taxo_multi


def validate_taxonomy_format(taxonomy):
    # type: (List[JSON]) -> None
    """
    Recursively validates that required fields of taxonomy class definitions are available for later process execution.

    :param taxonomy: taxonomy schema to be validated (multi-taxonomy list is expected)
    :raises: if the taxonomy or any underlying class definition format is invalid.
    """
    taxonomy_class_ids = []  # ids must be unique across taxonomies

    def check_taxonomy_class_format(taxonomy_class, taxonomy_id):
        if not isinstance(taxonomy_class, dict):
            raise TypeError("Invalid taxonomy class format definition.")
        if taxonomy_class.get("taxonomy_id") != taxonomy_id:
            raise ValueError("Invalid taxonomy ID doesn't match parent reference.")
        class_id = taxonomy_class.get("id")
        if not isinstance(class_id, int):
            raise ValueError("Missing or invalid class ID in taxonomy.")
        if class_id in taxonomy_class_ids:
            raise ValueError("Duplicate class ID found in taxonomies [class={}]".format(class_id))
        taxonomy_class_ids.append(class_id)
        children = taxonomy_class.get("children")
        if not isinstance(children, list):
            raise TypeError("Invalid taxonomy class children definition.")
        for c in children:
            check_taxonomy_class_format(c, taxonomy_id)

    if not isinstance(taxonomy, list):
        raise TypeError("Invalid taxonomy format definition.")
    if not len(taxonomy):
        raise ValueError("Missing taxonomy definitions.")
    for taxo in taxonomy:
        taxo_id = taxo.get("taxonomy_id")
        if not isinstance(taxo_id, int):
            raise ValueError("Invalid taxonomy ID value.")
        check_taxonomy_class_format(taxo, taxo_id)


def find_best_match_raster(rasters, feature):
    # type: (List[RasterDataType], FeatureType) -> RasterDataType
    """
    Attempts to find the best matching raster with the specified ``feature``.
    Search includes fuzzy matching of file name and overlapping geometry alignment.

    :raises RuntimeError: if no raster can be matched.
    """
    def fix_raster_search_name(name):
        return name.replace("8bits", "xbits").replace("16bits", "xbits")

    raster_data = None
    if "properties" in feature and "image_name" in feature["properties"]:
        raster_name = fix_raster_search_name(feature["properties"]["image_name"])
        for data in rasters:
            feature_path = fix_raster_search_name(data["file_path"])
            if raster_name in feature_path:
                LOGGER.info("matched raster/feature by filename: [{}, {}]".format(raster_name, feature_path))
                raster_data = data
                break
    if raster_data is None:
        for data in rasters:
            raster_geom = data["global_roi"]
            feature_geom = feature["geometry"]
            if raster_geom.contains(feature_geom):
                LOGGER.info("matched raster/feature by geometry: [{}, {}]".format(raster_geom, feature_geom))
                raster_data = data
                break
    if raster_data is None:
        raise RuntimeError("could not find proper raster for feature '{}'".format(feature["id"]))
    return raster_data


def create_batch_patches(annotations_meta,      # type: List[JSON]
                         taxonomy_meta,         # type: JSON
                         raster_search_paths,   # type: List[AnyStr]
                         dataset_store,         # type: DatasetStore
                         dataset_container,     # type: Dataset
                         dataset_latest,        # type: Optional[Dataset]
                         dataset_update_count,  # type: int
                         crop_fixed_size,       # type: Optional[Number]
                         crop_mode,             # type: Optional[Union[AnyStr, int]]
                         update_func,           # type: Callable[[AnyStr, Optional[Number]], None]
                         start_percent,         # type: Number
                         final_percent,         # type: Number
                         train_test_ratio,      # type: float
                         ):                     # type: (...) -> Dataset
    """
    Creates patches on disk using provided annotation metadata.
    Parses ``annotations_meta`` using format specified by `GeoImageNet API`.

    Patches are created by transforming the geometry into the appropriate coordinate system from ``annotations_meta``.

    Literal coordinate values as dimension limits of the patch are first applied.
    This creates patches of variable size, but corresponding to the original annotations.
    If ``crop_fixed_size`` is provided, `fixed` sized patches are afterwards created from the originals by cropping
    accordingly with dimensions of each patch's annotation coordinates.
    Otherwise, ``crop_mode`` indicates how to generate the crops dimensions from the original feature.

    .. seealso::
        - :func:`geoimagenet_ml.ml.utils.get_feature_bbox`
        - :func:`geoimagenet_ml.ml.utils.process_feature_crop`

    Created patches for the batch are then split into train/test sets per corresponding ``taxonomy_class_id``.

    .. note::

        - ``annotations_meta`` and ``taxonomy_meta`` formats are expected to resolve with provided example links.

    .. seealso::

        - `GeoImageNet API` annotation example: https://geoimagenetdev.crim.ca/api/v1/batches/annotations
        - `GeoImageNet API` taxonomy example: https://geoimagenetdev.crim.ca/api/v1/taxonomy_classes

    :param annotations_meta: metadata retrieved from URL(s) (see example link).
    :param taxonomy_meta: parent/child reference class IDs hierarchy matching annotation metadata (see example link).
    :param raster_search_paths: paths where to look for raster images matching annotation metadata.
    :param dataset_store: store connection where the updated dataset has to be written.
    :param dataset_container: dataset to be iteratively updated from extracted patches from rasters using annotations.
    :param dataset_latest: base dataset with existing patches from which to expand the new one (as desired).
    :param dataset_update_count: number of patches to generate until the dataset gets updated (save checkpoint).
    :param crop_fixed_size: dimension (in metres) to use the generate additional patches of constant dimension.
    :param update_func: ``function(message, percentage)`` called on each update operation for process execution logging.
    :param start_percent: execution percentage to use as starting value for the execution of this function.
    :param final_percent: execution percentage to reach a the end of the normal execution of this function.
    :param train_test_ratio: ratio to use for splitting patches into train/test groups for respective classes.
    :returns: updated ``dataset_container`` with metadata of created patches.
    """

    def select_split(splits, class_id, name=None):
        # type: (List[Tuple[AnyStr, ClassCounter]], Union[int, AnyStr], Optional[AnyStr]) -> AnyStr
        """
        Selects a split set from ``splits`` and decreases ``class_id`` in its corresponding counter
        if any 'instance' is left for selection within that class (counter didn't reach 0).
        Otherwise, the counter of the other split is decreased and its name is returned instead.

        :param splits: list of tuple with 'train'/'test' and its corresponding counter.
        :param class_id: class for which to update the selected split counter.
        :param name: select the set by 'name' value, or randomly if `None`.
        :returns: selected split 'name', updated counters by reference.
        """
        if name:
            chosen = list(filter(lambda s: s[0] == name, splits))[0]
        else:
            chosen = random.choice(splits)
        if chosen[1][class_id] <= 0:
            if chosen[0] == splits[0][0]:
                chosen = splits[1]
            else:
                chosen = splits[0]
        chosen[1][class_id] -= 1
        return chosen[0]

    # FIXME: although we support multiple GeoJSON URL as process input, functions below only expect 1
    #   - need to combine features by ID to avoid duplicates
    #   - need to handle different CRS for GeoTransform (update process runner docstring accordingly)
    if len(annotations_meta) != 1:
        raise NotImplementedError("Multiple GeoJSON parsing not implemented.")
    annotations_meta = annotations_meta[0]

    if not isinstance(dataset_update_count, int) or dataset_update_count < 1:
        raise AssertionError("invalid dataset update count value: {!s}".format(dataset_update_count))

    if isinstance(crop_fixed_size, (int, float)):
        if crop_fixed_size <= 0:
            raise ValueError(f"invalid crop fixed size value: {crop_fixed_size}, must be >0m")
        crop_formats = [(crop_fixed_size, None, "fixed")]
        update_func(f"validated dataset request for crop fixed size: {crop_fixed_size}m", start_percent)
    else:
        crop_mode = DATASET_CROP_MODES.get(crop_mode)
        if not isinstance(crop_mode, int):
            raise ValueError(f"invalid crop mode value: {crop_mode}, must be one of [{list(DATASET_CROP_MODES)}]")
        crop_mode_name = DATASET_CROP_MODE_NAMES[crop_mode]
        crop_formats = [(None, crop_mode, crop_mode_name)]
        update_func(f"validated dataset request for crop mode: {crop_mode_name}", start_percent)

    start_percent += 1
    update_func("updating taxonomy definition in dataset", start_percent)
    dataset_container.data["taxonomy"] = taxonomy_meta

    start_percent += 1
    update_func("parsing raster files", start_percent)
    srs = parse_coordinate_system(annotations_meta)
    rasters_data, raster_global_coverage = parse_rasters(raster_search_paths, default_srs=srs)

    start_percent += 1
    update_func("parsing GeoJSON metadata", start_percent)
    features, category_counter = parse_geojson(annotations_meta, srs_destination=srs, roi=raster_global_coverage)

    start_percent += 1
    patch_percent = final_percent - 1
    if dataset_latest:
        update_func("resolving incremental batch patches on top of [{!s}]".format(dataset_latest), start_percent)
        provided_features = set([feature["id"] for feature in features])
        mapped_features = {patch["feature"]: patch for patch in dataset_latest.data.get("patches", [])}
        matched_features = list(provided_features & set(mapped_features.keys()))
    else:
        update_func("creating batch patches from new dataset (not incremental)", start_percent)
        mapped_features = dict()
        matched_features = []

    train_counter, test_counter = category_counter.split(train_test_ratio)
    train_test_splits = [("train", train_counter), ("test", test_counter)]

    last_progress_offset, progress_scale = 0, float(patch_percent - start_percent) / len(features)
    dataset_container.data["patches"] = list()
    for feature_idx, feature in enumerate(features):
        progress_offset = int(start_percent + feature_idx * progress_scale)
        if progress_offset != last_progress_offset:
            last_progress_offset = progress_offset
            update_func("extracting patches for batch creation", progress_offset)

        # transfer patch data from previous batch, preserve selected split
        if feature["id"] in matched_features:
            patch_info = deepcopy(mapped_features.get(feature["id"]))
            if not isinstance(patch_info, dict) or DATASET_DATA_PATCH_CROPS_KEY not in patch_info:
                raise RuntimeError("Failed to retrieve presumably existing patch from previous batch (feature: {})"
                                   .format(feature["id"]))
            # copy information, but replace patch copies
            for i_crop, _ in enumerate(patch_info[DATASET_DATA_PATCH_CROPS_KEY]):
                for path_field in [DATASET_DATA_PATCH_PATH_KEY, DATASET_DATA_PATCH_MASK_KEY]:
                    old_patch_path = patch_info[DATASET_DATA_PATCH_CROPS_KEY][i_crop].get(path_field)
                    if not old_patch_path and path_field == DATASET_DATA_PATCH_MASK_KEY:
                        raise RuntimeError(f"Dataset [{dataset_latest.uuid}] is not compatible for incremental feature."
                                           " Missing mask image files.")
                    new_patch_path = old_patch_path.replace(dataset_latest.path, dataset_container.path)
                    if not new_patch_path.startswith(dataset_container.path):
                        raise RuntimeError("Invalid patch path from copy. Expected base: '{}', but got: '{}'"
                                           .format(dataset_container.path, new_patch_path))
                    patch_info[DATASET_DATA_PATCH_CROPS_KEY][i_crop][path_field] = new_patch_path
                    shutil.copy(old_patch_path, new_patch_path)
                    dataset_container.files.append(new_patch_path)
            dataset_container.data[DATASET_DATA_PATCH_KEY].append(patch_info)
            # update counter with previously selected split set
            select_split(train_test_splits, patch_info[DATASET_DATA_PATCH_CLASS_KEY],
                         name=patch_info[DATASET_DATA_PATCH_SPLIT_KEY])

        # new patch creation from feature specification, generate metadata and randomly select split
        else:
            raster_data = find_best_match_raster(rasters_data, feature)
            crop_class_id = feature.get("properties", {}).get("taxonomy_class_id")
            dataset_container.data[DATASET_DATA_PATCH_KEY].append({
                DATASET_DATA_PATCH_CROPS_KEY: [],  # updated gradually after
                DATASET_DATA_PATCH_IMAGE_KEY: raster_data["file_path"],
                DATASET_DATA_PATCH_CLASS_KEY: crop_class_id,
                DATASET_DATA_PATCH_SPLIT_KEY: select_split(train_test_splits, crop_class_id),
                DATASET_DATA_PATCH_FEATURE_KEY: feature.get("id"),
            })

            for crop_size, crop_mode, crop_type in crop_formats:
                crop, inv_crop, bbox = process_feature_crop(feature["geometry"], srs, raster_data, crop_size, crop_mode)
                if crop is not None:
                    if crop.ndim < 3 or crop.shape[2] != raster_data["band_count"]:
                        raise AssertionError("bad crop channel size")
                    output_geotransform = list(raster_data["offset_geotransform"])
                    output_geotransform[0], output_geotransform[3] = bbox[0], bbox[1]
                    output_driver = gdal.GetDriverByName("GTiff")
                    output_name = get_sane_name("{}_{}".format(feature["id"], crop_type), assert_invalid=False)
                    output_path = os.path.join(dataset_container.path, "{}_crop.tif".format(output_name))
                    output_mask = os.path.join(dataset_container.path, "{}_mask.png".format(output_name))
                    crop_image = Image.fromarray((255 * inv_crop.mask).astype(np.uint8)).convert("RGB")  # remove alpha
                    crop_image.save(output_mask)
                    if os.path.exists(output_path):
                        msg = "Output path [{}] already exists but is expected to not exist.".format(output_path)
                        update_func(msg + " Removing...", logging.WARNING)
                        os.remove(output_path)  # gdal raster creation fails if file already exists
                    output_dataset = output_driver.Create(output_path, crop.shape[1], crop.shape[0],
                                                          crop.shape[2], raster_data["data_type"])
                    for raster_band_idx in range(crop.shape[2]):
                        output_dataset.GetRasterBand(raster_band_idx + 1).SetNoDataValue(0)
                        output_dataset.GetRasterBand(raster_band_idx + 1).WriteArray(crop.data[:, :, raster_band_idx])
                    output_dataset.SetProjection(raster_data["srs"].ExportToWkt())
                    output_dataset.SetGeoTransform(output_geotransform)
                    output_dataset = None  # close output fd
                    dataset_container.files.append(output_path)
                    dataset_container.files.append(output_mask)
                    dataset_container.data[DATASET_DATA_PATCH_KEY][-1][DATASET_DATA_PATCH_CROPS_KEY].append({
                        "type": crop_type,
                        DATASET_DATA_PATCH_PATH_KEY: output_path,
                        DATASET_DATA_PATCH_MASK_KEY: output_mask,
                        "shape": list(crop.shape),
                        "data_type": raster_data["data_type"],
                        "coordinates": output_geotransform,
                    })
        if feature_idx // dataset_update_count:
            update_func("updating dataset definition (checkpoint: {}/{})"
                        .format(feature_idx, len(features)), progress_offset)
            dataset_store.save_dataset(dataset_container, overwrite=True)

    update_func("generating patches archived file", final_percent)
    dataset_container.zip()

    update_func("updating completed dataset definition", final_percent)
    dataset_container.mark_finished()
    return dataset_store.save_dataset(dataset_container, overwrite=True)


# FIXME: add definitions/implementations to support other task types (ex: object-detection)
MODEL_TASK_MAPPING = {
    fully_qualified_name(thelper.tasks.classif.Classification): {
        MAPPING_TASK:   fully_qualified_name(thelper.tasks.classif.Classification),
        MAPPING_LOADER: fully_qualified_name(BatchTestPatchesClassificationDatasetLoader),
        MAPPING_TESTER: fully_qualified_name(thelper.train.classif.ImageClassifTrainer),
        MAPPING_RESULT: classification_test_results_finder,  # type: Callable[[Dataset, JSON, AnyStr], JSON]
    },
    fully_qualified_name(thelper.tasks.segm.Segmentation): {
        MAPPING_TASK:   fully_qualified_name(thelper.tasks.segm.Segmentation),
        MAPPING_LOADER: fully_qualified_name(BatchTestPatchesSegmentationDatasetLoader),
        MAPPING_TESTER: fully_qualified_name(thelper.train.segm.ImageSegmTrainer),
        MAPPING_RESULT: classification_test_results_finder,  # FIXME: should point to a segmentation evaluation
    },
}
