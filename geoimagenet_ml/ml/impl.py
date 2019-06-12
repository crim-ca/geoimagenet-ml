from geoimagenet_ml.utils import get_sane_name
from geoimagenet_ml.ml.utils import parse_rasters, parse_geojson, parse_coordinate_system, process_feature_crop
from six.moves.urllib.parse import urlparse
from six.moves.urllib.request import urlopen
from copy import deepcopy
from io import BytesIO
import osgeo.gdal
import requests
import logging
import random
import shutil
import six
import ssl
import os
# noinspection PyPackageRequirements
import thelper
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geoimagenet_ml.store.datatypes import Job, Model, Dataset  # noqa: F401
    from geoimagenet_ml.store.interfaces import DatasetStore  # noqa: F401
    from geoimagenet_ml.typedefs import (  # noqa: F401
        Any, AnyStr, Callable, List, Tuple, Union, ParamsType, JSON, SettingsType, Number, Optional,
        FeatureType, RasterDataType
    )
    from geoimagenet_ml.utils import ClassCounter  # noqa: F401

# enforce GDAL exceptions (otherwise functions return None)
osgeo.gdal.UseExceptions()

LOGGER = logging.getLogger(__name__)

IMAGE_DATA_KEY = "data"


def load_model(model_file):
    # type: (Union[Any, AnyStr]) -> Tuple[bool, ParamsType, Optional[BytesIO], Optional[Exception]]
    """
    Tries to load a model checkpoint file from the file-like object, file path or URL.

    :return: tuple of (success, data, buffer, exception) accordingly.
    :raises: None (nothrow)
    """
    try:
        model_buffer = model_file
        if isinstance(model_file, six.string_types):
            if urlparse(model_file).scheme:
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


def get_test_data_runner(job, model_checkpoint_config, model, dataset, settings):
    # type: (Job, JSON, Model, Dataset, SettingsType) -> thelper.train.Trainer
    """
    Obtains a trainer specialized for testing data predictions using the provided model checkpoint and dataset loader.
    """
    test_config = test_loader_from_configs(model_checkpoint_config, model, dataset, settings)
    save_dir = os.path.join(settings.get("geoimagenet_ml.ml.jobs_path"), job.uuid)
    _, _, _, test_loader = thelper.data.utils.create_loaders(test_config["config"], save_dir=save_dir)
    # FIXME:
    #   If using different classes (from model.task) than ones defined by patches (dataset.task), need to filter/update
    #   mapping accordingly and generate task.
    #   [to be defined] how to handle mismatches: drop them, place in 'unknown' class and still evaluate them, etc.?
    #   (see: https://www.crim.ca/jira/browse/GEOIM-153)
    #   For now, use the dataset task directly, model must use exactly the same definition for functional operation.
    task = test_loader.dataset.task
    #task = thelper.tasks.utils.create_task(model_checkpoint_config["task"])  # enforce model task instead of dataset
    model = thelper.nn.create_model(test_config["config"], task, save_dir=save_dir, ckptdata=model_checkpoint_config)
    config = test_config["config"]
    loaders = None, None, test_loader   # type: thelper.typedefs.MultiLoaderType

    # session name as Job UUID will write data under '<geoimagenet_ml.ml.models_path>/<model-UUID>/output/<job-UUID>/'
    trainer = thelper.train.create_trainer(job.uuid, save_dir, config, model, loaders, model_checkpoint_config)
    return trainer


class BatchTestPatchesDatasetLoader(thelper.data.ImageFolderDataset):
    """
    Batch dataset parser that loads only patches from 'test' split.

    Uses :class:`thelper.data.ImageFolderDataset` ``__getitem__`` implementation to load image from a folder, but
    overrides the ``__init__`` to adapt the configuration to batch format.
    """
    def __init__(self, config=None, transforms=None):
        if not (isinstance(config, dict) and len(config)):
            raise ValueError("Expected dataset parameters as configuration input.")
        self.root = config["path"]
        # keys matching dataset config for easy loading and referencing to same fields
        self.image_key = IMAGE_DATA_KEY     # key employed by loader to extract image data (pixel values)
        self.label_key = "class"            # class id from API
        self.path_key = "path"              # actual file path
        self.idx_key = "index"              # increment for __getitem__
        # 'crops' for extra data such as coordinates
        # 'image' for original image path that was used to generate the patch from
        # 'feature' for annotation reference id
        meta_keys = [self.path_key, self.idx_key, "crops", "image", "feature"]
        class_ids = set()
        samples = []
        for patch_path, patch_info in zip(config["files"], config["data"]["patches"]):
            if patch_info["split"] == "test":
                class_name = patch_info["class"]
                class_ids.add(class_name)
                samples.append(deepcopy(patch_info))
                samples[-1][self.path_key] = patch_path
        if not len(class_ids):
            raise ValueError("No patch/class could be retrieved from batch for loading.")
        thelper.data.ClassificationDataset.__init__(
            self, class_names=list(class_ids), input_key=self.image_key,
            label_key=self.label_key, meta_keys=meta_keys, config=config, transforms=transforms)
        self.samples = samples


def test_loader_from_configs(model_checkpoint_config, model_config_override, dataset_config_override, settings):
    # type: (JSON, Model, Dataset, SettingsType) -> JSON
    """
    Obtains a simplified version of the configuration for 'test' task corresponding to the model and dataset.
    Removes parameters from the original file that would require additional unnecessary operations other than testing.
    Overrides checkpoint training configurations, model name and datasets to enforce with the ones passed.
    """

    # transfer required parts, omit training specific values or error-prone configurations
    test_config = deepcopy(model_checkpoint_config)     # type: JSON
    test_config["name"] = model_config_override["name"]
    for key in ["epoch", "iter", "sha1", "outputs", "optimizer"]:
        test_config.pop(key, None)

    # overrides of deployed model and dataset, json required because thelper dumps config
    test_dataset = dataset_config_override["name"]
    test_config["config"]["name"] = model_config_override["name"]
    test_config["config"]["datasets"] = {
        test_dataset: {
            "type": "{}.{}".format(BatchTestPatchesDatasetLoader.__module__, BatchTestPatchesDatasetLoader.__name__),
            "params": dataset_config_override.json()
        }
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
        test_dataset: 1.0    # use every single test dataset patch
    }
    trainer["use_tbx"] = False
    trainer["type"] = "{}.{}".format(thelper.train.classif.ImageClassifTrainer.__module__,
                                     thelper.train.classif.ImageClassifTrainer.__name__)
    for key in ["device", "train_device", "optimization", "monitor"]:
        trainer.pop(key, None)

    # enforce multiprocessing workers count according to settings
    # note:
    #   job worker process must be non-daemonic to allow data loader workers spawning
    # see:
    #   ``geoimagenet_ml.api.routes.processes.utils.process_ml_job_runner`` for worker setup
    loaders["workers"] = int(settings.get('geoimagenet_ml.ml.data_loader_workers', 0))

    # override metrics to retrieve only raw predictions
    trainer["metrics"] = {
        "predictions": {
            "type": "thelper.optim.metrics.RawPredictions",
        },
        "top_1_accuracy": {
            "type": "thelper.optim.metrics.CategoryAccuracy",
            "params": {
                "top_k": 1,
            }
        },
        "top_5_accuracy": {
            "type": "thelper.optim.metrics.CategoryAccuracy",
            "params": {
                "top_k": 5,
            }
        },
        "mean_absolute_error": {
            "type": "thelper.optim.metrics.MeanAbsoluteError",
            "params": {
                "reduction": "mean"
            }
        },
        "mean_squared_error": {
            "type": "thelper.optim.metrics.MeanSquaredError",
            "params": {
                "reduction": "mean"
            }
        }
    }
    return test_config


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
    # TODO: validate format
    return taxo


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
    If ``crop_fixed_size`` is provided, fixed sized patches are afterwards created by cropping accordingly with
    dimensions of each patch's annotation coordinates. Both `raw` and `crop` patches are preserved in this case.

    Created patches for the batch are then split into train/test sets per corresponding ``taxonomy_class_id``.

    .. note::

        - ``annotations_meta`` and ``taxonomy_meta`` formats are expected to resolve with provided example links.

    .. seealso::

        - `GeoImageNet API` example: https://geoimagenetdev.crim.ca/api/v1/batches/annotations
        - #TODO: add example taxonomy link (GEOIM-161)

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
    patches_crop = [(None, "raw")]
    if isinstance(crop_fixed_size, (int, float)):
        update_func("fixed sized crops [{}] also selected for creation".format(crop_fixed_size), start_percent)
        patches_crop.append((crop_fixed_size, "fixed"))

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
            if not isinstance(patch_info, dict) or "crops" not in patch_info:
                raise RuntimeError("Failed to retrieve presumably existing patch from previous batch (feature: {})"
                                   .format(feature["id"]))
            # copy information, but replace patch copies
            for i_crop, _ in enumerate(patch_info["crops"]):
                old_patch_path = patch_info["crops"][i_crop]["path"]
                new_patch_path = old_patch_path.replace(dataset_latest.path, dataset_container.path)
                if not new_patch_path.startswith(dataset_container.path):
                    raise RuntimeError("Invalid patch path from copy. Expected base: '{}', but got: '{}'"
                                       .format(dataset_container.path, new_patch_path))
                patch_info["crops"][i_crop]["path"] = new_patch_path
                shutil.copy(old_patch_path, new_patch_path)
                dataset_container.files.append(new_patch_path)
            dataset_container.data["patches"].append(patch_info)
            # update counter with previously selected split set
            select_split(train_test_splits, patch_info["class"], name=patch_info["split"])

        # new patch creation from feature specification, generate metadata and randomly select split
        else:
            raster_data = find_best_match_raster(rasters_data, feature)
            crop_class_id = feature.get("properties", {}).get("taxonomy_class_id")
            dataset_container.data["patches"].append({
                "crops": [],  # updated gradually after
                "image": raster_data["file_path"],
                "class": crop_class_id,
                "split": select_split(train_test_splits, crop_class_id),
                "feature": feature.get("id"),
            })

            for crop_size, crop_name in patches_crop:
                crop, _, bbox = process_feature_crop(feature["geometry"], srs, raster_data, crop_size)
                if crop is not None:
                    if crop.ndim < 3 or crop.shape[2] != raster_data["band_count"]:
                        raise AssertionError("bad crop channel size")
                    output_geotransform = list(raster_data["offset_geotransform"])
                    output_geotransform[0], output_geotransform[3] = bbox[0], bbox[1]
                    output_driver = osgeo.gdal.GetDriverByName("GTiff")
                    output_name = get_sane_name("{}_{}".format(feature["id"], crop_name), assert_invalid=False)
                    output_path = os.path.join(dataset_container.path, "{}.tif".format(output_name))
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
                    dataset_container.data["patches"][-1]["crops"].append({
                        "type": crop_name,
                        "path": output_path,
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
