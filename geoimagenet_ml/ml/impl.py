# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# FIXME:
#   Because of unresolved reference caused by GDAL, only `geoimagenet_ml.ml.utils` should import
#   it directly. Otherwise, any method in `geoimagenet_ml.ml.impl` should import GDAL inside the
#   corresponding function using it to avoid import error from elsewhere, and should be executed
#   only form a celery worker that knows how to map the library without breaking other imports.
#   |
#   see about unresolved reference error:
#       - https://github.com/conda-forge/libgdal-feedstock/pull/33
#       - https://github.com/conda-forge/fiona-feedstock/issues/68
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from geoimagenet_ml.utils import ClassCounter
from six.moves.urllib.parse import urlparse
from six.moves.urllib.request import urlopen
from copy import deepcopy
from io import BytesIO
import requests
import logging
import random
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
        Any, AnyStr, Callable, List, Tuple, Union, OptionType, JSON, SettingsType, Number, Optional,
    )

LOGGER = logging.getLogger(__name__)


def load_model(model_file):
    # type: (Union[Any, AnyStr]) -> Tuple[bool, OptionType, Union[BytesIO, None], Union[Exception, None]]
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
    save_dir = os.path.join(settings.get('geoimagenet_ml.api.models_path'), model.uuid)
    _, _, _, test_loader = thelper.data.utils.create_loaders(test_config["config"], save_dir=save_dir)
    task = thelper.tasks.utils.create_task(model_checkpoint_config["task"])   # enforce model task instead of dataset
    model = thelper.nn.create_model(test_config["config"], task, save_dir=save_dir, ckptdata=model_checkpoint_config)
    config = test_config["config"]
    loaders = None, None, test_loader   # type: thelper.typedefs.MultiLoaderType

    # session name as Job UUID will write data under '<geoimagenet_ml.api.models_path>/<model-UUID>/output/<job-UUID>/'
    trainer = thelper.train.create_trainer(job.uuid, save_dir, config, model, loaders, model_checkpoint_config)
    return trainer


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

    # overrides of deployed model and dataset
    test_config["config"]["name"] = model_config_override["name"]
    test_config["config"]["datasets"] = {
        dataset_config_override["name"]: dataset_config_override.params
    }

    # back-compatibility replacement
    test_config["config"]["loaders"] = test_config["config"].pop("data_config", test_config["config"].get("loaders"))
    if "loaders" not in test_config["config"]:
        raise ValueError("Missing 'loaders' configuration from model checkpoint.")

    dataset = test_config["config"]["datasets"][dataset_config_override["name"]]    # type: JSON
    loaders = test_config["config"]["loaders"]  # type: JSON
    trainer = test_config["config"]["trainer"]  # type: JSON

    # adjust root dir of dataset location to match version deployed on server
    dataset["params"]["root"] = dataset_config_override.path

    # remove categories to match model outputs defined during training task
    for key in ["category"]:
        dataset["params"].pop(key, None)

    # remove additional unnecessary sub-parts or error-prone configurations
    for key in ["sampler", "train_augments", "train_split", "valid_split"]:
        loaders.pop(key, None)

    # override required values with modified parameters and remove error-prone configurations
    loaders["test_split"] = {
        dataset_config_override["name"]: 1.0
    }
    trainer["use_tbx"] = False
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
            "type": "thelper.optim.RawPredictions",
        }
    }
    return test_config


def retrieve_annotations(geojson_urls):
    # type: (List[AnyStr]) -> List[JSON]
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


def create_batch_patches(annotations_meta,      # type: List[JSON]
                         raster_search_paths,   # type: List[AnyStr]
                         dataset_store,         # type: DatasetStore
                         dataset_container,     # type: Dataset
                         dataset_latest,        # type: Optional[Dataset]
                         dataset_update_count,  # type: int
                         crop_fixed_size,       # type: Optional[int]
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

    .. seealso::
        - `GeoImageNet API` format: https://geoimagenetdev.crim.ca/api/v1/ui/#/paths/~1batches/post
        - `GeoImageNet API` example: https://geoimagenetdev.crim.ca/api/v1/batches?taxonomy_id=1

    :returns: updated ``dataset_container`` with metadata of created patches.
    """
    # FIXME:
    #   imports that require libraries dynamically loaded
    #   only celery running the process is setup to load them properly
    from geoimagenet_ml.ml.utils import parse_rasters, parse_geojson, parse_coordinate_system, process_feature
    import osgeo.gdal

    def fix_raster_search_name(name):
        return name.replace("8bits", "xbits").replace("16bits", "xbits")

    def select_split(splits, class_id, name=None):
        # type: (List[Tuple[AnyStr, ClassCounter]], Union[int, AnyStr], Optional[AnyStr]) -> AnyStr
        """
        Selects a split set from ``splits`` and decreases ``class_id`` in its corresponding counter
        if any 'instance' is left for selection within that class (counter didn't reach 0).
        Otherwise, the counter of the other split decreased and its name is returned instead.

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
    #   - need to handle different CRS for GeoTransform
    if len(annotations_meta) != 1:
        raise NotImplementedError("Multiple GeoJSON parsing not implemented.")
    annotations_meta = annotations_meta[0]

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
        new_features = list(provided_features - set(mapped_features.keys()))
    else:
        update_func("creating batch patches from new dataset (not incremental)", start_percent)
        mapped_features = dict()
        new_features = []

    train_counter, test_counter = category_counter.split(train_test_ratio)
    train_test_splits = [('train', train_counter), ('test', test_counter)]
    patches_crop = [(None, 'raw')]
    if isinstance(crop_fixed_size, int):
        update_func("fixed sized crops [{}] also selected for creation".format(crop_fixed_size), start_percent)
        patches_crop.append((crop_fixed_size, 'fixed'))

    last_progress_offset, progress_scale = 0, int(patch_percent - start_percent / len(features))
    dataset_container.data["patches"] = list()
    for feature_idx, feature in enumerate(features):
        progress_offset = int(start_percent + feature_idx * progress_scale)
        if progress_offset != last_progress_offset:
            last_progress_offset = progress_offset
            update_func("extracting patches for batch creation", progress_offset)

        # transfer patch data from previous batch, preserve selected split
        if feature["id"] not in new_features:
            patch_info = mapped_features.get(feature["id"])
            if not patch_info:
                raise RuntimeError("Failed to retrieve presumably existing patch from previous batch (feature: {})"
                                   .format(feature["id"]))
            dataset_container.data["patches"].append(deepcopy(patch_info))
            # update counter with previously selected split set
            select_split(train_test_splits, patch_info["class"], name=patch_info["split"])

        # new patch creation from feature specification, generate metadata and randomly select split
        else:
            feature_geometry = feature["geometry"]
            raster_data = None
            if "properties" in feature and "image_name" in feature["properties"]:
                raster_name = fix_raster_search_name(feature["properties"]["image_name"])
                for data in rasters_data:
                    target_path = fix_raster_search_name(data["filepath"])
                    if raster_name in target_path:
                        raster_data = data
                        break
            if raster_data is None:
                for data in rasters_data:
                    if data["global_roi"].contains(feature_geometry):
                        raster_data = data
                        break
                if raster_data is None:
                    raise AssertionError("could not find proper raster for feature '{}'".format(feature["id"]))

            crop_class_id = feature.get("properties", {}).get("taxonomy_class_id")
            dataset_container.data["patches"].append({
                "crops": [],  # updated gradually after
                "image": feature.get("properties", {}).get("image_name"),
                "class": crop_class_id,
                "split": select_split(train_test_splits, crop_class_id),
                "feature": feature.get("id"),
            })

            for crop_size, crop_name in patches_crop:
                crop, crop_inv, bbox = process_feature(feature_geometry, srs, raster_data, crop_size)
                if crop is not None:
                    if crop.ndim < 3 or crop.shape[2] != raster_data["bandcount"]:
                        raise AssertionError("bad crop channel size")
                    output_geotransform = list(raster_data["offset_geotransform"])
                    output_geotransform[0], output_geotransform[3] = bbox[0], bbox[1]
                    output_driver = osgeo.gdal.GetDriverByName("GTiff")
                    output_name = "{}_{}".format(feature["id"], crop_name)
                    output_path = os.path.join(dataset_container.path, "{}.tif".format(output_name))
                    if os.path.exists(output_path):
                        msg = "Output path [{}] already exists but is expected to not exist.".format(output_path)
                        update_func(msg + " Removing...", logging.WARNING)
                        os.remove(output_path)  # gdal raster creation fails if file already exists
                    output_dataset = output_driver.Create(output_path, crop.shape[1], crop.shape[0],
                                                          crop.shape[2], raster_data["datatype"])
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
                        "datatype": raster_data["datatype"],
                        "coordinates": output_geotransform,
                    })
        if feature_idx // dataset_update_count:
            update_func("updating dataset definition (checkpoint: {}/{})"
                        .format(feature_idx, len(features)), patch_percent)
            dataset_store.save_dataset(dataset_container)

    update_func("generating patches archived file", final_percent)
    dataset_container.zip()

    update_func("updating completed dataset definition", final_percent)
    dataset_container.mark_finished()
    return dataset_store.save_dataset(dataset_container)
