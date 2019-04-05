from geoimagenet_ml.utils import ClassCounter, get_sane_name
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
        Any, AnyStr, Callable, List, Tuple, Union, OptionType, JSON, SettingsType, Number, Optional,
        FeatureType, RasterDataType
    )

# enforce GDAL exceptions (otherwise functions return None)
osgeo.gdal.UseExceptions()

LOGGER = logging.getLogger(__name__)


def load_model(model_file):
    # type: (Union[Any, AnyStr]) -> Tuple[bool, OptionType, Optional[BytesIO], Optional[Exception]]
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
    task = thelper.tasks.utils.create_task(model_checkpoint_config["task"])  # enforce model task instead of dataset
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
    #   - need to handle different CRS for GeoTransform
    if len(annotations_meta) != 1:
        raise NotImplementedError("Multiple GeoJSON parsing not implemented.")
    annotations_meta = annotations_meta[0]

    if not isinstance(dataset_update_count, int) or dataset_update_count < 1:
        raise AssertionError("invalid dataset update count value: {!s}".format(dataset_update_count))

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
    if isinstance(crop_fixed_size, int):
        update_func("fixed sized crops [{}] also selected for creation".format(crop_fixed_size), start_percent)
        patches_crop.append((crop_fixed_size, "fixed"))

    last_progress_offset, progress_scale = 0, int((patch_percent - start_percent) / len(features))
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
                        .format(feature_idx, len(features)), patch_percent)
            dataset_store.save_dataset(dataset_container)

    update_func("generating patches archived file", final_percent)
    dataset_container.zip()

    update_func("updating completed dataset definition", final_percent)
    dataset_container.mark_finished()
    return dataset_store.save_dataset(dataset_container)