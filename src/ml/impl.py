from geoimagenet_ml.typedefs import (
    Any, AnyStr, Tuple, Union, OptionDict, JsonDict, SettingDict, TYPE_CHECKING
)
from six.moves.urllib.parse import urlparse
from six.moves.urllib.request import urlopen
from copy import deepcopy
from io import BytesIO
import six
import ssl
import os
# noinspection PyPackageRequirements
import thelper

if TYPE_CHECKING:
    from geoimagenet_ml.store.datatypes import Job, Model, Dataset


def load_model(model_file):
    # type: (Union[Any, AnyStr]) -> Tuple[bool, OptionDict, Union[BytesIO, None], Union[Exception, None]]
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
    # type: (Job, JsonDict, Model, Dataset, SettingDict) -> thelper.train.Trainer
    """
    Obtains a trainer specialized for testing data predictions using the provided model checkpoint and dataset loader.
    """
    test_config = test_loader_from_configs(model_checkpoint_config, model, dataset, settings)
    save_dir = os.path.join(settings.get('geoimagenet_ml.api.models_path'), model.uuid)
    _, _, _, test_loader = thelper.data.utils.create_loaders(test_config["config"], save_dir=save_dir)
    task = thelper.tasks.utils.create_task(model_checkpoint_config["task"])   # enforce model task instead of dataset
    model = thelper.nn.create_model(test_config["config"], task, save_dir=save_dir, ckptdata=model_checkpoint_config)
    config = test_config["config"]
    loaders = None, None, test_loader

    # session name as Job UUID will write data under '<geoimagenet_ml.api.models_path>/<model-UUID>/output/<job-UUID>/'
    trainer = thelper.train.create_trainer(job.uuid, save_dir, config, model, loaders, model_checkpoint_config)
    return trainer


def test_loader_from_configs(model_checkpoint_config, model_config_override, dataset_config_override, settings):
    # type: (JsonDict, Model, Dataset, SettingDict) -> JsonDict
    """
    Obtains a simplified version of the configuration for 'test' task corresponding to the model and dataset.
    Removes parameters from the original file that would require additional unnecessary operations other than testing.
    Overrides checkpoint training configurations, model name and datasets to enforce with the ones passed.
    """

    # transfer required parts, omit training specific values or error-prone configurations
    test_config = deepcopy(model_checkpoint_config)     # type: JsonDict
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

    dataset = test_config["config"]["datasets"][dataset_config_override["name"]]    # type: JsonDict
    loaders = test_config["config"]["loaders"]  # type: JsonDict
    trainer = test_config["config"]["trainer"]  # type: JsonDict

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
