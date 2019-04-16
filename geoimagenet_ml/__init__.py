import os
import logging
logger = logging.getLogger(__name__)

# define common root paths
GEOIMAGENET_ML_SRC_DIR = os.path.abspath(os.path.dirname(__file__))
GEOIMAGENET_ML_API_DIR = os.path.abspath(os.path.dirname(__file__))
GEOIMAGENET_ML_ROOT_DIR = os.path.abspath(os.path.dirname(GEOIMAGENET_ML_SRC_DIR))
GEOIMAGENET_ML_CONFIG_DIR = os.path.abspath(os.path.join(GEOIMAGENET_ML_SRC_DIR, "config"))
GEOIMAGENET_ML_CONFIG_INI = os.path.abspath(os.path.join(GEOIMAGENET_ML_SRC_DIR, "config", "ml.ini"))


def includeme(config):
    logger.info("Adding GeoImageNet ML...")
    config.include("geoimagenet_ml.api")
    config.include("geoimagenet_ml.config")
    config.include("geoimagenet_ml.ml")
    config.include("geoimagenet_ml.processes")
    config.include("geoimagenet_ml.store")
