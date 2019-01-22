import json
import logging
import warnings
import os
import six

import ccfb.ml.utils
# noinspection PyPackageRequirements
import cv2 as cv
import numpy as np
# noinspection PyPackageRequirements
import thelper.data

logger = logging.getLogger(__name__)


class TreeBarkDataset(thelper.data.ImageDataset):

    def __init__(self, config=None, transforms=None):
        super(TreeBarkDataset, self).__init__(config=config, transforms=transforms)
        # each image should be associated with a mask in the 'masks' directory (next to root)
        # (masks & metadata should also be created by the image segmentation annotator app)
        self.mask_root = os.path.join(self.root, "masks")
        if not os.path.isdir(self.mask_root):
            raise AssertionError("invalid mask root dir '%s'" % self.mask_root)
        mask_root_metadata = os.path.join(self.mask_root, "metadata.log")
        if not os.path.isfile(mask_root_metadata):
            raise AssertionError("could not locate mask metadata file")
        self.mask_metadata = json.load(open(mask_root_metadata))
        mask_names = [os.path.basename(eval(sample)["path"]) for sample in self.mask_metadata["samples"]]
        self.mask_key = thelper.utils.get_key_def("mask_key", config, "mask")
        self.mask_path_key = thelper.utils.get_key_def("mask_path_key", config, "mask_path")
        self.mask_combine_transf = thelper.utils.str2bool(
            thelper.utils.get_key_def("mask_combine_transf", config, True)
        )
        thelper.data.logger.debug("parsing ouareau tree bark snapshot masks at '%s'..." % self.mask_root)
        for sample_idx in reversed(range(len(self.samples))):
            image_name = os.path.basename(self.samples[sample_idx][self.path_key])
            if image_name not in mask_names:
                self.samples.pop(sample_idx)
            else:
                mask_index = mask_names.index(image_name)
                mask_path = os.path.join(self.mask_root, "%06d.png" % mask_index)
                mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None
                if mask is None or np.count_nonzero(mask) == 0:
                    self.samples.pop(sample_idx)
                else:
                    self.samples[sample_idx][self.mask_path_key] = mask_path
        thelper.data.logger.debug("ouareau bark dataset: parsed %d samples" % len(self.samples))
        meta_keys = [self.path_key, self.idx_key, self.mask_key, self.mask_path_key]
        self.task = thelper.tasks.Task(self.image_key, None, meta_keys)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise AssertionError("sample index is out-of-range")
        image_path = self.samples[idx][self.path_key]
        image = cv.imread(image_path)
        if image is None:
            raise AssertionError("invalid image at '%s'" % image_path)
        mask_path = self.samples[idx][self.mask_path_key]
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE) if mask_path else None
        if self.transforms:
            if mask is not None and self.mask_combine_transf:
                image = self.transforms([image, mask])
            else:
                image = self.transforms(image)
        return {
            self.path_key: image_path,
            self.mask_path_key: mask_path,
            self.image_key: image,
            self.mask_key: mask,
            self.idx_key: idx
        }

    def get_task(self):
        return self.task


class GeotagTreeDataset(thelper.data.ClassificationDataset):
    def __init__(self, config, transforms=None):
        if "root" not in config or not isinstance(config["root"], six.string_types):
            warnings.warn("using default root dir: '{!s}'".format(os.path.abspath('.')))
            root = "./"
        else:
            root = config["root"]
        if not os.path.exists(root) or not os.path.isdir(root):
            raise AssertionError("dataset root folder at '%s' does not exist" % root)
        logger.debug("initializing geotagged tree dataset at path '%s'" % root)
        self.crop_img_size = int(config["crop_img_size"]) if "crop_img_size" in config else None
        self.crop_real_size = float(config["crop_real_size"]) if "crop_real_size" in config else None
        if (self.crop_img_size and self.crop_img_size <= 0) or (self.crop_real_size and self.crop_real_size <= 0):
            raise AssertionError("crop size must be positive")
        self.scaling_factor = thelper.utils.get_key_def("scaling_factor", config, default=1.0)
        if self.scaling_factor <= 0:
            raise AssertionError("invalid scaling factor (must be greater than zero)")
        if "rasterfile" not in config or not config["rasterfile"]:
            raise AssertionError("missing 'rasterfile' field in geotagged tree dataset config")
        self.norm_mean = np.asarray(config["norm_mean"]) if "norm_mean" in config else None
        self.norm_std = np.asarray(config["norm_std"]) if "norm_std" in config else None
        self.norm = self.norm_mean is not None and self.norm_std is not None
        rasterfile_paths = thelper.utils.get_file_paths(config["rasterfile"], root, allow_glob=True)
        raster_data = ccfb.ml.utils.parse_rasters(rasterfile_paths)
        raster_local_coverages, raster_geotransforms, self.raster_metadata = raster_data
        if len(raster_local_coverages) != len(raster_geotransforms) or \
                len(raster_local_coverages) != len(rasterfile_paths):
            raise AssertionError("all raster data vectors should be the same size...")
        self.rasters_data = []
        for raster_idx in range(len(raster_local_coverages)):
            logger.debug("parsed file '%s'" % rasterfile_paths[raster_idx])
            self.rasters_data.append({
                "roi": raster_local_coverages[raster_idx],
                "geotransform": raster_geotransforms[raster_idx],
                "filepath": rasterfile_paths[raster_idx]
            })
        logger.debug("parsed %d raster files" % len(self.rasters_data))
        if "shapefile" not in config or not config["shapefile"]:
            raise AssertionError("missing 'shapefile' field in geotagged tree dataset config")
        if "category_field_name" not in config:
            raise AssertionError("missing 'category_field_name' field in geotagged tree dataset config")
        target_category = config["category"] if "category" in config else None
        if target_category:
            logger.debug("will target feature category '%s'" % target_category)
        shapefile_paths = thelper.utils.get_file_paths(config["shapefile"], root, allow_glob=True)
        features, category_counter, feat_count = [], None, 0
        for shapefile_path in shapefile_paths:
            curr_features, category_counter, feat_count = ccfb.ml.utils.parse_shapefile(
                shapefile_path,
                self.raster_metadata["srs"],
                config["category_field_name"],
                count_offset=feat_count,
                roi=self.raster_metadata["roi"],
                target_category=target_category,
                category_counter=category_counter,
                only_polygon=False
            )
            features += curr_features
        real_categories = []
        for feature in features:
            category = feature["category"]
            if category in ccfb.ml.utils.SPECIES_MAP and ccfb.ml.utils.SPECIES_MAP[category][3] is not None:
                feature["category"] = ccfb.ml.utils.SPECIES_MAP[category][3]
                if feature["category"] not in real_categories:
                    real_categories.append(feature["category"])
            else:
                feature["category"] = None
        self.image_key = thelper.utils.get_key_def("image_key", config, "image")
        self.label_key = thelper.utils.get_key_def("label_key", config, "category")
        self.id_key = thelper.utils.get_key_def("id_key", config, "id")
        self.geo_tag_key = thelper.utils.get_key_def("geo_tag_key", config, "geo_tag")
        self.meta_keys = [self.id_key, self.geo_tag_key]
        super(GeotagTreeDataset, self).__init__(real_categories, self.image_key, self.label_key,
                                                meta_keys=self.meta_keys, config=config, transforms=transforms)
        logger.debug("validating features wrt surface constraints & roi bounds...")
        self.samples = []
        for feature in features:
            roi, _, _, _, _ = ccfb.ml.utils.get_feature_roi(feature["geometry"], self.raster_metadata,
                                                            self.crop_img_size, self.crop_real_size)
            if self.raster_metadata["roi"].contains(roi):
                if feature["category"] is not None:
                    self.samples.append(feature)
        logger.debug("retained %d samples after checking constraints" % len(self.samples))

    def __getitem__(self, idx):
        feature = self.samples[idx]
        if not feature["clean"]:
            raise AssertionError("should have been filtered earlier...")
        feature_geometry = feature["geometry"]
        logger.debug("processing feature [%d/%d] (name:'%s')" % (idx + 1, len(self.samples), feature["id"]))
        crop, crop_inv, bbox = ccfb.ml.utils.process_feature(
            feature_geometry, self.rasters_data, self.raster_metadata,
            crop_img_size=self.crop_img_size, crop_real_size=self.crop_real_size
        )
        if crop is None:
            raise AssertionError("should have cleaned up features before processing stage!")
        image = np.where(crop.mask, crop_inv.data, crop.data)
        if self.norm:
            image = ((image.astype(np.float32) - self.norm_mean) / self.norm_std)
        # keep only NIR-RGB image from 6-channel (BGR-RE-NIR) input
        image = image[..., [4, 2, 1, 0]]
        if self.scaling_factor != 1:
            image = cv.resize(image, (0, 0), fx=self.scaling_factor, fy=self.scaling_factor,
                              interpolation=cv.INTER_AREA)
        # PIL is some hot garbage, meaning simple operations are broken w/ 16 bit images (e.g. transpose)
        image = image.astype(np.float32)  # gotta force convert, even thought we might lose transform speed...
        # normalize only the image here (manually, as transforms will also apply to mask)
        # test_image = np.copy(image)
        # test_image = cv.normalize(test_image, None, 0.0, 1.0, cv.NORM_MINMAX)
        # cv.imshow("image_pre", cv.resize(test_image, (512, 512), interpolation=cv.INTER_NEAREST))
        if self.transforms:
            # note: cannot transform mask in parallel until augmentor refactoring w/ sample keys
            image = self.transforms(image)
        # test_image = np.copy(np.transpose(image.clone().numpy(), (1, 2, 0))[..., :-1])
        # test_image = cv.normalize(test_image, None, 0.0, 1.0, cv.NORM_MINMAX)
        # cv.imshow("image_post", cv.resize(test_image, (512, 512), interpolation=cv.INTER_NEAREST))
        # cv.waitKey(0)
        return {
            self.image_key: image,
            self.label_key: feature["category"],
            self.id_key: "ouareau"+feature["id"],
            self.geo_tag_key: [feature_geometry.x, feature_geometry.y, feature_geometry.z]
        }
