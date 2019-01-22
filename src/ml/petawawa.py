import os
import math
import logging
import numpy as np
import cv2 as cv
import thelper.data
import thelper.transforms as tt
import ccfb.ml.utils as mu

logger = logging.getLogger(__name__)


class CenteredCrownDataset(thelper.data.ClassificationDataset):
    def __init__(self, name, config, transforms=None):
        thelper.data.logger.info("pre-initializing tree crown dataset")
        if not self.root:
            self.root = "./"
        if not os.path.exists(self.root) or not os.path.isdir(self.root):
            raise AssertionError("dataset root folder at '%s' does not exist" % self.root)
        logger.info("initializing centered crown dataset at path '%s'" % self.root)
        self.crop_img_size = int(config["crop_img_size"]) if "crop_img_size" in config else None
        self.crop_real_size = float(config["crop_real_size"]) if "crop_real_size" in config else None
        if (self.crop_img_size and self.crop_img_size <= 0) or (self.crop_real_size and self.crop_real_size <= 0):
            raise AssertionError("crop size must be positive")
        self.area_min = float(config["area_min"]) if "area_min" in config else 0
        self.area_max = float(config["area_max"]) if "area_max" in config else math.inf
        if (self.area_min and self.area_min < 0) or (self.area_max and self.area_max < 0):
            raise AssertionError("minimum/maximum feature area threshold must be positive or null")
        if self.area_min > self.area_max:
            raise AssertionError("minimum feature area must be smaller than maximum feature area")
        self.uncertain_flags = config["uncertain_flags"].split(",") if "uncertain_flags" in config else []
        if "rasterfile" not in config or not config["rasterfile"]:
            raise AssertionError("missing 'rasterfile' field in centered crown dataset config")
        default_srs = config["default_srs"] if "default_srs" in config else None
        self.norm_mean = np.asarray(config["norm_mean"] if "norm_mean" in config else [0, 0, 0, 0]).astype(np.float32)
        self.norm_std = np.asarray(config["norm_std"] if "norm_std" in config else [1, 1, 1, 1]).astype(np.float32)
        rasterfile_paths = thelper.utils.get_file_paths(config["rasterfile"], self.root, allow_glob=True)
        raster_data = mu.parse_rasters(rasterfile_paths, default_srs)
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
        logger.info("parsed %d raster files" % len(self.rasters_data))
        if "shapefile" not in config or not config["shapefile"]:
            raise AssertionError("missing 'shapefile' field in centered crown dataset config")
        shapefile_paths = thelper.utils.get_file_paths(config["shapefile"], self.root, can_be_dir=True)
        if len(shapefile_paths) != 1:
            raise AssertionError("currently expecting only one shape file for all rasters")
        if "category_field_name" not in config:
            raise AssertionError("missing 'category_field_name' field in centered crown dataset config")
        if "identifier_field_name" not in config:
            raise AssertionError("missing 'identifier_field_name' field in centered crown dataset config")
        if "uncertain_flags" not in config:
            raise AssertionError("missing 'uncertain_flags' field in centered crown dataset config")
        target_category = config["category"] if "category" in config else None
        if target_category:
            logger.debug("will target feature category '%s'" % target_category)
        target_id = config["feature_id"] if "feature_id" in config else None
        if target_id:
            logger.debug("will target feature id '%s'" % target_id)
        self.features, self.category_counter = mu.parse_shapefile(
            shapefile_paths[0],
            self.raster_metadata["srs"],
            config["category_field_name"],
            config["identifier_field_name"],
            config["uncertain_flags"].split(","),
            roi=self.raster_metadata["roi"],
            target_category=target_category,
            target_id=target_id
        )
        logger.info("parsed %d clean features" % len(self.features))
        self.input_key = "image"
        self.label_key = "category"
        self.meta_keys = ["mask", "id", "geo_bbox"]
        super(CenteredCrownDataset, self).__init__(
            name, list(self.category_counter.keys()), self.input_key, self.label_key,
            meta_keys=self.meta_keys, config=config, transforms=transforms, bypass_deepcopy=True)
        logger.debug("validating features wrt surface constraints & roi bounds...")
        self.samples = [
            feature for feature in self.features if (
                    not (feature["geometry"].area < self.area_min or feature["geometry"].area > self.area_max)
                    # and mu.validate_feature_roi(feature["geometry"], self.raster_metadata,
                    #                             self.crop_img_size, self.crop_real_size)
            )
        ]
        logger.info("retained %d features after checking constraints" % len(self.features))

    def __getitem__(self, idx):
        feature = self.samples[idx]
        if not feature["clean"]:
            raise AssertionError("should have been filtered earlier...")
        feature_geometry = feature["geometry"]
        logger.debug("processing feature [%d/%d] (name:'%s')" % (idx + 1, len(self.features), feature["id"]))
        crop, crop_inv, bbox = mu.process_feature(feature_geometry, self.rasters_data, self.raster_metadata,
                                                  crop_img_size=self.crop_img_size, crop_real_size=self.crop_real_size)
        if crop is None:
            raise AssertionError("should have cleaned up features before processing stage!")
        # here, we reassemble the image and provide a mask instead of the masked arrays
        image = np.where(crop.mask, crop_inv.file, crop.file)
        # normalize only the image here (manually, as transforms will also apply to mask)
        image = ((image.astype(np.float32) - self.norm_mean) / self.norm_std)
        mask = np.expand_dims((np.any(np.bitwise_not(crop.mask), axis=2).astype(np.float32) - 0.5) * 2, 2)
        if self.transforms:
            # might not work for both if transforms not in list-compat interface
            # (thelper.transforms.ImageTransformWrapper)

            # HACK:
            #   Process Augmentor specific operations separately to allow special PIL operations
            aug_indices = [i for i, t in enumerate(self.transforms.transforms) if isinstance(t, tt.AugmentorWrapper)]
            if len(aug_indices):
                aug_oper_transforms = [self.transforms.transforms.pop(i) for i in aug_indices]
                aug_transforms = tt.Compose(aug_oper_transforms)
                # PIL only supports 1/3-channels transforms, split 4-channels in single images and merge results
                # Only geometric operations are applicable (no contrast, brightness, color dependant transforms)
                b, g, r, nir = cv.split(image)
                m = mask.reshape(mask.shape[:2])
                [b, g, r, nir, m] = aug_transforms([b, g, r, nir, m])
                image = cv.merge((b, g, r, nir))
                mask = np.expand_dims(m, axis=2)

            [image, mask] = self.transforms([image, mask])

        return {"image": image, "mask": mask, "category": feature["category"], "id": feature["id"], "geo_bbox": bbox}
