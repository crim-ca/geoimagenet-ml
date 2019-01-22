import gdal
import os
import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from apscheduler.schedulers.background import BackgroundScheduler

from ccfb02.ml import utils as mu
from thelper import utils as tu

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main():
    VERBOSITY_LEVEL = 0
    ccfb02_viz_logger = logging.getLogger("thelper.src.viz")
    default_uncertain_flags = "?,sick,dry,damaged,wet,dead,clump,Clump,snag,isol,shape,REPLICA,shaded,&"
    ap = argparse.ArgumentParser(description="geotiff shape/raster processing app")
    ap.add_argument("rasterfile", help="relative or absolute path to the raster file(s) to process (accepts wildcard)")
    ap.add_argument("shapefile", help="relative or absolute path to the shape file to process")
    ap.add_argument("-d", "--display", default=False, action="store_true",
                    help="toggles whether to display scaled raster via opencv")
    ap.add_argument("-g", "--display-graphs", default=False, action="store_true",
                    help="toggles whether to display analysis graphs via matplotlib")
    ap.add_argument("--data-root", default=r'../data',
                    help="default dataset root for relative file paths")
    ap.add_argument("-s", "--display-size", default="500x500",
                    help="display size in WIDTHxHEIGHT format (default=500x500)")
    ap.add_argument("--default-spatial-ref", default="26918",
                    help="default EPSG spatial ref code to use for undefined geotiff inputs (fallback=26918)")
    ap.add_argument("-v", "--verbose", action="count", default=VERBOSITY_LEVEL,
                    help="set output verbosity level")
    crop_group = ap.add_mutually_exclusive_group()
    crop_group.add_argument("-p", "--crop-img-size",
                            help="specific image size (in pixels) to crop to")
    crop_group.add_argument("-r", "--crop-real-size",
                            help="specific terrain size (in meters) to crop to " +
                                 "(will be rounded to the raster resolution unit)")
    ap.add_argument("-n", "--normalize", action="store_true",
                    help="toggles whether pixel values should be normalized across all rasters")
    target_group = ap.add_mutually_exclusive_group()
    target_group.add_argument("-c", "--category",
                              help="specifies which category to extract features from (default = all)")
    target_group.add_argument("-i", "--feature-id",
                              help="specifies which feature(s) to extract (via unique id, comma-separated)")
    ap.add_argument("-o", "--output",
                    help="path to the directory where crops should be saved")
    ap.add_argument("--area-min", help="minimum feature area threshold for exporting crops")
    ap.add_argument("--area-max", help="minimum feature area threshold for exporting crops")
    ap.add_argument("--category-field-name", default="Species",
                    help="specifies the layer definition field name to use for feature category parsing " +
                         "(default=Species)")
    ap.add_argument("--identifier-field-name", default="ID_Number",
                    help="specifies the layer definition field name to use for feature id parsing (default=ID_Number)")
    ap.add_argument("--uncertain-flags", default=default_uncertain_flags,
                    help="comma-separated list of uncertain/unclean flags to detect in category names " +
                         "(default=specific to src)")
    args = ap.parse_args()
    osgeo.gdal.UseExceptions()

    # read program args
    VERBOSITY_LEVEL = args.verbose
    display_size_str = args.display_size.split('x')
    if len(display_size_str) != 2:
        raise AssertionError("bad display size formatting")
    display_size = tuple([max(int(s), 1) for s in display_size_str])
    if (args.crop_img_size and int(args.crop_img_size) <= 0) \
            or (args.crop_real_size and float(args.crop_real_size) <= 0):
        raise AssertionError("crop size must be positive")
    if args.output and not os.path.isdir(args.output):
        raise AssertionError("output path must point to an existing directory")
    if args.area_min and float(args.area_min) < 0:
        raise AssertionError("minimum feature area threshold must be positive or null")
    if args.area_max and float(args.area_max) <= 0:
        raise AssertionError("maximum feature area threshold must be positive")
    if args.area_min and args.area_max and float(args.area_min) > float(args.area_max):
        raise AssertionError("minimum feature area must be smaller than maximum feature area")
    uncertain_flags = args.uncertain_flags.split(",")

    # read raster file(s) properties
    rasterfile_paths = tu.get_file_paths(args.rasterfile, args.data_root, allow_glob=True)
    raster_local_coverages, raster_geotransforms, raster_metadata = \
        mu.parse_rasters(rasterfile_paths, args.default_spatial_ref, args.normalize)
    if len(raster_local_coverages) != len(raster_geotransforms) or len(raster_local_coverages) != len(rasterfile_paths):
        raise AssertionError("all raster data vectors should be the same size...")
    rasters_data = []
    for raster_idx in range(len(raster_local_coverages)):
        rasters_data.append({
            "roi": raster_local_coverages[raster_idx],
            "geotransform": raster_geotransforms[raster_idx],
            "filepath": rasterfile_paths[raster_idx]
        })

    # shapefile parsing & feature extraction
    shapefile_paths = tu.get_file_paths(args.shapefile, args.data_root, can_be_dir=True)
    if len(shapefile_paths) != 1:
        raise AssertionError("currently expecting only one shape file for all rasters")
    features, category_counter = mu.parse_shapefile(
        shapefile_paths[0],
        raster_metadata["srs"],
        args.category_field_name,
        args.identifier_field_name,
        uncertain_flags,
        roi=raster_metadata["roi"],
        target_category=args.category,
        target_id=args.feature_id
    )
    feature_count = len(features)
    if args.display_graphs and feature_count > 1:
        tu.draw_histogram([feature["geometry"].area for feature in features], xlabel=r"Feature Surface ($m^2$)")
        tu.draw_popbars(category_counter.keys(), category_counter.values(), "Category")
        plt.show()

    # feature processing (cropping) loop
    ccfb02_viz_logger.info("Cropping %d features..." % feature_count)
    feature_idx = 0
    scheduler = None
    crop_sizes = list()
    if VERBOSITY_LEVEL > 0 and feature_count > 1:
        scheduler = BackgroundScheduler()
        scheduler.add_job(lambda: print("\t[%d%%]" % mu.percent(feature_idx, feature_count), end="\r"),
                          "interval", seconds=3)
        scheduler.start()
    oob_crop_feature_count = 0
    for feature in features:
        if not feature["clean"]:
            raise AssertionError("should have been filtered earlier...")
        feature_idx += 1
        feature_geometry = feature["geometry"]
        # first, check size constraints (surface, roi)
        if args.area_min and feature_geometry.area < float(args.area_min):
            oob_crop_feature_count += 1
            continue
        if args.area_max and feature_geometry.area > float(args.area_max):
            oob_crop_feature_count += 1
            continue
        ccfb02_viz_logger.debug("processing [%d/%d] : '%s' [surface=%.1f]"
                                % (feature_idx + 1, feature_count, feature["id"], feature_geometry.area))
        crop, crop_inv, bbox = mu.process_feature(feature_geometry, rasters_data, raster_metadata,
                                                  crop_img_size=args.crop_img_size, crop_real_size=args.crop_real_size)
        if crop is None:
            oob_crop_feature_count += 1
            continue
        elif args.display:
            crop_sizes.append(crop.shape)
        if args.output:
            if crop.ndim < 3 or crop.shape[2] != raster_metadata["bandcount"]:
                raise AssertionError("bad crop channel size")
            output_geotransform = list(raster_metadata["offset_geotransform"])
            output_geotransform[0], output_geotransform[3] = bbox[0], bbox[1]
            output_dataset = osgeo.gdal.GetDriverByName("GTiff").Create(os.path.join(args.output, feature["id"] + ".tif"),
                                                                  crop.shape[1], crop.shape[0], crop.shape[2],
                                                                  raster_metadata["datatype"])
            for raster_band_idx in range(crop.shape[2]):
                output_dataset.GetRasterBand(raster_band_idx + 1).SetNoDataValue(0)
                output_dataset.GetRasterBand(raster_band_idx + 1).WriteArray(crop.file[:, :, raster_band_idx])
            output_dataset.SetProjection(raster_metadata["srs"].ExportToWkt())
            output_dataset.SetGeoTransform(output_geotransform)
            output_dataset = None  # close output fd
        if args.display:
            import cv2 as cv
            nchannels = 1 if crop.shape[2] < 3 else 3
            cv.imshow("crop", cv.resize(cv.cvtColor(crop.file[:, :, 0:nchannels], cv.COLOR_RGB2BGR),
                                        display_size, interpolation=cv.INTER_NEAREST))
            cv.imshow("crop_inv", cv.resize(cv.cvtColor(crop_inv.file[:, :, 0:nchannels], cv.COLOR_RGB2BGR),
                                            display_size, interpolation=cv.INTER_NEAREST))
            cv.waitKey(1 if VERBOSITY_LEVEL < 2 else 0)

    if scheduler:
        scheduler.shutdown()
    ccfb02_viz_logger.info("done!")
    if oob_crop_feature_count > 0:
        ccfb02_viz_logger.info("skipped %d crops due to size constraint" % oob_crop_feature_count)
    if args.display:
        min_crop = np.min(crop_sizes)
        max_crop = np.max(crop_sizes)
        mean_crop = np.mean(crop_sizes)
        stddev_crop = np.std(crop_sizes)
        ccfb02_viz_logger.info("Min crop size: {}".format(min_crop))
        ccfb02_viz_logger.info("Max crop size: {}".format(max_crop))
        ccfb02_viz_logger.info("Mean crop size: {}".format(mean_crop))
        ccfb02_viz_logger.info("StdDev crop size: {}".format(stddev_crop))


if __name__ == "__main__":
    main()
