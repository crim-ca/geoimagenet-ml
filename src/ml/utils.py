import collections
import logging
import math
import os

import affine
import osgeo.gdal
import matplotlib.pyplot as plt
import numpy as np
import ogr
import osr
import shapely
import shapely.geometry
import shapely.ops
import shapely.wkt

import thelper.utils as tu

logger = logging.getLogger(__name__)

NUMPY2GDAL_TYPE_CONV = {
    np.uint8: osgeo.gdal.GDT_Byte,
    np.int8: osgeo.gdal.GDT_Byte,
    np.uint16: osgeo.gdal.GDT_UInt16,
    np.int16: osgeo.gdal.GDT_Int16,
    np.uint32: osgeo.gdal.GDT_UInt32,
    np.int32: osgeo.gdal.GDT_Int32,
    np.float32: osgeo.gdal.GDT_Float32,
    np.float64: osgeo.gdal.GDT_Float64,
    np.complex64: osgeo.gdal.GDT_CFloat32,
    np.complex128: osgeo.gdal.GDT_CFloat64,
}

GDAL2NUMPY_TYPE_CONV = {
    osgeo.gdal.GDT_Byte: np.uint8,
    osgeo.gdal.GDT_UInt16: np.uint16,
    osgeo.gdal.GDT_Int16: np.int16,
    osgeo.gdal.GDT_UInt32: np.uint32,
    osgeo.gdal.GDT_Int32: np.int32,
    osgeo.gdal.GDT_Float32: np.float32,
    osgeo.gdal.GDT_Float64: np.float64,
    osgeo.gdal.GDT_CInt16: np.complex64,
    osgeo.gdal.GDT_CInt32: np.complex64,
    osgeo.gdal.GDT_CFloat32: np.complex64,
    osgeo.gdal.GDT_CFloat64: np.complex128
}

# noinspection SpellCheckingInspection
SPECIES_MAP = {
    # BARKNET_CODE: (LATIN, FRENCH, ENGLISH, NRCAN_CODE)
    "BOJ": ("betula alleghaniensis", "bouleau jaune", "yellow birch", "By"),
    "BOP": ("betula papyrifera", "bouleau a papier", "white birch", "Bw"),
    "CHR": ("quercus rubra", "chene rouge", "red oak", "Or"),
    "EPB": ("picea glauca", "epinette blanche", "white spruce", "Sw"),
    "EPN": ("picea mariana", "epinette noire", "black spruce", "Sb"),
    "EPO": ("picea abies", "epinette communne", "norway spruce", None),
    "EPR": ("picea rubens", "epinette rouge", "red spruce", "Sr"),
    "ERB": ("acer platanoides", "erable plane", "norway maple", None),
    "ERR": ("acer rubrum", "erable rouge", "red maple", "Mr"),
    "ERS": ("acer saccharum", "erable a sucre", "sugar maple", "Mh"),
    "FRA": ("fraxinus americana", "frene blanc", "white ash", "Aw"),
    "HEG": ("fagus grandifolia", "hetre a grandes feuilles", "american beech", "Be"),
    "MEL": ("larix laricina", "meleze laricin", "tamarack", "La"),
    "ORA": ("ulmus americana", "orme d'amerique", "american elm", None),
    "OSV": ("ostrya virginiana", "ostryer de virginie", "ironwood", "Id"),
    "PEG": ("populus grandidentata", "peuplier a grandes dents", "large-tooth aspen", "Alt"),
    "PET": ("populus tremuloides", "peuplier faux-tremble", "trembling aspen", "At"),  # also Pt?
    "PIB": ("pinus strobus", "pin blanc", "white pine", "Pw"),
    "PID": ("pinus rigida", "pin rigide", "pitch pine", None),
    "PIR": ("pinus resinosa", "pin rouge", "red pine", "Pr"),
    "PRU": ("tsuga canadensis", "pruche du canada", "eastern hemlock", "He"),
    "SAB": ("abies balsamea", "sapin baumier", "balsam fir", "Bf"),
    "THO": ("thuja occidentalis", "thuya occidental", "eastern white cedar", "Ce"),
    # missing from barknet: ("pinus banksiana", "Pj")
    # missing from nrcan: ("picea abies", "EPO"), ("acer platanoides", "ERB"), ("ulmus americana", "ORA"), ("pinus rigida", "PID")  # noqa E501
    # unknown from nrcan: "Ba", "Abl", "Fb", "Pb", "Ps", "Lt", "XP", "Iw", "Ow"
}


def get_pxcoord(geotransform, x, y):
    inv_transform = ~affine.Affine.from_gdal(*geotransform)
    return inv_transform * (x, y)


def get_geocoord(geotransform, x, y):
    # orig_x,res_x,skew_x,orig_y,skew_y,res_y = geotransform
    # return (orig_x+x*res_x+y*skew_x,orig_y+x*skew_y+y*res_y)
    return affine.Affine.from_gdal(*geotransform) * (float(x), float(y))


def get_geoextent(geotransform, x, y, cols, rows):
    tl = get_geocoord(geotransform, x, y)
    bl = get_geocoord(geotransform, x, y + rows)
    br = get_geocoord(geotransform, x + cols, y + rows)
    tr = get_geocoord(geotransform, x + cols, y)
    return [tl, bl, br, tr]


def reproject_coords(coords, src_srs, tgt_srs):
    trans_coords = []
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    for x, y in coords:
        x, y, z = transform.TransformPoint(x, y)
        trans_coords.append([x, y])
    return trans_coords


def get_polygon_geojson(geom):
    output = [{'type': 'Polygon', 'coordinates': [[]]}]
    ring = geom.GetGeometryRef(0)
    for pt_idx in range(ring.GetPointCount()):
        pt_x, pt_y, pt_z = ring.GetPoint(pt_idx)
        output[0]["coordinates"][0].append((pt_x, pt_y))
    return output


def append_inters_polygons(base_shape, cut_shape, cut_idx, hit_list, geom_list):
    inters_geometry = base_shape.intersection(cut_shape)
    if not inters_geometry.is_empty:
        if inters_geometry.geom_type == "Polygon":
            hit_list.append(cut_idx)
            geom_list.append(inters_geometry)
        elif inters_geometry.geom_type != "MultiPolygon":
            for inters_subgeometry in inters_geometry.geoms:
                if inters_subgeometry.geom_type != "Polygon":
                    raise AssertionError("expected polygon intersection between vector and raster file")
                hit_list.append(cut_idx)
                geom_list.append(inters_subgeometry)
        else:
            raise AssertionError("unexpected geometry type")


def percent(count, total):
    return int(count * 100 // total)


def parse_shapefile(shapefile_path, srs_destination, category_field, id_field=None, count_offset=None,
                    uncertain_flags=None, roi=None, target_category=None, target_id=None, category_counter=None,
                    only_polygon=True):
    uncertain_flags = [] if uncertain_flags is None else uncertain_flags
    shapefile_driver = ogr.GetDriverByName("ESRI Shapefile")
    shapefile = shapefile_driver.Open(shapefile_path, 0)
    if shapefile is None:
        raise AssertionError("could not open vector data file at '%s'" % shapefile_path)
    logger.debug("Shapefile vector metadata printing below... %s" % str(shapefile))
    if len(shapefile) != 1:
        raise AssertionError("expected one layer, got multiple")
    layer = shapefile.GetLayer()
    layer_def = layer.GetLayerDefn()
    logger.debug("layer name: %s" % str(layer.GetName()))
    logger.debug("field count: %s" % str(layer_def.GetFieldCount()))
    got_category_field = False
    for field_idx in range(layer_def.GetFieldCount()):
        field_name = layer_def.GetFieldDefn(field_idx).GetName()
        field_type_code = layer_def.GetFieldDefn(field_idx).GetType()
        field_type = layer_def.GetFieldDefn(field_idx).GetFieldTypeName(field_type_code)
        field_width = layer_def.GetFieldDefn(field_idx).GetWidth()
        precision = layer_def.GetFieldDefn(field_idx).GetPrecision()
        logger.debug("field %d: {name:\"%s\", type:\"%s\", length:%s, precision:%s}"
                     % (field_idx, field_name, field_type, str(field_width), str(precision)))
        got_category_field = got_category_field or field_name == category_field
    if not got_category_field:
        raise AssertionError("could not find layer definition field with name '%s' to parse categories"
                             % category_field)
    if isinstance(srs_destination, str):  # assume in wkt format
        srs_destination = osr.SpatialReference(srs_destination)
    elif isinstance(srs_destination, int):  # assume EPSG code
        _srs_destination = osr.SpatialReference()
        _srs_destination.ImportFromEPSG(srs_destination)
        srs_destination = _srs_destination
    shapes_srs_transform = osr.CoordinateTransformation(layer.GetSpatialRef(), srs_destination)
    feature_count = layer.GetFeatureCount()
    logger.info("total shapefile feature count: %s" % str(feature_count))
    oob_feature_count = 0
    features = []
    logger.debug("scanning parsed features for out-of-bounds cases...")
    feat_count = 0 if count_offset is None else count_offset
    for feature in layer:
        ogr_geometry = feature.GetGeometryRef()
        ogr_geometry.Transform(shapes_srs_transform)
        feature_geometry = shapely.wkt.loads(ogr_geometry.ExportToWkt())
        if roi and not roi.contains(feature_geometry):
            oob_feature_count += 1
        else:
            feature_id = feature.GetFieldAsString(id_field).strip() if id_field else str(feat_count)
            feature_category = feature.GetFieldAsString(category_field).strip()
            features.append({
                "id": feature_id,
                "category": feature_category,
                "geometry": feature_geometry,
            })
        feat_count += 1
    layer.ResetReading()
    logger.info("out-of-bounds features: %d  (%d%%)"
                % (oob_feature_count, int(oob_feature_count * 100 // feature_count)))
    unlabeled_feature_count = 0
    uncertain_feature_count = 0
    bad_shape_feature_count = 0
    if category_counter is None:
        category_counter = collections.Counter()
    for feature in features:
        if len(feature["category"]) == 0:
            unlabeled_feature_count += 1
            uncertain_feature_count += 1
            feature["clean"] = False
        elif any(flag in feature["category"] for flag in uncertain_flags):
            uncertain_feature_count += 1
            feature["clean"] = False
        elif only_polygon and feature["geometry"].geom_type != "Polygon":
            bad_shape_feature_count += 1
            feature["clean"] = False
        else:
            feature["clean"] = True
            category_counter[feature["category"]] += 1
    n_features = len(features)
    logger.info("bad shape features: %d  (%d%%)"
                % (bad_shape_feature_count, percent(bad_shape_feature_count, n_features)))
    logger.info("unlabeled features: %d  (%d%%)"
                % (unlabeled_feature_count, percent(unlabeled_feature_count, n_features)))
    logger.info("uncertain features: %d  (%d%%)"
                % (uncertain_feature_count, percent(uncertain_feature_count, n_features)))
    logger.info("clean features: %s" % sum(category_counter.values()))
    logger.info("unique+clean feature categories: %s" % len(category_counter.keys()))
    for cat in category_counter:
        logger.debug("  %s = %d" % (cat, category_counter[cat]))
    if target_category:
        if isinstance(target_category, str):
            target_category = target_category.split(",")
        if not isinstance(target_category, list):
            raise AssertionError("unexpected target category type")
        for tgt in target_category:
            if tgt in category_counter:
                category_percent = percent(category_counter[tgt], sum(category_counter.values()))
                logger.info("selected category '%s' raw feature count: %d  (%d%%)"
                            % (tgt, category_counter[tgt], category_percent))
        features = [feature for feature in features if feature["category"] in target_category and feature["clean"]]
        if not features:
            raise AssertionError("no clean feature found under category '%s'" % target_category)
    elif target_id:
        feature_ids = target_id.split(",")
        features = [feature for feature in features if feature["id"] in feature_ids]
        if not features:
            raise AssertionError("could not find any feature(s) with id '%s'" % target_id)
        features = [feature for feature in features if feature["clean"]]
        if not features:
            raise AssertionError("no clean feature(s) found with id '%s'" % target_id)
    else:
        features = [feature for feature in features if feature["clean"]]
        if not features:
            raise AssertionError("no clean feature(s) found in shapefile" % target_id)
    return features, category_counter, feat_count


def parse_rasters(rasterfile_paths, default_srs=None, normalize=False):
    raster_local_coverages = []
    raster_geotransforms = []
    raster_stats_map = []
    raster_srs = None
    raster_skew = None
    raster_resolution = None
    raster_bandcount = None
    raster_datatype = None
    raster_cols_max = 0
    raster_rows_max = 0
    for rasterfile_path in rasterfile_paths:
        rasterfile = osgeo.gdal.Open(rasterfile_path, osgeo.gdal.GA_ReadOnly)
        if rasterfile is None:
            raise AssertionError("could not open raster data file at '%s'" % rasterfile_path)
        logger.debug("Raster '%s' metadata printing below..." % rasterfile_path)
        logger.debug("%s" % str(rasterfile))
        logger.debug("%s" % str(rasterfile.GetMetadata()))
        logger.debug("band count: %s" % str(rasterfile.RasterCount))
        raster_cols_max = max(rasterfile.RasterXSize, raster_cols_max)
        raster_rows_max = max(rasterfile.RasterYSize, raster_rows_max)
        raster_geotransform = rasterfile.GetGeoTransform()
        raster_extent = get_geoextent(raster_geotransform, 0, 0, rasterfile.RasterXSize, rasterfile.RasterYSize)
        logger.debug("extent: %s" % str(raster_extent))
        raster_curr_srs = osr.SpatialReference()
        raster_curr_srs_str = rasterfile.GetProjectionRef()
        if "unknown" not in raster_curr_srs_str:
            raster_curr_srs.ImportFromWkt(raster_curr_srs_str)
        else:
            if default_srs:
                raster_curr_srs.ImportFromEPSG(int(default_srs))
            else:
                raise AssertionError("raster did not provide an srs, and no default EPSG srs provided")
        logger.debug("spatial ref:\n%s" % str(raster_curr_srs))
        if not raster_srs:
            raster_srs = raster_curr_srs
        elif not raster_srs.IsSame(raster_curr_srs):
            raise AssertionError("all input rasters should already be in the same spatial ref system")
        px_width, px_height = raster_geotransform[1], raster_geotransform[5]
        skew_x, skew_y = raster_geotransform[2], raster_geotransform[4]
        if not raster_resolution:
            raster_resolution = (px_width, px_height)
            raster_skew = (skew_x, skew_y)
        elif raster_resolution != (px_width, px_height):
            raise AssertionError("expected identical data resolutions in all bands & rasters")
        elif raster_skew != (skew_x, skew_y):
            raise AssertionError("expected identical grid skew in all bands & rasters")
        if not raster_bandcount:
            raster_bandcount = rasterfile.RasterCount
        elif raster_bandcount != rasterfile.RasterCount:
            raise AssertionError("expected identical band counts for all rasters")
        raster_bands_stats = []
        for raster_band_idx in range(raster_bandcount):
            curr_band = rasterfile.GetRasterBand(raster_band_idx + 1)  # offset, starts at 1
            if curr_band is None:
                raise AssertionError("found invalid raster band")
            # lines below cause crashes on python 3.6m on windows w/ gdal from precomp wheel
            # curr_band_stats = curr_band.GetStatistics(True,True)
            # if curr_band_stats is None:
            #    raise AssertionError("could not compute band statistics")
            if not raster_datatype:
                raster_datatype = curr_band.DataType
            elif raster_datatype != curr_band.DataType:
                raise AssertionError("expected identical data types in all bands & rasters")
            if normalize:
                logger.debug("computing band #%d statistics..." % (raster_band_idx + 1))
                band_array = curr_band.ReadAsArray()
                band_nodataval = curr_band.GetNoDataValue()
                band_ma = np.ma.array(band_array.astype(np.float32),
                                      mask=np.where(band_array != band_nodataval, False, True))
                raster_bands_stats.append([
                    np.ma.min(band_ma),
                    np.ma.max(band_ma),
                    np.ma.std(band_ma),
                    np.ma.mean(band_ma)])
        if normalize:
            raster_stats_map.append({"name": os.path.split(rasterfile_path)[1], "stats": raster_bands_stats})
        raster_local_coverage = shapely.geometry.Polygon([list(pt) for pt in raster_extent]).buffer(0.01)
        raster_local_coverages.append(raster_local_coverage)
        raster_geotransforms.append(raster_geotransform)
        # noinspection PyUnusedLocal
        rasterfile = None  # close input fd
    raster_offset_geotransform = (0, raster_resolution[0], raster_skew[0], 0, raster_skew[1], raster_resolution[1])
    raster_global_coverage = shapely.ops.cascaded_union(raster_local_coverages)
    if normalize:
        if not raster_stats_map:
            import pickle
            with open("raster_errbars_labels.pickle", "rb") as handle:
                raster_errbars_labels = pickle.load(handle)
            with open("raster_errbars_stats.pickle", "rb") as handle:
                raster_errbars_stats = pickle.load(handle)
        else:
            raster_errbars_labels = [raster_stats["name"] for raster_stats in raster_stats_map]
            raster_errbars_stats = [raster_stats["stats"] for raster_stats in raster_stats_map]
        raster_errbars_min = np.array([[band[:][0] for band in file] for file in raster_errbars_stats])
        raster_errbars_max = np.array([[band[:][1] for band in file] for file in raster_errbars_stats])
        raster_errbars_stddev = np.array([[band[:][2] for band in file] for file in raster_errbars_stats])
        raster_errbars_mean = np.array([[band[:][3] for band in file] for file in raster_errbars_stats])
        tu.draw_errbars(raster_errbars_labels, raster_errbars_min, raster_errbars_max, raster_errbars_stddev,
                        raster_errbars_mean, xlabel="Filename")
        logger.info("overall mean = %s" % str(np.mean(raster_errbars_mean, axis=0)))
        logger.info("overall stddev = %s" % str(np.mean(raster_errbars_stddev, axis=0)))
        # normalization impl below still missing, output is still in orig datatype without range modifications
        plt.show()
    common_metadata = {
        "srs": raster_srs.ExportToWkt(),
        "skew": raster_skew,
        "resolution": raster_resolution,
        "bandcount": raster_bandcount,
        "datatype": raster_datatype,
        "cols_max": raster_cols_max,
        "rows_max": raster_rows_max,
        "offset_geotransform": raster_offset_geotransform,
        "roi": raster_global_coverage
    }
    return raster_local_coverages, raster_geotransforms, common_metadata


def get_feature_bbox(geom, offsets=None):
    if offsets and len(offsets) != 2:
        raise AssertionError("offset param must be 2d")
    bounds = geom.bounds
    if offsets:
        centroid = geom.centroid
        roi_tl = (centroid.x - offsets[0], centroid.y + offsets[1])
        roi_br = (centroid.x + offsets[0], centroid.y - offsets[1])
    else:
        roi_tl = (bounds[0], bounds[3])
        roi_br = (bounds[2], bounds[1])
    return roi_tl, roi_br


def get_feature_roi(geom, raster_metadata, crop_img_size=None, crop_real_size=None):
    if crop_img_size and crop_real_size:
        raise AssertionError("should only provide one type of crop resolution, or none")
    offset_geotransform = raster_metadata["offset_geotransform"]
    if crop_img_size or crop_real_size:
        if crop_img_size:
            crop_size = int(crop_img_size)
            x_offset, y_offset = get_geocoord(offset_geotransform, crop_size, crop_size)
            x_offset, y_offset = abs(x_offset / 2), abs(y_offset / 2)
        elif crop_real_size:
            x_offset = y_offset = float(crop_real_size) / 2
        else:
            raise ValueError()
        roi_tl, roi_br = get_feature_bbox(geom, (x_offset, y_offset))
    else:
        roi_tl, roi_br = get_feature_bbox(geom)
    roi_tl_offsetpx_real = get_pxcoord(offset_geotransform, roi_tl[0], roi_tl[1])
    roi_tl_offsetpx = (int(math.floor(roi_tl_offsetpx_real[0])), int(math.floor(roi_tl_offsetpx_real[1])))
    if crop_img_size:
        crop_width = crop_height = int(crop_img_size)
        roi_br_offsetpx = (roi_tl_offsetpx[0] + crop_width, roi_tl_offsetpx[1] + crop_height)
    else:
        roi_br_offsetpx_real = get_pxcoord(offset_geotransform, roi_br[0], roi_br[1])
        roi_br_offsetpx = (int(math.ceil(roi_br_offsetpx_real[0])), int(math.ceil(roi_br_offsetpx_real[1])))
        crop_width = max(roi_br_offsetpx[0] - roi_tl_offsetpx[0], 1)
        crop_height = max(roi_br_offsetpx[1] - roi_tl_offsetpx[1], 1)
    roi_tl = get_geocoord(offset_geotransform, roi_tl_offsetpx[0], roi_tl_offsetpx[1])
    roi_br = get_geocoord(offset_geotransform, roi_br_offsetpx[0], roi_br_offsetpx[1])
    roi = shapely.geometry.Polygon([roi_tl, (roi_br[0], roi_tl[1]), roi_br, (roi_tl[0], roi_br[1])])
    return roi, roi_tl, roi_br, crop_width, crop_height


def process_feature(geom, rasters_data, raster_metadata, crop_img_size=None, crop_real_size=None):
    roi, roi_tl, roi_br, crop_width, crop_height = get_feature_roi(geom, raster_metadata, crop_img_size, crop_real_size)
    if not raster_metadata["roi"].contains(roi):
        raise AssertionError("roi not fully contained in rasters, should have been filtered out")
    crop_datatype = GDAL2NUMPY_TYPE_CONV[raster_metadata["datatype"]]
    crop_size = (crop_height, crop_width, raster_metadata["bandcount"])
    crop = np.ma.array(np.zeros(crop_size, dtype=crop_datatype), mask=np.ones(crop_size, dtype=np.uint8))
    crop_inv = np.ma.copy(crop)
    roi_hits = []
    roi_segms = []
    # raster data parsing loop (will test all regions that touch the selected feature)
    for raster_idx, raster_data in enumerate(rasters_data):
        append_inters_polygons(raster_data["roi"], roi, raster_idx, roi_hits, roi_segms)
    for raster_idx in roi_hits:
        rasterfile_path = rasters_data[raster_idx]["filepath"]
        rasterfile = osgeo.gdal.Open(rasterfile_path, osgeo.gdal.GA_ReadOnly)
        if rasterfile is None:
            raise AssertionError("could not open raster data file at '%s'" % rasterfile_path)
        raster_geotransform = rasters_data[raster_idx]["geotransform"]
        local_roi_tl_px_real = get_pxcoord(raster_geotransform, roi_tl[0], roi_tl[1])
        local_roi_tl_px = (int(max(round(local_roi_tl_px_real[0]), 0)),
                           int(max(round(local_roi_tl_px_real[1]), 0)))
        local_roi_br_px_real = get_pxcoord(raster_geotransform, roi_br[0], roi_br[1])
        local_roi_br_px = (int(min(round(local_roi_br_px_real[0]), rasterfile.RasterXSize)),
                           int(min(round(local_roi_br_px_real[1]), rasterfile.RasterYSize)))
        local_roi_offset = (local_roi_tl_px[1] - int(round(local_roi_tl_px_real[1])),
                            local_roi_tl_px[0] - int(round(local_roi_tl_px_real[0])))
        local_roi_cols = min(local_roi_br_px[0] - local_roi_tl_px[0], crop_width - local_roi_offset[1])
        local_roi_rows = min(local_roi_br_px[1] - local_roi_tl_px[1], crop_height - local_roi_offset[0])
        if local_roi_cols <= 0 or local_roi_rows <= 0:
            # most likely just intersected the edge with less than one pixel worth of info
            if len(roi_hits) <= 1:
                raise AssertionError("unexpected empty intersection with no fallback raster")
            continue
        local_roi_tl_real = get_geocoord(raster_geotransform, *local_roi_tl_px)
        local_geotransform = list(raster_metadata["offset_geotransform"])
        local_geotransform[0], local_geotransform[3] = local_roi_tl_real[0], local_roi_tl_real[1]
        local_target_ds = osgeo.gdal.GetDriverByName("MEM").Create(
            '', local_roi_cols, local_roi_rows, 2, osgeo.gdal.GDT_Byte)  # one band for mask, one inv mask
        local_target_ds.SetGeoTransform(local_geotransform)
        raster_srs = osr.SpatialReference(raster_metadata["srs"])
        local_target_ds.SetProjection(raster_metadata["srs"])
        local_target_ds.GetRasterBand(1).WriteArray(np.zeros((local_roi_rows, local_roi_cols), dtype=np.uint8))
        ogr_dataset = ogr.GetDriverByName("Memory").CreateDataSource("masks")
        ogr_layer = ogr_dataset.CreateLayer("feature_mask", srs=raster_srs)
        ogr_feature = ogr.Feature(ogr_layer.GetLayerDefn())
        ogr_geometry = ogr.CreateGeometryFromWkt(geom.wkt)
        ogr_feature.SetGeometry(ogr_geometry)
        ogr_layer.CreateFeature(ogr_feature)
        osgeo.gdal.RasterizeLayer(local_target_ds, [1], ogr_layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"])
        local_feature_mask_array = local_target_ds.GetRasterBand(1).ReadAsArray()
        if local_feature_mask_array is None:
            raise AssertionError("layer rasterization failed")
        local_target_ds.GetRasterBand(2).WriteArray(np.ones((local_roi_rows, local_roi_cols), dtype=np.uint8))
        ogr_layer_inv = ogr_dataset.CreateLayer("bg_mask", srs=raster_srs)
        ogr_feature_inv = ogr.Feature(ogr_layer_inv.GetLayerDefn())
        ogr_feature_inv.SetGeometry(ogr_geometry)
        ogr_layer_inv.CreateFeature(ogr_feature_inv)
        osgeo.gdal.RasterizeLayer(local_target_ds, [2], ogr_layer_inv, burn_values=[0], options=["ALL_TOUCHED=TRUE"])
        local_bg_mask_array = local_target_ds.GetRasterBand(2).ReadAsArray()
        if local_bg_mask_array is None:
            raise AssertionError("layer rasterization failed")
        for raster_band_idx in range(raster_metadata["bandcount"]):
            curr_band = rasterfile.GetRasterBand(raster_band_idx + 1)
            band_nodataval = curr_band.GetNoDataValue()
            raw_band_crop_data = curr_band.ReadAsArray(local_roi_tl_px[0], local_roi_tl_px[1],
                                                       local_roi_cols, local_roi_rows)
            if raw_band_crop_data is None:
                raise AssertionError("raster crop data read failed")
            band_crop_inv_data = np.where(local_bg_mask_array > 0, raw_band_crop_data, band_nodataval)
            band_crop_data = np.where(local_feature_mask_array > 0, raw_band_crop_data, band_nodataval)
            crop_i_min = local_roi_offset[0]
            crop_i_max = local_roi_offset[0] + local_roi_rows
            crop_j_min = local_roi_offset[1]
            crop_j_max = local_roi_offset[1] + local_roi_cols
            local_crop_inv_data = crop_inv.data[crop_i_min:crop_i_max, crop_j_min:crop_j_max, raster_band_idx]
            local_crop_inv_mask = crop_inv.mask[crop_i_min:crop_i_max, crop_j_min:crop_j_max, raster_band_idx]
            local_crop_data = crop.data[crop_i_min:crop_i_max, crop_j_min:crop_j_max, raster_band_idx]
            local_crop_mask = crop.mask[crop_i_min:crop_i_max, crop_j_min:crop_j_max, raster_band_idx]
            if local_crop_inv_mask.shape != band_crop_inv_data.shape:
                raise AssertionError("crop/roi mask size mismatch, probably rounding error somewhere")
            local_copy_inv_mask = np.where(
                np.bitwise_and(local_crop_inv_mask, band_crop_inv_data != band_nodataval), True, False)
            local_copy_mask = np.where(np.bitwise_and(local_crop_mask, band_crop_data != band_nodataval), True, False)
            # could also blend already-written pixels? (ignored for now)
            np.copyto(local_crop_inv_data, np.where(local_copy_inv_mask, raw_band_crop_data, local_crop_inv_data))
            np.copyto(local_crop_data, np.where(local_copy_mask, raw_band_crop_data, local_crop_data))
            np.bitwise_and(local_crop_inv_mask, np.invert(local_copy_inv_mask), out=local_crop_inv_mask)
            np.bitwise_and(local_crop_mask, np.invert(local_copy_mask), out=local_crop_mask)
    # ogr_dataset = None # close local fd
    # noinspection PyUnusedLocal
    local_target_ds = None  # close local fd
    # noinspection PyUnusedLocal
    rasterfile = None  # close input fd
    return crop, crop_inv, np.asarray(list(roi_tl) + list(roi_br))
