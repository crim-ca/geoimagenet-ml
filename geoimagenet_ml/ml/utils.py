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
from typing import TYPE_CHECKING
import logging
import math
import os

# === PACKAGE PROVIDED BY THELPER ===
# noinspection PyPackageRequirements
import affine
# noinspection PyPackageRequirements
import osgeo.gdal
# noinspection PyPackageRequirements
import numpy as np
# noinspection PyPackageRequirements
import ogr
# noinspection PyPackageRequirements
import osr
# noinspection PyPackageRequirements
import shapely
# noinspection PyPackageRequirements
import shapely.geometry
# noinspection PyPackageRequirements
import shapely.ops
# noinspection PyPackageRequirements
import shapely.wkt

if TYPE_CHECKING:
    from geoimagenet_ml.typedefs import JSON, List, Tuple, Optional

LOGGER = logging.getLogger(__name__)

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


def parse_coordinate_system(body):
    # type: (JSON) -> osr.SpatialReference
    crs_body = body.get("crs") or body.get("srs")
    crs_type = crs_body.get("type", "").upper()
    crs_opts = list(crs_body.get("properties").values())
    srs = ogr.osr.SpatialReference()
    csr = None
    if crs_type == "EPSG":
        csr = srs.ImportFromEPSG(*crs_opts)
    if crs_type == "EPSGA":
        csr = srs.ImportFromEPSGA(*crs_opts)
    if crs_type == "ERM":
        csr = srs.ImportFromERM(*crs_opts)
    if crs_type == "ESRI":
        csr = srs.ImportFromESRI(*crs_opts)
    if crs_type == "ESRI":
        csr = srs.ImportFromESRI(*crs_opts)
    if crs_type == "PCI":
        csr = srs.ImportFromPCI(*crs_opts)
    if not csr:
        LOGGER.error("Could not identify CRS/SRS type.")
        raise NotImplementedError("Could not identify CRS/SRS type.")
    return csr


def parse_geojson(geojson,          # type: JSON
                  srs_destination,  # type: osr.SpatialReference
                  roi=None,         # type: Optional[shapely.geometry.base.BaseGeometry]
                  ):                # type: (...) -> Tuple[List[JSON], ClassCounter]
    if geojson is None or not isinstance(geojson, dict):
        raise AssertionError("unexpected geojson type")
    if "features" not in geojson or not isinstance(geojson["features"], list):
        raise AssertionError("unexpected geojson feature list type")
    features = geojson["features"]
    LOGGER.info("total geojson feature count: %d" % len(features))
    if "crs" not in geojson or geojson["crs"]["type"] != "EPSG":
        raise AssertionError("geojson is missing its crs/epsg fields")
    srs_origin = ogr.osr.SpatialReference()
    srs_origin.ImportFromEPSG(geojson["crs"]["properties"]["code"])
    shapes_srs_transform = None
    if not srs_origin.IsSame(srs_destination):
        shapes_srs_transform = osr.CoordinateTransformation(srs_origin, srs_destination)
    kept_features = []
    LOGGER.debug("scanning parsed features for out-of-bounds cases...")
    for feature in features:
        raw_geometry = feature["geometry"]
        if raw_geometry["type"] == "Polygon":
            coords = raw_geometry["coordinates"]
            if not isinstance(coords, list):
                raise AssertionError("unexpected poly coords type")
            if len(coords) != 1:
                raise AssertionError("unexpected coords embedding; should be list-of-list-of-points w/ unique ring")
            if not all([isinstance(c, list) and len(c) == 2 for c in coords[0]]) or len(coords[0]) < 4:
                raise AssertionError("unexpected poly coord format")
            poly = shapely.geometry.Polygon(coords[0])
            if shapes_srs_transform is not None:
                ogr_geometry = ogr.CreateGeometryFromWkb(poly.wkb)
                ogr_geometry.Transform(shapes_srs_transform)
                poly = shapely.wkt.loads(ogr_geometry.ExportToWkt())
            feature["geometry"] = poly
        else:
            raise AssertionError("unhandled raw geometry type [{}]".format(raw_geometry["type"]))
        if roi is not None and roi and roi.contains(feature["geometry"]):
            kept_features.append(feature)
    LOGGER.info("kept features: {} ({}%)".format(len(kept_features), int(len(kept_features) * 100 // len(features))))
    features = kept_features
    category_counter = ClassCounter()
    for feature in features:
        category_counter[feature["properties"]["taxonomy_class_id"]] += 1
    LOGGER.info("unique + clean feature categories: {}".format(len(category_counter.keys())))
    LOGGER.debug(str(category_counter))
    return features, category_counter


def parse_shapefile(shapefile_path, srs_destination, category_field, id_field,
                    uncertain_flags=None, roi=None, target_category=None, target_id=None):
    uncertain_flags = [] if uncertain_flags is None else uncertain_flags
    shapefile_driver = ogr.GetDriverByName("ESRI Shapefile")
    shapefile = shapefile_driver.Open(shapefile_path, 0)
    if shapefile is None:
        raise AssertionError("could not open vector data file at '%s'" % shapefile_path)
    LOGGER.debug("Shapefile vector metadata printing below... %s" % str(shapefile))
    if len(shapefile) != 1:
        raise AssertionError("expected one layer, got multiple")
    layer = shapefile.GetLayer()
    layer_def = layer.GetLayerDefn()
    LOGGER.debug("layer name: %s" % str(layer.GetName()))
    LOGGER.debug("field count: %s" % str(layer_def.GetFieldCount()))
    got_category_field = False
    for field_idx in range(layer_def.GetFieldCount()):
        field_name = layer_def.GetFieldDefn(field_idx).GetName()
        field_type_code = layer_def.GetFieldDefn(field_idx).GetType()
        field_type = layer_def.GetFieldDefn(field_idx).GetFieldTypeName(field_type_code)
        field_width = layer_def.GetFieldDefn(field_idx).GetWidth()
        precision = layer_def.GetFieldDefn(field_idx).GetPrecision()
        LOGGER.debug("field %d: {name:\"%s\", type:\"%s\", length:%s, precision:%s}"
                     % (field_idx, field_name, field_type, str(field_width), str(precision)))
        got_category_field = got_category_field or field_name == category_field
    if not got_category_field:
        raise AssertionError("could not find layer definition field with name '%s' to parse categories"
                             % category_field)
    shapes_srs_transform = osr.CoordinateTransformation(layer.GetSpatialRef(), srs_destination)
    feature_count = layer.GetFeatureCount()
    LOGGER.info("total shapefile feature count: %s" % str(feature_count))
    oob_feature_count = 0
    features = []
    LOGGER.debug("scanning parsed features for out-of-bounds cases...")
    for feature in layer:
        ogr_geometry = feature.GetGeometryRef()
        ogr_geometry.Transform(shapes_srs_transform)
        feature_geometry = shapely.wkt.loads(ogr_geometry.ExportToWkt())
        if roi and not roi.contains(feature_geometry):
            oob_feature_count += 1
        else:
            feature_id = feature.GetFieldAsString(id_field).strip()
            feature_category = feature.GetFieldAsString(category_field).strip()
            features.append({
                "id": feature_id,
                "category": feature_category,
                "geometry": feature_geometry,
            })
    LOGGER.info("out-of-bounds features: %d  (%d%%)"
                % (oob_feature_count, int(oob_feature_count * 100 // feature_count)))
    unlabeled_feature_count = 0
    uncertain_feature_count = 0
    bad_shape_feature_count = 0
    category_counter = ClassCounter()
    for feature in features:
        if len(feature["category"]) == 0:
            unlabeled_feature_count += 1
            uncertain_feature_count += 1
            feature["clean"] = False
        elif any(flag in feature["category"] for flag in uncertain_flags):
            uncertain_feature_count += 1
            feature["clean"] = False
        elif feature["geometry"].geom_type != "Polygon":
            bad_shape_feature_count += 1
            feature["clean"] = False
        else:
            feature["clean"] = True
            category_counter[feature["category"]] += 1

    n_features = len(features)
    LOGGER.info("bad shape features: %d  (%d%%)"
                % (bad_shape_feature_count, percent(bad_shape_feature_count, n_features)))
    LOGGER.info("unlabeled features: %d  (%d%%)"
                % (unlabeled_feature_count, percent(unlabeled_feature_count, n_features)))
    LOGGER.info("uncertain features: %d  (%d%%)"
                % (uncertain_feature_count, percent(uncertain_feature_count, n_features)))
    LOGGER.info("clean features: %s" % sum(category_counter.values()))
    LOGGER.info("unique+clean feature categories: %s" % len(category_counter.keys()))
    LOGGER.debug("%s" % str(category_counter.keys()))
    if target_category:
        if target_category not in category_counter:
            raise AssertionError("could not find specified category '%s' in parsed features" % target_category)
        category_percent = percent(category_counter[target_category], sum(category_counter.values()))
        LOGGER.info("selected category raw feature count: %d  (%d%%)"
                    % (category_counter[target_category], category_percent))
        features = [feature for feature in features if feature["category"] == target_category and feature["clean"]]
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
    return features, category_counter


def parse_rasters(rasterfile_paths, default_srs=None, normalize=False):
    rasters_data = []
    global_rois = []
    raster_stats_map = []
    for rasterfile_path in rasterfile_paths:
        rasterfile = osgeo.gdal.Open(rasterfile_path, osgeo.gdal.GA_ReadOnly)
        if rasterfile is None:
            raise AssertionError("could not open raster data file at '%s'" % rasterfile_path)
        LOGGER.debug("Raster '%s' metadata printing below..." % rasterfile_path)
        LOGGER.debug("%s" % str(rasterfile))
        LOGGER.debug("%s" % str(rasterfile.GetMetadata()))
        LOGGER.debug("band count: %s" % str(rasterfile.RasterCount))
        raster_geotransform = rasterfile.GetGeoTransform()
        raster_extent = get_geoextent(raster_geotransform, 0, 0, rasterfile.RasterXSize, rasterfile.RasterYSize)
        LOGGER.debug("extent: %s" % str(raster_extent))
        raster_curr_srs = osr.SpatialReference()
        raster_curr_srs_str = rasterfile.GetProjectionRef()
        if "unknown" not in raster_curr_srs_str:
            raster_curr_srs.ImportFromWkt(raster_curr_srs_str)
        else:
            if default_srs is None:
                raise AssertionError("raster did not provide an SRS, and no default EPSG SRS provided")
            raster_curr_srs = default_srs
        LOGGER.debug("spatial ref:\n%s" % str(raster_curr_srs))
        px_width, px_height = raster_geotransform[1], raster_geotransform[5]
        skew_x, skew_y = raster_geotransform[2], raster_geotransform[4]
        raster_bands_stats = []
        raster_datatype = None
        for raster_band_idx in range(rasterfile.RasterCount):
            curr_band = rasterfile.GetRasterBand(raster_band_idx + 1)  # offset, starts at 1
            if curr_band is None:
                raise AssertionError("found invalid raster band")
            # lines below cause crashes on python 3.6m on windows w/ gdal from pre-compiled wheel
            # curr_band_stats = curr_band.GetStatistics(True,True)
            # if curr_band_stats is None:
            #    raise AssertionError("could not compute band statistics")
            if not raster_datatype:
                raster_datatype = curr_band.DataType
            elif raster_datatype != curr_band.DataType:
                raise AssertionError("expected identical data types in all bands & rasters")
            if normalize:
                LOGGER.debug("computing band #%d statistics..." % (raster_band_idx + 1))
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
        local_roi = shapely.geometry.Polygon([list(pt) for pt in raster_extent]).buffer(0.01)
        if not raster_curr_srs.IsSame(default_srs):
            shapes_srs_transform = osr.CoordinateTransformation(raster_curr_srs, default_srs)
            ogr_geometry = ogr.CreateGeometryFromWkb(local_roi.wkb)
            ogr_geometry.Transform(shapes_srs_transform)
            global_roi = shapely.wkt.loads(ogr_geometry.ExportToWkt())
        else:
            global_roi = local_roi
        global_rois.append(global_roi)
        rasters_data.append({
            "srs": raster_curr_srs,
            "geotransform": raster_geotransform,
            "offset_geotransform": (0, px_width, skew_x, 0, skew_y, px_height),
            "extent": raster_extent,
            "skew": (skew_x, skew_y),
            "resolution": (px_width, px_height),
            "bandcount": rasterfile.RasterCount,
            "cols": rasterfile.RasterXSize,
            "rows": rasterfile.RasterYSize,
            "datatype": raster_datatype,
            "local_roi": local_roi,
            "global_roi": global_roi,
            "filepath": rasterfile_path,
        })
        rasterfile = None  # close input fd
    coverage = shapely.ops.cascaded_union(global_rois)
    return rasters_data, coverage


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


def process_feature(geom, geom_srs, raster_data, crop_fixed_size=None):
    if not raster_data["global_roi"].contains(geom):
        return None, None, None  # exact shape should be fully contained in a single raster
    if not raster_data["srs"].IsSame(geom_srs):
        shapes_srs_transform = osr.CoordinateTransformation(geom_srs, raster_data["srs"])
        ogr_geometry = ogr.CreateGeometryFromWkb(geom.wkb)
        ogr_geometry.Transform(shapes_srs_transform)
        geom = shapely.wkt.loads(ogr_geometry.ExportToWkt())
    if crop_fixed_size:
        offset = float(crop_fixed_size) / 2
        roi_tl, roi_br = get_feature_bbox(geom, (offset, offset))
    else:
        roi_tl, roi_br = get_feature_bbox(geom)
    # round projected geometry bounds to nearest pixel in raster
    offset_geotransform = raster_data["offset_geotransform"]
    roi_tl_offsetpx_real = get_pxcoord(offset_geotransform, roi_tl[0], roi_tl[1])
    roi_tl_offsetpx = (int(math.floor(roi_tl_offsetpx_real[0])), int(math.floor(roi_tl_offsetpx_real[1])))
    roi_br_offsetpx_real = get_pxcoord(offset_geotransform, roi_br[0], roi_br[1])
    roi_br_offsetpx = (int(math.ceil(roi_br_offsetpx_real[0])), int(math.ceil(roi_br_offsetpx_real[1])))
    crop_width = max(roi_br_offsetpx[0] - roi_tl_offsetpx[0], 1)
    crop_height = max(roi_br_offsetpx[1] - roi_tl_offsetpx[1], 1)
    roi_tl = get_geocoord(offset_geotransform, roi_tl_offsetpx[0], roi_tl_offsetpx[1])
    roi_br = get_geocoord(offset_geotransform, roi_br_offsetpx[0], roi_br_offsetpx[1])
    roi = shapely.geometry.Polygon([roi_tl, (roi_br[0], roi_tl[1]), roi_br, (roi_tl[0], roi_br[1])])
    if not raster_data["local_roi"].contains(roi):
        return None, None, None  # asking for a larger crop can clip raster bounds
    crop_datatype = GDAL2NUMPY_TYPE_CONV[raster_data["datatype"]]
    crop_size = (crop_height, crop_width, raster_data["bandcount"])
    crop = np.ma.array(np.zeros(crop_size, dtype=crop_datatype), mask=np.ones(crop_size, dtype=np.uint8))
    crop_inv = np.ma.copy(crop)
    bounds = np.asarray(list(roi_tl) + list(roi_br))
    rasterfile_path = raster_data["filepath"]
    rasterfile = osgeo.gdal.Open(rasterfile_path, osgeo.gdal.GA_ReadOnly)
    if rasterfile is None:
        raise AssertionError("could not open raster data file at '%s'" % rasterfile_path)
    raster_geotransform = raster_data["geotransform"]
    # edge handling code below leftover from CCFB02 impl
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
        return None, None, None
    local_roi_tl_real = get_geocoord(raster_geotransform, *local_roi_tl_px)
    local_geotransform = list(offset_geotransform)
    local_geotransform[0], local_geotransform[3] = local_roi_tl_real[0], local_roi_tl_real[1]
    local_target_ds = osgeo.gdal.GetDriverByName("MEM").Create(
        '', local_roi_cols, local_roi_rows, 2, osgeo.gdal.GDT_Byte)  # one band for mask, one inv mask
    local_target_ds.SetGeoTransform(local_geotransform)
    local_target_ds.SetProjection(raster_data["srs"].ExportToWkt())
    local_target_ds.GetRasterBand(1).WriteArray(np.zeros((local_roi_rows, local_roi_cols), dtype=np.uint8))
    ogr_dataset = ogr.GetDriverByName("Memory").CreateDataSource("masks")
    ogr_layer = ogr_dataset.CreateLayer("feature_mask", srs=raster_data["srs"])
    ogr_feature = ogr.Feature(ogr_layer.GetLayerDefn())
    ogr_geometry = ogr.CreateGeometryFromWkt(geom.wkt)
    ogr_feature.SetGeometry(ogr_geometry)
    ogr_layer.CreateFeature(ogr_feature)
    osgeo.gdal.RasterizeLayer(local_target_ds, [1], ogr_layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"])
    local_feature_mask_array = local_target_ds.GetRasterBand(1).ReadAsArray()
    if local_feature_mask_array is None:
        raise AssertionError("layer rasterization failed")
    local_target_ds.GetRasterBand(2).WriteArray(np.ones((local_roi_rows, local_roi_cols), dtype=np.uint8))
    ogr_layer_inv = ogr_dataset.CreateLayer("bg_mask", srs=raster_data["srs"])
    ogr_feature_inv = ogr.Feature(ogr_layer_inv.GetLayerDefn())
    ogr_feature_inv.SetGeometry(ogr_geometry)
    ogr_layer_inv.CreateFeature(ogr_feature_inv)
    osgeo.gdal.RasterizeLayer(local_target_ds, [2], ogr_layer_inv, burn_values=[0], options=["ALL_TOUCHED=TRUE"])
    local_bg_mask_array = local_target_ds.GetRasterBand(2).ReadAsArray()
    if local_bg_mask_array is None:
        raise AssertionError("layer rasterization failed")
    for raster_band_idx in range(raster_data["bandcount"]):
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
    local_target_ds = None  # close local fd
    rasterfile = None  # close input fd
    return crop, crop_inv, bounds
