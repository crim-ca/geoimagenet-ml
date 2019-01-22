import argparse
import csv
import os

import osgeo.ogr as ogr
import osgeo.osr as osr

import ccfb.ml.utils

uncertain_flags = "?,sick,dry,damaged,wet,dead,clump,Clump,snag,isol,shape,REPLICA,shaded,&"


def cvt_label(label_str):
    for sid, species in ccfb.ml.utils.SPECIES_MAP.items():
        if label_str == species[3]:
            return species[0]
    return label_str + "?"


def read_logfile(path):
    samples = []
    with open(path) as fd:
        csv_reader = csv.reader(fd, delimiter=",")
        feature_id_idx, line_idx = -1, 0
        preds_idxs = []
        for row in csv_reader:
            if line_idx == 0:
                feature_id_idx = row.index("id")
                pred_idx = 0
                while True:
                    pred_label_name_str = "pred_label_name_" + str(pred_idx)
                    pred_label_score_str = "pred_label_score_" + str(pred_idx)
                    if pred_label_name_str in row and pred_label_score_str in row:
                        if row.index(pred_label_name_str) != row.index(pred_label_score_str) - 1:
                            raise AssertionError("unexpected label pred ordering")
                        preds_idxs += [row.index(pred_label_name_str)]
                    else:
                        break
                    pred_idx += 1
            else:
                samples.append({
                    "id": int(row[feature_id_idx]),
                    "preds": [
                        (row[pred_idx], row[pred_idx + 1])
                        for pred_idx in preds_idxs
                    ]
                })
            line_idx += 1
    return samples


def main():
    ap = argparse.ArgumentParser(description="shapefile polyline annotator app")
    ap.add_argument("logfile_train", type=str, help="path to the thelper train logger output file")
    ap.add_argument("logfile_valid", type=str, help="path to the thelper valid logger output file")
    ap.add_argument("input_shapefile", type=str, help="path to the input shape file")
    ap.add_argument("output_shapefile", type=str, help="path to the output shape file")

    args = ap.parse_args()
    if not os.path.isfile(args.logfile_train) or not os.path.isfile(args.logfile_valid):
        raise AssertionError("invalid log file")

    samples = {}
    len_preds = None

    train_samples = read_logfile(args.logfile_train)
    for sample in train_samples:
        if len_preds is None:
            len_preds = len(sample["preds"])
        elif len_preds != len(sample["preds"]):
            raise AssertionError("unexpected pred length")
        if sample["id"] not in samples:
            samples[sample["id"]] = {
                **sample,
                "train": True
            }

    valid_samples = read_logfile(args.logfile_valid)
    for sample in valid_samples:
        if len_preds is None:
            len_preds = len(sample["preds"])
        elif len_preds != len(sample["preds"]):
            raise AssertionError("unexpected pred length")
        if sample["id"] in samples:
            raise AssertionError("valid samples should never overlap?")  # unless test-time augments were used
        samples[sample["id"]] = {
            **sample,
            "train": False
        }

    features, _, _ = ccfb.ml.utils.parse_shapefile(args.input_shapefile, srs_destination=26918,
                                                   category_field="Species", id_field="ID_Number",
                                                   uncertain_flags=uncertain_flags.split(","))

    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource(args.output_shapefile)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26918)
    layer = data_source.CreateLayer("crowns", srs, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn("train", ogr.OFTInteger))
    field_gt_label = ogr.FieldDefn("gt_label", ogr.OFTString)
    field_gt_label.SetWidth(24)
    layer.CreateField(field_gt_label)
    field_gt_label = ogr.FieldDefn("gt_acr", ogr.OFTString)
    field_gt_label.SetWidth(24)
    layer.CreateField(field_gt_label)
    for pred_idx in range(len_preds):
        field_pred_label = ogr.FieldDefn("pred_lbl_%d" % (pred_idx + 1), ogr.OFTString)
        field_pred_label.SetWidth(24)
        layer.CreateField(field_pred_label)
        field_pred_label = ogr.FieldDefn("pred_acr_%d" % (pred_idx + 1), ogr.OFTString)
        field_pred_label.SetWidth(24)
        layer.CreateField(field_pred_label)
        layer.CreateField(ogr.FieldDefn("score_%d" % (pred_idx + 1), ogr.OFTReal))

    for in_feat in features:
        feat_id = int(in_feat["id"])
        if feat_id in samples:
            sample = samples[feat_id]
            out_feat = ogr.Feature(layer.GetLayerDefn())
            out_feat.SetField("id", feat_id)
            out_feat.SetField("train", int(sample["train"]))
            out_feat.SetField("gt_label", cvt_label(in_feat["category"]))
            out_feat.SetField("gt_acr", in_feat["category"])
            for pred_idx in range(len_preds):
                out_feat.SetField("pred_lbl_%d" % (pred_idx + 1), cvt_label(sample["preds"][pred_idx][0]))
                out_feat.SetField("pred_acr_%d" % (pred_idx + 1), sample["preds"][pred_idx][0])
                out_feat.SetField("score_%d" % (pred_idx + 1), sample["preds"][pred_idx][1])
            out_feat.SetGeometry(ogr.CreateGeometryFromWkt(in_feat["geometry"].wkt))
            layer.CreateFeature(out_feat)
            out_feat = None
    data_source = None

    print("all done")


if __name__ == "__main__":
    main()
