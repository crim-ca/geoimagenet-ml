import argparse
import csv
import os

import cv2 as cv
import numpy as np

barknet_tree_names = {
    "BOJ": ("Betula alleghaniensis", "bouleau jaune"),
    "BOP": ("Betula papyrifera", "bouleau a papier"),
    "CHR": ("Quercus rubra", "chene rouge"),
    "EPB": ("Picea glauca", "epinette blanche"),
    "EPN": ("Picea mariana", "epinette noire"),
    "EPO": ("Picea abies", "epinette communne"),
    "EPR": ("Picea rubens", "epinette rouge"),
    "ERB": ("Acer platanoides", "erable plane"),
    "ERR": ("Acer rubrum", "erable rouge"),
    "ERS": ("Acer saccharum", "erable a sucre"),
    "FRA": ("Fraxinus americana", "frene blanc"),
    "HEG": ("Fagus grandifolia", "hetre a grandes feuilles"),
    "MEL": ("Larix laricina", "meleze laricin"),
    "ORA": ("Ulmus americana", "orme d'amerique"),
    "OSV": ("Ostrya virginiana", "ostryer de virginie"),
    "PEG": ("Populus grandidentata", "peuplier a grandes dents"),
    "PET": ("Populus tremuloides", "peuplier faux-tremble"),
    "PIB": ("Pinus strobus", "pin blanc"),
    "PID": ("Pinus rigida", "pin rigide"),
    "PIR": ("Pinus resinosa", "pin rouge"),
    "PRU": ("Tsuga canadensis", "pruche du canada"),
    "SAB": ("Abies balsamea", "sapin baumier"),
    "THO": ("Thuja occidentalis", "thuya occidental"),
}


def main():
    ap = argparse.ArgumentParser(description="snapshot image annotator app")
    ap.add_argument("logfile", type=str, help="path to the thelper logger output file")
    ap.add_argument("--bar-size", type=int, help="display prediction text bar size", default=50)
    ap.add_argument("--output-dir", type=str, help="output directory where to save the annotated images", default=None)
    ap.add_argument("--resize-factor", type=float, help="resize factor for display images (float scalar)", default=1.0)
    ap.add_argument("--remap-tree-names", action="store_true", help="force-remap tree names from acronyms", default=False)
    args = ap.parse_args()
    if not os.path.isfile(args.logfile):
        raise AssertionError("invalid log file")
    with open(args.logfile) as fd:
        csv_reader = csv.reader(fd, delimiter=",")
        path_idx, line_idx = -1, 0
        preds_idxs = []
        samples = []
        for row in csv_reader:
            if line_idx == 0:
                path_idx = row.index("path")
                pred_idx = 0
                while True:
                    pred_label_idx = "pred_label_idx_" + str(pred_idx)
                    if pred_label_idx in row:
                        preds_idxs += [row.index(pred_label_idx)]
                    else:
                        break
                    pred_idx += 1
            else:
                samples.append({
                    "path": row[path_idx],
                    "preds": [
                        (row[pred_idx], row[pred_idx + 1], row[pred_idx + 2])
                        for pred_idx in preds_idxs
                    ]
                })
            line_idx += 1
    for sample in samples:
        if not os.path.isfile(sample["path"]):
            raise AssertionError("cannot find file at '%s'" % sample["path"])
        image = cv.imread(sample["path"])
        if image is None:
            raise AssertionError("cannot open file at '%s'" % sample["path"])
        if args.resize_factor != 1.0:
            image = cv.resize(image, (0, 0), fx=args.resize_factor, fy=args.resize_factor)
        display = None
        display_bar_size = args.bar_size
        for idx, pred in enumerate(sample["preds"]):
            if display is None:
                display = np.zeros((display_bar_size * len(sample["preds"]), image.shape[1], 3), dtype=np.uint8)
            class_name = pred[1]
            if args.remap_tree_names:
                full_name = barknet_tree_names[class_name]
                class_name = full_name[0] + " (%s) " % full_name[1]
            display_str = " Prediction #%d : %s @ %2.1f%%" % (idx + 1, class_name, float(pred[2]) * 100)
            cv.putText(display, display_str, (10, 25 + display_bar_size * idx),
                       cv.FONT_HERSHEY_DUPLEX, 0.50, (255, 255, 255), 1, cv.LINE_AA)
        display = np.concatenate((display, image), axis=0)
        if args.output_dir:
            if not os.path.isdir(args.output_dir):
                raise AssertionError("invalid output dir '%s'" % args.output_dir)
            cv.imwrite(os.path.join(args.output_dir, os.path.basename(sample["path"])), display)
        else:
            cv.imshow("display", display)
            cv.waitKey(0)


if __name__ == "__main__":
    main()
