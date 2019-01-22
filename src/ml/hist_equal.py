import argparse
import pickle
import os

import cv2 as cv
import gdal
import matplotlib.pyplot as plt
import numpy as np

import thelper.utils


def get_hists(image, bins):
    if not isinstance(image, np.ndarray) or image.ndim > 2:
        raise AssertionError("invalid input")
    arr = image.compressed() if isinstance(image, np.ma.masked_array) else image.ravel()
    if image.dtype == np.uint8:
        hist, bin_edges = np.histogram(arr, bins, [0, 2 ** 8 - 1])
    elif image.dtype == np.uint16:
        hist, bin_edges = np.histogram(arr, bins, [0, 2 ** 16 - 1])
    else:
        raise AssertionError("invalid image type")
    return hist.astype(np.float64) / arr.size, np.cumsum(hist.astype(np.float64) / arr.size), bin_edges


def get_band_lut(shifted_bin_centers, default_bin_centers, array_type):
    if shifted_bin_centers.shape != default_bin_centers.shape:
        raise AssertionError("bin arrays shape mismatch")
    if array_type == np.uint8:
        shifted_bin_centers = np.insert(np.append(shifted_bin_centers.astype(np.float64), shifted_bin_centers[-1]), 0, shifted_bin_centers[0])
        default_bin_centers = np.insert(np.append(default_bin_centers.astype(np.float64), 2 ** 8 - 1), 0, 0)
        lut = np.interp(np.arange(0, 2 ** 8, dtype=np.uint8), default_bin_centers, shifted_bin_centers)
        return np.rint(lut).astype(np.uint8)
    elif array_type == np.uint16:
        shifted_bin_centers = np.insert(np.append(shifted_bin_centers.astype(np.float64), shifted_bin_centers[-1]), 0, shifted_bin_centers[0])
        default_bin_centers = np.insert(np.append(default_bin_centers.astype(np.float64), 2 ** 16 - 1), 0, 0)
        lut = np.interp(np.arange(0, 2 ** 16, dtype=np.uint16), default_bin_centers, shifted_bin_centers)
        return np.rint(lut).astype(np.uint16)
    else:
        raise AssertionError("invalid array type")


def get_ref_hist_from_pickle(paths):
    if not isinstance(paths, list):
        paths = [paths]
    ref_cumhists = []
    bin_edges = None
    for path in paths:
        with open(path, "rb") as fd:
            stats = pickle.load(fd)
            if not isinstance(stats, dict) or "cumhist" not in stats or "bin_edges" not in stats:
                raise AssertionError("invalid pickle file")
            if bin_edges is not None:
                if not np.array_equal(bin_edges,stats["bin_edges"]):
                    raise AssertionError("mismatched bin edges")
            else:
                bin_edges = stats["bin_edges"]
            ref_cumhists.append(stats["cumhist"].astype(np.float64))
    ref_cumhist = np.sum(ref_cumhists, axis=0) / len(ref_cumhists)
    return ref_cumhist, bin_edges


def parse_rasters(rasterfile_paths, display=False, output=None, bins=100,
                  plot_y_max=None, match_refs=None):
    raster_infos = []
    raster_bandcount = None
    raster_datatype = None
    for rasterfile_path in rasterfile_paths:
        rasterfile = osgeo.gdal.Open(rasterfile_path, osgeo.gdal.GA_ReadOnly)
        if rasterfile is None:
            raise AssertionError("could not open raster data file at '%s'" % rasterfile_path)
        print("Raster '%s' metadata printing below..." % rasterfile_path)
        print("%s" % str(rasterfile))
        print("%s" % str(rasterfile.GetMetadata()))
        print("band count: %s" % str(rasterfile.RasterCount))
        if not raster_bandcount:
            raster_bandcount = rasterfile.RasterCount
        elif raster_bandcount != rasterfile.RasterCount:
            raise AssertionError("expected indentical band counts for all rasters")
        raster_bands_stats = []
        fig_hist = plt.figure(num="hist", figsize=(5, 5), dpi=320, facecolor="w", edgecolor="k")
        fig_hist.clf()
        ax_hist = fig_hist.add_subplot(2, 1, 1)
        ax_cumhist = fig_hist.add_subplot(2, 1, 2)
        output_rasterfile = None
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
            print("reading band #%d..." % (raster_band_idx + 1))
            band_array = curr_band.ReadAsArray()
            band_nodataval = curr_band.GetNoDataValue()
            if band_nodataval is None:
                band_nodataval = 2 ** (16 if band_array.dtype == np.uint16 else 8) - 1
            band_ma = np.ma.array(band_array,
                                  mask=np.where(band_array != band_nodataval, False, True))
            print("computing band #%d statistics..." % (raster_band_idx + 1))

            band_hist, band_cumhist, band_bin_edges = get_hists(band_ma, bins)
            bin_centers = (band_bin_edges[:-1] + band_bin_edges[1:]) / 2
            if match_refs and output:
                print("writing corrected band #%d output..." % (raster_band_idx + 1))
                if raster_band_idx in match_refs:
                    ref_cumhist, ref_bin_edges = match_refs[raster_band_idx]
                    if not np.array_equal(band_bin_edges, ref_bin_edges):
                        raise AssertionError("bin edges mismatch")
                    shifted_bin_centers = np.interp(band_cumhist, ref_cumhist, bin_centers)
                    band_shift_lut = get_band_lut(shifted_bin_centers, bin_centers, band_ma.dtype)
                    fig_curve = plt.figure("curve")
                    fig_curve.clf()
                    ax_curve = fig_curve.add_subplot(1, 1, 1)
                    ax_curve.plot(np.arange(0, band_shift_lut.size), band_shift_lut)
                    fig_curve.show()
                    band_ma = np.ma.masked_array(band_shift_lut[band_ma.data], mask=band_ma.mask)
                    band_hist, band_cumhist, band_bin_edges = get_hists(band_ma, bins)
                if output_rasterfile is None:
                    driver = osgeo.gdal.GetDriverByName("GTiff")
                    orig_rasterfile_name = os.path.split(rasterfile_path)[1].replace("tif", "")
                    output_rasterfile_path = os.path.join(output, orig_rasterfile_name + "corrected.tif")
                    rasterfile_rows, rasterfile_cols = band_array.shape
                    if os.path.exists(output_rasterfile_path):
                        os.remove(output_rasterfile_path)
                    output_rasterfile = driver.Create(output_rasterfile_path, rasterfile_cols, rasterfile_rows,
                                                      raster_bandcount, curr_band.DataType)
                    output_rasterfile.SetGeoTransform(rasterfile.GetGeoTransform())
                    output_rasterfile.SetProjection(rasterfile.GetProjection())
                output_band = output_rasterfile.GetRasterBand(raster_band_idx + 1)
                output_band.WriteArray(band_ma)
                if band_nodataval is not None:
                    output_band.SetNoDataValue(band_nodataval)
                output_band.FlushCache()
                output_band = None
            stats = {
                "min": np.ma.min(band_ma),
                "max": np.ma.max(band_ma),
                "std": np.ma.std(band_ma),
                "mean": np.ma.mean(band_ma),
                "hist": band_hist,
                "cumhist": band_cumhist,
                "bin_edges": band_bin_edges,
            }
            colors = ["m", "r", "g", "b", "c", "y", "k"]
            color = colors[raster_band_idx % len(colors)]
            ax_hist.plot(bin_centers, band_hist, color + "o-")
            ax_cumhist.plot(bin_centers, band_cumhist, color + "o-")
            ax_hist.set_ylim(bottom=0)
            ax_hist.set_title("histogram")
            ax_cumhist.set_ylim(bottom=0, top=1)
            ax_cumhist.set_title("cumulative histogram")
            if plot_y_max is not None:
                ax_hist.set_ylim(top=plot_y_max)
            raster_bands_stats.append(stats)
        fig_hist.tight_layout()
        if display:
            fig_hist.show()
        raster_info = {
            "name": os.path.split(rasterfile_path)[1],
            "stats": raster_bands_stats,
            "hist": thelper.utils.fig2array(fig_hist),
        }
        if output is not None:
            hist_img_path = os.path.join(output, raster_info["name"] + "hists.png")
            cv.imwrite(hist_img_path, np.copy(raster_info["hist"][..., [2, 1, 0]]))
            meta_file_path = os.path.join(output, raster_info["name"] + ".meta")
            stats_file_path = os.path.join(output, raster_info["name"])
            with open(meta_file_path, "w") as fd:
                stats_keys = ["min", "max", "std", "mean"]
                for stats_key in stats_keys:
                    fd.write("%s :\n" % stats_key)
                    for raster_band_idx in range(raster_bandcount):
                        fd.write(str(raster_bands_stats[raster_band_idx][stats_key]) + "\n")
                    fd.write("\n")
            if match_refs is None:
                for raster_band_idx in range(raster_bandcount):
                    with open(stats_file_path + "." + str(raster_band_idx) + ".pickle", "wb") as fd:
                        pickle.dump(raster_bands_stats[raster_band_idx], fd)
        rasterfile = None  # close input fd
        if match_refs:
            output_rasterfile.FlushCache()
            output_rasterfile = None  # close output fd
        raster_infos.append(raster_info)
    return raster_infos


def main():
    ap = argparse.ArgumentParser(description="geotiff raster histogram equalizer app")
    ap.add_argument("rasterfile", help="relative or absolute path to the raster file(s) to process (accepts wildcard)")
    ap.add_argument("-v", "--display", default=False, action="store_true", help="display histograms via matplotlib")
    ap.add_argument("-d", "--data-root", default="data", type=str, help="default dataset root for relative file paths")
    ap.add_argument("-o", "--output", default=None, type=str, help="output directory used to save stats & histograms")
    ap.add_argument("-b", "--bins", default=100, type=int, help="default histogram bin count")
    ap.add_argument("-r", "--reference", default=None, type=str, help="reference bands for hist matching")
    ap.add_argument("-s", "--magic-swap", default=False, action="store_true", help="swap ref channels order expecting NRGB->BGRXNL")
    ap.add_argument("--plot-y-max", default=None, type=float, help="maximum y limit used in drawn plots")
    args = ap.parse_args()
    osgeo.gdal.UseExceptions()

    if args.output is not None:
        if not os.path.isdir(args.output):
            os.makedirs(args.output)

    rasterfile_paths = thelper.utils.get_file_paths(args.rasterfile, args.data_root, allow_glob=True)
    print("found %d rasters" % len(rasterfile_paths))

    match_refs = None
    if args.reference:
        reference_paths = thelper.utils.get_file_paths(args.reference, args.data_root, allow_glob=True)
        print("found %d reference files" % len(reference_paths))
        if any([not ref.endswith(".pickle") for ref in reference_paths]):
            raise AssertionError("got some unsupported reference file types")
        reference_band_paths = {}
        for ref in reference_paths:
            ref_segms = ref.split(".")
            if ref_segms[-1] != "pickle":
                raise AssertionError("unexpected file ext")
            if len(ref_segms) < 3:
                raise AssertionError("unexpected file name formatting")
            band_idx = int(ref_segms[-2])
            if band_idx not in reference_band_paths:
                reference_band_paths[band_idx] = []
            reference_band_paths[band_idx].append(ref)
        match_refs = {}
        fig_curve = plt.figure("cumhist_ref")
        fig_curve.clf()
        ax_cumhist = fig_curve.add_subplot(1, 1, 1)
        for band, paths in reference_band_paths.items():
            match_refs[band] = get_ref_hist_from_pickle(paths)
            bin_centers = (match_refs[band][1][:-1] + match_refs[band][1][1:]) / 2
            colors = ["m", "r", "g", "b", "c", "y", "k"]
            color = colors[band % len(colors)]
            ax_cumhist.plot(bin_centers, match_refs[band][0], color + "o-")
        ax_cumhist.set_ylim(bottom=0, top=1)
        fig_curve.show()

        if args.magic_swap:
            # this is a dirty dirty hack that assumes ref = petawawa, target = ouareau
            match_refs = {
                4: match_refs[0],  # bandidx#0 in petawawa = nir = bandidx#4 in ouareau
                2: match_refs[1],  # bandidx#1 in petawawa = red = bandidx#2 in ouareau
                1: match_refs[2],  # bandidx#2 in petawawa = green = bandidx#1 in ouareau
                0: match_refs[3],  # bandidx#3 in petawawa = blue = bandidx#0 in ouareau
            }

    parse_rasters(rasterfile_paths, display=args.display, output=args.output,
                  bins=args.bins, plot_y_max=args.plot_y_max, match_refs=match_refs)


if __name__ == "__main__":
    main()
