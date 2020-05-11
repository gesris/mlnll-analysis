#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Dumbledraw.dumbledraw as dd
import Dumbledraw.rootfile_parser as rootfile_parser
import Dumbledraw.styles as styles
import ROOT

import argparse
import copy
import yaml
import distutils.util
import logging
import os

logger = logging.getLogger("")

from utils import config as cfg


def setup_logging(output_file, level=logging.DEBUG):
    logger.setLevel(level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler(output_file, "w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def main(args):
    channel_categories = {
        "mt": [str(i) for i in range(len(cfg.ml_classes))],
    }
    channel_dict = {
        "ee": "ee",
        "em": "e#mu",
        "et": "e#tau_{h}",
        "mm": "#mu#mu",
        "mt": "#mu#tau_{h}",
        "tt": "#tau_{h}#tau_{h}"
    }
    category_dict = {str(i): c for i, c in enumerate(cfg.ml_classes)}
    linear = False
    normalize_by_bin_width = False
    if linear:
        split_value = 0
    else:
        if normalize_by_bin_width:
            split_value = 10001
        else:
            split_value = 101

    split_dict = {c: split_value for c in ["et", "mt", "tt", "em"]}

    bkg_processes = [
        "QCD",
        "VVT",
        "VVL",
        "VVJ",
        "W",
        "TTT",
        "TTL",
        "TTJ",
        "ZJ",
        "ZL",
        "ZTT"]
    all_bkg_processes = [b for b in bkg_processes]
    legend_bkg_processes = copy.deepcopy(bkg_processes)
    legend_bkg_processes.reverse()

    era = 2018
    plots = []
    channel = "mt"
    for category in channel_categories[channel]:
        rootfile = rootfile_parser.Rootfile_parser(os.path.join(args.workdir, "shapes_prefit.root"))
        bkg_processes = [b for b in all_bkg_processes]
        legend_bkg_processes = copy.deepcopy(bkg_processes)
        legend_bkg_processes.reverse()
        # create plot
        width = 600
        if linear:
            plot = dd.Plot(
                [0.3, [0.3, 0.28]], "ModTDR", r=0.04, l=0.14, width=width)
        else:
            plot = dd.Plot(
                [0.5, [0.3, 0.28]], "ModTDR", r=0.04, l=0.14, width=width)

        # get background histograms
        for process in bkg_processes:
            try:
                plot.add_hist(
                    rootfile.get(
                        era,
                        channel,
                        category,
                        process),
                    process,
                    "bkg")
                plot.setGraphStyle(
                    process, "hist", fillcolor=styles.color_dict[process])
            except BaseException:
                pass

        # get signal histograms
        plot_idx_to_add_signal = [0, 2] if linear else [1, 2]
        for i in plot_idx_to_add_signal:
            plot.subplot(i).add_hist(
                rootfile.get(era, channel, category, "ggH"), "ggH")
            plot.subplot(i).add_hist(
                rootfile.get(era, channel, category, "ggH"), "ggH_top")
            plot.subplot(i).add_hist(
                rootfile.get(era, channel, category, "qqH"), "qqH")
            plot.subplot(i).add_hist(
                rootfile.get(era, channel, category, "qqH"), "qqH_top")

        # get observed data and total background histograms
        # NOTE: With CMSSW_8_1_0 the TotalBkg definition has changed.
        plot.add_hist(
            rootfile.get(era, channel, category, "data_obs"), "data_obs")
        total_bkg = rootfile.get(era, channel, category, "TotalBkg")
        #ggHHist = rootfile.get(era, channel, category, "ggH")
        #qqHHist = rootfile.get(era, channel, category, "qqH")
        #total_bkg.Add(ggHHist, -1)
        # if qqHHist:
        #     total_bkg.Add(qqHHist, -1)
        plot.add_hist(total_bkg, "total_bkg")

        plot.subplot(0).setGraphStyle("data_obs", "e0")
        plot.subplot(0 if linear else 1).setGraphStyle(
            "ggH", "hist", linecolor=styles.color_dict["ggH"], linewidth=3)
        plot.subplot(
            0 if linear else 1).setGraphStyle(
            "ggH_top", "hist", linecolor=0)
        plot.subplot(0 if linear else 1).setGraphStyle(
            "qqH", "hist", linecolor=styles.color_dict["qqH"], linewidth=3)
        plot.subplot(
            0 if linear else 1).setGraphStyle(
            "qqH_top", "hist", linecolor=0)
        plot.subplot(0 if linear else 1).setGraphStyle(
            "VH", "hist", linecolor=styles.color_dict["VH"], linewidth=3)
        plot.subplot(
            0 if linear else 1).setGraphStyle(
            "VH_top", "hist", linecolor=0)
        plot.subplot(0 if linear else 1).setGraphStyle(
            "ttH", "hist", linecolor=styles.color_dict["ttH"], linewidth=3)
        plot.subplot(
            0 if linear else 1).setGraphStyle(
            "ttH_top", "hist", linecolor=0)
        plot.subplot(0 if linear else 1).setGraphStyle(
            "HWW", "hist", linecolor=styles.color_dict["HWW"], linewidth=3)
        plot.subplot(
            0 if linear else 1).setGraphStyle(
            "HWW_top", "hist", linecolor=0)
        plot.setGraphStyle(
            "total_bkg",
            "e2",
            markersize=0,
            fillcolor=styles.color_dict["unc"],
            linecolor=0)

        # assemble ratio
        bkg_ggH = plot.subplot(2).get_hist("ggH")
        bkg_qqH = plot.subplot(2).get_hist("qqH")
        bkg_ggH.Add(plot.subplot(2).get_hist("total_bkg"))
        if bkg_qqH:
            bkg_qqH.Add(plot.subplot(2).get_hist("total_bkg"))
        plot.subplot(2).add_hist(bkg_ggH, "bkg_ggH")
        plot.subplot(2).add_hist(bkg_ggH, "bkg_ggH_top")
        plot.subplot(2).add_hist(bkg_qqH, "bkg_qqH")
        plot.subplot(2).add_hist(bkg_qqH, "bkg_qqH_top")
        plot.subplot(2).setGraphStyle(
            "bkg_ggH",
            "hist",
            linecolor=styles.color_dict["ggH"],
            linewidth=3)
        plot.subplot(2).setGraphStyle("bkg_ggH_top", "hist", linecolor=0)
        plot.subplot(2).setGraphStyle(
            "bkg_qqH",
            "hist",
            linecolor=styles.color_dict["qqH"],
            linewidth=3)
        plot.subplot(2).setGraphStyle("bkg_qqH_top", "hist", linecolor=0)

        plot.subplot(2).normalize([
            "total_bkg", "bkg_ggH", "bkg_ggH_top", "bkg_qqH",
            "bkg_qqH_top", "data_obs"
        ], "total_bkg")

        # stack background processes
        plot.create_stack(bkg_processes, "stack")

        # normalize stacks by bin-width
        if normalize_by_bin_width:
            plot.subplot(0).normalizeByBinWidth()
            plot.subplot(1).normalizeByBinWidth()

        # set axes limits and labels
        plot.subplot(0).setYlims(
            split_dict[channel],
            max(2 * plot.subplot(0).get_hist("total_bkg").GetMaximum(),
                split_dict[channel] * 2))

        plot.subplot(2).setYlims(0.75, 1.35)
        if not linear:
            plot.subplot(1).setYlims(0.1, split_dict[channel])
            plot.subplot(1).setLogY()
            plot.subplot(1).setYlabel(
                "")  # otherwise number labels are not drawn on axis
        plot.subplot(2).setXlabel(styles.x_label_dict[channel][cfg.analysis_variable])
        if normalize_by_bin_width:
            plot.subplot(0).setYlabel("dN/d(NN output)")
        else:
            plot.subplot(0).setYlabel("N_{events}")

        plot.subplot(2).setYlabel("")

        # plot.scaleXTitleSize(0.8)
        # plot.scaleXLabelSize(0.8)
        # plot.scaleYTitleSize(0.8)
        plot.scaleYLabelSize(0.8)
        # plot.scaleXLabelOffset(2.0)
        plot.scaleYTitleOffset(1.1)

        #plot.subplot(2).setNYdivisions(3, 5)

        # if not channel == "tt" and category in ["11", "12", "13", "14", "15", "16"]:
        #    plot.subplot(2).changeXLabels(["0.2", "0.4", "0.6", "0.8", "1.0"])

        # draw subplots. Argument contains names of objects to be drawn in
        # corresponding order.
        procs_to_draw = [
            "stack",
            "total_bkg",
            "ggH",
            "ggH_top",
            "qqH",
            "qqH_top",
            "data_obs"] if linear else [
            "stack",
            "total_bkg",
            "data_obs"]
        plot.subplot(0).Draw(procs_to_draw)
        if not linear:
            plot.subplot(1).Draw(
                ["stack", "total_bkg", "ggH", "ggH_top", "qqH", "qqH_top",
                 "data_obs"])
        plot.subplot(2).Draw([
            "total_bkg", "bkg_ggH", "bkg_ggH_top", "bkg_qqH",
            "bkg_qqH_top", "data_obs"
        ])

        # create legends
        suffix = ["", "_top"]
        for i in range(2):

            plot.add_legend(width=0.6, height=0.15)
            for process in legend_bkg_processes:
                try:
                    plot.legend(i).add_entry(
                        0, process, styles.legend_label_dict
                        [process.replace("TTL", "TT").replace(
                            "VVL", "VV")],
                        'f')
                except BaseException:
                    pass
            plot.legend(i).add_entry(0, "total_bkg", "Bkg. unc.", 'f')
            plot.legend(i).add_entry(
                0 if linear else 1, "ggH%s" %
                suffix[i], "gg#rightarrowH", 'l')
            plot.legend(i).add_entry(
                0 if linear else 1, "qqH%s" %
                suffix[i], "qq#rightarrowH", 'l')
            plot.legend(i).add_entry(0, "data_obs", "Data", 'PE')
            plot.legend(i).setNColumns(3)
        plot.legend(0).Draw()
        plot.legend(1).setAlpha(0.0)
        plot.legend(1).Draw()

        for i in range(2):
            plot.add_legend(
                reference_subplot=2, pos=1, width=0.5, height=0.03)
            plot.legend(i + 2).add_entry(0, "data_obs", "Data", 'PE')
            plot.legend(
                i +
                2).add_entry(
                0 if linear else 1,
                "ggH%s" %
                suffix[i],
                "ggH+bkg.",
                'l')
            plot.legend(
                i +
                2).add_entry(
                0 if linear else 1,
                "qqH%s" %
                suffix[i],
                "qqH+bkg.",
                'l')
            plot.legend(i + 2).add_entry(0, "total_bkg", "Bkg. unc.", 'f')
            plot.legend(i + 2).setNColumns(4)
        plot.legend(2).Draw()
        plot.legend(3).setAlpha(0.0)
        plot.legend(3).Draw()

        # draw additional labels
        #plot.DrawCMS()
        plot.DrawLumi("59.7 fb^{-1} (2018, 13 TeV)")

        plot.DrawChannelCategoryLabel(
            "%s, %s" % (channel_dict[channel], category_dict[category]),
            begin_left=None)

        # save plot
        postfix = "prefit"
        for filetype in ["png", "pdf"]:
            plot.save(
                "%s/%s_%s_%s_%s.%s" %
                (args.workdir,
                 "2018",
                 channel,
                 category,
                 postfix,
                 filetype))
        # work around to have clean up seg faults only at the end of the
        # script
        plots.append(plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'plot_analysis.log'), logging.INFO)
    main(args)
