
import os
import argparse

from utils import config as cfg
import ROOT

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rc("font", size=16, family="serif")

def main(args):
    filepath = os.path.join(args.workdir, 'cmb/common/htt_input_2018.root')
    #filepath = '/work/gristo/mlnll-analysis/output/4_bins_mig01x100_nosysimpl_shapes_temp/cmb/common/htt_input_2018.root'
    dirpath = 'htt_mt_0_2018'
    nominal = 'ggH125'
    up = 'ggH125_THU_ggH_Mig01Up'
    down = 'ggH125_THU_ggH_Mig01Down'
    tfile = ROOT.TFile(filepath, 'UPDATE')
    folder = tfile.Get(dirpath)
    folder.cd()
    
    ## Assign hists
    h_nominal = folder.Get(nominal)
    h_up    = folder.Get(up)
    h_down  = folder.Get(down)
    h_qqH   = folder.Get('qqH125')
    h_ztt   = folder.Get('ZTT')
    h_zl    = folder.Get('ZL')
    h_zj    = folder.Get('ZJ')
    h_w     = folder.Get('W')
    h_ttt   = folder.Get('TTT')
    h_ttj   = folder.Get('TTJ')
    h_ttl   = folder.Get('TTL')
    h_vvj   = folder.Get('VVJ')
    h_vvt   = folder.Get('VVT')
    h_vvl   = folder.Get('VVL')
    h_qcd   = folder.Get('QCD')
    nbins = h_nominal.GetNbinsX()
    scale = args.scale

    Htt = []
    Htt_up = []
    Htt_down = []
    Ztt = []
    W = []
    ttbar = []
    vv = []
    qcd = []

    for i in range(1, nbins+1):
        ## Assign bincontent
        c_nominal = h_nominal.GetBinContent(i)
        c_up = h_up.GetBinContent(i)
        c_down = h_down.GetBinContent(i)
        c_qqH = h_qqH.GetBincontent(i)
        c_ztt = h_ztt.GetBinContent(i)
        c_zl = h_zl.GetBinContent(i)
        c_zj = h_zj.GetBinContent(i)
        c_w = h_w.GetBinContent(i)
        c_ttt = h_ttt.GetBinContent(i)
        c_ttj = h_ttj.GetBinContent(i)
        c_ttl = h_ttl.GetBinContent(i)
        c_vvt = h_vvt.GetBinContent(i)
        c_vvj = h_vvj.GetBinContent(i)
        c_vvl = h_vvl.GetBinContent(i)
        c_qcd = h_qcd.GetBinContent(i)

        ## calculate shift * scale
        diff_up = c_up - c_nominal
        diff_down = c_down - c_nominal
        h_up.SetBinContent(i, c_nominal + scale * diff_up)
        h_down.SetBinContent(i, c_nominal + scale * diff_down)
        c_up = h_up.GetBinContent(i)
        c_down = h_down.GetBinContent(i)

        ## Append content to arrays
        Htt.append(np.max(c_nominal + c_qqH, 1e-2))
        Htt_up.append(np.max(c_up + c_qqH, 1e-2))
        Htt_down.append(np.max(c_down + c_qqH, 1e-2))
        Ztt.append(np.max(c_zj + c_zl + c_ztt, 1e-2))
        W.append(np.max(c_w, 1e-2))
        ttbar.append(np.max(c_ttj + c_ttl + c_ttt, 1e-2))
        vv.append(np.max(c_vvj + c_vvl + c_vvt, 1e-2))
        qcd.append(np.max(c_qcd, 1e-2))
    tfile.Close()    

    ## Plot configs
    bins = cfg.analysis_binning
    upper_edges, lower_edges = bins[1:], bins[:-1]
    bins_center = []
    for i in range(0, len(bins) - 1):
        bins_center.append(bins[i] + (bins[i + 1] - bins[i]) / 2)

    ## Plot hists    
    plt.figure(figsize=(7, 6))
    plt.hist(bins_center, weights=Htt, bins=bins, histtype="step", lw=2, color="C0")
    plt.hist(bins_center, weights=Htt_up, bins=bins, histtype="step", lw=2, ls=':', color="C0")
    plt.hist(bins_center, weights=Htt_down, bins=bins, histtype="step", lw=2, ls=':', color="C0")
    plt.plot([0], [0], lw=2, color="C0", label="Htt")
    plt.plot([0], [0], lw=2, ls=':', color="C0", label="Htt Up")
    plt.plot([0], [0], lw=2, ls='--', color="C0", label="Htt Down")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., prop={'size': 14})
    plt.xlabel("$f$")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.savefig(os.path.join(args.workdir, 'histogram_combine.png'), bbox_inches = "tight")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('scale', type=int, help='Nuisance scale')
    args = parser.parse_args()
    main(args)
