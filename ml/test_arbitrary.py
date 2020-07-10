import ROOT

path = '/work/gristo/mlnll-analysis/output/32_bins_shapes/cmb/common/htt_input_2018.root'
f = ROOT.TFile(path)
d = f.Get('htt_mt_0_2018')
procs = {}
hdata = []
hcount = []
for key in d.GetListOfKeys():
    name = key.GetName()
    if 'ggH125_THU_ggH_' in name:
        h = d.Get(name)
        procs[name] = [h.Integral()]
        h.Draw("HIST SAME")

        hdata.append(name)
        hcount.append(procs[name][0])

for i in range(0, len(hdata)):
    print("Name: {},        Yield: {:.3f}".format(hdata[i], hcount[i]))


