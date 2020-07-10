import ROOT

path = '/work/gristo/mlnll-analysis/output/32_bins_shapes/cmb/common/htt_input_2018.root'
f = ROOT.TFile(path)
d = f.Get('htt_mt_0_2018')
procs = {}
for key in d.GetListOfKeys():
    name = key.GetName()
    if 'ggH125_THU_ggH' in name:
        continue
    h = d.Get(name)
    procs[name] = [h.Integrate()]
hdata = []
hcount = []

hdata.append(procs[name])
hcount.append(procs[name][0])

