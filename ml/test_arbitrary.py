import ROOT

path = '/work/gristo/mlnll-analysis/output/32_bins_shapes/cmb/common/htt_input_2018.root'
f = ROOT.TFile(path)
d = f.Get('htt_mt_0_2018')
procs = {}
for key in d.GetListOfKeys():
    name = key.GetName()
    if 'qqH125_THU_qqH' in name:
        h = d.Get(name)
        procs[name] = [h.Integral()]
hdata = []
hcount = []

hdata.append(procs[name])
hcount.append(procs[name][0])
print(procs[name])
print(procs[name][0])

