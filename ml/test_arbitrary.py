import ROOT

path = '/work/gristo/mlnll-analysis/output/32_bins_shapes/cmb/common/htt_input_2018.root'
f = ROOT.TFile(path)
d = f.Get('htt_mt_0_2018')
procs = {}
for key in d.GetListOfKeys():
    name = key.GetName()
    if 'qqH125_THU_qqH_JET01' in name:
        h = d.Get(name)
        print(h.Integrate())
        #procs[name] = [h.Integrate()]
#hdata = []
#hcount = []

#hdata.append(procs[name])
#hcount.append(procs[name][0])

