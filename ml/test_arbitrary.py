import ROOT

path = '/work/gristo/mlnll-analysis/output/32_bins_shapes/cmb/common/htt_input_2018.root'
f = ROOT.TFile(path)
d = f.Get('htt_mt_0_2018')
nbins = 32

yields = {}
bincounts = {}

hnames = []
hyields = []

for key in d.GetListOfKeys():
    name = key.GetName()
    if 'ggH125' in name:
        h = d.Get(name)

        yields[name] = [h.Integral()]

        hnames.append(name)
        hyields.append(yields[name][0])
        bincounts[name] = [h.GetBinContent(i + 1) for i in range(nbins)]
        print("1",bincounts[name])
        print("2",bincounts)

hist = []


#for i in range(0, len(hnames)):
#    print("Name: {},        Yield: {:.3f}".format(hnames[i], hyields[i]))


