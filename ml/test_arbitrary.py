import ROOT

path = '/work/gristo/mlnll-analysis/output/32_bins_shapes/cmb/common/htt_input_2018.root'
f = ROOT.TFile(path)
d = f.Get('htt_mt_0_2018')
nbins = 32

yields = {}
bincounts = {}

hnames = []     # contains all names of root histograms
hyields = []    # contains yields of root histograms
hists = []      # contains python histograms of root histograms

for key in d.GetListOfKeys():
    name = key.GetName()
    if 'ggH125' in name:
        h = d.Get(name)

        yields[name] = [h.Integral()]

        hnames.append(name)
        hyields.append(yields[name][0])
        bincounts[name] = [h.GetBinContent(i + 1) for i in range(nbins)]
        hists.append(bincounts[name])

# first histogram is nominal ggH125, rest systematics
nominal_ggH125 = hists[0]
systematics_ggH125 = hists[1:]
print(nominal_ggH125)
print(systematics_ggH125)

#for i in range(0, len(hnames)):
#    print("Name: {},        Yield: {:.3f}\nHistogram: {}".format(hnames[i], hyields[i], hists[i]))


