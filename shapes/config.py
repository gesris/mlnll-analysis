from ntuple_processor.utils import Selection
from ntuple_processor.utils import Cut
from ntuple_processor.utils import Weight
from ntuple_processor.variations import ReplaceCut

# Base path to main ntuples
ntuples_base = "/ceph/htautau/deeptau_02-20/2018/ntuples/"

# No friend trees
friends_base = []

# File list
files = {
    'singlemuon': [
        'SingleMuon_Run2018A_17Sep2018v2_13TeV_MINIAOD',
        'SingleMuon_Run2018B_17Sep2018v1_13TeV_MINIAOD',
        'SingleMuon_Run2018C_17Sep2018v1_13TeV_MINIAOD',
        'SingleMuon_Run2018D_22Jan2019v2_13TeV_MINIAOD'
        ],
    'wjets': [
        'W1JetsToLNu_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2',
        'W2JetsToLNu_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2',
        'W3JetsToLNu_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2',
        'W4JetsToLNu_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2',
        'WJetsToLNu_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2'
        ],
    'dy': [
        'DY1JetsToLLM50_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2',
        'DY2JetsToLLM50_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2',
        'DY3JetsToLLM50_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2',
        'DY4JetsToLLM50_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v1',
        'DYJetsToLLM10to50_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2',
        'DYJetsToLLM50_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v1'
        ],
    'tt': [
        'TTTo2L2Nu_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'TTToHadronic_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'TTToSemiLeptonic_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1'
        ],
    'vv': [
        'WW_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_pythia8_v2',
        'ZZ_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_pythia8_v2',
        'WZ_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_pythia8_v3',
        'STt-channelantitop4fInclusiveDecaysTuneCP5_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'STt-channeltop4fInclusiveDecaysTuneCP5_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'STtWantitop5finclusiveDecaysTuneCP5_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_ext1-v1',
        'STtWtop5finclusiveDecaysTuneCP5_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_ext1-v1'
        ],
    'ggh': [
        'GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2',
        'GluGluHToTauTauHTXSFilterSTXS1p1Bin104to105M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'GluGluHToTauTauHTXSFilterSTXS1p1Bin106M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2',
        'GluGluHToTauTauHTXSFilterSTXS1p1Bin107to109M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'GluGluHToTauTauHTXSFilterSTXS1p1Bin110to113M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2',
        'GluGluHToTauTauM125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2'
        ],
    'qqh': [
        'VBFHToTauTauHTXSFilterSTXS1p1Bin203to205M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'VBFHToTauTauHTXSFilterSTXS1p1Bin206M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'VBFHToTauTauHTXSFilterSTXS1p1Bin207to210M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'VBFHToTauTauM125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_ext1-v1',
        'ZHToTauTauM125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2',
        'WminusHToTauTauM125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2',
        'WplusHToTauTauM125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2'
        ]
    }

# Selections

channel = Selection(name = "mt",
               cuts = [
                       ("flagMETFilter == 1", "METFilter"),
                       ("extraelec_veto<0.5", "extraelec_veto"),
                       ("extramuon_veto<0.5", "extramuon_veto"),
                       ("dilepton_veto<0.5", "dilepton_veto"),
                       ("againstMuonTight3_2>0.5", "againstMuonDiscriminator"),
                       ("againstElectronVLooseMVA6_2>0.5", "againstElectronDiscriminator"),
                       ("byTightIsolationMVArun2017v2DBoldDMwLT2017_2>0.5", "tau_iso"),
                       ("iso_1<0.15", "muon_iso"),
                       ("q_1*q_2<0", "os"),
                       ("mt_1<50", "m_t"),
                       ("pt_2>30 && ((trg_singlemuon_27 == 1) || (trg_singlemuon_24 == 1) || (pt_1 < 25 && trg_crossmuon_mu20tau27 == 1))", "trg_selection")
               ])

lumi_weight = ("59.7 * 1000.0", "lumi")

w = Selection(name = "w",
        weights = [
            ("generatorWeight", "generatorWeight"),
            ("((0.00092600048*((npartons <= 0 || npartons >= 5)*1.0 + (npartons == 1)*0.1647043928 + (npartons == 2)*0.128547226623 + (npartons == 3)*0.0767138313139 + (npartons == 4)*0.0631529545476)) * (genbosonmass>=0.0) + numberGeneratedEventsWeight * crossSectionPerEventWeight * (genbosonmass<0.0))", "wj_stitching_weight"),
            ("puweight", "puweight"),
            ("idWeight_1*idWeight_2", "idweight"),
            ("isoWeight_1*isoWeight_2", "isoweight"),
            ("trackWeight_1*trackWeight_2", "trackweight"),
            ("(crossTriggerMCWeight_1*(crossTriggerMCWeight_1<10 && crossTriggerMCWeight_1>0.1)+(crossTriggerMCWeight_1>10 || crossTriggerMCWeight_1<0.1))*(pt_1<25) + (trigger_24_27_Weight_1*(pt_1>25))", "triggerweight"),
            ("eleTauFakeRateWeight*muTauFakeRateWeight", "leptonTauFakeRateWeight"),
            ("((gen_match_2 == 5)*0.90 + (gen_match_2 != 5))", "taubyIsoIdWeight"),
            lumi_weight
            ]
        )

dy = Selection(name = "dy",
        weights = [
            ("generatorWeight", "generatorWeight"),
            ("((genbosonmass >= 50.0)*0.00005754202*((npartons == 0 || npartons >= 5)*1.0 + (npartons == 1)*0.194267667208 + (npartons == 2)*0.21727746547 + (npartons == 3)*0.26760465744 + (npartons == 4)*0.294078683662) + (genbosonmass < 50.0)*numberGeneratedEventsWeight*crossSectionPerEventWeight)", "z_stitching_weight"),
            ("puweight", "puweight"),
            ("idWeight_1*idWeight_2", "idweight"),
            ("isoWeight_1*isoWeight_2", "isoweight"),
            ("trackWeight_1*trackWeight_2", "trackweight"),
            ("(crossTriggerMCWeight_1*(crossTriggerMCWeight_1<10 && crossTriggerMCWeight_1>0.1)+(crossTriggerMCWeight_1>10 || crossTriggerMCWeight_1<0.1))*(pt_1<25) + (trigger_24_27_Weight_1*(pt_1>25))", "triggerweight"),
            ("((gen_match_2 == 5)*0.90 + (gen_match_2 != 5))", "taubyIsoIdWeight"),
            ("zPtReweightWeight", "zPtReweightWeight"),
            lumi_weight
            ]
        )

ztt = Selection(name = "ztt",
        cuts = [
            ("gen_match_1==4 && gen_match_2==5", "ztt_cut")
            ]
        )

zl = Selection(name = "zl",
        cuts = [
            ("!(gen_match_1==4 && gen_match_2==5) && !(gen_match_2 == 6)", "zl_cut")
            ]
        )

zj = Selection(name = "zj",
        cuts = [
            ("gen_match_2 == 6", "zj_cut")
            ]
        )

tt = Selection(name = "tt",
        weights = [
            ("generatorWeight", "generatorWeight"),
            ("numberGeneratedEventsWeight", "numberGeneratedEventsWeight"),
            ("(abs(crossSectionPerEventWeight - 380.1) < 0.1)*377.96 + (abs(crossSectionPerEventWeight - 87.31) < 0.1)*88.29 + (abs(crossSectionPerEventWeight - 364.4) < 0.1)*365.35", "crossSectionPerEventWeight"),
            ("puweight", "puweight"),
            ("idWeight_1*idWeight_2", "idweight"),
            ("isoWeight_1*isoWeight_2", "isoweight"),
            ("trackWeight_1*trackWeight_2", "trackweight"),
            ("topPtReweightWeight", "topPtReweightWeight"),
            ("(crossTriggerMCWeight_1*(crossTriggerMCWeight_1<10 && crossTriggerMCWeight_1>0.1)+(crossTriggerMCWeight_1>10 || crossTriggerMCWeight_1<0.1))*(pt_1<25) + (trigger_24_27_Weight_1*(pt_1>25))", "triggerweight"),
            ("eleTauFakeRateWeight*muTauFakeRateWeight", "leptonTauFakeRateWeight"),
            ("((gen_match_2 == 5)*0.90 + (gen_match_2 != 5))", "taubyIsoIdWeight"),
            lumi_weight
            ]
        )

ttt = Selection(name = "ttt",
        cuts = [
            ("gen_match_1==4 && gen_match_2==5", "ttt_cut")
            ]
        )

ttl = Selection(name = "ttl",
        cuts = [
            ("!(gen_match_1==4 && gen_match_2==5) & !(gen_match_2 == 6)", "ttl_cut")
            ]
        )

ttj = Selection(name = "ttj",
        cuts = [
            ("(gen_match_2 == 6 && gen_match_2 == 6)", "ttj_cut")
            ]
        )

vv = Selection(name = "vv",
        weights = [
            ("generatorWeight", "generatorWeight"),
            ("numberGeneratedEventsWeight", "numberGeneratedEventsWeight"),
            ("puweight", "puweight"),
            ("idWeight_1*idWeight_2", "idweight"),
            ("isoWeight_1*isoWeight_2", "isoweight"),
            ("trackWeight_1*trackWeight_2", "trackweight"),
            ("(crossTriggerMCWeight_1*(crossTriggerMCWeight_1<10 && crossTriggerMCWeight_1>0.1)+(crossTriggerMCWeight_1>10 || crossTriggerMCWeight_1<0.1))*(pt_1<25) + (trigger_24_27_Weight_1*(pt_1>25))", "triggerweight"),
            ("eleTauFakeRateWeight*muTauFakeRateWeight", "leptonTauFakeRateWeight"),
            ("((gen_match_2 == 5)*0.90 + (gen_match_2 != 5))", "taubyIsoIdWeight"),
            ("crossSectionPerEventWeight", "crossSectionPerEventWeight"),
            lumi_weight
            ]
        )

vvt = Selection(name = "vvt",
        cuts = [
            ("gen_match_1==4 && gen_match_2==5", "vvt_cut")
            ]
        )
vvl = Selection(name = "vvl",
        cuts = [
            ("!(gen_match_1==4 && gen_match_2==5) & !(gen_match_2 == 6)", "vvl_cut")
            ]
        )

vvj = Selection(name = "vvj",
        cuts = [
            ("(gen_match_2 == 6 && gen_match_2 == 6)", "vvj_cut")
            ]
        )

htt = Selection(name = "htt",
        weights = [
            ("generatorWeight", "generatorWeight"),
            ("puweight", "puweight"),
            ("idWeight_1*idWeight_2", "idweight"),
            ("isoWeight_1*isoWeight_2", "isoweight"),
            ("trackWeight_1*trackWeight_2", "trackweight"),
            ("(crossTriggerMCWeight_1*(crossTriggerMCWeight_1<10 && crossTriggerMCWeight_1>0.1)+(crossTriggerMCWeight_1>10 || crossTriggerMCWeight_1<0.1))*(pt_1<25) + (trigger_24_27_Weight_1*(pt_1>25))", "triggerweight"),
            ("eleTauFakeRateWeight*muTauFakeRateWeight", "leptonTauFakeRateWeight"),
            ("((gen_match_2 == 5)*0.90 + (gen_match_2 != 5))", "taubyIsoIdWeight"),
            lumi_weight
            ]
        )

ggh = Selection(name = "ggh",
        weights = [
            ("ggh_NNLO_weight", "gghNNLO"),
            ("1.01", "bbh_inclusion_weight"),
            ("(((htxs_stage1p1cat==100||htxs_stage1p1cat==102||htxs_stage1p1cat==103)*crossSectionPerEventWeight*numberGeneratedEventsWeight+"
             "(htxs_stage1p1cat==101)*2.09e-8+"
             "(htxs_stage1p1cat==104||htxs_stage1p1cat==105)*4.28e-8+"
             "(htxs_stage1p1cat==106)*1.39e-8+"
             "(htxs_stage1p1cat>=107&&htxs_stage1p1cat<=109)*4.90e-8+"
             "(htxs_stage1p1cat>=110&&htxs_stage1p1cat<=113)*9.69e-9"
             ")*(abs(crossSectionPerEventWeight - 0.00538017) > 1e-5) + numberGeneratedEventsWeight*crossSectionPerEventWeight*(abs(crossSectionPerEventWeight - 0.00538017) < 1e-5))", "ggh_stitching_weight")
            ],
        cuts = [
            ("(htxs_stage1p1cat>=100)&&(htxs_stage1p1cat<=113)", "htxs_cut")
            ]
        )

qqh = Selection(name = "qqh",
        weights = [
            ("((htxs_stage1p1cat>=200&&htxs_stage1p1cat<=202)||abs(crossSectionPerEventWeight-0.04774)<0.001||abs(crossSectionPerEventWeight-0.052685)<0.001||abs(crossSectionPerEventWeight-0.03342)<0.001)*crossSectionPerEventWeight*numberGeneratedEventsWeight+(abs(crossSectionPerEventWeight-0.04774)>=0.001&&abs(crossSectionPerEventWeight-0.052685)>=0.001&&abs(crossSectionPerEventWeight-0.03342)>=0.001)*("
             "(htxs_stage1p1cat>=203&&htxs_stage1p1cat<=205)*9.41e-9+"
             "(htxs_stage1p1cat==206)*8.52e-9+"
             "(htxs_stage1p1cat>=207&&htxs_stage1p1cat<=210)*1.79e-8"
             ")", "qqh_stitching_weight")
            ],
        cuts = [
            ("(htxs_stage1p1cat>=200)&&(htxs_stage1p1cat<=210)", "htxs_cut")
            ]
        )

# Variations

same_sign = ReplaceCut("same_sign", "os", Cut("q_1*q_2>0", "ss"))
