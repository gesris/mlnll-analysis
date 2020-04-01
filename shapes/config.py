from ntuple_processor.utils import Selection

# Base path to main ntuples
ntuples_base = "/ceph/htautau/deeptau_02-20/2018/ntuples/"

# No friend trees
friends_base = []

# Files associated to processes
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
