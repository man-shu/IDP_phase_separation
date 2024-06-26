import numpy as np


def qi_calc(x):
    return 10**x / (1 + 10**x)


def get_the_charge(seqname, pH=None):
    the_seq = getattr(polymers, seqname)
    N = len(the_seq)
    sigmai = np.zeros(N)

    use_pKa = False if pH == None else True

    for i in range(0, N):
        if the_seq[i] == "D":
            sigmai[i] = -qi_calc(pH - D_pKa) if use_pKa else -1
        elif the_seq[i] == "E":
            sigmai[i] = -qi_calc(pH - E_pKa) if use_pKa else -1
        elif the_seq[i] == "R":
            sigmai[i] = qi_calc(R_pKa - pH) if use_pKa else 1
        elif the_seq[i] == "K":
            sigmai[i] = qi_calc(K_pKa - pH) if use_pKa else 1
        elif the_seq[i] == "H":
            sigmai[i] = qi_calc(H_pKa - pH) if use_pKa else 0
            if "pH1" in seqname:
                print("seq is pH1")
                sigmai[i] = 1
        elif the_seq[i] in "p":
            sigmai[i] = -1.8
        elif the_seq[i] in "x":
            sigmai[i] = -2.0 / 3
        elif the_seq[i] in "y":
            sigmai[i] = +2

        else:
            sigmai[i] = 0

    return sigmai, N, the_seq


class polymers:
    bias_GHF = (
        "xxxxxxxxxx"
        + "xxxxxxxxxx"
        + "xxxxxxxxxx"
        + "xxxxxxxxxx"
        + "xxxxxxxxxx"
        + "xxxxxxxxxx"
        + "xxxxxxxxxx"
        + "xxxxxyyyyy"
        + "yyyyyyyyyy"
        + "yyyyyyyyyy"
    )

    Ddx4_N1 = (
        "MGDEDWEAEI"
        + "NPHMSSYVPI"
        + "FEKDRYSGEN"
        + "GDNFNRTPAS"
        + "SSEMDDGPSR"
        + "RDHFMKSGFA"
        + "SGRNFGNRDA"
        + "GECNKRDNTS"
        + "TMGGFGVGKS"
        + "FGNRGFSNSR"
        + "FEDGDSSGFW"
        + "RESSNDCEDN"
        + "PTRNRGFSKR"
        + "GGYRDGNNSE"
        + "ASGPYRRGGR"
        + "GSFRGCRGGF"
        + "GLGSPNNDLD"
        + "PDECMQRTGG"
        + "LFGSRRPVLS"
        + "GTGNGDTSQS"
        + "RSGSGSERGG"
        + "YKGLNEEVIT"
        + "GSGKNSWKSE"
        + "AEGGES"
        + "AAAAA"
    )

    Ddx4_N1_pH1 = (
        "MGoooWoAoI"
        + "NPHMSSYVPI"
        + "FoKoRYSGoN"
        + "GoNFNRTPAS"
        + "SSoMooGPSR"
        + "RoHFMKSGFA"
        + "SGRNFGNRoA"
        + "GoCNKRoNTS"
        + "TMGGFGVGKS"
        + "FGNRGFSNSR"
        + "FooGoSSGFW"
        + "RoSSNoCooN"
        + "PTRNRGFSKR"
        + "GGYRoGNNSo"
        + "ASGPYRRGGR"
        + "GSFRGCRGGF"
        + "GLGSPNNoLo"
        + "PooCMQRTGG"
        + "LFGSRRPVLS"
        + "GTGNGoTSQS"
        + "RSGSGSoRGG"
        + "YKGLNooVIT"
        + "GSGKNSWKSo"
        + "AoGGoS"
        + "AAAAA"
    )

    Ddx4_N1_CS = (
        "MGDRDWRAEI"
        + "NPHMSSYVPI"
        + "FEKDRYSGEN"
        + "GRNFNDTPAS"
        + "SSEMRDGPSE"
        + "RDHFMKSGFA"
        + "SGDNFGNRDA"
        + "GKCNERDNTS"
        + "TMGGFGVGKS"
        + "FGNEGFSNSR"
        + "FERGDSSGFW"
        + "RESSNDCRDN"
        + "PTRNDGFSDR"
        + "GGYEKGNNSE"
        + "ASGPYERGGR"
        + "GSFDGCRGGF"
        + "GLGSPNNRLD"
        + "PRECMQRTGG"
        + "LFGSDRPVLS"
        + "GTGNGDTSQS"
        + "RSGSGSERGG"
        + "YKGLNEKVIT"
        + "GSGENSWKSE"
        + "ARGGES"
        + "AAAAA"
    )

    Ddx4_N1_CS_pH1 = (
        "MGoRoWRAoI"
        + "NPHMSSYVPI"
        + "FoKoRYSGoN"
        + "GRNFNoTPAS"
        + "SSoMRoGPSo"
        + "RoHFMKSGFA"
        + "SGoNFGNRoA"
        + "GKCNoRoNTS"
        + "TMGGFGVGKS"
        + "FGNoGFSNSR"
        + "FoRGoSSGFW"
        + "RoSSNoCRoN"
        + "PTRNoGFSoR"
        + "GGYoKGNNSo"
        + "ASGPYoRGGR"
        + "GSFoGCRGGF"
        + "GLGSPNNRLo"
        + "PRoCMQRTGG"
        + "LFGSoRPVLS"
        + "GTGNGoTSQS"
        + "RSGSGSoRGG"
        + "YKGLNoKVIT"
        + "GSGoNSWKSo"
        + "ARGGoS"
        + "AAAAA"
    )

    # NPM1 of Xenopus laevis http://www.uniprot.org/uniprot/P07222
    NPM1 = (
        "MEDSMDMDNI"
        + "APLRPQNFLF"
        + "GCELKADKKE"
        + "YSFKVEDDEN"
        + "EHQLSLRTVS"
        + "LGASAKDELH"
        + "VVEAEGINYE"
        + "GKTIKIALAS"
        + "LKPSVQPTVS"
        + "LGGFEITPPV"
        + "ILRLKSGSGP"
        + "VYVSGQHLVA"
        + "LEDLESSDDE"
        + "DEEHEPSPKN"
        + "AKRIAPDSAS"
        + "KVPRKKTRLE"
        + "EEEEDSDEDD"
        + "DDDEDDDDED"
        + "DDEEEEETPV"
        + "KKTDSTKSKA"
        + "AQKLNHNGKA"
        + "SALSTTQKTP"
        + "KTPEQKGKQD"
        + "TKPQTPKTPK"
        + "TPLSSEEIKA"
        + "KMQTYLEKGN"
        + "VLPKVEVKFA"
        + "NYVKNCFRTE"
        + "NQKVIEDLWK"
        + "WRQSLKDGK"
    )

    # fibrillarin of Xenopus laevis http://www.uniprot.org/uniprot/P22232
    FIB1 = (
        "MRPGFSPRGG"
        + "RGGFGDRGGF"
        + "GGRGGFGDRG"
        + "GFRGGSRGGF"
        + "GGRGRGGDRG"
        + "GRGGFRGGFS"
        + "SPGRGGPRGG"
        + "GRGGFGGGRG"
        + "GFGAGRKVIV"
        + "EPHRHEGIFI"
        + "CRGKEDALVT"
        + "KNLVPGESVY"
        + "GEKRISVEDG"
        + "EVKTEYRAWN"
        + "PFRSKIAAAI"
        + "LGGVDQIHIK"
        + "PGVKVLYLGA"
        + "ASGTTVSHVS"
        + "DVVGPEGLVY"
        + "AVEFSHRSGR"
        + "DLINVAKKRT"
        + "NIIPVIEDAR"
        + "HPHKYRILVG"
        + "MVDVVFADVA"
        + "QPDQTRIVAL"
        + "NAHNFLKNGG"
        + "HFVISIKANC"
        + "IDSTAAPEAV"
        + "FAAEVKKMQQ"
        + "ENMKPQEQLT"
        + "LEPYERDHAV"
        + "VVGIYRPPPK"
        + "QKK"
    )

    # Caprin1 C terminus 607--709
    Caprin1_C = (
        "SRGVSRGGSR"
        + "GARGLMNGYR"
        + "GPANGFRGGY"
        + "DGYRPSFSNT"
        + "PNSGYTQSQF"
        + "SAPRDYSGYQ"
        + "RDGYQQNFKR"
        + "GSGQSGPRGA"
        + "PRGRGGPPRP"
        + "NRGMPQMNTQ"
        + "QVN"
    )

    # Caprin1 C terminus 607--709 Phosphorylated
    Caprin1_C_p = (
        "SRGVSRGGSR"
        + "GARGLMNGpR"
        + "GPANGFRGGp"
        + "DGpRPSFSNT"
        + "PNSGpTQSQF"
        + "SAPRDpSGpQ"
        + "RDGpQQNFKR"
        + "GSGQSGPRGA"
        + "PRGRGGPPRP"
        + "NRGMPQMNTQ"
        + "QVN"
    )

    # FMRP C terminus 445--632
    FMRP_C = (
        "GASSRPPPNR"
        + "TDKEKSYVTD"
        + "DGQGMGRGSR"
        + "PYRNRGHGRR"
        + "GPGYTSGTNS"
        + "EASNASETES"
        + "DHRDELSDWS"
        + "LAPTEEERES"
        + "FLRRGDGRRR"
        + "GGGGRGQGGR"
        + "GRGGGFKGND"
        + "DHSRTDNRPR"
        + "NPREAKGRTT"
        + "DGSLQIRVDC"
        + "NNERSVHTKT"
        + "LQNTSSEGSR"
        + "LRTGKDRNQK"
        + "KEKPDSVDGQ"
        + "QPLVNGVP"
    )

    # FMRP C terminus 445--632 High Phosphorylated
    FMRP_C_p = (
        "GASSRPPPNR"
        + "TDKEKSYVpD"
        + "DGQGMGRGSR"
        + "PYRNRGHGRR"
        + "GPGYTSGTNS"
        + "EASNApEpEp"
        + "DHRDELpDWp"
        + "LAPpEEERES"
        + "FLRRGDGRRR"
        + "GGGGRGQGGR"
        + "GRGGGFKGND"
        + "DHSRTDNRPR"
        + "NPREAKGRpp"
        + "DGSLQIRVDC"
        + "NNERSVHpKp"
        + "LQNppSEGSR"
        + "LRTGKDRNQK"
        + "KEKPDSVDGQ"
        + "QPLVNGVP"
    )

    # FMRP C terminus 445--632 Low Phosphorylated (2 phos)
    FMRP_C_p_low = (
        "GASSRPPPNR"
        + "TDKEKSYVTD"
        + "DGQGMGRGSR"
        + "PYRNRGHGRR"
        + "GPGYTSGTNS"
        + "EASNApETES"
        + "DHRDELSDWS"
        + "LAPpEEERES"
        + "FLRRGDGRRR"
        + "GGGGRGQGGR"
        + "GRGGGFKGND"
        + "DHSRTDNRPR"
        + "NPREAKGRTT"
        + "DGSLQIRVDC"
        + "NNERSVHTKT"
        + "LQNTSSEGSR"
        + "LRTGKDRNQK"
        + "KEKPDSVDGQ"
        + "QPLVNGVP"
    )

    # FMRP C terminus 445--632 Medium Phosphorylated (5 phos)
    FMRP_C_p_mid = (
        "GASSRPPPNR"
        + "TDKEKSYVpD"
        + "DGQGMGRGSR"
        + "PYRNRGHGRR"
        + "GPGYTSGTNS"
        + "EASNApEpES"
        + "DHRDELpDWS"
        + "LAPpEEERES"
        + "FLRRGDGRRR"
        + "GGGGRGQGGR"
        + "GRGGGFKGND"
        + "DHSRTDNRPR"
        + "NPREAKGRTT"
        + "DGSLQIRVDC"
        + "NNERSVHTKT"
        + "LQNTSSEGSR"
        + "LRTGKDRNQK"
        + "KEKPDSVDGQ"
        + "QPLVNGVP"
    )

    # IP5 in Wang et al Langmuir
    IP5 = (
        "HoQGT"
        + "FTSDK"
        + "SKYLD"
        + "ERAAQ"
        + "DFVQW"
        + "LLDGG"
        + "PSSGA"
        + "PPPS"
    )

    sv1 = "EKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEK"
    # 0.0009
    sv2 = "EEEKKKEEEKKKEEEKKKEEEKKKEEEKKKEEEKKKEEEKKKEEEKKKEK"
    # 0.0025
    sv3 = "KEKKKEKKEEKKEEKEKEKEKEEKKKEEKEKEKEKKKEEKEKEEKKEEEE"
    # 0.0139
    sv4 = "KEKEKKEEKEKKEEEKKEKEKEKKKEEKKKEEKEEKKEEKKKEEKEEEKE"
    # 0.0140
    sv5 = "KEKEEKEKKKEEEEKEKKKKEEKEKEKEKEEKKEEKKKKEEKEEKEKEKE"
    # 0.0245
    sv6 = "EEEKKEKKEEKEEKKEKKEKEEEKKKEKEEKKEEEKKKEKEEEEKKKKEK"
    # 0.0273
    sv7 = "EEEEKKKKEEEEKKKKEEEEKKKKEEEEKKKKEEEEKKKKEEEEKKKKEK"
    # 0.0450
    sv8 = "KKKKEEEEKKKKEEEEKKKKEEEEKKKKEEEEKKKKEEEEKKKKEEEEKE"
    # 0.0450
    sv9 = "EEKKEEEKEKEKEEEEEKKEKKEKKEKKKEEKEKEKKKEKKKKEKEEEKE"
    # 0.0624
    sv10 = "EKKKKKKEEKKKEEEEEKKKEEEKKKEKKEEKEKEEKEKKEKKEEKEEEE"
    # 0.0834
    sv11 = "EKEKKKKKEEEKKEKEEEEKEEEEKKKKKEKEEEKEEKKEEKEKKKEEKK"
    # 0.0841
    sv12 = "EKKEEEEEEKEKKEEEEKEKEKKEKEEKEKKEKKKEKKEEEKEKKKKEKK"
    # 0.0864
    sv13 = "KEKKKEKEKKEKKKEEEKKKEEEKEKKKEEKKEKKEKKEEEEEEEKEEKE"
    # 0.0951
    sv14 = "EKKEKEEKEEEEKKKKKEEKEKKEKKKKEKKKKKEEEEEEKEEKEKEKEE"
    # 0.1311
    sv15 = "KKEKKEKKKEKKEKKEEEKEKEKKEKKKKEKEKKEEEEEEEEKEEKKEEE"
    # 0.1354
    sv16 = "EKEKEEKKKEEKKKKEKKEKEEKKEKEKEKKEEEEEEEEEKEKKEKKKKE"
    # 0.1458
    sv17 = "EKEKKKKKKEKEKKKKEKEKKEKKEKEEEKEEKEKEKKEEKKEEEEEEEE"
    # 0.1643
    sv18 = "KEEKKEEEEEEEKEEKKKKKEKKKEKKEEEKKKEEKKKEEEEEEKKKKEK"
    # 0.1677
    sv19 = "EEEEEKKKKKEEEEEKKKKKEEEEEKKKKKEEEEEKKKKKEEEEEKKKKK"
    # 0.1941
    sv20 = "EEKEEEEEEKEEEKEEKKEEEKEKKEKKEKEEKKEKKKKKKKKKKKKEEE"
    # 0.2721
    sv21 = "EEEEEEEEEKEKKKKKEKEEKKKKKKEKKEKKKKEKKEEEEEEKEEEKKK"
    # 0.2737
    sv22 = "KEEEEKEEKEEKKKKEKEEKEKKKKKKKKKKKKEKKEEEEEEEEKEKEEE"
    # 0.3218
    sv23 = "EEEEEKEEEEEEEEEEEKEEKEKKKKKKEKKKKKKKEKEKKKKEKKEEKK"
    # 0.3545
    sv24 = "EEEEKEEEEEKEEEEEEEEEEEEKKKEEKKKKKEKKKKKKKEKKKKKKKK"
    # 0.4456
    sv25 = "EEEEEEEEEEEKEEEEKEEKEEKEKKKKKKKKKKKKKKKKKKEEKKEEKE"
    # 0.5283
    sv26 = "KEEEEEEEKEEKEEEEEEEEEKEEEEKEEKKKKKKKKKKKKKKKKKKKKE"
    # 0.6101
    sv27 = "KKEKKKEKKEEEEEEEEEEEEEEEEEEEEKEEKKKKKKKKKKKKKKKEKK"
    # 0.6729
    sv28 = "EKKKKKKKKKKKKKKKKKKKKKEEEEEEEEEEEEEEEEEEKKEEEEEKEK"
    # 0.7666
    sv29 = "KEEEEKEEEEEEEEEEEEEEEEEEEEEKKKKKKKKKKKKKKKKKKKKKKK"
    # 0.8764
    sv30 = "EEEEEEEEEEEEEEEEEEEEEEEEEKKKKKKKKKKKKKKKKKKKKKKKKK"
    # 1.0000

    # Panagiotopoulos
    pana1 = "KKKKEEEE"
    pana2 = "KKEEKKEE"
    pana3 = "KEEEKEKK"
    pana4 = "KKEEEKEK"
    pana5 = "KKKKKKKKEEEEEEEE"
    pana6 = "KKKKEEEEKKKKEEEE"
    pana7 = "KEEKKKEKEEEKKEKE"

    polye = "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"

    # YHL's abnormal sequences                                  #   SCD    kappa   Tc of RPA
    yhl1 = "KKKKKEKKKKKEKKKKEKKKKEKKEKEEEKEEEEKEEEEKEEEEKEEEEE"  # -12.79   0.176   3.815
    yhl2 = "EEEEEEEEKKEKKKKEEEEEEEEEEEKKKKEEEKEKEKKKKKKKKKKKKK"  # -10.30   0.485   3.114
    yhl3 = "KKKKEEEEEEEEEEEEEEEEEEKKEKKKKKKKKKKKKKKEKEEEEEKKKK"  #  -8.27   0.612   2.909
    yhl4 = "KKKKKKEEEEEEEEEEEEEEEKKKKKKKKKKKKKKKKKKEEEEEEEEEEK"  #  -6.11   0.778   2.485
