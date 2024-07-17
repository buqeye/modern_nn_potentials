import numpy as np
from cheftgp.gaussianprocess import ScaleSchemeBunch

# for each choice of scale and scheme, sets the total possible orders and nomenclature
EKM0p8fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_EKM_R-0p8fm.h5",
    np.array([0, 2, 3, 4, 5]),
    ["Oranges", "Greens", "Blues", "Reds"],
    "SCS",
    "0p8fm",
    orders_labels_dict={
        5: r"N$^{4}$LO",
        4: r"N$^{3}$LO",
        3: r"N$^{2}$LO",
        2: r"NLO",
    },
    orders_names_dict={
        5: "N4LO",
        4: "N3LO",
        3: "N2LO",
        2: "NLO",
    },
)
EKM0p9fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_EKM_R-0p9fm.h5",
    np.array([0, 2, 3, 4, 5]),
    ["Oranges", "Greens", "Blues", "Reds"],
    "SCS",
    "0p9fm",
    orders_labels_dict={
        5: r"N$^{4}$LO",
        4: r"N$^{3}$LO",
        3: r"N$^{2}$LO",
        2: r"NLO",
    },
    orders_names_dict={
        5: "N4LO",
        4: "N3LO",
        3: "N2LO",
        2: "NLO",
    },
)
EKM1p0fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_EKM_R-1p0fm.h5",
    np.array([0, 2, 3, 4, 5]),
    ["Oranges", "Greens", "Blues", "Reds"],
    "SCS",
    "1p0fm",
    orders_labels_dict={
        5: r"N$^{4}$LO",
        4: r"N$^{3}$LO",
        3: r"N$^{2}$LO",
        2: r"NLO",
    },
    orders_names_dict={
        5: "N4LO",
        4: "N3LO",
        3: "N2LO",
        2: "NLO",
    },
)
EKM1p1fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_EKM_R-1p1fm.h5",
    np.array([0, 2, 3, 4, 5]),
    ["Oranges", "Greens", "Blues", "Reds"],
    "SCS",
    "1p1fm",
    orders_labels_dict={
        5: r"N$^{4}$LO",
        4: r"N$^{3}$LO",
        3: r"N$^{2}$LO",
        2: r"NLO",
    },
    orders_names_dict={
        5: "N4LO",
        4: "N3LO",
        3: "N2LO",
        2: "NLO",
    },
)
EKM1p2fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_EKM_R-1p2fm.h5",
    np.array([0, 2, 3, 4, 5]),
    ["Oranges", "Greens", "Blues", "Reds"],
    "SCS",
    "1p2fm",
    orders_labels_dict={
        5: r"N$^{4}$LO",
        4: r"N$^{3}$LO",
        3: r"N$^{2}$LO",
        2: r"NLO",
    },
    orders_names_dict={
        5: "N4LO",
        4: "N3LO",
        3: "N2LO",
        2: "NLO",
    },
)

RKE400MeV = ScaleSchemeBunch(
    "../observables_data/scattering_observables_RKE_L-400MeV.h5",
    np.array([0, 2, 3, 4, 5, 6]),
    ["Oranges", "Greens", "Blues", "Reds", "Purples"],
    "SMS",
    "400MeV",
    orders_labels_dict={
        6: r"N$^{4}$LO$^{+}$",
        5: r"N$^{4}$LO",
        4: r"N$^{3}$LO",
        3: r"N$^{2}$LO",
        2: r"NLO",
    },
    orders_names_dict={
        6: "N4LO+",
        5: "N4LO",
        4: "N3LO",
        3: "N2LO",
        2: "NLO",
    },
)
RKE450MeV = ScaleSchemeBunch(
    "../observables_data/scattering_observables_RKE_L-450MeV.h5",
    np.array([0, 2, 3, 4, 5, 6]),
    ["Oranges", "Greens", "Blues", "Reds", "Purples"],
    "SMS",
    "450MeV",
    orders_labels_dict={
        6: r"N$^{4}$LO$^{+}$",
        5: r"N$^{4}$LO",
        4: r"N$^{3}$LO",
        3: r"N$^{2}$LO",
        2: r"NLO",
    },
    orders_names_dict={
        6: "N4LO+",
        5: "N4LO",
        4: "N3LO",
        3: "N2LO",
        2: "NLO",
    },
)
RKE500MeV = ScaleSchemeBunch(
    "../observables_data/scattering_observables_RKE_L-500MeV.h5",
    np.array([0, 2, 3, 4, 5, 6]),
    ["Oranges", "Greens", "Blues", "Reds", "Purples"],
    "SMS",
    "500MeV",
    orders_labels_dict={
        6: r"N$^{4}$LO$^{+}$",
        5: r"N$^{4}$LO",
        4: r"N$^{3}$LO",
        3: r"N$^{2}$LO",
        2: r"NLO",
    },
    orders_names_dict={
        6: "N4LO+",
        5: "N4LO",
        4: "N3LO",
        3: "N2LO",
        2: "NLO",
    },
)
RKE550MeV = ScaleSchemeBunch(
    "../observables_data/scattering_observables_RKE_L-550MeV.h5",
    np.array([0, 2, 3, 4, 5, 6]),
    ["Oranges", "Greens", "Blues", "Reds", "Purples"],
    "SMS",
    "550MeV",
    orders_labels_dict={
        6: r"N$^{4}$LO$^{+}$",
        5: r"N$^{4}$LO",
        4: r"N$^{3}$LO",
        3: r"N$^{2}$LO",
        2: r"NLO",
    },
    orders_names_dict={
        6: "N4LO+",
        5: "N4LO",
        4: "N3LO",
        3: "N2LO",
        2: "NLO",
    },
)

EMN450MeV = ScaleSchemeBunch(
    "../observables_data/scattering_observables_EM-450MeV.h5",
    np.array([0, 2, 3, 4, 5]),
    ["Oranges", "Greens", "Blues", "Reds"],
    "EMN",
    "450MeV",
    orders_labels_dict={
        5: r"N$^{4}$LO",
        4: r"N$^{3}$LO",
        3: r"N$^{2}$LO",
        2: r"NLO",
    },
    orders_names_dict={
        5: "N4LO",
        4: "N3LO",
        3: "N2LO",
        2: "NLO",
    },
)
EMN500MeV = ScaleSchemeBunch(
    "../observables_data/scattering_observables_EM-500MeV.h5",
    np.array([0, 2, 3, 4, 5]),
    ["Oranges", "Greens", "Blues", "Reds"],
    "EMN",
    "500MeV",
    orders_labels_dict={
        5: r"N$^{4}$LO",
        4: r"N$^{3}$LO",
        3: r"N$^{2}$LO",
        2: r"NLO",
    },
    orders_names_dict={
        5: "N4LO",
        4: "N3LO",
        3: "N2LO",
        2: "NLO",
    },
)
EMN550MeV = ScaleSchemeBunch(
    "../observables_data/scattering_observables_EM-550MeV.h5",
    np.array([0, 2, 3, 4, 5]),
    ["Oranges", "Greens", "Blues", "Reds"],
    "EMN",
    "550MeV",
    orders_labels_dict={
        5: r"N$^{4}$LO",
        4: r"N$^{3}$LO",
        3: r"N$^{2}$LO",
        2: r"NLO",
    },
    orders_names_dict={
        5: "N4LO",
        4: "N3LO",
        3: "N2LO",
        2: "NLO",
    },
)

GT0p9fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_Gezerlis-0p9fm.h5",
    np.array([0, 2, 3]),
    ["Oranges", "Greens"],
    "GT",
    "0p9fm",
    orders_labels_dict={
        3: r"N$^{2}$LO",
        2: r"NLO",
    },
    orders_names_dict={
        3: "N2LO",
        2: "NLO",
    },
)
GT1p0fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_Gezerlis-1p0fm.h5",
    np.array([0, 2, 3]),
    ["Oranges", "Greens"],
    "GT",
    "1p0fm",
    orders_labels_dict={
        3: r"N$^{2}$LO",
        2: r"NLO",
    },
    orders_names_dict={
        3: "N2LO",
        2: "NLO",
    },
)
GT1p1fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_Gezerlis-1p1fm.h5",
    np.array([0, 2, 3]),
    ["Oranges", "Greens"],
    "GT",
    "1p1fm",
    orders_labels_dict={
        3: r"N$^{2}$LO",
        2: r"NLO",
    },
    orders_names_dict={
        3: "N2LO",
        2: "NLO",
    },
)
GT1p2fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_Gezerlis-1p2fm.h5",
    np.array([0, 2, 3]),
    ["Oranges", "Greens"],
    "GT",
    "1p2fm",
    orders_labels_dict={
        3: r"N$^{2}$LO",
        2: r"NLO",
    },
    orders_names_dict={
        3: "N2LO",
        2: "NLO",
    },
)
