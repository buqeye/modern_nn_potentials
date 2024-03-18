import numpy as np
import matplotlib.pyplot as plt
from cheftgp.gaussianprocess import ScaleSchemeBunch

# for each choice of scale and scheme, sets the total possible orders and nomenclature
EKM0p8fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_EKM_R-0p8fm.h5",
    np.array([0, 2, 3, 4, 5]),
    [plt.get_cmap(name) for name in ["Oranges", "Greens", "Blues", "Reds"]],
    "SCS",
    "0p8fm",
)
EKM0p9fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_EKM_R-0p9fm.h5",
    np.array([0, 2, 3, 4, 5]),
    [plt.get_cmap(name) for name in ["Oranges", "Greens", "Blues", "Reds"]],
    "SCS",
    "0p9fm",
)
EKM1p0fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_EKM_R-1p0fm.h5",
    np.array([0, 2, 3, 4, 5]),
    [plt.get_cmap(name) for name in ["Oranges", "Greens", "Blues", "Reds"]],
    "SCS",
    "1p0fm",
)
EKM1p1fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_EKM_R-1p1fm.h5",
    np.array([0, 2, 3, 4, 5]),
    [plt.get_cmap(name) for name in ["Oranges", "Greens", "Blues", "Reds"]],
    "SCS",
    "1p1fm",
)
EKM1p2fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_EKM_R-1p2fm.h5",
    np.array([0, 2, 3, 4, 5]),
    [plt.get_cmap(name) for name in ["Oranges", "Greens", "Blues", "Reds"]],
    "SCS",
    "1p2fm",
)

RKE400MeV = ScaleSchemeBunch(
    "../observables_data/scattering_observables_RKE_L-400MeV.h5",
    np.array([0, 2, 3, 4, 5, 6]),
    [plt.get_cmap(name) for name in ["Oranges", "Greens", "Blues", "Reds", "Purples"]],
    "SMS",
    "400MeV",
)
RKE450MeV = ScaleSchemeBunch(
    "../observables_data/scattering_observables_RKE_L-450MeV.h5",
    np.array([0, 2, 3, 4, 5, 6]),
    [plt.get_cmap(name) for name in ["Oranges", "Greens", "Blues", "Reds", "Purples"]],
    "SMS",
    "450MeV",
)
RKE500MeV = ScaleSchemeBunch(
    "../observables_data/scattering_observables_RKE_L-500MeV.h5",
    np.array([0, 2, 3, 4, 5, 6]),
    [plt.get_cmap(name) for name in ["Oranges", "Greens", "Blues", "Reds", "Purples"]],
    "SMS",
    "500MeV",
    extra = ["Oranges", "Greens", "Blues", "Reds", "Purples"],
)
RKE550MeV = ScaleSchemeBunch(
    "../observables_data/scattering_observables_RKE_L-550MeV.h5",
    np.array([0, 2, 3, 4, 5, 6]),
    [plt.get_cmap(name) for name in ["Oranges", "Greens", "Blues", "Reds", "Purples"]],
    "SMS",
    "550MeV",
)

EMN450MeV = ScaleSchemeBunch(
    "../observables_data/scattering_observables_EM-450MeV.h5",
    np.array([0, 2, 3, 4, 5]),
    [plt.get_cmap(name) for name in ["Oranges", "Greens", "Blues", "Reds"]],
    "EMN",
    "450MeV",
)
EMN500MeV = ScaleSchemeBunch(
    "../observables_data/scattering_observables_EM-500MeV.h5",
    np.array([0, 2, 3, 4, 5]),
    [plt.get_cmap(name) for name in ["Oranges", "Greens", "Blues", "Reds"]],
    "EMN",
    "500MeV",
)
EMN550MeV = ScaleSchemeBunch(
    "../observables_data/scattering_observables_EM-550MeV.h5",
    np.array([0, 2, 3, 4, 5]),
    [plt.get_cmap(name) for name in ["Oranges", "Greens", "Blues", "Reds"]],
    "EMN",
    "550MeV",
)

GT0p9fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_Gezerlis-0p9fm.h5",
    np.array([0, 2, 3]),
    [plt.get_cmap(name) for name in ["Oranges", "Greens"]],
    "GT",
    "0p9fm",
)
GT1p0fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_Gezerlis-1p0fm.h5",
    np.array([0, 2, 3]),
    [plt.get_cmap(name) for name in ["Oranges", "Greens"]],
    "GT",
    "1p0fm",
)
GT1p1fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_Gezerlis-1p1fm.h5",
    np.array([0, 2, 3]),
    [plt.get_cmap(name) for name in ["Oranges", "Greens"]],
    "GT",
    "1p1fm",
)
GT1p2fm = ScaleSchemeBunch(
    "../observables_data/scattering_observables_Gezerlis-1p2fm.h5",
    np.array([0, 2, 3]),
    [plt.get_cmap(name) for name in ["Oranges", "Greens"]],
    "GT",
    "1p2fm",
)