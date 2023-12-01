from generator_fns import *

# import gc
#
# from cheftgp.constants import *
# from cheftgp.eft import *
# from cheftgp.gaussianprocess_refactored import *
# from cheftgp.graphs import *
# from cheftgp.scattering import *
# from cheftgp.utils import *
# from cheftgp.potentials import *
# from cheftgp.traintestsplits import *
#
# # sets rcParams for plotting
# setup_rc_params()

# def generate_posteriors(
#     nn_interaction="np",
#     scale_scheme_bunch_array=[RKE500MeV],
#     Q_param_method_array=["sum"],
#     p_param_method_array=["Qofprel"],
#     input_space_deg=["cos"],
#     input_space_tlab=["prel"],
#     t_lab_train_pts = np.array([]),
#     degrees_train_pts = np.array([]),
#     orders_excluded=[],
#     orders_names_dict=None,
#     orders_labels_dict=None,
#     LengthScaleTlabInput = LengthScale("1/16-1_fitted", 0.25, 0.25, 4, whether_fit=True),
#     LengthScaleDegInput = LengthScale("1/16-1_fitted", 0.25, 0.25, 4, whether_fit=True),
#     m_pi_eff=138,
#     Lambdab=600,
#     print_all_classes=False,
#     savefile_type="pdf",
#     plot_lambdapost_pointwise_bool=False,
#     plot_lambdapost_curvewise_bool=False,
#     save_lambdapost_pointwise_bool=False,
#     save_lambdapost_curvewise_bool=False,
#     filename_addendum="",
# ):
#     """
#     nn_interaction (str): what pair of nucleons are interacting?
#     Built-in options: "nn", "np", "pp"
#     Default: "np"
#
#     scale_scheme_bunch_array (ScaleSchemeBunch list): potential/cutoff
#         combinations for evaluation.
#     Built-in options:
#         EKM0p8fm, EKM0p9fm, EKM1p0fm, EKM1p1fm, EKM1p2fm
#         ( https://doi.org/10.1140/epja/i2015-15053-8 );
#         RKE400MeV, RKE450MeV, RKE500MeV, RKE550MeV
#         ( https://doi.org/10.1140/epja/i2018-12516-4 );
#         EMN450MeV, EMN500MeV, EMN550MeV
#         ( https://doi.org/10.1103/PhysRevC.96.024004 );
#         GT0p9fm, GT1p0fm, GT1p1fm, GT1p2fm
#         ( https://doi.org/10.1103/PhysRevC.90.054323 )
#     Default: [RKE500MeV]
#
#     Q_param_method_array (str list): methods of parametrizing the dimensionless
#         expansion parameter Q for evaluation.
#     Built-in options: "smax", "max", "sum", or "rawsum"
#     Default: ["sum"]
#
#     p_param_method_array (str list): methods of parametrizing the characteristic
#         momentum p in Q(p) for evaluation.
#     Built-in options: "Qofprel", "Qofqcm", "Qofpq"
#     Default: ["Qofprel"]
#
#     input_space_deg (str): angle-dependent input space for evaluating the posterior
#         pdf for the breakdown scale, effective soft scale, length scales, etc.
#     Built-in options: "deg", "cos", "qcm", "qcm2"
#     Default: "cos"
#
#     input_space_tlab (str): energy-dependent input space for evaluating the posterior
#         pdf for the breakdown scale, effective soft scale, length scales, etc.
#     Built-in options: "Elab", "prel"
#     Default: "prel"
#
#     t_lab_train_pts (float NumPy array): lab energies (in MeV) where the TruncationTP
#         object will be trained. Will be converted to input_space_tlab by another function.
#     Default: []
#
#     degrees_train_pts (float NumPy array): scattering angles (in degrees) where the
#         TruncationTP object will be trained. Will be converted to input_space_deg by
#         another function.
#     Default: []
#
#     orders_excluded (int list): list of *coefficient* orders to be excluded from
#         the fitting procedure. 2 corresponds to c2, 3 to c3, etc.
#     Default: []
#
#     orders_names_dict (dict): dictionary method linking the numerical indices (int)
#         of EFT orders and their corresponding abbreviations (str)
#     Default: None
#
#     orders_labels_dict (dict): dictionary method linking the numerical indices (int)
#         of EFT orders and their corresponding math-mode-formatted labels (str)
#     Default: None
#
#     LengthScaleTlabInput (LengthScale): initial guess for the correlation
#         length in the kernel (as a factor of the total size of the energy-
#         dependent input space) plus boundaries of the fit procedure as factors
#         of the initial guess for the correlation length. Fitting may be bypassed
#         when whether_fit = False.
#     Default: LengthScale(0.25, 0.25, 4, whether_fit = True)
#
#     LengthScaleDegInput (LengthScale): initial guess for the correlation
#         length in the kernel (as a factor of the total size of the angle-
#         dependent input space) plus boundaries of the fit procedure as factors
#         of the initial guess for the correlation length. Fitting may be bypassed
#         when whether_fit = False.
#     Default: LengthScale(0.25, 0.25, 4, whether_fit = True)
#
#     m_pi_eff (float): effective pion mass for the theory (in MeV).
#     Default: 138
#
#     Lambdab (float): breakdown scale for the theory (in MeV).
#     Default: 600
#
#     savefile_type (str): string for specifying the type of file to be saved.
#     Default: 'png'
#
#     plot_..._bool (bool): boolean for whether to plot different figures
#     Default: True
#
#     save_..._bool (bool): boolean for whether to save different figures
#     Default: True
#
#     filename_addendum (str): string for distinguishing otherwise similarly named
#         files.
#     Default: ''
#     """
#     mpl.rc(
#         "savefig",
#         transparent=False,
#         bbox="tight",
#         pad_inches=0.05,
#         dpi=300,
#         format=savefile_type,
#     )
#
#     # try:
#     # runs through the potentials
#     for o, ScaleScheme in enumerate(scale_scheme_bunch_array):
#         # gets observable data from a local file
#         # default location is the same as this program's
#         SGT = ScaleScheme.get_data("SGT")
#         DSG = ScaleScheme.get_data("DSG")
#         AY = ScaleScheme.get_data("PB")
#         A = ScaleScheme.get_data("A")
#         D = ScaleScheme.get_data("D")
#         AXX = ScaleScheme.get_data("AXX")
#         AYY = ScaleScheme.get_data("AYY")
#         t_lab = ScaleScheme.get_data("t_lab")
#         degrees = ScaleScheme.get_data("degrees")
#
#         # creates the bunch for each observable to be plotted against angle
#         SGTBunch = ObservableBunch(
#             "SGT",
#             SGT,
#             [],
#             [],
#             "\sigma_{\mathrm{tot}}",
#             "dimensionful",
#         )
#         DSGBunch = ObservableBunch(
#             "DSG",
#             DSG,
#             [],
#             [],
#             "d \sigma / d \Omega",
#             "dimensionful",
#         )
#         AYBunch = ObservableBunch(
#             "AY",
#             AY,
#             [],
#             [],
#             "A_{y}",
#             "dimensionless",
#             constraint=[[0, -1], [0, 0], "angle"],
#         )
#         ABunch = ObservableBunch(
#             "A",
#             A,
#             [],
#             [],
#             "A",
#             "dimensionless",
#             constraint=[[0], [0], "angle"],
#         )
#         DBunch = ObservableBunch(
#             "D", D, [], [], "D", "dimensionless"
#         )
#         DBunch_dimensionful = ObservableBunch(
#             "D_dimensionful", D, [], [], "D", "dimensionful"
#         )
#         AXXBunch = ObservableBunch(
#             "AXX", AXX, [], [], "A_{xx}", "dimensionless"
#         )
#         AYYBunch = ObservableBunch(
#             "AYY", AYY, [], [], "A_{yy}", "dimensionless"
#         )
#
#         observable_array = [
#             SGTBunch,
#             DSGBunch,
#             AYBunch,
#             ABunch,
#             DBunch,
#             DBunch_dimensionful,
#             AXXBunch,
#             AYYBunch,
#         ]
#         # runs through the parametrizations of p in Q(p)
#         for p, PParamMethod in enumerate(p_param_method_array):
#             # creates the bunches for the vs-angle input spaces
#             DegBunch = InputSpaceBunch(
#                 "deg",
#                 deg_fn,
#                 p_approx(
#                     PParamMethod,
#                     E_to_p(t_lab, interaction=nn_interaction),
#                     degrees,
#                 ),
#                 r"$\theta$ (deg)",
#                 [
#                     r"$",
#                     None,
#                     r"(\theta, E_{\mathrm{lab}}= ",
#                     None,
#                     "\,\mathrm{MeV})$",
#                 ],
#             )
#             CosBunch = InputSpaceBunch(
#                 "cos",
#                 neg_cos,
#                 p_approx(
#                     PParamMethod,
#                     E_to_p(t_lab, interaction=nn_interaction),
#                     degrees,
#                 ),
#                 r"$-\mathrm{cos}(\theta)$",
#                 [
#                     r"$",
#                     None,
#                     r"(-\mathrm{cos}(\theta), E_{\mathrm{lab}}= ",
#                     None,
#                     "\,\mathrm{MeV})$",
#                 ],
#             )
#             SinBunch = InputSpaceBunch(
#                 "sin",
#                 sin_thing,
#                 p_approx(
#                     PParamMethod,
#                     E_to_p(t_lab, interaction=nn_interaction),
#                     degrees,
#                 ),
#                 r"$\mathrm{sin}(\theta)$",
#                 [
#                     r"$",
#                     None,
#                     r"(\mathrm{sin}(\theta), E_{\mathrm{lab}}= ",
#                     None,
#                     "\,\mathrm{MeV})$",
#                 ],
#             )
#             QcmBunch = InputSpaceBunch(
#                 "qcm",
#                 deg_to_qcm,
#                 p_approx(
#                     PParamMethod,
#                     E_to_p(t_lab, interaction=nn_interaction),
#                     degrees,
#                 ),
#                 r"$q_{\mathrm{cm}}$ (MeV)",
#                 [
#                     r"$",
#                     None,
#                     r"(q_{\mathrm{cm}}, E_{\mathrm{lab}}= ",
#                     None,
#                     "\,\mathrm{MeV})$",
#                 ],
#             )
#             Qcm2Bunch = InputSpaceBunch(
#                 "qcm2",
#                 deg_to_qcm2,
#                 p_approx(
#                     PParamMethod,
#                     E_to_p(t_lab, interaction=nn_interaction),
#                     degrees,
#                 ),
#                 r"$q_{\mathrm{cm}}^{2}$ (MeV$^{2}$)",
#                 [
#                     r"$",
#                     None,
#                     r"(q_{\mathrm{cm}}^{2}, E_{\mathrm{lab}}= ",
#                     None,
#                     "\,\mathrm{MeV})$",
#                 ],
#             )
#
#             vsquantity_array_deg = [
#                 DegBunch,
#                 CosBunch,
#                 QcmBunch,
#                 Qcm2Bunch,
#                 SinBunch,
#             ]
#
#             ElabBunch = InputSpaceBunch(
#                 "Elab",
#                 Elab_fn,
#                 p_approx(
#                     "Qofprel",
#                     E_to_p(t_lab, interaction=nn_interaction),
#                     degrees,
#                 ),
#                 r"$E_{\mathrm{lab}}$ (MeV)",
#                 [r"$", None, r"(E_{\mathrm{lab}})$"],
#             )
#
#             PrelBunch = InputSpaceBunch(
#                 "prel",
#                 E_to_p,
#                 p_approx(
#                     "Qofprel",
#                     E_to_p(t_lab, interaction=nn_interaction),
#                     degrees,
#                 ),
#                 r"$p_{\mathrm{rel}}$ (MeV)",
#                 [r"$", None, r"(p_{\mathrm{rel}})$"],
#             )
#
#             vsquantity_array_tlab = [ElabBunch, PrelBunch]
#
#             # creates each input space bunch's title
#             for bunch in vsquantity_array_deg:
#                 bunch.make_title()
#             for bunch in vsquantity_array_tlab:
#                 bunch.make_title()
#
#             vsquantity_posterior_array_deg = [
#                 b for b in vsquantity_array_deg if b.name in input_space_deg
#             ]
#             vsquantity_posterior_array_tlab = [
#                 b for b in vsquantity_array_tlab if b.name in input_space_tlab
#             ]
#
#             # runs through the parametrization methods
#             for k, QParamMethod in enumerate(Q_param_method_array):
#                 # runs through the angle-based input spaces
#                 for j, VsQuantityPosteriorTlab in enumerate(vsquantity_posterior_array_tlab):
#                     # runs through the angle-based input spaces
#                     for i, VsQuantityPosteriorDeg in enumerate(vsquantity_posterior_array_deg):
#                         # creates the posterior bounds for the Lambda-ell
#                         # posterior probability distribution function scaled using
#                         # the current value of Lambdab and an estimate of
#                         # 1/4 of the total input space size for the correlation
#                         # length
#
#                         # sets the meshes for the random variable arrays
#                         mpi_vals = np.linspace(50, 300, 30, dtype=np.dtype('f4'))
#                         # mpi_vals = 138 * np.array([0.9999, 1.0001])
#                         ls_deg_vals = np.linspace(0.01,
#                                               1.5 * (VsQuantityPosteriorDeg.input_space(
#                                                   **{"p_input": E_to_p(np.max(t_lab_train_pts), nn_interaction),
#                                                      "deg_input": max(degrees),
#                                                      "interaction": nn_interaction}) -
#                                                    VsQuantityPosteriorDeg.input_space(
#                                                        **{"p_input": E_to_p(np.min(t_lab_train_pts), nn_interaction),
#                                                           "deg_input": min(degrees),
#                                                           "interaction": nn_interaction})),
#                                               30)
#                         ls_tlab_vals = np.linspace(1, 150, 30, dtype=np.dtype('f4'))
#                         lambda_vals = np.linspace(300, 900, 30, dtype=np.dtype('f4'))
#                         # lambda_vals = np.linspace(300, 1200, 500, dtype=np.dtype('f4'))
#                         # lambda_vals = 600 * np.array([0.9999, 1.0001])
#
#                         mesh_cart = gm.cartesian(lambda_vals, np.log(ls_deg_vals), np.log(ls_tlab_vals), mpi_vals)
#                         mesh_cart_sgt = np.delete(mesh_cart, 1, 1)
#
#                         # sets the RandomVariable objects
#                         LambdabVariable = RandomVariable(var=lambda_vals,
#                                                          user_val=Lambdab,
#                                                          name='Lambdab',
#                                                          label="\Lambda_{b}",
#                                                          units="MeV",
#                                                          ticks=[450, 600, 750],
#                                                          logprior=Lb_logprior(lambda_vals),
#                                                          logprior_name="uniformprior",
#                                                          marg_bool = True)
#                         LsDegVariable = RandomVariable(var=ls_deg_vals,
#                                                     user_val=None,
#                                                     name='lsdeg',
#                                                     label="\ell_{\Theta}",
#                                                     units="",
#                                                     ticks=[],
#                                                     logprior=np.zeros(len(ls_deg_vals)),
#                                                     logprior_name="noprior",
#                                                     marg_bool=False)
#                         LsTlabVariable = RandomVariable(var=ls_tlab_vals,
#                                                     user_val=None,
#                                                     name='lstlab',
#                                                     label="\ell_{T}",
#                                                     units="MeV",
#                                                     ticks=[],
#                                                     logprior=np.zeros(len(ls_tlab_vals)),
#                                                     logprior_name="noprior",
#                                                     marg_bool=False)
#                         MpieffVariable = RandomVariable(var=mpi_vals,
#                                                         user_val=m_pi_eff,
#                                                         name='mpieff',
#                                                         label="m_{\pi}",
#                                                         units="MeV",
#                                                         ticks=[100, 150, 200, 250, 300, 350],
#                                                         logprior=mpieff_logprior(mpi_vals),
#                                                         logprior_name="uniformprior",
#                                                         marg_bool = True)
#                         variables_array = np.array([LambdabVariable, LsDegVariable, LsTlabVariable, MpieffVariable])
#
#                         # chooses a starting guess for the GP length scale optimization procedure
#                         # LengthScaleGuess = length_scale_input
#                         # if E_angle_pair[0]:
#                         #     LengthScaleGuess = LengthScaleDegInput
#                         #     LengthScaleGuess.make_guess(
#                         #         VsQuantity.input_space(
#                         #             **{
#                         #                 "deg_input": degrees,
#                         #                 "p_input": E_to_p(
#                         #                     E_lab, interaction=nn_interaction
#                         #                 ),
#                         #             }
#                         #         )
#                         #     )
#                         # else:
#                         #     LengthScaleGuess = LengthScaleTlabInput
#                         #     LengthScaleGuess.make_guess(
#                         #         VsQuantity.input_space(
#                         #             **{
#                         #                 "E_lab": t_lab,
#                         #                 "interaction": nn_interaction,
#                         #             }
#                         #         )
#                         #     )
#
#                         center = 0
#                         df = 1
#                         disp = 0
#                         std_scale = 1
#
#                         # information for naming the savefiles
#                         FileName = FileNaming(
#                             ScaleScheme.potential_string,
#                             ScaleScheme.cutoff_string,
#                             QParamMethod,
#                             PParamMethod,
#                             VsQuantityPosteriorDeg.name + 'x' + VsQuantityPosteriorTlab.name,
#                             filename_addendum=filename_addendum,
#                         )
#                         #
#                         # information on the orders for each potential
#                         Orders = OrderInfo(
#                             ScaleScheme.orders_full,
#                             [0] + orders_excluded,
#                             ScaleScheme.colors,
#                             ScaleScheme.light_colors,
#                             orders_names_dict=orders_names_dict,
#                             orders_labels_dict=orders_labels_dict,
#                         )
#
#                         if plot_lambdapost_curvewise_bool or plot_lambdapost_pointwise_bool:
#                             obs_dict = {"SGT": SGTBunch, "DSG": DSGBunch, "D": DBunch, "AXX": AXXBunch, "AYY": AYYBunch, "A": ABunch, "AY": AYBunch}
#
#                             # # just SGT
#                             # plot_obs_list = [["SGT"]]
#                             # obs_name_grouped_list = ["SGT"]
#                             # obs_labels_grouped_list = [r'$\sigma$']
#                             # mesh_cart_grouped_list = [mesh_cart_sgt]
#
#                             # # just DSG
#                             # plot_obs_list = [["DSG"]]
#                             # obs_name_grouped_list = ["DSG"]
#                             # obs_labels_grouped_list = [r'$\displaystyle\frac{d\sigma}{d\Omega}$']
#                             # mesh_cart_grouped_list = [mesh_cart]
#
#                             # # just D
#                             # plot_obs_list = [["D"]]
#                             # obs_name_grouped_list = ["D"]
#                             # obs_labels_grouped_list = [r'$D$']
#
#                             # # just AXX
#                             # plot_obs_list = [["AXX"]]
#                             # obs_name_grouped_list = ["AXX"]
#                             # obs_labels_grouped_list = [r'$A_{xx}$']
#
#                             # # just AYY
#                             # plot_obs_list = [["AYY"]]
#                             # obs_name_grouped_list = ["AYY"]
#                             # obs_labels_grouped_list = [r'$A_{yy}$']
#
#                             # # just A
#                             # plot_obs_list = [["A"]]
#                             # obs_name_grouped_list = ["A"]
#                             # obs_labels_grouped_list = [r'$A$']
#
#                             # # just AY
#                             # plot_obs_list = [["AY"]]
#                             # obs_name_grouped_list = ["AY"]
#                             # obs_labels_grouped_list = [r'$A_{y}$']
#
#                             # # for equalizing SGT and DSG
#                             # plot_obs_list = [["SGT"], ["DSG"]]
#                             # obs_name_grouped_list = ["SGT", "DSG"]
#                             # obs_labels_grouped_list = [r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$']
#                             # mesh_cart_grouped_list = [mesh_cart_sgt, mesh_cart]
#
#                             # for equalizing SGT and DSG
#                             plot_obs_list = [["SGT"], ["DSG"], ["SGT", "DSG"]]
#                             obs_name_grouped_list = ["SGT", "DSG", "TWOOBS"]
#                             obs_labels_grouped_list = [r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$', r'2Obs.']
#                             # mesh_cart_grouped_list = [mesh_cart, mesh_cart, mesh_cart]
#                             mesh_cart_grouped_list = [[mesh_cart_sgt], [mesh_cart], [mesh_cart_sgt, mesh_cart]]
#
#                             # # for SGT, DSG, and D
#                             # plot_obs_list = [["SGT"], ["DSG"], ["D"]]
#                             # obs_name_grouped_list = ["SGT", "DSG", "D"]
#                             # obs_labels_grouped_list = [r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$', r'$D$']
#
#                             # # spins
#                             # plot_obs_list = [["D", "AXX", "AYY", "A", "AY"]]
#                             # obs_name_grouped_list = ["spins"]
#                             # obs_labels_grouped_list = [r'$X_{pqik}$']
#
#                             # # ALLOBS for energy input spaces
#                             # plot_obs_list = [["SGT", "DSG", "D", "AXX", "AYY", "A", "AY"]]
#                             # obs_name_grouped_list = ["ALLOBS"]
#                             # obs_labels_grouped_list = [r'Obs.']
#
#                             # # ALLOBS for angle input spaces
#                             # plot_obs_list = [["DSG", "D", "AXX", "AYY", "A", "AY"]]
#                             # obs_name_grouped_list = ["ALLOBS"]
#                             # obs_labels_grouped_list = [r'Obs.']
#
#                             # # SGT, DSG, spins, ALLOBS for energy input spaces
#                             # plot_obs_list = [["SGT"], ["DSG"], ["D", "AXX", "AYY", "A", "AY"],
#                             #                  ["SGT", "DSG", "D", "AXX", "AYY", "A", "AY"]]
#                             # obs_name_grouped_list = ["SGT", "DSG", "spins", "ALLOBS"]
#                             # obs_labels_grouped_list = [r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$', r'$X_{pqik}$',
#                             #                            r'Obs.']
#
#                             # # DSG and spins
#                             # plot_obs_list = [["DSG"], ["D", "AXX", "AYY", "A", "AY"]]
#                             # obs_name_grouped_list = ["DSG", "spins"]
#                             # obs_labels_grouped_list = [r'$\displaystyle\frac{d\sigma}{d\Omega}$',
#                             #                            r'$X_{pqik}$']
#
#                             # # EACHOBS for energy input spaces
#                             # plot_obs_list = [["SGT"], ["DSG"], ["D"], ["AXX"], ["AYY"], ["A"], ["AY"]]
#                             # obs_name_grouped_list = ["SGT", "DSG", "D", "AXX", "AYY", "A", "AY"]
#                             # obs_labels_grouped_list = [r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$',
#                             #                            r'$D$', r'$A_{xx}$', r'$A_{yy}$', r'$A$', r'$A_{y}$']
#                             # mesh_cart_grouped_list = [[mesh_cart_sgt], [mesh_cart], [mesh_cart], [mesh_cart], [mesh_cart],
#                             #                           [mesh_cart], [mesh_cart]]
#
#                             # # EACHOBS and ALLOBS for energy input spaces
#                             # plot_obs_list = [["SGT"], ["DSG"], ["D"], ["AXX"], ["AYY"], ["A"], ["AY"],
#                             #                  ["SGT", "DSG", "D", "AXX", "AYY", "A", "AY"]]
#                             # obs_name_grouped_list = ["SGT", "DSG", "D", "AXX", "AYY", "A", "AY", "ALLOBS"]
#                             # obs_labels_grouped_list = [r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$',
#                             #                            r'$D$', r'$A_{xx}$', r'$A_{yy}$', r'$A$', r'$A_{y}$', r'Obs.']
#                             # mesh_cart_grouped_list = [[mesh_cart_sgt], [mesh_cart], [mesh_cart], [mesh_cart],
#                             #                           [mesh_cart], [mesh_cart], [mesh_cart],
#                             #                           [mesh_cart_sgt, mesh_cart, mesh_cart, mesh_cart,
#                             #                           mesh_cart, mesh_cart, mesh_cart]]
#
#                             # # ALLOBS for energy input spaces
#                             # plot_obs_list = [["SGT", "DSG", "D", "AXX", "AYY", "A", "AY"]]
#                             # obs_name_grouped_list = ["ALLOBS"]
#                             # obs_labels_grouped_list = [r'Obs.']
#
#                             # # # EACHOBS and ALLOBS for angle input spaces
#                             # plot_obs_list = [["DSG"], ["D"], ["AXX"], ["AYY"], ["A"], ["AY"],
#                             #                  ["DSG", "D", "AXX", "AYY", "A", "AY"]]
#                             # obs_name_grouped_list = ["DSG", "D", "AXX", "AYY", "A", "AY", "ALLOBS"]
#                             # obs_labels_grouped_list = [r'$\displaystyle\frac{d\sigma}{d\Omega}$',
#                             #                            r'$D$', r'$A_{xx}$', r'$A_{yy}$', r'$A$', r'$A_{y}$',
#                             #                            r'Obs.']
#
#                             # # # EACHOBS and ALLOBS for angle input spaces
#                             # plot_obs_list = [["DSG"],
#                             #                  ["D"], ["AXX"], ["AYY"], ["A"], ["AY"]]
#                             # obs_name_grouped_list = ["DSG", "D", "AXX", "AYY", "A", "AY"]
#                             # obs_labels_grouped_list = [r'$\displaystyle\frac{d\sigma}{d\Omega}$',
#                             #                            r'$D$', r'$A_{xx}$', r'$A_{yy}$', r'$A$', r'$A_{y}$']
#
#                             obs_grouped_list = [
#                                 [obs_dict[obs_name] for obs_name in obs_sublist] for
#                                 obs_sublist in plot_obs_list]
#
#                             LengthScaleTlabInput.make_guess(
#                                 VsQuantityPosteriorTlab.input_space(
#                                     **{
#                                         "E_lab": t_lab,
#                                         "interaction": nn_interaction,
#                                     }
#                                 )
#                             )
#                             LengthScaleDegInput.make_guess(
#                                 VsQuantityPosteriorDeg.input_space(
#                                     **{
#                                         "deg_input": degrees,
#                                         "p_input": E_to_p(
#                                             np.average(t_lab_train_pts), interaction=nn_interaction
#                                         ),
#                                     }
#                                 )
#                             )
#
#                             if plot_lambdapost_curvewise_bool:
#                                 plot_posteriors_curvewise(
#                                     # order stuff
#                                     light_colors = Orders.lightcolors_array,
#                                     nn_orders_array = Orders.orders_restricted,
#                                     nn_orders_full_array = Orders.orders_full,
#                                     excluded = Orders.excluded,
#                                     orders_labels_dict = {6: r'N$^{4}$LO$^{+}$', 5: r'N$^{4}$LO',
#                                                            4: r'N$^{3}$LO', 3: r'N$^{2}$LO',
#                                                            2: r'NLO'},
#                                     orders_names_dict={6: 'N4LO+', 5: 'N4LO',
#                                                         4: 'N3LO', 3: 'N2LO',
#                                                         2: 'NLO'},
#                                     # strings
#                                     # p_param = PParamMethod,
#                                     # Q_param = QParamMethod,
#                                     nn_interaction = nn_interaction,
#                                     # hyperparameters
#                                     center = center,
#                                     disp = disp,
#                                     df = df,
#                                     std_est = std_scale,
#                                     # filename stuff
#                                     obs_data_grouped_list = obs_grouped_list,
#                                     obs_name_grouped_list = obs_name_grouped_list,
#                                     obs_labels_grouped_list = obs_labels_grouped_list,
#                                     mesh_cart_grouped_list = mesh_cart_grouped_list,
#                                     t_lab=t_lab,
#                                     t_lab_train_pts=t_lab_train_pts,
#                                     # t_lab_pts=np.array([5, 21, 48, 85, 133, 192]),
#                                     # t_lab_pts=np.array([5, 21, 48, 85, 133, 192, 261]),
#                                     # t_lab_train_pts=np.array([1, 12, 33, 65, 108, 161, 225, 300]), # set0 / refactor
#                                     # t_lab_pts=np.array([25, 75, 125, 175, 225, 275, 325]), # set1
#                                     # t_lab_pts=np.array([1, 10, 28, 55, 90, 133, 185]), # set2
#                                     # t_lab_pts=np.array([1, 9, 23, 45, 73, 108, 150]), # set3
#                                     # t_lab_pts=np.array([1, 8, 19, 36, 58, 85, 118]),  # set4
#                                     # t_lab_pts=np.array([1, 11, 31, 61, 100, 150]),  # set5
#                                     # t_lab_pts=np.array([1, 6, 15, 28, 45, 65, 90, 118, 150]),  # set6
#                                     # t_lab_train_pts=np.array([36, 58, 85, 118, 155, 198, 246, 300]),  # set7
#                                     # t_lab_pts=np.array([1, 12, 33, 65]),
#                                     # t_lab_pts=np.array([108, 161, 225, 300]),
#                                     # t_lab_pts=np.array([1, 5, 12, 21]),
#                                     # t_lab_pts=np.array([33, 48, 65, 85]),
#                                     # t_lab_pts=np.array([108, 133, 161, 192]),
#                                     # t_lab_pts=np.array([42, 65, 94, 128, 167, 211, 261]),
#
#                                     # t_lab_pts=np.array([50, 100, 150, 200, 250, 300]),
#                                     # t_lab_pts=np.array([1, 5, 12, 21, 33, 48]),
#                                     # t_lab_pts=np.array([1, 10, 25, 48]),                                        # t_lab_pts=np.array([1, 10, 25]),
#                                     # t_lab_pts=np.array([65, 85, 108, 133, 161, 192]),
#                                     # t_lab_pts=np.array([65, 100, 143, 192]),
#                                     # t_lab_pts=np.array([100, 143, 192]),
#                                     # t_lab_train_pts=np.array([4, 20, 47, 81, 129, 188, 249]),
#                                     # t_lab_test_pts=np.array([4, 20, 47, 81, 129, 188, 249]),
#                                     InputSpaceTlab=VsQuantityPosteriorTlab,
#                                     LsTlab=LengthScaleTlabInput,
#                                     degrees=degrees,
#                                     degrees_train_pts=degrees_train_pts,
#                                     InputSpaceDeg=VsQuantityPosteriorDeg,
#                                     LsDeg=LengthScaleDegInput,
#                                     variables_array=variables_array,
#
#                                     mom_fn=E_to_p,
#                                     mom_fn_kwargs={"interaction" : "np"},
#
#                                     scaling_fn=scaling_fn,
#                                     scaling_fn_kwargs={},
#
#                                     ratio_fn=ratio_fn_curvewise,
#                                     ratio_fn_kwargs={
#                                         "p_param": PParamMethod,
#                                         "Q_param": QParamMethod,
#                                         "mpi_var": m_pi_eff,
#                                         "lambda_var": Lambdab
#                                     },
#                                     log_likelihood_fn=log_likelihood,
#                                     log_likelihood_fn_kwargs={
#                                         "p_param": PParamMethod,
#                                         "Q_param": QParamMethod
#                                     },
#
#                                     orders=2,
#
#                                     FileName = FileName,
#
#                                     whether_use_data=True,
#                                     whether_save_data=False,
#                                     whether_save_plots=save_lambdapost_curvewise_bool,
#                                 )
#                             if plot_lambdapost_pointwise_bool:
#                                 plot_posteriors_pointwise(
#                                     # order stuff
#                                     light_colors=Orders.lightcolors_array,
#                                     nn_orders_array=Orders.orders_restricted,
#                                     nn_orders_full_array=Orders.orders_full,
#                                     excluded=Orders.excluded,
#                                     orders_labels_dict={6: r'N$^{4}$LO$^{+}$', 5: r'N$^{4}$LO',
#                                                         4: r'N$^{3}$LO', 3: r'N$^{2}$LO',
#                                                         2: r'NLO'},
#                                     # orders_names_dict={6: 'N4LO+', 5: 'N4LO',
#                                     #                    4: 'N3LO', 3: 'N2LO',
#                                     #                    2: 'NLO'},
#                                     # strings
#                                     # p_param=PParamMethod,
#                                     # Q_param=QParamMethod,
#                                     # filename stuff
#                                     obs_data_grouped_list=obs_grouped_list,
#                                     obs_name_grouped_list=obs_name_grouped_list,
#                                     obs_labels_grouped_list=obs_labels_grouped_list,
#                                     t_lab=t_lab,
#                                     t_lab_train_pts=t_lab_train_pts,
#                                     # t_lab_pts=np.array([5, 21, 48, 85, 133, 192]),
#                                     # t_lab_pts=np.array([5, 21, 48, 85, 133, 192, 261]),
#                                     # t_lab_train_pts=np.array([1, 12, 33, 65, 108, 161, 225, 300]), # set0 / refactor
#                                     # t_lab_pts=np.array([25, 75, 125, 175, 225, 275, 325]), # set1
#                                     # t_lab_pts=np.array([1, 10, 28, 55, 90, 133, 185]), # set2
#                                     # t_lab_pts=np.array([1, 9, 23, 45, 73, 108, 150]), # set3
#                                     # t_lab_pts=np.array([1, 8, 19, 36, 58, 85, 118]),  # set4
#                                     # t_lab_pts=np.array([1, 11, 31, 61, 100, 150]),  # set5
#                                     # t_lab_pts=np.array([1, 6, 15, 28, 45, 65, 90, 118, 150]),  # set6
#                                     # t_lab_train_pts=np.array([36, 58, 85, 118, 155, 198, 246, 300]),  # set7
#                                     # t_lab_pts=np.array([1, 12, 33, 65]),
#                                     # t_lab_pts=np.array([108, 161, 225, 300]),
#                                     # t_lab_pts=np.array([1, 5, 12, 21]),
#                                     # t_lab_pts=np.array([33, 48, 65, 85]),
#                                     # t_lab_pts=np.array([108, 133, 161, 192]),
#                                     # t_lab_pts=np.array([42, 65, 94, 128, 167, 211, 261]),
#
#                                     # t_lab_pts=np.array([50, 100, 150, 200, 250, 300]),
#                                     # t_lab_pts=np.array([1, 5, 12, 21, 33, 48]),
#                                     # t_lab_pts=np.array([1, 10, 25, 48]),                                        # t_lab_pts=np.array([1, 10, 25]),
#                                     # t_lab_pts=np.array([65, 85, 108, 133, 161, 192]),
#                                     # t_lab_pts=np.array([65, 100, 143, 192]),
#                                     # t_lab_pts=np.array([100, 143, 192]),
#                                     # t_lab_train_pts=np.array([4, 20, 47, 81, 129, 188, 249]),
#                                     # t_lab_test_pts=np.array([4, 20, 47, 81, 129, 188, 249]),
#                                     InputSpaceTlab=VsQuantityPosteriorTlab,
#                                     degrees=degrees,
#                                     degrees_train_pts=degrees_train_pts,
#                                     InputSpaceDeg=VsQuantityPosteriorDeg,
#                                     variables_array=np.array([LambdabVariable]),
#
#                                     mom_fn_tlab=E_to_p,
#                                     mom_fn_tlab_kwargs={"interaction": "np"},
#
#                                     mom_fn_degrees=mom_fn_degrees,
#                                     mom_fn_degrees_kwargs={},
#
#                                     p_fn=p_approx,
#                                     p_fn_kwargs={"p_name" : PParamMethod,
#                                                    },
#
#                                     ratio_fn=Q_approx,
#                                     ratio_fn_kwargs={
#                                         "Q_parametrization": QParamMethod,
#                                         "m_pi": m_pi_eff,
#                                     },
#
#                                     orders=3,
#
#                                     FileName=FileName,
#
#                                     whether_save_plots=save_lambdapost_pointwise_bool,
#                                 )
#
#     # except:
#     #     print("Error encountered in running loop.")
#
#     # prints all instances of the classes relevant for the arguments of
#     # GPAnalysis()
#     if print_all_classes:
#         scalescheme_current_list = []
#         observable_current_list = []
#         inputspace_current_list = []
#         traintest_current_list = []
#         lengthscale_current_list = []
#
#         for obj in gc.get_objects():
#             if isinstance(obj, ScaleSchemeBunch):
#                 scalescheme_current_list.append(obj.name)
#             elif isinstance(obj, ObservableBunch):
#                 observable_current_list.append(obj.name)
#             elif isinstance(obj, InputSpaceBunch):
#                 inputspace_current_list.append(obj.name)
#             elif isinstance(obj, TrainTestSplit):
#                 traintest_current_list.append(obj.name)
#             elif isinstance(obj, LengthScale):
#                 lengthscale_current_list.append(obj.name)
#
#         print("\n\n************************************")
#         print("Available potentials: " + str(scalescheme_current_list))
#         print("Available observables: " + str(observable_current_list))
#         print("Available Q parametrizations: ['smax', 'max', 'sum', 'rawsum']")
#         print("Available input spaces: " + str(inputspace_current_list))
#         print("Available train/test splits: " + str(traintest_current_list))
#         print("Available length scales: " + str(lengthscale_current_list))
#         print("************************************")


generate_posteriors(
    nn_interaction="np",
    scale_scheme_bunch_array=[RKE500MeV],
    Q_param_method_array=["sum"],
    p_param_method_array=["Qofprel"],
    input_space_deg=["cos"],
    input_space_tlab=["prel"],
    t_lab_train_pts=np.array([1, 12, 33, 65, 108, 161, 225, 300]),  # set0 / refactor
    # t_lab_pts=np.array([25, 75, 125, 175, 225, 275, 325]), # set1
    # t_lab_pts=np.array([1, 10, 28, 55, 90, 133, 185]), # set2
    # t_lab_pts=np.array([1, 9, 23, 45, 73, 108, 150]), # set3
    # t_lab_pts=np.array([1, 8, 19, 36, 58, 85, 118]),  # set4
    # t_lab_pts=np.array([1, 11, 31, 61, 100, 150]),  # set5
    # t_lab_pts=np.array([1, 6, 15, 28, 45, 65, 90, 118, 150]),  # set6
    # t_lab_train_pts=np.array([36, 58, 85, 118, 155, 198, 246, 300]),  # set7
    # t_lab_train_pts=np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]),  # 1704
    degrees_train_pts=np.array([41, 60, 76, 90, 104, 120, 139]), # evencos
    # degrees_pts=np.array(
    #     [15, 31, 50, 90, 130, 149, 165]
    # ),  # evensin
    # degrees_train_pts=np.array([40, 60, 80, 100, 120, 140]), # 1704
    orders_excluded=[],
    orders_names_dict=None,
    orders_labels_dict=None,
    LengthScaleTlabInput=LengthScale("1/16-1_fitted", 0.25, 0.25, 4, whether_fit=True),
    LengthScaleDegInput=LengthScale("1/16-1_fitted", 0.25, 0.25, 4, whether_fit=True),
    m_pi_eff=141,
    Lambdab=480,
    print_all_classes=False,
    savefile_type="png",
    plot_lambdapost_pointwise_bool=True,
    plot_lambdapost_curvewise_bool=True,
    save_lambdapost_pointwise_bool=False,
    save_lambdapost_curvewise_bool=False,
    filename_addendum="_cluster",
)