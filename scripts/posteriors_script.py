from generator_fns import *

# sets the meshes for the random variable arrays
mpi_vals = np.linspace(1, 350, 30, dtype=np.dtype('f4'))
# mpi_vals = 200 * np.array([0.9999, 1.0001])
ls_tlab_vals = np.linspace(1, 150, 30, dtype=np.dtype('f4'))
ls_deg_vals = np.linspace(0.01, 4, 30, dtype=np.dtype('f4'))
# ls_deg_vals = np.linspace(50, 100, 30, dtype=np.dtype('f4'))
lambda_vals = np.linspace(200, 900, 30, dtype=np.dtype('f4'))
# lambda_vals = 600 * np.array([0.9999, 1.0001])

mesh_cart = gm.cartesian(lambda_vals, np.log(ls_tlab_vals), np.log(ls_deg_vals), mpi_vals)
mesh_cart_sgt = np.delete(mesh_cart, 2, 1)
mesh_cart_ang = np.delete(mesh_cart, 1, 1)
# print(mesh_cart_ang)

# mpi_vals = np.linspace(50, 210, 20, dtype=np.dtype('f4'))
# ls_tlab_vals = np.linspace(41, 115, 15, dtype=np.dtype('f4'))
# ls_deg_mag_vals = np.linspace(20, 400, 50, dtype=np.dtype('f4'))
# ls_deg_exp_vals = np.linspace(0.5, 1.5, 11, dtype=np.dtype('f4'))
# lambda_vals = np.linspace(200, 900, 30, dtype=np.dtype('f4'))
#
# mesh_cart = gm.cartesian(lambda_vals, np.log(ls_tlab_vals), np.log(ls_deg_mag_vals), np.log(ls_deg_exp_vals), mpi_vals)

# # just SGT
# plot_obs_list = [["SGT"]]
# obs_name_grouped_list = ["SGT"]
# obs_labels_grouped_list = [r'$\sigma$']
# mesh_cart_grouped_list = [[mesh_cart_sgt]]

# just DSG
plot_obs_list = [["DSG"]]
obs_name_grouped_list = ["DSG"]
obs_labels_grouped_list = [r'$\displaystyle\frac{d\sigma}{d\Omega}$']
mesh_cart_grouped_list = [[mesh_cart]]
# mesh_cart_grouped_list = [[mesh_cart_q]]

# # just D
# plot_obs_list = [["D"]]
# obs_name_grouped_list = ["D"]
# obs_labels_grouped_list = [r'$D$']
# mesh_cart_grouped_list = [[mesh_cart]]

# # just AXX
# plot_obs_list = [["AXX"]]
# obs_name_grouped_list = ["AXX"]
# obs_labels_grouped_list = [r'$A_{xx}$']
# mesh_cart_grouped_list = [[mesh_cart]]

# # just AYY
# plot_obs_list = [["AYY"]]
# obs_name_grouped_list = ["AYY"]
# obs_labels_grouped_list = [r'$A_{yy}$']
# mesh_cart_grouped_list = [[mesh_cart]]

# # just A
# plot_obs_list = [["A"]]
# obs_name_grouped_list = ["A"]
# obs_labels_grouped_list = [r'$A$']
# mesh_cart_grouped_list = [[mesh_cart]]

# # just AY
# plot_obs_list = [["AY"]]
# obs_name_grouped_list = ["AY"]
# obs_labels_grouped_list = [r'$A_{y}$']
# mesh_cart_grouped_list = [[mesh_cart]]

# # for equalizing SGT and DSG
# plot_obs_list = [["SGT"], ["DSG"]]
# obs_name_grouped_list = ["SGT", "DSG"]
# obs_labels_grouped_list = [r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$']
# mesh_cart_grouped_list = [[mesh_cart_sgt], [mesh_cart]]

# # EACHOBS for energy input spaces
# plot_obs_list = [["DSG21"]]
# obs_name_grouped_list = ["DSG21"]
# obs_labels_grouped_list = [r'$\displaystyle\frac{d\sigma}{d\Omega}(\Theta, p = 100\,\mathrm{MeV})$', ]
# mesh_cart_grouped_list = [[mesh_cart_ang]]

# # for equalizing SGT and DSG
# plot_obs_list = [["SGT"], ["DSG"], ["SGT", "DSG"]]
# obs_name_grouped_list = ["SGT", "DSG", "TWOOBS"]
# obs_labels_grouped_list = [r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$', r'2Obs.']
# # mesh_cart_grouped_list = [mesh_cart, mesh_cart, mesh_cart]
# mesh_cart_grouped_list = [[mesh_cart_sgt], [mesh_cart], [mesh_cart_sgt, mesh_cart]]

#
# plot_obs_list = [["SGT"], ["SGTnosine"], ["DSGsine"], ["DSG"]]
# obs_name_grouped_list = ["SGT", "SGTnosine", "DSGsine", "DSG"]
# obs_labels_grouped_list = [r'$\sigma$', r'$\Sigma$',
#                            r'$\displaystyle\frac{d\sigma}{d\Omega}\mathrm{sin}(\theta)$',
#                            r'$\displaystyle\frac{d\sigma}{d\Omega}$']
# mesh_cart_grouped_list = [[mesh_cart_sgt], [mesh_cart_sgt], [mesh_cart], [mesh_cart]]

# #
# plot_obs_list = [["DSGsine"], ["DSG"]]
# obs_name_grouped_list = ["DSGsine", "DSG"]
# obs_labels_grouped_list = [
#                            r'$\displaystyle\frac{d\sigma}{d\Omega}\mathrm{sin}(\theta)$',
#                            r'$\displaystyle\frac{d\sigma}{d\Omega}$']
# mesh_cart_grouped_list = [[mesh_cart], [mesh_cart]]

# # for SGT, DSG, and D
# plot_obs_list = [["SGT"], ["DSG"], ["D"]]
# obs_name_grouped_list = ["SGT", "DSG", "D"]
# obs_labels_grouped_list = [r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$', r'$D$']

# #
# plot_obs_list = [["SGT"], ["DSG"], ["DSGsb"], ["D"], ["Dint"], ["Dsb"],
#                  ["AXX"], ["AXXint"], ["AXXsb"], ["AYY"], ["AYYint"], ["AYYsb"],
#                  ["A"], ["Aint"], ["Asb"], ["AY"], ["AYint"], ["AYsb"],]
# obs_name_grouped_list = ["SGT", "DSG", "DSGsb", "D", "Dint", "Dsb", "AXX", "AXXint", "AXXsb",
#                          "AYY", "AYYint", "AYYsb", "A", "Aint", "Asb" "AY", "AYint", "AYsb"]
# obs_labels_grouped_list = [r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$',
#                            r'$\displaystyle\frac{d\sigma}{d\Omega}_{sb}$', r'$D$',
#                            r'$\int D$', r'$D_{sb}$', r'$A_{xx}$', r'$\int A_{xx}$', r'$A_{xxsb}$',
#                            r'$A_{yy}$', r'$\int A_{yy}$', r'$A_{yysb}$',
#                            r'$A$', r'$\int A$', r'$A_{sb}$', r'$A_{y}$', r'$\int A_{y}$', r'$A_{ysb}$', ]
# mesh_cart_grouped_list = [[mesh_cart_sgt], [mesh_cart], [mesh_cart_sgt], [mesh_cart],
#                           [mesh_cart_sgt], [mesh_cart_sgt],
#                           [mesh_cart], [mesh_cart_sgt], [mesh_cart_sgt], [mesh_cart], [mesh_cart_sgt],
#                           [mesh_cart_sgt], [mesh_cart], [mesh_cart_sgt], [mesh_cart_sgt],
#                           [mesh_cart], [mesh_cart_sgt], [mesh_cart_sgt], ]

# plot_obs_list = [["D"], ["Dint"], ["Dsb"]]
# obs_name_grouped_list = ["D", "Dint", "Dsb"]
# obs_labels_grouped_list = [r'$D$', r'$\int D$', r'$D_{sb}$']
# mesh_cart_grouped_list = [[mesh_cart], [mesh_cart_sgt], [mesh_cart_sgt],]

# # spins
# plot_obs_list = [["D", "AXX", "AYY", "A", "AY"]]
# obs_name_grouped_list = ["spins"]
# obs_labels_grouped_list = [r'$X_{pqik}$']

# # ALLOBS for energy input spaces
# plot_obs_list = [["SGT", "DSG", "D", "AXX", "AYY", "A", "AY"]]
# obs_name_grouped_list = ["ALLOBS"]
# obs_labels_grouped_list = [r'Obs.']

# # ALLOBS for angle input spaces
# plot_obs_list = [["DSG", "D", "AXX", "AYY", "A", "AY"]]
# obs_name_grouped_list = ["ALLOBS"]
# obs_labels_grouped_list = [r'Obs.']
# mesh_cart_grouped_list = [[mesh_cart, mesh_cart, mesh_cart, mesh_cart, mesh_cart, mesh_cart]]

# # SGT, DSG, spins, ALLOBS for energy input spaces
# plot_obs_list = [["SGT"], ["DSG"], ["D", "AXX", "AYY", "A", "AY"],
#                  ["SGT", "DSG", "D", "AXX", "AYY", "A", "AY"]]
# obs_name_grouped_list = ["SGT", "DSG", "spins", "ALLOBS"]
# obs_labels_grouped_list = [r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$', r'$X_{pqik}$',
#                            r'Obs.']

# # DSG and spins
# plot_obs_list = [["DSG"], ["D", "AXX", "AYY", "A", "AY"]]
# obs_name_grouped_list = ["DSG", "spins"]
# obs_labels_grouped_list = [r'$\displaystyle\frac{d\sigma}{d\Omega}$',
#                            r'$X_{pqik}$']

# # EACHOBS for energy input spaces
# plot_obs_list = [["SGT"], ["DSG"], ["D"], ["AXX"], ["AYY"], ["A"], ["AY"]]
# obs_name_grouped_list = ["SGT", "DSG", "D", "AXX", "AYY", "A", "AY"]
# obs_labels_grouped_list = [r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$',
#                            r'$D$', r'$A_{xx}$', r'$A_{yy}$', r'$A$', r'$A_{y}$']
# mesh_cart_grouped_list = [[mesh_cart_sgt], [mesh_cart], [mesh_cart], [mesh_cart], [mesh_cart],
#                           [mesh_cart], [mesh_cart]]

# # EACHOBS and ALLOBS for energy input spaces
# plot_obs_list = [["SGT"], ["DSG"], ["D"], ["AXX"], ["AYY"], ["A"], ["AY"],
#                  ["DSG", "D", "AXX", "AYY", "A", "AY"]]
# obs_name_grouped_list = ["SGT", "DSG", "D", "AXX", "AYY", "A", "AY", "ALLOBS"]
# obs_labels_grouped_list = [r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$',
#                            r'$D$', r'$A_{xx}$', r'$A_{yy}$', r'$A$', r'$A_{y}$', r'$\Pi$Obs.']
# mesh_cart_grouped_list = [[mesh_cart_sgt], [mesh_cart], [mesh_cart], [mesh_cart],
#                           [mesh_cart], [mesh_cart], [mesh_cart],
#                           [mesh_cart, mesh_cart, mesh_cart,
#                           mesh_cart, mesh_cart, mesh_cart]]

# # EACHOBS and ALLOBS for energy input spaces
# plot_obs_list = [["SGT"], ["DSG", "D", "AXX", "AYY", "A", "AY"]]
# obs_name_grouped_list = ["SGT", "ALLOBS"]
# obs_labels_grouped_list = [r'$\sigma$', r'$\Pi$Obs.']
# mesh_cart_grouped_list = [[mesh_cart_sgt],
#                           [mesh_cart, mesh_cart, mesh_cart,
#                            mesh_cart, mesh_cart, mesh_cart]]

# # ALLOBS for energy input spaces
# plot_obs_list = [["SGT", "DSG", "D", "AXX", "AYY", "A", "AY"]]
# obs_name_grouped_list = ["ALLOBS"]
# obs_labels_grouped_list = [r'Obs.']

# # # EACHOBS and ALLOBS for angle input spaces
# plot_obs_list = [["DSG"], ["D"], ["AXX"], ["AYY"], ["A"], ["AY"],
#                  ["DSG", "D", "AXX", "AYY", "A", "AY"]]
# obs_name_grouped_list = ["DSG", "D", "AXX", "AYY", "A", "AY", "ALLOBS"]
# obs_labels_grouped_list = [r'$\displaystyle\frac{d\sigma}{d\Omega}$',
#                            r'$D$', r'$A_{xx}$', r'$A_{yy}$', r'$A$', r'$A_{y}$',
#                            r'Obs.']
# mesh_cart_grouped_list = [[mesh_cart_q], [mesh_cart_q], [mesh_cart_q],
#                           [mesh_cart_q], [mesh_cart_q], [mesh_cart_q],
#                           [mesh_cart_q, mesh_cart_q, mesh_cart_q, mesh_cart_q,
#                           mesh_cart_q, mesh_cart_q]]

# # # ALLOBS for angle input spaces
# plot_obs_list = [["DSG", "D", "AXX", "AYY", "A", "AY"]]
# obs_name_grouped_list = ["ALLOBS"]
# obs_labels_grouped_list = [r'Obs.']
# mesh_cart_grouped_list = [[mesh_cart_q, mesh_cart_q, mesh_cart_q, mesh_cart_q,
#                            mesh_cart_q, mesh_cart_q]]

# # # EACHOBS and ALLOBS for angle input spaces
# plot_obs_list = [["DSG"],
#                  ["D"], ["AXX"], ["AYY"], ["A"], ["AY"]]
# obs_name_grouped_list = ["DSG", "D", "AXX", "AYY", "A", "AY"]
# obs_labels_grouped_list = [r'$\displaystyle\frac{d\sigma}{d\Omega}$',
#                            r'$D$', r'$A_{xx}$', r'$A_{yy}$', r'$A$', r'$A_{y}$']

# # EACHOBS for energy input spaces
# plot_obs_list = [["SGT"], ["DSG"],
#                  ["DSG30"], ["DSG60"], ["DSG90"], ["DSG120"], ["DSG150"],
#                  ["DSG30", "DSG60", "DSG90", "DSG120", "DSG150"],
#                  ["DSG5"], ["DSG21"], ["DSG48"], ["DSG85"], ["DSG133"],
#                  ["DSG5", "DSG21", "DSG48", "DSG85", "DSG133"],
#                  ]
# obs_name_grouped_list = ["SGT", "DSG",
#                          "DSG30", "DSG60", "DSG90", "DSG120", "DSG150", "DSGESUM",
#                          "DSG5", "DSG21", "DSG48", "DSG85", "DSG133", "DSGTHETASUM", ]
# obs_labels_grouped_list = [r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$',
#                            r'$\displaystyle\frac{d\sigma}{d\Omega}(E, \Theta = 30^{\circ})$',
#                            r'$\displaystyle\frac{d\sigma}{d\Omega}(E, \Theta = 60^{\circ})$',
#                            r'$\displaystyle\frac{d\sigma}{d\Omega}(E, \Theta = 90^{\circ})$',
#                            r'$\displaystyle\frac{d\sigma}{d\Omega}(E, \Theta = 120^{\circ})$',
#                            r'$\displaystyle\frac{d\sigma}{d\Omega}(E, \Theta = 150^{\circ})$',
#                            r'$\displaystyle\Pi\frac{d\sigma}{d\Omega}(E)$',
#                            r'$\displaystyle\frac{d\sigma}{d\Omega}(\Theta, p = 50\,\mathrm{MeV})$',
#                            r'$\displaystyle\frac{d\sigma}{d\Omega}(\Theta, p = 100\,\mathrm{MeV})$',
#                            r'$\displaystyle\frac{d\sigma}{d\Omega}(\Theta, p = 150\,\mathrm{MeV})$',
#                            r'$\displaystyle\frac{d\sigma}{d\Omega}(\Theta, p = 200\,\mathrm{MeV})$',
#                            r'$\displaystyle\frac{d\sigma}{d\Omega}(\Theta, p = 250\,\mathrm{MeV})$',
#                            r'$\displaystyle\Pi\frac{d\sigma}{d\Omega}(\Theta)$'
#                            ]
# mesh_cart_grouped_list = [[mesh_cart_sgt], [mesh_cart],
#                           [mesh_cart_sgt], [mesh_cart_sgt], [mesh_cart_sgt],
#                           [mesh_cart_sgt], [mesh_cart_sgt],
#                           [mesh_cart_sgt, mesh_cart_sgt, mesh_cart_sgt, mesh_cart_sgt, mesh_cart_sgt],
#                           [mesh_cart_ang], [mesh_cart_ang], [mesh_cart_ang],
#                           [mesh_cart_ang], [mesh_cart_ang],
#                           [mesh_cart_ang, mesh_cart_ang, mesh_cart_ang, mesh_cart_ang,
#                            mesh_cart_ang]
#                           ]

# # ALLOBS
# plot_obs_list = [["DSG", "D", "AXX", "AYY", "A", "AY"]]
# obs_name_grouped_list = ["ALLOBS"]
# obs_labels_grouped_list = [r'$\Pi$Obs.']
# mesh_cart_grouped_list = [[mesh_cart, mesh_cart, mesh_cart,
#                            mesh_cart, mesh_cart, mesh_cart]]

# sets the RandomVariable objects
LambdabVariable = RandomVariable(var=lambda_vals,
                                 user_val=None,
                                 name='Lambdab',
                                 label="\Lambda_{b}",
                                 units="MeV",
                                 ticks=[300, 400, 500, 600, 700, 800],
                                 logprior=Lb_logprior(lambda_vals),
                                 logprior_name="uniformprior",
                                 marg_bool = True)
LsDegVariable = RandomVariable(var=ls_deg_vals,
                            user_val=None,
                            name='lsdeg',
                            label="\ell_{\Theta}",
                            units="",
                            ticks=[],
                            logprior=np.zeros(len(ls_deg_vals)),
                            logprior_name="noprior",
                            marg_bool=True)
# LsDegMagVariable = RandomVariable(var=ls_deg_mag_vals,
#                             user_val=None,
#                             name='lsdegmag',
#                             label="a",
#                             units="",
#                             ticks=[],
#                             logprior=np.zeros(len(ls_deg_mag_vals)),
#                             logprior_name="noprior",
#                             marg_bool=True)
# LsDegExpVariable = RandomVariable(var=ls_deg_exp_vals,
#                             user_val=None,
#                             name='lsdegexp',
#                             label="b",
#                             units="",
#                             ticks=[],
#                             logprior=np.zeros(len(ls_deg_exp_vals)),
#                             logprior_name="noprior",
#                             marg_bool=True)
LsTlabVariable = RandomVariable(var=ls_tlab_vals,
                            user_val=None,
                            name='lstlab',
                            label="\ell_{T}",
                            units="MeV",
                            ticks=[],
                            logprior=np.zeros(len(ls_tlab_vals)),
                            logprior_name="noprior",
                            marg_bool=True)
MpieffVariable = RandomVariable(var=mpi_vals,
                                user_val=None,
                                name='mpieff',
                                label="m_{\pi}",
                                units="MeV",
                                ticks=[50, 100, 150, 200, 250, 300],
                                logprior=mpieff_logprior(mpi_vals),
                                logprior_name="uniformprior",
                                marg_bool = True)
variables_array = np.array([LambdabVariable, LsTlabVariable, LsDegVariable, MpieffVariable])
# variables_array = np.array([LambdabVariable, LsTlabVariable, LsDegMagVariable, LsDegExpVariable, MpieffVariable])

# ls_deg_vals = np.linspace(0.01, 4, 100, dtype=np.dtype('f4'))
# q_vals = np.linspace(0.01, 1.01, 100, dtype=np.dtype('f4'))
#
# mesh_cart_q = gm.cartesian(q_vals, np.log(ls_deg_vals))
#
# QVariable = RandomVariable(var=q_vals,
#                                 user_val=0.3,
#                                 name='Q',
#                                 label="Q",
#                                 units="",
#                                 ticks=[0.2, 0.4, 0.6, 0.8],
#                                 logprior=np.zeros(len(q_vals)),
#                                 logprior_name="noprior",
#                                 marg_bool=True)
# LsDegVariable = RandomVariable(var=ls_deg_vals,
#                                user_val=None,
#                                name='lsdeg',
#                                label="\ell_{\Theta}",
#                                units="",
#                                ticks=[],
#                                logprior=np.zeros(len(ls_deg_vals)),
#                                logprior_name="noprior",
#                                marg_bool=False)
# variables_array = np.array([QVariable, LsDegVariable])

ratio_fn=ratio_fn_curvewise
ratio_fn_kwargs={
    "p_param": "pprel",
    "Q_param": "sum",
    "mpi_var": 138,
    "lambda_var": 570,
    "single_expansion": False,
}
log_likelihood_fn=log_likelihood
log_likelihood_fn_kwargs={
    "p_param": "pprel",
    "Q_param": "sum",
    "single_expansion": False,
}

def warping_fn(pts_array):
    pts_array_shape = np.shape(pts_array)
    pts_array = np.reshape(pts_array, (np.prod(pts_array_shape[:-1]), ) + (pts_array_shape[-1], ))
    print("pts_array_shape = " + str(pts_array_shape))
    print("pts_array has shape " + str(np.shape(pts_array)))
    try:
        pass
        # for pt_idx, pt in enumerate(pts_array):
        #     pts_array[pt_idx, :] = np.array([pts_array[pt_idx, 0],
        #                                      pts_array[pt_idx, 1]])
        # for pt_idx, pt in enumerate(pts_array):
        #     pts_array[pt_idx, :] = np.array([pts_array[pt_idx, 0],
        #                                      pts_array[pt_idx, 1] * 0.23 / (990 * (pts_array[pt_idx, 0])**(-1.4)),])
        # for pt_idx, pt in enumerate(pts_array):
        #     pts_array[pt_idx, :] = np.array([pts_array[pt_idx, 0],
        #                                      pts_array[pt_idx, 1] * 0.28 / (110 * (pts_array[pt_idx, 0])**(-1.)),])
        # for pt_idx, pt in enumerate(pts_array):
        #     pts_array[pt_idx, :] = np.array([pts_array[pt_idx, 0],
        #                                      pts_array[pt_idx, 1] * 0.37 / (100 * (pts_array[pt_idx, 0])**(-0.94)),])
        # for pt_idx, pt in enumerate(pts_array):
        #     pts_array[pt_idx, :] = np.array([pts_array[pt_idx, 0],
        #                                      pts_array[pt_idx, 1] * 0.26 / (340 * (pts_array[pt_idx, 0])**(-1.2)),])

#         for pt_idx, pt in enumerate(pts_array):
#             pts_array[pt_idx, :] = np.array([pts_array[pt_idx, 0],])
#                     for pt_idx, pt in enumerate(pts_array):
#                         pts_array[pt_idx, :] = np.array([pts_array[pt_idx, 0]**(2) / 500, ])
#                     for pt_idx, pt in enumerate(pts_array):
#                         pts_array[pt_idx, :] = np.array([pts_array[pt_idx, 0]**(3) / 2, ])
    except:
        pass

    pts_array = np.reshape(pts_array, pts_array_shape)

    return pts_array

warping_fn_kwargs = {}

def scaling_fn(X,
               ls_array,
               ):
    X_shape = np.shape(X)
    X = np.reshape(X, (np.prod(X_shape[:-1]), ) + (X_shape[-1], ))
    ls = np.array([])
    try:
        for pt_idx, pt in enumerate(X):
            # ls = np.append(ls, np.array([ls_array[0], ls_array[1] * X[pt_idx, 0]**(-1. * ls_array[2])
            #                              ]))
            ls = np.append(ls, ls_array)
    except:
        pass

    ls = np.reshape(ls, X_shape)
    # print("ls_array = " + str(ls_array))
    # print("ls = " + str(ls))

    return ls

scaling_fn_kwargs={}

generate_posteriors(
    scale_scheme_bunch_array=[RKE500MeV],
    Q_param_method_array=["sum"],
    p_param_method_array=["pprel"],
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
    # t_lab_train_pts=np.array([12, 33, 65, 108, 161, 225, 300]),  # set8
    # t_lab_train_pts=np.array([33, 65, 108, 161, 225, 300]),  # set9
    # t_lab_train_pts=np.array([65, 108, 161, 225,  300]),  # set10
    # t_lab_train_pts=np.array([108, 161, 225, 300]),  # set11
    # t_lab_train_pts=np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]),  # 1704
    # t_lab_train_pts=np.array([42, 69, 103, 144, 192]),  # setdrp5
    # t_lab_train_pts=np.array([42, 62, 85, 113, 145, 179, 218]),  # setdrp7
    # t_lab_train_pts=np.array([21, 48, 85, 133, 192, 261]),  # set12
    # t_lab_train_pts=np.array([12, 33, 65, 108, 161, 225]),  # set13
    degrees_train_pts=np.array([41, 60, 76, 90, 104, 120, 139]), # evencos
    # degrees_train_pts=np.array(
    #     [15, 31, 50, 90, 130, 149, 165]
    # ),  # evensin
    # degrees_train_pts=np.array([40, 60, 80, 100, 120, 140]), # 1704
    # degrees_train_pts=np.array([60, 70, 80, 90, 100, 110, 120]), # morecos
    # degrees_train_pts=np.array([1, 2, 4, 90, 176, 178, 179]), # evencsc
    orders_from_ho=1,
    orders_excluded=[],
    orders_names_dict=None,
    orders_labels_dict=None,
    # LengthScaleTlabInput=LengthScale("1/16-1_fitted", 0.25, 0.25, 4, whether_fit=True),
    # LengthScaleDegInput=LengthScale("1/16-1_fitted", 0.25, 0.25, 4, whether_fit=True),
    length_scale_list = [NSKernelParam(60, [10, 200]),
                         NSKernelParam(0.4, [0.05, 3])],
    # length_scale_list = [NSKernelParam(60, [10, 200]),
    #                      NSKernelParam(100, [10, 1000]),
    #                      NSKernelParam(1, [0.2, 2])],
    m_pi_eff=138,
    Lambdab=570,
    print_all_classes=False,
    savefile_type="png",

    plot_posterior_curvewise_bool=True,
    plot_corner_curvewise_bool=True,
    use_data_curvewise_bool=False,
    save_data_curvewise_bool=False,
    save_posterior_curvewise_bool=False,

    plot_obs_list = plot_obs_list,
    obs_name_grouped_list = obs_name_grouped_list,
    obs_labels_grouped_list = obs_labels_grouped_list,
    mesh_cart_grouped_list = mesh_cart_grouped_list,
    variables_array_curvewise = variables_array,

    ratio_fn_posterior=ratio_fn,
    ratio_fn_kwargs_posterior=ratio_fn_kwargs,
    log_likelihood_fn_posterior=log_likelihood_fn,
    log_likelihood_fn_kwargs_posterior=log_likelihood_fn_kwargs,
    warping_fn = warping_fn,
    warping_fn_kwargs = warping_fn_kwargs,
    scaling_fn = scaling_fn,
    scaling_fn_kwargs = scaling_fn_kwargs,

    plot_posterior_pointwise_bool=True,
    save_posterior_pointwise_bool=False,

    variables_array_pointwise = np.array([LambdabVariable]),

    filename_addendum="_nsrbf",
)