from generator_fns import *

# sets the meshes for the random variable arrays
mpi_vals = np.linspace(1, 301, 150, dtype=np.dtype('f4'))
ls_tlab_vals = np.linspace(1, 100, 100, dtype=np.dtype('f4'))
ls_deg_mag_vals = np.linspace(1, 601, 150, dtype=np.dtype('f4'))
lambda_vals = np.linspace(200, 900, 150, dtype=np.dtype('f4'))

mesh_cart = gm.cartesian(lambda_vals, np.log(ls_tlab_vals), np.log(ls_deg_mag_vals), mpi_vals)

# ALLOBS
plot_obs_list = [["DSG", "D", "AXX", "AYY", "A", "AY"]]
obs_name_grouped_list = ["ALLOBS"]
obs_labels_grouped_list = [r'$\Pi$Obs.']
mesh_cart_grouped_list = [[mesh_cart, mesh_cart, mesh_cart,
                           mesh_cart, mesh_cart, mesh_cart]]

# sets the RandomVariable objects
LambdabVariable = RandomVariable(var=lambda_vals,
                                 user_val=None,
                                 name='Lambdab',
                                 label="\Lambda_{b}",
                                 units="MeV",
                                 ticks=[300, 450, 600, 750],
                                 logprior=Lb_logprior(lambda_vals),
                                 logprior_name="uniformprior",
                                 marg_bool = True)
LsTlabVariable = RandomVariable(var=ls_tlab_vals,
                            user_val=None,
                            name='lstlab',
                            label="\ell_{T}",
                            units="MeV",
                            ticks=[],
                            logprior=np.zeros(len(ls_tlab_vals)),
                            logprior_name="noprior",
                            marg_bool=False)
LsDegMagVariable = RandomVariable(var=ls_deg_mag_vals,
                            user_val=None,
                            name='lsdegmag',
                            label="a",
                            units="",
                            ticks=[],
                            logprior=np.zeros(len(ls_deg_mag_vals)),
                            logprior_name="noprior",
                            marg_bool=False)
MpieffVariable = RandomVariable(var=mpi_vals,
                                user_val=None,
                                name='mpieff',
                                label="m_{\pi}",
                                units="MeV",
                                ticks=[50, 100, 150, 200, 250, 300],
                                logprior=mpieff_logprior(mpi_vals),
                                logprior_name="uniformprior",
                                marg_bool = True)
variables_array = np.array([LambdabVariable, LsTlabVariable, LsDegMagVariable, MpieffVariable])

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
    try:
        pass
    except:
        pass

    pts_array = np.reshape(pts_array, pts_array_shape)

    return pts_array

warping_fn_kwargs = {}

def scaling_fn(X,
               ls_array,
               exponent = 0):
    X_shape = np.shape(X)
    X = np.reshape(X, (np.prod(X_shape[:-1]), ) + (X_shape[-1], ))
    ls = np.array([])
    try:
        for pt_idx, pt in enumerate(X):
            ls = np.append(ls, np.array([ls_array[0], ls_array[1] * X[pt_idx, 0]**(-1. * exponent)
                                         ]))
    except:
        pass

    ls = np.reshape(ls, X_shape)

    return ls

scaling_fn_kwargs={"exponent" : 0.99}

def cbar_fn(X,
               cbar_array = np.array([1]),
               scaling = 1,
               offset = 0.5):
    X_shape = np.shape(X)
    X = np.reshape(X, (np.prod(X_shape[:-1]), ) + (X_shape[-1], ))
    cbar = np.array([])
    try:
        for pt_idx, pt in enumerate(X):
            R = np.max(X[:, 0]) - np.min(X[:, 0])
            cbar = np.append(cbar, np.array([(1 + (scaling / R * (pt[0] - offset * R)) ** (2)) ** (-0.5)
                                             ]))
    except:
        pass

    cbar = np.reshape(cbar, X_shape[:-1])

    return cbar

cbar_fn_kwargs={"scaling" : 1.7,
                "offset" : 0.71}

generate_posteriors(
    scale_scheme_bunch_array=[RKE500MeV],
    Q_param_method_array=["sum"],
    p_param_method_array=["pprel"],
    input_space_deg=["cos"],
    input_space_tlab=["prel"],
    t_lab_train_pts=np.array([1, 12, 33, 65, 108, 161, 225, 300]),
    degrees_train_pts=np.array([41, 60, 76, 90, 104, 120, 139]),
    orders_from_ho=4,
    orders_excluded=[],
    orders_names_dict=None,
    orders_labels_dict=None,
    length_scale_list=[NSKernelParam(60, [10, 200]),
                       NSKernelParam(100, [10, 1000])],
    length_scale_fixed=False,
    cbar_list=[NSKernelParam(1.0, [0.1, 10])],
    cbar_fixed=True,
    m_pi_eff=138,
    Lambdab=570,
    print_all_classes=False,
    savefile_type="png",

    plot_posterior_curvewise_bool=True,
    plot_marg_curvewise_bool=False,
    plot_corner_curvewise_bool=False,
    use_data_curvewise_bool=False,
    save_data_curvewise_bool=True,
    save_posterior_curvewise_bool=False,

    plot_obs_list=plot_obs_list,
    obs_name_grouped_list=obs_name_grouped_list,
    obs_labels_grouped_list=obs_labels_grouped_list,
    mesh_cart_grouped_list=mesh_cart_grouped_list,
    variables_array_curvewise=variables_array,

    ratio_fn_posterior=ratio_fn,
    ratio_fn_kwargs_posterior=ratio_fn_kwargs,
    log_likelihood_fn_posterior=log_likelihood_fn,
    log_likelihood_fn_kwargs_posterior=log_likelihood_fn_kwargs,

    warping_fn = warping_fn,
    warping_fn_kwargs = warping_fn_kwargs,
    cbar_fn = cbar_fn,
    cbar_fn_kwargs = cbar_fn_kwargs,
    scaling_fn = scaling_fn,
    scaling_fn_kwargs = scaling_fn_kwargs,

    plot_posterior_pointwise_bool=False,
    save_posterior_pointwise_bool=False,

    variables_array_pointwise=np.array([LambdabVariable]),

    filename_addendum="_cluster3",
)