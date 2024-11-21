from generator_fns import *

# sets the meshes for the random variable arrays
# mpi_vals = np.linspace(1, 301, 150, dtype=np.dtype('f4'))
cbar_scale_vals = np.linspace(3.5, 7.5, 30)
cbar_offset_vals = np.linspace(1.5, 2.5, 25)
ls_tlab_vals = np.linspace(40, 90, 20, dtype=np.dtype('f4'))
ls_deg_vals = np.linspace(0.11, 0.81, 20, dtype=np.dtype('f4'))
lambda_vals = np.linspace(250, 850, 150, dtype=np.dtype('f4'))
LambdabVariable = RandomVariable(var=lambda_vals,
                                 user_val=None,
                                 name='Lambdab',
                                 label="\Lambda_{b}",
                                 units="MeV",
                                 ticks=[300, 450, 600, 750],
                                 logprior=Lb_logprior(lambda_vals),
                                 logprior_name="uniformprior",
                                 marg_bool = True)

mesh_cart = gm.cartesian(np.log(cbar_scale_vals), np.log(cbar_offset_vals), np.log(ls_tlab_vals), np.log(ls_deg_vals))

# ALLOBS
plot_obs_list = [["DSG", "D", "AXX", "AYY", "A", "AY"]]
obs_name_grouped_list = ["ALLOBS"]
obs_labels_grouped_list = [r'$\Pi$Obs.']
mesh_cart_grouped_list = [[mesh_cart, mesh_cart, mesh_cart,
                           mesh_cart, mesh_cart, mesh_cart]]

# sets the RandomVariable objects

CbarScaleVariable = RandomVariable(var=cbar_scale_vals,
                                   user_val=2,
                                   name='cbarscale',
                                   label="k",
                                   units="",
                                   ticks=[],
                                   logprior=np.zeros(len(cbar_scale_vals)),
                                   logprior_name="noprior",
                                   marg_bool=True)
CbarOffsetVariable = RandomVariable(var=cbar_offset_vals,
                                    user_val=0.5,
                                    name='cbaroffset',
                                    label="\delta",
                                    units="",
                                    ticks=[],
                                    logprior=np.zeros(len(cbar_offset_vals)),
                                    logprior_name="noprior",
                                    marg_bool=True)
LsTlabVariable = RandomVariable(var=ls_tlab_vals,
                                user_val=None,
                                name='lstlab',
                                label="\ell_{T}",
                                units="MeV",
                                ticks=[],
                                logprior=np.zeros(len(ls_tlab_vals)),
                                logprior_name="noprior",
                                marg_bool=False)
LsDegVariable = RandomVariable(var=ls_deg_vals,
                            user_val=None,
                            name='lsdeg',
                            label="\ell_{\Theta}",
                            units="",
                            ticks=[],
                            logprior=np.zeros(len(ls_deg_vals)),
                            logprior_name="noprior",
                            marg_bool=False)
variables_array = np.array([
                            CbarScaleVariable, CbarOffsetVariable,
                            LsTlabVariable, LsDegVariable,
                            ])


def ratio_fn_new(X, p_grid_train, p_param, p_shape, Q_param, mpi_var, lambda_var,
                 single_expansion=False):
    """
    Function for interpolating between the input space and the ratio across that input space.

    Parameters
    ----------
    X (array) : array of input-space values. These are never called but must be passed here anyway due to GSUM.
    p_grid_train : array of momenta for evaluating ratio
    p_param (str) : type of p parametrization.
        Can be "Qofprel", "Qofqcm", or "Qofpq"
    p_shape (tuple) : shape for momentum before calculating ratio.
    Q_param (str) : type of Q parametrization.
        Can be "smoothmax", "max", or "sum".
    mpi_var (float) : value of the (effective) pion mass (in MeV) for calculating the ratio.
    lambda_var (float) : value of the breakdown scale (in MeV) for calculating the ratio.
    single_expansion (bool) : if True, then mpi_var is set to 0 within Q_approx
        Default : False
    """
    p = np.array([])
    for pt in p_grid_train:
        try:
            # p = np.append(p, p_approx(p_name = p_param, degrees = np.array([pt[0]]), prel = np.array([pt[1]])))
            p = np.append(
                p,
                p_approx(
                    p_name=p_param, degrees=np.array([pt[1]]), prel=np.array([pt[0]])
                ),
            )
        except:
            p = np.append(
                p,
                p_approx(p_name=p_param, degrees=np.array([0]), prel=np.array([pt[0]])),
            )

    return Q_approx(p=np.reshape(p, p_shape),
                    Q_parametrization=Q_param, Lambda_b=lambda_var,
                    m_pi=mpi_var, single_expansion=single_expansion)


@ray.remote
def log_likelihood_new(gp_fitted,
                       mesh_points,
                       log_likelihood_fn_kwargs
                       ):
    """
    Calculates the log-likelihood for a set of inputs.

    Parameters
    ----------
    gp_fitted (TruncationGP) : fitted Gaussian process object.
    mesh_points (float array) : array of tuples of random variables at which to evaluate the log-likelihood.
        Must be in the order (lambda_var, all length scales, mpi_var).
    log_likelihood_fn_kwargs (dict) : keyword arguments for log_likelihood.
    """
    return [gp_fitted.log_marginal_likelihood([pt[n] for n in range(len(pt))],
                                              **{**log_likelihood_fn_kwargs}

                                              ) for pt in mesh_points]

ratio_fn=ratio_fn_new
ratio_fn_kwargs={
    "p_param": "pprel",
    "Q_param": "sum",
    "mpi_var": 138,
    "lambda_var": 570,
    "single_expansion": False,
}
log_likelihood_fn=log_likelihood_new
log_likelihood_fn_kwargs={
    "p_param": "pprel",
    "Q_param": "sum",
    "mpi_var": 138,
    "lambda_var": 570,
    "single_expansion": False,
}

def warping_fn(pts_array,
               magnitude = 1,
               exponent = -1):
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
               ls_array = np.array([1]),
               exponent = 0):
    X_shape = np.shape(X)
    X = np.reshape(X, (np.prod(X_shape[:-1]), ) + (X_shape[-1], ))
    ls = np.array([])

    for pt_idx, pt in enumerate(X):
        ls = np.append(ls, ls_array)

    ls = np.reshape(ls, X_shape)

    return ls

scaling_fn_kwargs={}

def cbar_fn(X,
               cbar_array = np.array([1]),
               scaling = 1,
               offset = 0.5):
    X_shape = np.shape(X)
    X = np.reshape(X, (np.prod(X_shape[:-1]), ) + (X_shape[-1], ))
    cbar = np.array([])
    try:
        for pt_idx, pt in enumerate(X):
            R = 406
            # cbar = np.append(cbar, cbar_array)
            cbar = np.append(cbar, np.array([(1 + (cbar_array[0] / R * (pt[0] - cbar_array[1] * R)) ** (2)) ** (-0.5)
                                             ]))
    except:
        pass

    cbar = np.reshape(cbar, X_shape[:-1])

    return cbar

cbar_fn_kwargs={}

generate_posteriors(
    scale_scheme_bunch_array=[RKE500MeV],
    Q_param_method_array=["sum"],
    p_param_method_array=["pprel"],
    input_space_deg=["cos"],
    input_space_tlab=["prel"],
    t_lab_train_pts=np.array([1, 12, 33, 65, 108, 161, 225, 300]),
    degrees_train_pts=np.array([41, 60, 76, 90, 104, 120, 139]),
    orders_from_ho=1,
    orders_excluded=[],
    orders_names_dict=None,
    orders_labels_dict=None,
    length_scale_list=[NSKernelParam(60, [10, 200]),
                       NSKernelParam(0.2 * 2,
                                     [0.05 * 2,
                                      1 * 2])],
    length_scale_fixed=False,
    cbar_list=[NSKernelParam(1.0, [0.1, 10]),
               NSKernelParam(0.5, [0.01, 0.99])],
    cbar_fixed=False,
    m_pi_eff=138,
    Lambdab=570,
    print_all_classes=False,
    savefile_type="png",

    plot_posterior_curvewise_bool=True,
    plot_marg_curvewise_bool=True,
    plot_corner_curvewise_bool=True,
    use_data_curvewise_bool=False,
    save_data_curvewise_bool=False,
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

    filename_addendum="_cluster4",
)