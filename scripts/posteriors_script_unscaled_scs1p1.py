from generator_fns import *

# sets the meshes for the random variable arrays
mpi_vals = np.linspace(1, 350, 150, dtype=np.dtype('f4'))
ls_deg_vals = np.linspace(0.01, 4, 150, dtype=np.dtype('f4'))
ls_tlab_vals = np.linspace(1, 150, 150, dtype=np.dtype('f4'))
lambda_vals = np.linspace(200, 900, 150, dtype=np.dtype('f4'))

mesh_cart = gm.cartesian(lambda_vals, np.log(ls_tlab_vals), np.log(ls_deg_vals), mpi_vals)
mesh_cart_sgt = np.delete(mesh_cart, 2, 1)

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
LsDegVariable = RandomVariable(var=ls_deg_vals,
                            user_val=None,
                            name='lsdeg',
                            label="\ell_{\Theta}",
                            units="",
                            ticks=[],
                            logprior=np.zeros(len(ls_deg_vals)),
                            logprior_name="noprior",
                            marg_bool=False)
LsTlabVariable = RandomVariable(var=ls_tlab_vals,
                            user_val=None,
                            name='lstlab',
                            label="\ell_{T}",
                            units="MeV",
                            ticks=[],
                            logprior=np.zeros(len(ls_tlab_vals)),
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
variables_array = np.array([LambdabVariable, LsTlabVariable, LsDegVariable, MpieffVariable])

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

generate_posteriors(
    scale_scheme_bunch_array=[EKM1p1fm],
    Q_param_method_array=["sum"],
    p_param_method_array=["pprel"],
    input_space_deg=["cos"],
    input_space_tlab=["prel"],
    t_lab_train_pts=np.array([1, 12, 33, 65, 108, 161, 225, 300]),
    degrees_train_pts=np.array([41, 60, 76, 90, 104, 120, 139]),
    orders_from_ho=3,
    orders_excluded=[],
    orders_names_dict=None,
    orders_labels_dict=None,
    length_scale_list=[NSKernelParam(60, [10, 200]),
                       NSKernelParam(0.4, [0.05, 3])],
    length_scale_fixed=False,
    cbar_list=[NSKernelParam(1.0, [0.1, 10])],
    cbar_fixed=True,
    m_pi_eff=141,
    Lambdab=480,
    print_all_classes=False,
    savefile_type="png",

    plot_posterior_curvewise_bool=True,
    plot_marg_curvewise_bool=True,
    plot_corner_curvewise_bool=True,
    use_data_curvewise_bool=True,
    save_data_curvewise_bool=True,
    save_posterior_curvewise_bool=True,

    plot_obs_list=plot_obs_list,
    obs_name_grouped_list=obs_name_grouped_list,
    obs_labels_grouped_list=obs_labels_grouped_list,
    mesh_cart_grouped_list=mesh_cart_grouped_list,
    variables_array_curvewise=variables_array,

    ratio_fn_posterior=ratio_fn,
    ratio_fn_kwargs_posterior=ratio_fn_kwargs,
    log_likelihood_fn_posterior=log_likelihood_fn,
    log_likelihood_fn_kwargs_posterior=log_likelihood_fn_kwargs,

    plot_posterior_pointwise_bool=False,
    save_posterior_pointwise_bool=False,

    variables_array_pointwise=np.array([LambdabVariable]),

    filename_addendum="_cluster",
)