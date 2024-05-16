import gc

from cheftgp.constants import *
from cheftgp.eft import *
from cheftgp.gaussianprocess import *
from cheftgp.graphs import *
from cheftgp.scattering import *
from cheftgp.utils import *
from cheftgp.potentials import *
from cheftgp.traintestsplits import *

# sets rcParams for plotting
setup_rc_params()


def generate_diagnostics(
    scale_scheme_bunch_array=[RKE500MeV],
    observable_input=["DSG"],
    x_quantities_array=[[150], []],
    Q_param_method_array=["sum"],
    p_param_method_array=["Qofprel"],
    input_space_input=["cos"],
    train_test_split_array=[Fullspaceanglessplit1],
    orders_excluded=[],
    orders_names_dict=None,
    orders_labels_dict=None,
    LengthScale_list=[LengthScale("1/16-1_fitted", 0.25, 0.25, 4, whether_fit=True)],
    fixed_sd=None,
    m_pi_eff=138,
    Lambdab=600,
    print_all_classes=False,
    savefile_type="pdf",
    plot_coeffs_bool=True,
    plot_md_bool=True,
    plot_pc_bool=True,
    plot_ci_bool=True,
    plot_pdf_bool=True,
    plot_trunc_bool=True,
    plot_plotzilla_bool=True,
    save_coeffs_bool=True,
    save_md_bool=True,
    save_pc_bool=True,
    save_ci_bool=True,
    save_pdf_bool=True,
    save_trunc_bool=True,
    save_plotzilla_bool=True,
    filename_addendum="",
):
    """
    scale_scheme_bunch_array (ScaleSchemeBunch list): potential/cutoff
        combinations for evaluation.
    Built-in options:
        EKM0p8fm, EKM0p9fm, EKM1p0fm, EKM1p1fm, EKM1p2fm
        ( https://doi.org/10.1140/epja/i2015-15053-8 );
        RKE400MeV, RKE450MeV, RKE500MeV, RKE550MeV
        ( https://doi.org/10.1140/epja/i2018-12516-4 );
        EMN450MeV, EMN500MeV, EMN550MeV
        ( https://doi.org/10.1103/PhysRevC.96.024004 );
        GT0p9fm, GT1p0fm, GT1p1fm, GT1p2fm
        ( https://doi.org/10.1103/PhysRevC.90.054323 )
    Default: [RKE500MeV]

    observable_input (str list): observables for evaluation. Note that SGT
        should not be in the same list as other observables.
    Built-in options: "SGT", "DSG", "AY", "A", "D", "AXX", "AYY"
    Default: ["DSG"]

    x_quantities_array (list of lists of empty or 1 int): values (if any) at which to fix observables.
        Note that SGT must be treated differently since it is not evaluated at one energy at a time.
    Must be [] for SGT
    If no evaluation at fixed quantity is desired, set equal to []
    Default: [[150], []]

    Q_param_method_array (str list): methods of parametrizing the dimensionless
        expansion parameter Q for evaluation.
    Built-in options: "smax", "max", "sum", or "rawsum"
    Default: ["sum"]

    p_param_method_array (str list): methods of parametrizing the characteristic
        momentum p in Q(p) for evaluation.
    Built-in options: "Qofprel", "Qofqcm", "Qofpq"
    Default: ["Qofprel"]

    input_space_input (str list): input spaces for evaluation. Note that SGT
        must be treated differently since it is not evaluated at one energy at
        a time.
    Built-in options: "Elab", "prel" for energy-dependent input spaces
    Built-in options: "deg", "cos", "qcm", "qcm2" for angle-dependent input spaces
    Default: ["cos"]

    train_test_split_array (TrainTestSplit list): splits of training and
        testing points for evaluation. Note that SGT must be treated
        differently since it is not evaluated at one energy at a time.
    Built-in options: Nolowenergysplit, Yeslowenergysplit for SGT
    Built-in options: Fullspaceanglessplit, Forwardanglessplit,
        Backwardanglessplit for all other observables
    Default: Fullspaceanglessplit

    orders_excluded (int list): list of *coefficient* orders to be excluded from
        the fitting procedure. 2 corresponds to c2, 3 to c3, etc.
    Default: []

    orders_names_dict (dict): dictionary method linking the numerical indices (int)
        of EFT orders and their corresponding abbreviations (str)
    Default: None

    orders_labels_dict (dict): dictionary method linking the numerical indices (int)
        of EFT orders and their corresponding math-mode-formatted labels (str)
    Default: None

    LengthScale_list (LengthScale list): initial guess(es) for the correlation
        length in the kernel (as a factor of the total size of the energy-
        dependent input space) plus boundaries of the fit procedure as factors
        of the initial guess for the correlation length. Fitting may be bypassed
        when whether_fit = False.
    Default: [LengthScale(0.25, 0.25, 4, whether_fit = True)]

    fixed_sd (float): fixed standard deviation for the Gaussian process fit.
        May be any positive float. If None, then there is no fixed standard
        deviation and it is calculated by the fitting procedure.
    Default: None

    m_pi_eff (float): effective pion mass for the theory (in MeV).
    Default: 138

    Lambdab (float): breakdown scale for the theory (in MeV).
    Default: 600

    print_all_classes (bool): prints out all instances of potential, observable, Q parametrization, input space,
        train/test split, and length scale objects
    Default: False

    savefile_type (str): string for specifying the type of file to be saved.
    Default: 'png'

    plot_..._bool (bool): boolean for whether to plot different figures
    Default: True

    save_..._bool (bool): boolean for whether to save different figures
    Default: True

    filename_addendum (str): string for distinguishing otherwise similarly named
        files. Argument of the same name to FileNaming.__init__.
    Default: ''
    """
    mpl.rc(
        "savefig",
        transparent=False,
        bbox="tight",
        pad_inches=0.05,
        dpi=300,
        format=savefile_type,
    )

    # try:

    # runs through the potentials
    for o, ScaleScheme in enumerate(scale_scheme_bunch_array):
        # gets observable data from a local file
        # default location is the same as this program's
        SGT = ScaleScheme.get_data("SGT")
        DSG = ScaleScheme.get_data("DSG")
        AY = ScaleScheme.get_data("PB")
        A = ScaleScheme.get_data("A")
        D = ScaleScheme.get_data("D")
        AXX = ScaleScheme.get_data("AXX")
        AYY = ScaleScheme.get_data("AYY")
        t_lab = ScaleScheme.get_data("t_lab")
        degrees = ScaleScheme.get_data("degrees")

        for xq_idx in range(len(x_quantities_array)):
            if x_quantities_array[xq_idx] == []:
                x_quantities_array[xq_idx] = None

        x_quantities_tuples = gm.cartesian(*x_quantities_array)

        # runs through the x quantities at which to evaluate the observables
        for j, x_quantities_tuple in enumerate(x_quantities_tuples):
            E_lab_val = x_quantities_tuple[0]
            angle_lab_val = x_quantities_tuple[1]
            E_angle_vals_pair = [E_lab_val, angle_lab_val]

            # sets E_lab to full t_lab array when E_lab_val is None
            if E_lab_val is not None:
                E_lab = np.array([E_lab_val])
            else:
                E_lab = t_lab

            # sets angle_lab to full degrees array when angle_lab_val is None
            if angle_lab_val is not None:
                angle_lab = np.array([angle_lab_val])
            else:
                angle_lab = degrees

            # creates the bunch for each observable to be plotted against angle
            SGTBunch = ObservableBunch(
                "SGT",
                SGT,
                E_angle_vals_pair,
                "\sigma_{\mathrm{tot}}",
                "dimensionful",
                nn_interaction="np",
                unit_string="mb",
            )
            DSGBunch = ObservableBunch(
                "DSG",
                DSG,
                E_angle_vals_pair,
                "d \sigma / d \Omega",
                "dimensionful",
                nn_interaction="np",
                unit_string="mb",
            )
            AYBunch = ObservableBunch(
                "AY",
                AY,
                E_angle_vals_pair,
                "A_{y}",
                "dimensionless",
                nn_interaction="np",
                constraint=[[0, -1], [0, 0], "angle"],
            )
            ABunch = ObservableBunch(
                "A",
                A,
                E_angle_vals_pair,
                "A",
                "dimensionless",
                nn_interaction="np",
                constraint=[[0], [0], "angle"],
            )
            DBunch = ObservableBunch(
                "D",
                D,
                E_angle_vals_pair,
                "D",
                "dimensionless",
                nn_interaction="np",
            )
            DBunch_dimensionful = ObservableBunch(
                "D_dimensionful",
                D,
                E_angle_vals_pair,
                "D",
                "dimensionful",
                nn_interaction="np",
            )
            AXXBunch = ObservableBunch(
                "AXX",
                AXX,
                E_angle_vals_pair,
                "A_{xx}",
                "dimensionless",
                nn_interaction="np",
            )
            AYYBunch = ObservableBunch(
                "AYY",
                AYY,
                E_angle_vals_pair,
                "A_{yy}",
                "dimensionless",
                nn_interaction="np",
            )

            observable_array = [
                SGTBunch,
                DSGBunch,
                AYBunch,
                ABunch,
                DBunch,
                DBunch_dimensionful,
                AXXBunch,
                AYYBunch,
            ]

            observable_array = [
                b for b in observable_array if b.name in observable_input
            ]

            # runs through the observables
            for m, Observable in enumerate(observable_array):
                # runs through the parametrizations of p in Q(p)
                for p, PParamMethod in enumerate(p_param_method_array):
                    # instantiates input spaces
                    DegBunch = InputSpaceBunch(
                        "deg",
                        deg_fn,
                        p_approx(
                            PParamMethod,
                            E_to_p(E_lab, interaction=Observable.nn_interaction),
                            angle_lab,
                        ),
                        r"$\theta$ (deg)",
                        [
                            r"$",
                            Observable.title,
                            r"(\theta, E_{\mathrm{lab}}= ",
                            E_lab_val,
                            "\,\mathrm{MeV})$",
                        ],
                    )
                    CosBunch = InputSpaceBunch(
                        "cos",
                        neg_cos,
                        p_approx(
                            PParamMethod,
                            E_to_p(E_lab, interaction=Observable.nn_interaction),
                            angle_lab,
                        ),
                        r"$-\mathrm{cos}(\theta)$",
                        [
                            r"$",
                            Observable.title,
                            r"(-\mathrm{cos}(\theta), E_{\mathrm{lab}}= ",
                            E_lab_val,
                            "\,\mathrm{MeV})$",
                        ],
                    )
                    SinBunch = InputSpaceBunch(
                        "sin",
                        sin_thing,
                        p_approx(
                            PParamMethod,
                            E_to_p(E_lab, interaction=Observable.nn_interaction),
                            angle_lab,
                        ),
                        r"$\mathrm{sin}(\theta)$",
                        [
                            r"$",
                            Observable.title,
                            r"(\mathrm{sin}(\theta), E_{\mathrm{lab}}= ",
                            E_lab_val,
                            "\,\mathrm{MeV})$",
                        ],
                    )
                    QcmBunch = InputSpaceBunch(
                        "qcm",
                        deg_to_qcm,
                        p_approx(
                            PParamMethod,
                            E_to_p(E_lab, interaction=Observable.nn_interaction),
                            angle_lab,
                        ),
                        r"$q_{\mathrm{cm}}$ (MeV)",
                        [
                            r"$",
                            Observable.title,
                            r"(q_{\mathrm{cm}}, E_{\mathrm{lab}}= ",
                            E_lab_val,
                            "\,\mathrm{MeV})$",
                        ],
                    )
                    Qcm2Bunch = InputSpaceBunch(
                        "qcm2",
                        deg_to_qcm2,
                        p_approx(
                            PParamMethod,
                            E_to_p(E_lab, interaction=Observable.nn_interaction),
                            angle_lab,
                        ),
                        r"$q_{\mathrm{cm}}^{2}$ (MeV$^{2}$)",
                        [
                            r"$",
                            Observable.title,
                            r"(q_{\mathrm{cm}}^{2}, E_{\mathrm{lab}}= ",
                            E_lab_val,
                            "\,\mathrm{MeV})$",
                        ],
                    )

                    ElabBunch = InputSpaceBunch(
                        "Elab",
                        Elab_fn,
                        p_approx(
                            "Qofprel",
                            E_to_p(E_lab, interaction=Observable.nn_interaction),
                            angle_lab,
                        ),
                        r"$E_{\mathrm{lab}}$ (MeV)",
                        [
                            r"$",
                            Observable.title,
                            r"(E_{\mathrm{lab}}, \theta= ",
                            angle_lab_val,
                            "^{\circ})$",
                        ],
                    )

                    PrelBunch = InputSpaceBunch(
                        "prel",
                        E_to_p,
                        p_approx(
                            "Qofprel",
                            E_to_p(E_lab, interaction=Observable.nn_interaction),
                            angle_lab,
                        ),
                        r"$p_{\mathrm{rel}}$ (MeV)",
                        [
                            r"$",
                            Observable.title,
                            r"(p_{\mathrm{rel}}, \theta= ",
                            angle_lab_val,
                            "^{\circ})$",
                        ],
                    )

                    vsquantity_array = [
                        DegBunch,
                        CosBunch,
                        SinBunch,
                        QcmBunch,
                        Qcm2Bunch,
                        ElabBunch,
                        PrelBunch,
                    ]

                    vsquantity_array = np.reshape(
                        np.array(
                            [
                                b
                                for b in vsquantity_array
                                if b.name in np.array(input_space_input).flatten()
                            ]
                        ),
                        np.shape(np.array(input_space_input)),
                    )

                    # creates each input space bunch's title
                    for bunch in vsquantity_array.flatten():
                        try:
                            bunch.make_title()
                        except:
                            pass

                    # runs through the parametrization methods
                    for k, QParamMethod in enumerate(Q_param_method_array):
                        # runs through the input spaces
                        for i, VsQuantity in enumerate(vsquantity_array):
                            # runs through the training and testing masks
                            for l, TrainingTestingSplit in enumerate(
                                train_test_split_array
                            ):
                                # calculates initial guesses for the GP length scale, with bounds for fitting
                                for ls, vsq in zip(LengthScale_list, VsQuantity):
                                    ls.make_guess(
                                        vsq.input_space(
                                            **{
                                                "deg_input": angle_lab,
                                                "p_input": E_to_p(
                                                    E_lab, interaction="np"
                                                ),
                                                "E_lab": E_lab,
                                                "interaction": "np",
                                            }
                                        )
                                    )

                                # creates the GP with all its hyperparameters
                                ratio = Q_approx(
                                    p_approx(
                                        PParamMethod, E_to_p(E_lab, "np"), angle_lab
                                    ),
                                    QParamMethod,
                                    Lambda_b=Lambdab,
                                    m_pi=m_pi_eff,
                                ).T
                                center = 0
                                df = 1
                                disp = 0
                                std_scale = 1
                                GPHyper = GPHyperparameters(
                                    LengthScale_list,
                                    center,
                                    ratio,
                                    df=df,
                                    disp=disp,
                                    scale=std_scale,
                                    seed=None,
                                    sd=fixed_sd,
                                )

                                # information for naming the savefiles
                                FileName = FileNaming(
                                    QParamMethod,
                                    PParamMethod,
                                    filename_addendum=filename_addendum,
                                )

                                # information on the orders for each potential
                                Orders = OrderInfo(
                                    ScaleScheme.orders_full,
                                    [0] + orders_excluded,
                                    ScaleScheme.colors,
                                    ScaleScheme.light_colors,
                                    orders_names_dict=orders_names_dict,
                                    orders_labels_dict=orders_labels_dict,
                                )

                                # creates the object used to generate and plot statistical diagnostics
                                if Observable.name == "SGT":
                                    x_quantity = [["energy", E_lab, t_lab, "MeV"]]
                                else:
                                    x_quantity = [
                                        ["energy", E_lab, t_lab, "MeV"],
                                        ["angle", angle_lab, degrees, "degrees"],
                                    ]

                                MyPlot = GSUMDiagnostics(
                                    schemescale=ScaleScheme,
                                    observable=Observable,
                                    inputspace=VsQuantity,
                                    traintestsplit=TrainingTestingSplit,
                                    gphyperparameters=GPHyper,
                                    orderinfo=Orders,
                                    filenaming=FileName,
                                    x_quantity=x_quantity,
                                )

                                # plots figures
                                if plot_coeffs_bool:
                                    MyPlot.plot_coefficients(
                                        whether_save=save_coeffs_bool
                                    )
                                if plot_md_bool:
                                    MyPlot.plot_md(whether_save=save_md_bool)
                                if plot_pc_bool:
                                    MyPlot.plot_pc(whether_save=save_pc_bool)
                                if plot_ci_bool:
                                    MyPlot.plot_credible_intervals(
                                        whether_save=save_ci_bool
                                    )
                                if plot_pdf_bool:
                                    MyPlot.plot_posterior_pdf(
                                        whether_save=save_pdf_bool
                                    )
                                if plot_trunc_bool:
                                    # gets observables data from NN Online ("true" values for those observables)
                                    online_data_dict = get_nn_online_data()
                                    get_nn_online_data()

                                    MyPlot.plot_truncation_errors(
                                        online_data_dict[Observable.name],
                                        whether_save=save_trunc_bool,
                                        residual_plot=True,
                                    )

                            if plot_plotzilla_bool:
                                MyPlot.plotzilla(whether_save=save_plotzilla_bool)

    # except:
    #     print("Error encountered in running loop.")

    # prints all instances of the classes relevant for the arguments of
    # GPAnalysis()
    if print_all_classes:
        scalescheme_current_list = []
        observable_current_list = []
        inputspace_current_list = []
        traintest_current_list = []
        lengthscale_current_list = []

        for obj in gc.get_objects():
            if isinstance(obj, ScaleSchemeBunch):
                scalescheme_current_list.append(obj.name)
            elif isinstance(obj, ObservableBunch):
                observable_current_list.append(obj.name)
            elif isinstance(obj, InputSpaceBunch):
                inputspace_current_list.append(obj.name)
            elif isinstance(obj, TrainTestSplit):
                traintest_current_list.append(obj.name)
            elif isinstance(obj, LengthScale):
                lengthscale_current_list.append(obj.name)

        print("\n\n************************************")
        print("Available potentials: " + str(scalescheme_current_list))
        print("Available observables: " + str(observable_current_list))
        print("Available Q parametrizations: ['smax', 'max', 'sum', 'rawsum']")
        print("Available input spaces: " + str(inputspace_current_list))
        print("Available train/test splits: " + str(traintest_current_list))
        print("Available length scales: " + str(lengthscale_current_list))
        print("************************************")


def generate_posteriors(
    scale_scheme_bunch_array=[RKE500MeV],
    Q_param_method_array=["sum"],
    p_param_method_array=["Qofprel"],
    input_space_deg=["cos"],
    input_space_tlab=["prel"],
    Elab_slice=None,
    deg_slice=None,
    t_lab_train_pts=np.array([]),
    degrees_train_pts=np.array([]),
    orders_from_ho=1,
    orders_excluded=[],
    orders_names_dict=None,
    orders_labels_dict=None,
    length_scale_list=[None],
    cbar_list=[None],
    m_pi_eff=138,
    Lambdab=600,
    print_all_classes=False,
    savefile_type="pdf",
    plot_obs_list=[None],
    obs_name_grouped_list=[None],
    obs_labels_grouped_list=[None],
    mesh_cart_grouped_list=[None],
    variables_array_curvewise=[None],
    ratio_fn_posterior=None,
    ratio_fn_kwargs_posterior=None,
    log_likelihood_fn_posterior=None,
    log_likelihood_fn_kwargs_posterior=None,
    warping_fn=None,
    warping_fn_kwargs=None,
    cbar_fn=None,
    cbar_fn_kwargs=None,
    scaling_fn=None,
    scaling_fn_kwargs=None,
    variables_array_pointwise=[None],
    plot_posterior_pointwise_bool=False,
    plot_posterior_curvewise_bool=False,
    plot_corner_curvewise_bool=False,
    use_data_curvewise_bool=False,
    save_data_curvewise_bool=False,
    save_posterior_pointwise_bool=False,
    save_posterior_curvewise_bool=False,
    filename_addendum="",
):
    """
    nn_interaction (str): what pair of nucleons are interacting?
    Built-in options: "nn", "np", "pp"
    Default: "np"

    scale_scheme_bunch_array (ScaleSchemeBunch list): potential/cutoff
        combinations for evaluation.
    Built-in options:
        EKM0p8fm, EKM0p9fm, EKM1p0fm, EKM1p1fm, EKM1p2fm
        ( https://doi.org/10.1140/epja/i2015-15053-8 );
        RKE400MeV, RKE450MeV, RKE500MeV, RKE550MeV
        ( https://doi.org/10.1140/epja/i2018-12516-4 );
        EMN450MeV, EMN500MeV, EMN550MeV
        ( https://doi.org/10.1103/PhysRevC.96.024004 );
        GT0p9fm, GT1p0fm, GT1p1fm, GT1p2fm
        ( https://doi.org/10.1103/PhysRevC.90.054323 )
    Default: [RKE500MeV]

    Q_param_method_array (str list): methods of parametrizing the dimensionless
        expansion parameter Q for evaluation.
    Built-in options: "smax", "max", "sum", or "rawsum"
    Default: ["sum"]

    p_param_method_array (str list): methods of parametrizing the characteristic
        momentum p in Q(p) for evaluation.
    Built-in options: "Qofprel", "Qofqcm", "Qofpq"
    Default: ["Qofprel"]

    input_space_deg (str): angle-dependent input space for evaluating the posterior
        pdf for the breakdown scale, effective soft scale, length scales, etc.
    Built-in options: "deg", "cos", "qcm", "qcm2"
    Default: "cos"

    input_space_tlab (str): energy-dependent input space for evaluating the posterior
        pdf for the breakdown scale, effective soft scale, length scales, etc.
    Built-in options: "Elab", "prel"
    Default: "prel"

    Elab_slice (int): lab energy (in MeV) at which to slice the observables for the purposes
        of calculating posteriors.
    If None, every observable is left as is.
    Default : None

    deg_slice (int): scattering angle (in degrees) at which to slice the observables for the
        purposes of calculating posteriors.
    If None, every observable is left as is.
    Default : None

    t_lab_train_pts (float NumPy array): lab energies (in MeV) where the TruncationTP
        object will be trained. Will be converted to input_space_tlab by another function.
    Default: []

    degrees_train_pts (float NumPy array): scattering angles (in degrees) where the
        TruncationTP object will be trained. Will be converted to input_space_deg by
        another function.
    Default: []

    orders_from_ho (int): number of orders, including the highest order, for which the
        posterior will be evaluated.
    Default: 1

    orders_excluded (int list): list of *coefficient* orders to be excluded from
        the fitting procedure. 2 corresponds to c2, 3 to c3, etc.
    Default: []

    orders_names_dict (dict): dictionary method linking the numerical indices (int)
        of EFT orders and their corresponding abbreviations (str)
    Default: None

    orders_labels_dict (dict): dictionary method linking the numerical indices (int)
        of EFT orders and their corresponding math-mode-formatted labels (str)
    Default: None

    LengthScaleTlabInput (LengthScale): initial guess for the correlation
        length in the kernel (as a factor of the total size of the energy-
        dependent input space) plus boundaries of the fit procedure as factors
        of the initial guess for the correlation length. Fitting may be bypassed
        when whether_fit = False.
    Default: LengthScale(0.25, 0.25, 4, whether_fit = True)

    LengthScaleDegInput (LengthScale): initial guess for the correlation
        length in the kernel (as a factor of the total size of the angle-
        dependent input space) plus boundaries of the fit procedure as factors
        of the initial guess for the correlation length. Fitting may be bypassed
        when whether_fit = False.
    Default: LengthScale(0.25, 0.25, 4, whether_fit = True)

    m_pi_eff (float): effective pion mass for the theory (in MeV).
    Default: 138

    Lambdab (float): breakdown scale for the theory (in MeV).
    Default: 600

    print_all_classes (bool): prints out all instances of potential, observable, Q parametrization, input space,
        train/test split, and length scale objects
    Default: False

    savefile_type (str): string for specifying the type of file to be saved.
    Default: 'png'

    plot_obs_list (str list): list of strings corresponding to observables in obs_dict, grouped by
        whether they will be combined.
    For instance, [["SGT"], ["DSG"], ["SGT", "DSG"]] means that SGT and DSG will have their log-
        likelihoods calculated separately, and then they will be calculated again and summed.
    Default: [None]

    obs_name_grouped_list (str list): list of strings for naming the observables in filenames.
    Default: [None]

    obs_labels_grouped_list (str list): list of strings for labeling the observables in figures.
    Default: [None]

    mesh_cart_grouped_list (array list): list of arrays generated from the random-variable meshes,
        using GSUM's cartesian function.
    Order matters here, and is determined by ratio_fn and log_likelihood_fn.
    Default: [None]

    variables_array_curvewise (RandomVariable array): NumPy array of random variables.
    Order matters here, and is determined by ratio_fn and log_likelihood_fn.
    Default: [None]

    ratio_fn_posterior (function): function that maps momentum and scale values (from
        variables_array_curvewise) onto values of the ratio Q.
    Default: None

    ratio_fn_kwargs_posterior (dict): keyword arguments for ratio_fn_posterior.
    Default: None

    log_likelihood_fn_posterior (function): function that maps values of the random variables
        onto a log-likelihood value.
    Default: None

    log_likelihood_fn_kwargs_posterior (dict): keyword arguments for log_likelihood_fn_posterior.
    Default: None

    variables_array_pointwise (RandomVariable array): NumPy array of random variables.
    Order matters here, and is determined by ratio_fn and log_likelihood_fn.
    Should, in all likelihood, just be np.array([LambdabVariable]).
    Default: [None]

    plot_..._bool (bool): boolean for whether to plot different figures
    Default: True

    save_..._bool (bool): boolean for whether to save different figures
    Default: True

    filename_addendum (str): string for distinguishing otherwise similarly named
        files.
    Default: ''
    """

    mpl.rc(
        "savefig",
        transparent=False,
        bbox="tight",
        pad_inches=0.05,
        dpi=300,
        format=savefile_type,
    )

    # array for the MAP values, means, and standard deviations of the 1D pdfs
    stats_array_curvewise = np.array([])
    stats_array_pointwise = np.array([])

    # try:
    # runs through the potentials
    for o, ScaleScheme in enumerate(scale_scheme_bunch_array):
        # gets observable data from a local file
        # default location is the same as this program's
        SGT = ScaleScheme.get_data("SGT")
        DSG = ScaleScheme.get_data("DSG")
        AY = ScaleScheme.get_data("PB")
        A = ScaleScheme.get_data("A")
        D = ScaleScheme.get_data("D")
        AXX = ScaleScheme.get_data("AXX")
        AYY = ScaleScheme.get_data("AYY")
        t_lab = ScaleScheme.get_data("t_lab")
        degrees = ScaleScheme.get_data("degrees")

        # creates the bunch for each observable to be plotted
        SGTBunch = ObservableBunch(
            "SGT",
            SGT,
            [],
            "\sigma_{\mathrm{tot}}",
            "dimensionful",
            nn_interaction="np",
        )

        # no slicing; normal procedure.
        if (Elab_slice is None) and (deg_slice is None):
            DSGBunch = ObservableBunch(
                "DSG",
                DSG,
                [],
                "d \sigma / d \Omega",
                "dimensionful",
                nn_interaction="np",
            )
            AYBunch = ObservableBunch(
                "AY",
                AY,
                # np.reshape(AY[:, np.isin(t_lab, E_slice_Q), :], (np.shape(AY)[0], len(degrees))),
                [],
                "A_{y}",
                "dimensionless",
                nn_interaction="np",
                constraint=[[0, -1], [0, 0], "angle"],
            )
            ABunch = ObservableBunch(
                "A",
                A,
                [],
                "A",
                "dimensionless",
                nn_interaction="np",
                constraint=[[0], [0], "angle"],
            )
            DBunch = ObservableBunch(
                "D",
                D,
                [],
                "D",
                "dimensionless",
                nn_interaction="np",
            )
            # version of DBunch that treats D as dimensionful for the purposes of Fig. 22
            DBunch_dimensionful = ObservableBunch(
                "D_dimensionful",
                D,
                [],
                "D",
                "dimensionful",
                nn_interaction="np",
            )
            AXXBunch = ObservableBunch(
                "AXX",
                AXX,
                [],
                "A_{xx}",
                "dimensionless",
                nn_interaction="np",
            )
            AYYBunch = ObservableBunch(
                "AYY",
                AYY,
                [],
                "A_{yy}",
                "dimensionless",
                nn_interaction="np",
            )
        # sliced in energy; used for calculations with constant Q.
        elif (Elab_slice is not None) and (deg_slice is None):
            DSGBunch = ObservableBunch(
                "DSG",
                np.reshape(
                    DSG[:, np.isin(t_lab, Elab_slice), :],
                    (np.shape(DSG)[0], len(degrees)),
                ),
                [],
                "d \sigma / d \Omega",
                "dimensionful",
                nn_interaction="np",
            )
            AYBunch = ObservableBunch(
                "AY",
                # AY,
                np.reshape(
                    AY[:, np.isin(t_lab, Elab_slice), :],
                    (np.shape(AY)[0], len(degrees)),
                ),
                # np.reshape(AY[:, np.isin(t_lab, E_slice_Q), :], (np.shape(AY)[0], len(degrees))),
                [],
                "A_{y}",
                "dimensionless",
                nn_interaction="np",
                constraint=[[0, -1], [0, 0], "angle"],
            )
            ABunch = ObservableBunch(
                "A",
                np.reshape(
                    A[:, np.isin(t_lab, Elab_slice), :], (np.shape(A)[0], len(degrees))
                ),
                [],
                "A",
                "dimensionless",
                nn_interaction="np",
                constraint=[[0], [0], "angle"],
            )
            DBunch = ObservableBunch(
                "D",
                np.reshape(
                    D[:, np.isin(t_lab, Elab_slice), :], (np.shape(D)[0], len(degrees))
                ),
                [],
                "D",
                "dimensionless",
                nn_interaction="np",
            )
            # version of DBunch that treats D as dimensionful for the purposes of Fig. 22
            DBunch_dimensionful = ObservableBunch(
                "D_dimensionful",
                np.reshape(
                    D[:, np.isin(t_lab, Elab_slice), :], (np.shape(D)[0], len(degrees))
                ),
                [],
                "D",
                "dimensionful",
                nn_interaction="np",
            )
            AXXBunch = ObservableBunch(
                "AXX",
                np.reshape(
                    AXX[:, np.isin(t_lab, Elab_slice), :],
                    (np.shape(AXX)[0], len(degrees)),
                ),
                [],
                "A_{xx}",
                "dimensionless",
                nn_interaction="np",
            )
            AYYBunch = ObservableBunch(
                "AYY",
                np.reshape(
                    AYY[:, np.isin(t_lab, Elab_slice), :],
                    (np.shape(AYY)[0], len(degrees)),
                ),
                [],
                "A_{yy}",
                "dimensionless",
                nn_interaction="np",
            )
        # sliced in angle; not used for anything.
        elif (Elab_slice is None) and (deg_slice is not None):
            DSGBunch = ObservableBunch(
                "DSG",
                np.reshape(
                    DSG[:, :, np.isin(degrees, deg_slice)],
                    (np.shape(DSG)[0], len(t_lab)),
                ),
                [],
                "d \sigma / d \Omega",
                "dimensionful",
                nn_interaction="np",
            )
            AYBunch = ObservableBunch(
                "AY",
                # AY,
                np.reshape(
                    AY[:, :, np.isin(degrees, deg_slice)], (np.shape(AY)[0], len(t_lab))
                ),
                # np.reshape(AY[:, np.isin(t_lab, E_slice_Q), :], (np.shape(AY)[0], len(degrees))),
                [],
                "A_{y}",
                "dimensionless",
                nn_interaction="np",
                constraint=[[0, -1], [0, 0], "angle"],
            )
            ABunch = ObservableBunch(
                "A",
                np.reshape(
                    A[:, :, np.isin(degrees, deg_slice)], (np.shape(A)[0], len(t_lab))
                ),
                [],
                "A",
                "dimensionless",
                nn_interaction="np",
                constraint=[[0], [0], "angle"],
            )
            DBunch = ObservableBunch(
                "D",
                np.reshape(
                    D[:, :, np.isin(degrees, deg_slice)], (np.shape(D)[0], len(t_lab))
                ),
                [],
                "D",
                "dimensionless",
                nn_interaction="np",
            )
            # version of DBunch that treats D as dimensionful for the purposes of Fig. 22
            DBunch_dimensionful = ObservableBunch(
                "D_dimensionful",
                np.reshape(
                    D[:, :, np.isin(degrees, deg_slice)], (np.shape(D)[0], len(t_lab))
                ),
                [],
                "D",
                "dimensionful",
                nn_interaction="np",
            )
            AXXBunch = ObservableBunch(
                "AXX",
                np.reshape(
                    AXX[:, :, np.isin(degrees, deg_slice)],
                    (np.shape(AXX)[0], len(t_lab)),
                ),
                [],
                "A_{xx}",
                "dimensionless",
                nn_interaction="np",
            )
            AYYBunch = ObservableBunch(
                "AYY",
                np.reshape(
                    AYY[:, :, np.isin(degrees, deg_slice)],
                    (np.shape(AYY)[0], len(t_lab)),
                ),
                [],
                "A_{yy}",
                "dimensionless",
                nn_interaction="np",
            )

        observable_array = [
            SGTBunch,
            DSGBunch,
            AYBunch,
            ABunch,
            DBunch,
            DBunch_dimensionful,
            AXXBunch,
            AYYBunch,
        ]
        # runs through the parametrizations of p in Q(p)
        for p, PParamMethod in enumerate(p_param_method_array):
            # creates the bunches for the vs-angle input spaces
            DegBunch = InputSpaceBunch(
                "deg",
                deg_fn,
                p_approx(
                    PParamMethod,
                    E_to_p(t_lab, interaction="np"),
                    degrees,
                ),
                r"$\theta$ (deg)",
                [
                    r"$",
                    None,
                    r"(\theta, E_{\mathrm{lab}}= ",
                    None,
                    "\,\mathrm{MeV})$",
                ],
            )
            CosBunch = InputSpaceBunch(
                "cos",
                neg_cos,
                p_approx(
                    PParamMethod,
                    E_to_p(t_lab, interaction="np"),
                    degrees,
                ),
                r"$-\mathrm{cos}(\theta)$",
                [
                    r"$",
                    None,
                    r"(-\mathrm{cos}(\theta), E_{\mathrm{lab}}= ",
                    None,
                    "\,\mathrm{MeV})$",
                ],
            )
            SinBunch = InputSpaceBunch(
                "sin",
                sin_thing,
                p_approx(
                    PParamMethod,
                    E_to_p(t_lab, interaction="np"),
                    degrees,
                ),
                r"$\mathrm{sin}(\theta)$",
                [
                    r"$",
                    None,
                    r"(\mathrm{sin}(\theta), E_{\mathrm{lab}}= ",
                    None,
                    "\,\mathrm{MeV})$",
                ],
            )
            QcmBunch = InputSpaceBunch(
                "qcm",
                deg_to_qcm,
                p_approx(
                    PParamMethod,
                    E_to_p(t_lab, interaction="np"),
                    degrees,
                ),
                r"$q_{\mathrm{cm}}$ (MeV)",
                [
                    r"$",
                    None,
                    r"(q_{\mathrm{cm}}, E_{\mathrm{lab}}= ",
                    None,
                    "\,\mathrm{MeV})$",
                ],
            )
            Qcm2Bunch = InputSpaceBunch(
                "qcm2",
                deg_to_qcm2,
                p_approx(
                    PParamMethod,
                    E_to_p(t_lab, interaction="np"),
                    degrees,
                ),
                r"$q_{\mathrm{cm}}^{2}$ (MeV$^{2}$)",
                [
                    r"$",
                    None,
                    r"(q_{\mathrm{cm}}^{2}, E_{\mathrm{lab}}= ",
                    None,
                    "\,\mathrm{MeV})$",
                ],
            )

            vsquantity_array_deg = [
                DegBunch,
                CosBunch,
                QcmBunch,
                Qcm2Bunch,
                SinBunch,
            ]

            ElabBunch = InputSpaceBunch(
                "Elab",
                Elab_fn,
                p_approx(
                    "Qofprel",
                    E_to_p(t_lab, interaction="np"),
                    degrees,
                ),
                r"$E_{\mathrm{lab}}$ (MeV)",
                [r"$", None, r"(E_{\mathrm{lab}})$"],
            )

            PrelBunch = InputSpaceBunch(
                "prel",
                E_to_p,
                p_approx(
                    "Qofprel",
                    E_to_p(t_lab, interaction="np"),
                    degrees,
                ),
                r"$p_{\mathrm{rel}}$ (MeV)",
                [r"$", None, r"(p_{\mathrm{rel}})$"],
            )

            vsquantity_array_tlab = [ElabBunch, PrelBunch]

            # creates each input space bunch's title
            for bunch in vsquantity_array_deg:
                bunch.make_title()
            for bunch in vsquantity_array_tlab:
                bunch.make_title()

            vsquantity_posterior_array_deg = [
                b for b in vsquantity_array_deg if b.name in input_space_deg
            ]
            vsquantity_posterior_array_tlab = [
                b for b in vsquantity_array_tlab if b.name in input_space_tlab
            ]

            # runs through the parametrization methods
            for k, QParamMethod in enumerate(Q_param_method_array):
                # runs through the angle-based input spaces
                for j, VsQuantityPosteriorTlab in enumerate(
                    vsquantity_posterior_array_tlab
                ):
                    # runs through the angle-based input spaces
                    for i, VsQuantityPosteriorDeg in enumerate(
                        vsquantity_posterior_array_deg
                    ):
                        center = 0
                        df = 1
                        disp = 0
                        std_scale = 1

                        # information for naming the savefiles
                        FileName = FileNaming(
                            QParamMethod,
                            PParamMethod,
                            scheme=ScaleScheme.potential_string,
                            scale=ScaleScheme.cutoff_string,
                            vs_what=VsQuantityPosteriorDeg.name
                            + "x"
                            + VsQuantityPosteriorTlab.name,
                            filename_addendum=filename_addendum,
                        )
                        #
                        # information on the orders for each potential
                        Orders = OrderInfo(
                            ScaleScheme.orders_full,
                            [0] + orders_excluded,
                            ScaleScheme.colors,
                            ScaleScheme.light_colors,
                            orders_names_dict=orders_names_dict,
                            orders_labels_dict=orders_labels_dict,
                        )

                        if (
                            plot_posterior_curvewise_bool
                            or plot_posterior_pointwise_bool
                        ):
                            ratio_sb_2d = Q_approx(
                                p_approx(
                                    PParamMethod,
                                    E_to_p(t_lab, interaction="np"),
                                    degrees,
                                ),
                                QParamMethod,
                                Lambda_b=Lambdab,
                                m_pi=m_pi_eff,
                            )
                            ratio_sb_1d = Q_approx(
                                p_approx(
                                    PParamMethod,
                                    E_to_p(t_lab, interaction="np"),
                                    np.array([1]),
                                ),
                                QParamMethod,
                                Lambda_b=Lambdab,
                                m_pi=m_pi_eff,
                            )

                            obs_dict = {
                                "SGT": SGTBunch,
                                "DSG": DSGBunch,
                                "D": DBunch,
                                "AXX": AXXBunch,
                                "AYY": AYYBunch,
                                "A": ABunch,
                                "AY": AYBunch,
                                "Ddimensional": DBunch_dimensionful,
                            }

                            obs_grouped_list = [
                                [obs_dict[obs_name] for obs_name in obs_sublist]
                                for obs_sublist in plot_obs_list
                            ]

                            if plot_posterior_curvewise_bool:
                                stats_array_curvewise = np.append(
                                    stats_array_curvewise,
                                    plot_posteriors_curvewise(
                                        # order stuff
                                        light_colors=Orders.lightcolors_array,
                                        nn_orders_array=Orders.orders_restricted,
                                        nn_orders_full_array=Orders.orders_full,
                                        excluded=Orders.excluded,
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
                                        # strings
                                        nn_interaction="np",
                                        # hyperparameters
                                        center=center,
                                        disp=disp,
                                        df=df,
                                        std_est=std_scale,
                                        # filename stuff
                                        obs_data_grouped_list=obs_grouped_list,
                                        obs_name_grouped_list=obs_name_grouped_list,
                                        obs_labels_grouped_list=obs_labels_grouped_list,
                                        mesh_cart_grouped_list=mesh_cart_grouped_list,
                                        t_lab=t_lab,
                                        t_lab_train_pts=t_lab_train_pts,
                                        InputSpaceTlab=VsQuantityPosteriorTlab,
                                        degrees=degrees,
                                        degrees_train_pts=degrees_train_pts,
                                        InputSpaceDeg=VsQuantityPosteriorDeg,
                                        length_scale_list=length_scale_list,
                                        cbar_list=cbar_list,
                                        variables_array=variables_array_curvewise,
                                        mom_fn=E_to_p,
                                        mom_fn_kwargs={"interaction": "np"},
                                        warping_fn=warping_fn,
                                        warping_fn_kwargs=warping_fn_kwargs,
                                        cbar_fn=cbar_fn,
                                        cbar_fn_kwargs=cbar_fn_kwargs,
                                        scaling_fn=scaling_fn,
                                        scaling_fn_kwargs=scaling_fn_kwargs,
                                        ratio_fn=ratio_fn_posterior,
                                        ratio_fn_kwargs=ratio_fn_kwargs_posterior,
                                        log_likelihood_fn=log_likelihood_fn_posterior,
                                        log_likelihood_fn_kwargs=log_likelihood_fn_kwargs_posterior,
                                        orders=orders_from_ho,
                                        FileName=FileName,
                                        whether_plot_posteriors=plot_posterior_curvewise_bool,
                                        whether_plot_corner=plot_corner_curvewise_bool,
                                        whether_use_data=use_data_curvewise_bool,
                                        whether_save_data=save_data_curvewise_bool,
                                        whether_save_plots=save_posterior_curvewise_bool,
                                    ),
                                )
                            if plot_posterior_pointwise_bool:
                                stats_array_pointwise = np.append(
                                    stats_array_pointwise,
                                    plot_posteriors_pointwise(
                                        # order stuff
                                        light_colors=Orders.lightcolors_array,
                                        nn_orders_array=Orders.orders_restricted,
                                        nn_orders_full_array=Orders.orders_full,
                                        excluded=Orders.excluded,
                                        orders_labels_dict={
                                            6: r"N$^{4}$LO$^{+}$",
                                            5: r"N$^{4}$LO",
                                            4: r"N$^{3}$LO",
                                            3: r"N$^{2}$LO",
                                            2: r"NLO",
                                        },
                                        # strings
                                        # filename stuff
                                        obs_data_grouped_list=obs_grouped_list,
                                        obs_name_grouped_list=obs_name_grouped_list,
                                        obs_labels_grouped_list=obs_labels_grouped_list,
                                        t_lab=t_lab,
                                        t_lab_train_pts=t_lab_train_pts,
                                        InputSpaceTlab=VsQuantityPosteriorTlab,
                                        degrees=degrees,
                                        degrees_train_pts=degrees_train_pts,
                                        InputSpaceDeg=VsQuantityPosteriorDeg,
                                        variables_array=variables_array_pointwise,
                                        mom_fn_tlab=E_to_p,
                                        mom_fn_tlab_kwargs={"interaction": "np"},
                                        mom_fn_degrees=mom_fn_degrees,
                                        mom_fn_degrees_kwargs={},
                                        p_fn=p_approx,
                                        p_fn_kwargs={
                                            "p_name": PParamMethod,
                                        },
                                        ratio_fn=Q_approx,
                                        ratio_fn_kwargs={
                                            "Q_parametrization": QParamMethod,
                                            "m_pi": m_pi_eff,
                                        },
                                        orders=orders_from_ho,
                                        FileName=FileName,
                                        whether_save_plots=save_posterior_pointwise_bool,
                                    ),
                                )

    # except:
    #     print("Error encountered in running loop.")

    # prints all instances of the classes relevant for the arguments of
    # GPAnalysis()
    if print_all_classes:
        scalescheme_current_list = []
        observable_current_list = []
        inputspace_current_list = []
        traintest_current_list = []
        lengthscale_current_list = []

        for obj in gc.get_objects():
            if isinstance(obj, ScaleSchemeBunch):
                scalescheme_current_list.append(obj.name)
            elif isinstance(obj, ObservableBunch):
                observable_current_list.append(obj.name)
            elif isinstance(obj, InputSpaceBunch):
                inputspace_current_list.append(obj.name)
            elif isinstance(obj, TrainTestSplit):
                traintest_current_list.append(obj.name)
            elif isinstance(obj, LengthScale):
                lengthscale_current_list.append(obj.name)

        print("\n\n************************************")
        print("Available potentials: " + str(scalescheme_current_list))
        print("Available observables: " + str(observable_current_list))
        print("Available Q parametrizations: ['smax', 'max', 'sum', 'rawsum']")
        print("Available input spaces: " + str(inputspace_current_list))
        print("Available train/test splits: " + str(traintest_current_list))
        print("Available length scales: " + str(lengthscale_current_list))
        print("************************************")

    return stats_array_curvewise, stats_array_pointwise
