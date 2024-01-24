import os.path

import numpy as np
from scipy.interpolate import interp1d, interpn
from .utils import versatile_train_test_split, compute_posterior_intervals, sig_figs, correlation_coefficient, \
    round_to_same_digits, mean_and_stddev
from .scattering import E_to_p
from .graphs import draw_summary_statistics, corner_plot, offset_xlabel, joint_plot, setup_rc_params, plot_marg_posteriors, \
    plot_corner_posteriors, softblack, gray, edgewidth, text_bbox
from .eft import Q_approx, Lb_logprior, mpieff_logprior, p_approx
import h5py
import ray
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import gsum as gm
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import itertools
import time

# softblack = 'k'  # Looks better when printed on tex file
# gray = '0.7'
# edgewidth = 0.6
#
# mpl.rcParams['figure.dpi'] = 180
# mpl.rcParams['font.size'] = 9
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['font.family'] = 'serif'
#
# mpl.rcParams['axes.labelsize'] = 14 # 9
# mpl.rcParams['axes.edgecolor'] = softblack
# mpl.rcParams['axes.xmargin'] = 0
# mpl.rcParams['axes.labelcolor'] = softblack
# mpl.rcParams['axes.linewidth']
#
# mpl.rcParams['ytick.direction'] = 'in'
# mpl.rcParams['xtick.direction'] = 'in'
# mpl.rcParams['xtick.labelsize'] = 11 # 9
# mpl.rcParams['ytick.labelsize'] = 11 # 9
# mpl.rcParams['xtick.color'] = softblack
# mpl.rcParams['ytick.color'] = softblack
# mpl.rcParams['xtick.minor.size'] = 2.4
# mpl.rcParams['ytick.minor.size'] = 2.4
#
# mpl.rcParams['legend.title_fontsize'] = 9
# mpl.rcParams['legend.fontsize'] = 14 # 9
# mpl.rcParams['legend.edgecolor'] = 'inherit'  # inherits from axes.edgecolor, to match
# mpl.rcParams['legend.facecolor'] = (1, 1, 1, 0.6)  # Set facecolor with its own alpha, so edgecolor is unaffected
# mpl.rcParams['legend.fancybox'] = True
# mpl.rcParams['legend.borderaxespad'] = 0.8
# mpl.rcParams['legend.framealpha'] = None  # Do not set overall alpha (affects edgecolor). Handled by facecolor above
# mpl.rcParams['patch.linewidth'] = 0.8  # This is for legend edgewidth, since it does not have its own option
#
# mpl.rcParams['lines.markersize'] = 10
#
# text_bbox = dict(boxstyle='round', fc=(1, 1, 1, 0.6), ec=softblack, lw=0.8)
# mpl.rc('savefig', transparent=False, bbox='tight', pad_inches=0.05, dpi=300, format='pdf')
setup_rc_params()

class GPHyperparameters:
    def __init__(self, ls_class, center, ratio, nugget=1e-10, seed=None, df=np.inf,
                 disp=0, scale=1, sd=None):
        """
        Information necessary for Gaussian process hyperparameters.

        Parameters
        ----------
        ls_class (LengthScale) : LengthScale object with relevant information.
        center (float) : initial guess for the mean of the distribution.
        ratio (array) : array of values for the dimensionless expansion parameter
        nugget (float) : small number used for allowing white-noise error into fitting procedures.
            default : 1e-10
        seed (int) : seed for RNGs in fitting procedure.
            default : None
        df (int) : number of degrees of freedom.
            default : np.inf
        disp (int) : initial guess for the standard deviation.
            default : 0
        scale (float) : estimate for the standard deviation.
            default : 1
        sd (float) : fixed standard deviation for fitting procedure.
            default : None (i.e., it will calculate one afresh using the fitting algorithm)
        """
        self.ls = ls_class.ls_guess
        self.ls_lower = ls_class.ls_bound_lower
        self.ls_upper = ls_class.ls_bound_upper
        self.whether_fit = ls_class.whether_fit
        self.center = center
        self.ratio = ratio
        self.nugget = nugget
        self.seed = seed
        self.df = df
        self.disp = disp
        self.scale = scale
        self.sd = sd


class FileNaming:
    def __init__(self, scheme, scale, Q_param, p_param, input_space, filename_addendum=""):
        """
        Information necessary to name files for output figures.

        Parameters
        ----------
        scheme (str) : name of the scheme
        scale (str) : name of the scale
        Q_param (str) : name of the Q parametrization
        p_param (str) : name of the p parametrization
        input_space (str) : name of the input space (x-variable)
        filename_addendum (str) : optional extra string
            default : ""
        """
        self.scheme = scheme
        self.scale = scale
        self.Q_param = Q_param
        self.p_param = p_param
        self.vs_what = input_space
        self.filename_addendum = filename_addendum


class PosteriorBounds:
    def __init__(self, x_lower, x_upper, x_n, y_lower, y_upper, y_n):
        """
        Class for the boundaries of the 2D posterior PDF plot and the mesh on which it is plotted.
        """
        self.x_vals = np.linspace(x_lower, x_upper, x_n)
        self.y_vals = np.linspace(y_lower, y_upper, y_n)


class RandomVariable:
    def __init__(self, var, user_val, name, label, units, ticks, logprior, logprior_name, marg_bool = True):
        """
        Instantiates the information in a class necessary for a random variable for Bayesian parameter estimation.

        Parameters
        ----------
        var (array) : 1-d array of the variable's values.
        user_val (float) : user-set value for the variable.
        name (str) : name for the random variable.
        label (str) : markdown-formatted label for plots.
        units (str) : abbreviation for the variable's units.
        ticks (array) : location of ticks for 1-d plots of the variable.
        logprior (array) : 1-d array of the log-prior for the variable.
        logprior_name (str) : name for the log-prior
        marg_bool (str) : boolean for whether the variable will be plotted in corner plot (False to marginalize)
            default : True
        """
        self.var = var
        self.user_val = user_val
        self.name = name
        self.label = label
        self.units = units
        self.ticks = ticks
        self.logprior = logprior
        self.logprior_name = logprior_name
        self.marg_bool = marg_bool

class OrderInfo:
    def __init__(self, orders_array, excluded, colors_array, lightcolors_array,
                 orders_names_dict=None, orders_labels_dict=None):
        """
        Class for the number of orders under consideration and the color for each.

        Parameters
        ----------
        orders_array (array): list of orders at which the potential CAN BE evaluated
        excluded (array): list of orders to be excluded from analysis
        colors_array (array): list of colors corresponding to each order
        lightcolors_array (array): list of lighter versions of colors_array
        orders_names_dict (dict): dictionary method linking the numerical indices (int)
            of EFT orders and their corresponding abbreviations (str)
            default : None
        orders_names_dict (dict): dictionary method linking the numerical indices (int)
            of EFT orders and their corresponding math-mode-formatted labels (str)
            default: None
        """
        self.orders_full = np.array(orders_array)
        self.excluded = np.array(excluded)
        self.mask_restricted = ~np.isin(self.orders_full, self.excluded)
        self.orders_restricted = self.orders_full[self.mask_restricted]
        self.mask_eval = self.mask_restricted[1:]
        self.colors_array = list(np.array(colors_array)[self.mask_eval])
        self.lightcolors_array = list(np.array(lightcolors_array)[self.mask_eval])
        self.orders_names_dict = orders_names_dict
        self.orders_labels_dict = orders_labels_dict


class InputSpaceBunch:
    def __init__(self, name, input_space, mom, caption, title_pieces):
        """
        Class for an input space (i.e., x-coordinate)

        name (string) : (abbreviated) name for the input space
        input_space (array) : x-coordinate mesh points for evaluation
        mom (array) : momenta for the purpose of calculating the ratio
        caption (string) : caption for the x-axis of the coefficient plots for that input space
        title_pieces (array) : information to be concatenated into the coefficient plot's title
        """
        self.name = name
        self.input_space = input_space
        self.mom = mom
        self.caption = caption
        self.title_pieces = title_pieces

    def make_title(self):
        """
        Concatenates the entries of title_pieces into a plot title
        """
        self.title = ''
        for piece in self.title_pieces: self.title += str(piece)
        return self.title


class ObservableBunch:
    def __init__(self, name, data, energies, angles, title, ref_type, constraint=None):
        """
        Class for an observable
        name (string) : (abbreviated) name for the observable
        data (array) : coefficient values at each order over the mesh
        energies (array) : energies at which the observable will be evaluated (should be None for observables
            plotted against angle)
        angles (array) : angles at which the observable will be evaluated (should be None for observables
            plotted against energy)
        title (string) : title for the coefficient plot
        ref_type (string) : tells whether the reference scale (to be divided out of the coefficient
            values) has dimension (e.g., the case of the cross section) or not (e.g., the case of the
        spin observables). Can only be "dimensionless" or "dimensionful".
        constraint (array or None): constraint on the values of the observable, including the
            name of the quantity for which the constraint applies.
            For dimensionful (i.e., cross-section) observables, should be None.
        """
        self.name = name
        self.data = data
        self.energies = energies
        self.angles = angles
        self.title = title
        self.ref_type = ref_type
        self.constraint = constraint
        if (ref_type != "dimensionless") and (ref_type != "dimensionful"):
            raise Exception("ref_type must be dimensionless or dimensionful.")


class Interpolation:
    def __init__(self, x, y, kind='cubic'):
        """
        Class for an interpolater
        x (array) : x-coordinate data
        y (array) : y-coordinate data
        kind (string) : scipy.interpolate.interp1d interpolater 'kind'
        """
        self.x = x
        self.y = y
        self.kind = kind
        self.f_interp = interp1d(self.x, self.y, kind=self.kind)


class TrainTestSplit:
    def __init__(self, name, n_train, n_test_inter, isclose_factor=0.01,
                 offset_train_min_factor=0, offset_train_max_factor=0,
                 xmin_train_factor=0, xmax_train_factor=1,
                 offset_test_min_factor=0, offset_test_max_factor=0,
                 xmin_test_factor=0, xmax_test_factor=1,
                 train_at_ends=True, test_at_ends=False):
        """
        Class for an input space (i.e., x-coordinate)

        name (str) : (abbreviated) name for the combination of training and testing masks
        n_train (int) : number of intervals into which to split x, with training points at the
            edges of each interval
        n_test_inter (int) : number of subintervals into which to split the intervals between
            training points, with testing points at the edges of each subinterval
        isclose_factor (float) : fraction of the total input space for the tolerance of making
            sure that training and testing points don't coincide
        offset_train_min_factor (float) : fraction above the minimum of the input space where
            the first potential training point ought to go
        offset_train_max_factor (float) : fraction below the maximum of the input space where
            the last potential training point ought to go
        xmin_train_factor (float) : fraction of the input space below which there ought not to
            be training points
        xmax_train_factor (float) : fraction of the input space above which there ought not to
            be training points
        offset_test_min_factor (float) : fraction above the minimum of the input space where
            the first potential testing point ought to go
        offset_test_max_factor (float) : fraction below the maximum of the input space where
            the last potential testing point ought to go
        xmin_test_factor (float) : fraction of the input space below which there ought not to
            be testing points
        xmax_test_factor (float) : fraction of the input space above which there ought not to
            be testing points
        train_at_ends (bool) : whether training points should be allowed at or near the
            endpoints of x
        test_at_ends (bool) : whether testing points should be allowed at or near the endpoints
            of x
        """
        self.name = name
        self.n_train = n_train
        self.n_test_inter = n_test_inter
        self.isclose_factor = isclose_factor
        self.offset_train_min_factor = offset_train_min_factor
        self.offset_train_max_factor = offset_train_max_factor
        self.xmin_train_factor = xmin_train_factor
        self.xmax_train_factor = xmax_train_factor
        self.offset_test_min_factor = offset_test_min_factor
        self.offset_test_max_factor = offset_test_max_factor
        self.xmin_test_factor = xmin_test_factor
        self.xmax_test_factor = xmax_test_factor
        self.train_at_ends = train_at_ends
        self.test_at_ends = test_at_ends

    def make_masks(self, x, y):
        """Returns the training and testing points in the input space and the corresponding
        (interpolated) data values after calculating the actual values for xmin, xmax, and
        offsets using the corresponding factors and the input space

        Parameters
        ----------
        x (1D array) : input space
        y (ND array) : data points at each input space value, with N>1 dimensions for N
            orders
        """
        self.x = x
        self.y = y

        # calculates the actual value for each offset, xmin, and xmax
        self.offset_train_min = self.offset_train_min_factor \
                                * (np.max(self.x) - np.min(self.x))
        self.offset_train_max = self.offset_train_max_factor \
                                * (np.max(self.x) - np.min(self.x))
        self.xmin_train = np.min(self.x) + self.xmin_train_factor * \
                          (np.max(self.x) - np.min(self.x))
        self.xmax_train = np.min(self.x) + self.xmax_train_factor * \
                          (np.max(self.x) - np.min(self.x))
        self.offset_test_min = self.offset_test_min_factor \
                               * (np.max(self.x) - np.min(self.x))
        self.offset_test_max = self.offset_test_max_factor \
                               * (np.max(self.x) - np.min(self.x))
        self.xmin_test = np.min(self.x) + self.xmin_test_factor * \
                         (np.max(self.x) - np.min(self.x))
        self.xmax_test = np.min(self.x) + self.xmax_test_factor * \
                         (np.max(self.x) - np.min(self.x))

        self.interp_obj = Interpolation(self.x, self.y, kind='cubic')

        # creates the x and y training and testing points
        self.x_train, self.x_test, self.y_train, self.y_test = \
            versatile_train_test_split(self.interp_obj,
                                       self.n_train, n_test_inter=self.n_test_inter,
                                       isclose_factor=self.isclose_factor,
                                       offset_train_min=self.offset_train_min,
                                       offset_train_max=self.offset_train_max,
                                       xmin_train=self.xmin_train, xmax_train=self.xmax_train,
                                       offset_test_min=self.offset_test_min,
                                       offset_test_max=self.offset_test_max,
                                       xmin_test=self.xmin_test, xmax_test=self.xmax_test,
                                       train_at_ends=self.train_at_ends, test_at_ends=self.test_at_ends)

        return self.x_train, self.x_test, self.y_train, self.y_test


class ScaleSchemeBunch:
    # os.path.join(os.path.abspath(__file__), os.pardir)
    def __init__(self, file_name, orders_full, cmaps, potential_string, cutoff_string,
                 dir_path=""):
        """
        Information relevant to a particular scheme (regulator choice) and scale (cutoff choice).

        Parameters
        ----------
        file_name (str) : name for files that includes information about the scale and scheme.
        orders_full (int array) : array with the full range of orders for that scheme and scale.
        cmaps (cmap) : array of matplotlib cmap objects corresponding to each order of coefficient.
        potential_string (str) : name of potential (scheme).
        cutoff_string (str) : name of cutoff (scale).
        dir_path (str) : path to directory where data is stored on each scale/scheme combination.
            default : ""
        """
        self.file_name = file_name
        self.orders_full = orders_full
        self.cmaps = cmaps
        self.potential_string = potential_string
        self.cutoff_string = cutoff_string
        self.name = self.potential_string + self.cutoff_string
        self.dir_path = dir_path

        self.full_path = self.dir_path + self.file_name

        self.colors = [cmap(0.55 - 0.1 * (i == 0)) for i, cmap in enumerate(self.cmaps)]
        self.light_colors = [cmap(0.35) for cmap in self.cmaps]

    def get_data(self, observable_string):
        """
        Parameters
        ----------
        observable_string : abbreviation for the observable of interest.

        Returns
        -------
        obs_data (array) : array of observable data.
        """
        try:
            response = h5py.File(self.full_path, "r")
            obs_data = np.array(response[observable_string][:])
            response.close()
            return obs_data
        except:
            raise Exception("Data could not be found at " + self.full_path + ".")


class LengthScale:
    def __init__(self, name, ls_guess_factor, ls_bound_lower_factor,
                 ls_bound_upper_factor, whether_fit=True):
        """
        Class for setting a guess for the Gaussian process correlation length scale and its
        bounds.

        Parameters
        ----------
        name (str) : name for the instance.
        ls_guess_factor (float) : ls_guess_factor * total length of the input space = initial guess for length scale.
        ls_bound_lower_factor (float) : ls_bound_lower_factor * total length of the input space = lower bound for length scale fitting.
        ls_bound_upper_factor (float) : ls_bound_upper_factor * total length of the input space = upper bound for length scale fitting.
        whether_fit (bool) : should the length scale be fitted or kept constant?
            default : True
        """
        self.name = name
        self.ls_guess_factor = ls_guess_factor
        self.ls_bound_lower_factor = ls_bound_lower_factor
        self.ls_bound_upper_factor = ls_bound_upper_factor
        self.whether_fit = whether_fit

    def make_guess(self, x):
        """
        Parameters
        ----------
        x (array) : input space.
        """
        self.ls_guess = (np.max(x) - np.min(x)) * self.ls_guess_factor

        if self.whether_fit:
            self.ls_bound_lower = self.ls_bound_lower_factor * self.ls_guess
            self.ls_bound_upper = self.ls_bound_upper_factor * self.ls_guess
        else:
            self.ls_bound_lower = self.ls_guess.copy()
            self.ls_bound_upper = self.ls_guess.copy()


class GSUMDiagnostics:
    def __init__(self, nn_interaction, observable, Lambda_b, inputspace, traintestsplit,
                 gphyperparameters, orderinfo, filenaming,
                 fixed_quantity=[None, None, None, None],
                 x_quantity=[None, None, None], posteriorgrid=None):
        """
        Class for everything involving Jordan Melendez's GSUM library for observables that
        can be plotted against angle.

        Parameters
        ----------
        nn_interaction (str) : two-letter string for two nucleons interacting in observables.
        observable (ObservableBunch) : observable being plotted.
        Lambda_b (float) : breakdown scale (in MeV).
        inputspace (InputSpaceBunch) : input space against which the observable is plotted.
        traintestsplit (TrainTestSplit) : training and testing masks.
        gphyperparameters (GPHyperparameters) : parameters for fitted Gaussian process.
        orderinfo (OrderInfo) : information about the EFT orders and their colors.
        filenaming (FileNaming) : strings for naming the save files.
        fixed_quantity (list) : [fixed_quantity name (str), fixed_quantity value (float), fixed_quantity array (array), fixed_quantity units (str)]
        x_quantity (list) : [x_quantity name (str), x_quantity array (array), x_quantity units (str)]
        posteriorgrid (PosteriorBounds) : xy-grid over which to plot the Lambda-ell posterior pdf.
        """
        self.nn_interaction = nn_interaction

        # information on the observable
        self.observable = observable
        self.observable_name = self.observable.name
        self.data = self.observable.data
        self.ref_type = self.observable.ref_type
        self.constraint = self.observable.constraint

        # cutoff scale
        self.Lambda_b = Lambda_b

        # energy or angle at which the observable is evaluated, along with all
        # possible energies or angles for evaluation
        self.fixed_quantity_name = fixed_quantity[0]
        self.fixed_quantity_value = fixed_quantity[1]
        self.fixed_quantity_array = fixed_quantity[2]
        self.fixed_quantity_units = fixed_quantity[3]

        # angle or energy mesh
        self.x_quantity_name = x_quantity[0]
        self.x_quantity_array = x_quantity[1]
        self.x_quantity_units = x_quantity[2]

        # information on the input space
        self.inputspace = inputspace
        self.vs_what = self.inputspace.name
        self.x = self.inputspace.input_space(**{"deg_input": self.x_quantity_array,
                                                "p_input": E_to_p(self.fixed_quantity_value,
                                                                  interaction=self.nn_interaction),
                                                "E_lab": self.x_quantity_array,
                                                "interaction": self.nn_interaction})
        self.X = self.x[:, None]
        self.caption_coeffs = self.inputspace.caption
        self.title_coeffs = self.inputspace.title

        # information on the train/test split
        self.traintestsplit = traintestsplit
        self.train_pts_loc = self.traintestsplit.name
        self.x_train = self.traintestsplit.x_train
        self.n_train_pts = len(self.x_train)
        self.x_test = self.traintestsplit.x_test
        self.n_test_pts = len(self.x_test)
        self.y_train = self.traintestsplit.y_train
        self.y_test = self.traintestsplit.y_test

        # information on the GP hyperparameters
        self.gphyperparameters = gphyperparameters
        self.ls = self.gphyperparameters.ls
        self.ls_lower = self.gphyperparameters.ls_lower
        self.ls_upper = self.gphyperparameters.ls_upper
        self.whether_fit = self.gphyperparameters.whether_fit
        self.center = self.gphyperparameters.center
        self.ratio = self.gphyperparameters.ratio
        self.nugget = self.gphyperparameters.nugget
        self.seed = self.gphyperparameters.seed
        self.df = self.gphyperparameters.df
        self.disp = self.gphyperparameters.disp
        self.std_est = self.gphyperparameters.scale
        self.sd = self.gphyperparameters.sd

        # information on the orders at which the potential is evaluated

        self.orderinfo = orderinfo
        self.nn_orders_full = self.orderinfo.orders_full
        self.excluded = self.orderinfo.excluded
        self.colors = self.orderinfo.colors_array
        self.light_colors = self.orderinfo.lightcolors_array
        self.mask_restricted = self.orderinfo.mask_restricted
        self.orders_restricted = self.orderinfo.orders_restricted
        self.mask_eval = self.orderinfo.mask_eval

        if self.orderinfo.orders_names_dict is None:
            self.orders_names_dict = {
                6: "N4LO+",
                5: "N4LO",
                4: "N3LO",
                3: "N2LO",
                2: "NLO",
            }
        else:
            self.orders_names_dict = self.orderinfo.orders_names_dict
        if self.orderinfo.orders_labels_dict is None:
            self.orders_labels_dict = {6: r'N$^{4}$LO$^{+}$', 5: r'N$^{4}$LO',
                                       4: r'N$^{3}$LO', 3: r'N$^{2}$LO',
                                       2: r'NLO'}
        else:
            self.orders_labels_dict = self.orderinfo.orders_labels_dict

        # information for naming the file
        self.filenaming = filenaming
        self.scheme = self.filenaming.scheme
        self.scale = self.filenaming.scale
        self.Q_param = self.filenaming.Q_param
        self.p_param = self.filenaming.p_param
        self.filename_addendum = self.filenaming.filename_addendum

        # posterior pdf bounds
        self.posteriorgrid = posteriorgrid

        # for plotting observables at a fixed energy
        if self.fixed_quantity_name == "energy":
            self.fixed_idx = np.nonzero(self.fixed_quantity_array == self.fixed_quantity_value)[0][0]

            self.data = self.data[:, self.fixed_idx, :].T

            self.X_train = self.x_train[:, None]
            self.y_train = self.y_train[:, self.fixed_idx, :].T
            self.X_test = self.x_test[:, None]
            self.y_test = self.y_test[:, self.fixed_idx, :].T

            # determines the reference scale for the truncation-error model, including for
            # training and testing
            if self.ref_type == "dimensionless":
                self.ref = np.ones(len(self.x)) * 1
                self.ref_train = np.ones(len(self.x_train)) * 1
                self.ref_test = np.ones(len(self.x_test)) * 1

            elif self.ref_type == "dimensionful":
                self.ref = self.data[:, -1]

                self.interp_f_ref = interp1d(self.x, self.ref)
                self.ref_train = self.interp_f_ref(self.x_train)
                self.ref_test = self.interp_f_ref(self.x_test)

        # for plotting observables at a fixed angle
        elif self.fixed_quantity_name == "angle":
            if self.fixed_quantity_value == 0:
                self.X_train = self.x_train[:, None]
                self.y_train = self.y_train.T
                self.X_test = self.x_test[:, None]
                self.y_test = self.y_test.T
            else:
                self.fixed_idx = np.nonzero(self.fixed_quantity_array == self.fixed_quantity_value)[0][0]

                self.data = self.data[:, :, self.fixed_idx].T

                self.X_train = self.x_train[:, None]
                self.y_train = self.y_train[:, self.fixed_idx, :].T
                self.X_test = self.x_test[:, None]
                self.y_test = self.y_test[:, self.fixed_idx, :].T

            # determines the reference scale for the truncation-error model, including for
            # training and testing
            if self.ref_type == "dimensionless":
                self.ref = np.ones(len(self.x)) * 1
                self.ref_train = np.ones(len(self.x_train)) * 1
                self.ref_test = np.ones(len(self.x_test)) * 1
            elif self.ref_type == "dimensionful":
                if self.fixed_quantity_value == 0:
                    self.ref = self.data[-1]
                    self.data = self.data.T
                else:
                    self.ref = self.data[:, -1]

                self.interp_f_ref = interp1d(self.x, self.ref)
                self.ref_train = self.interp_f_ref(self.x_train)
                self.ref_test = self.interp_f_ref(self.x_test)

        # uses interpolation to find the proper reference scales
        self.interp_f_ref = interp1d(self.x, self.ref)

        # Extract the coefficients and define kernel
        self.coeffs = gm.coefficients(self.data, ratio=self.ratio,
                                      ref=self.ref, orders=self.nn_orders_full)

        # uses interpolation to find the proper ratios for training and testing
        self.interp_f_ratio = interp1d(self.x, self.ratio * np.ones(len(self.x)))
        self.ratio_train = self.interp_f_ratio(self.x_train)
        self.coeffs_train = gm.coefficients(self.y_train, ratio=self.ratio_train,
                                            ref=self.ref_train,
                                            orders=self.nn_orders_full)
        self.ratio_test = self.interp_f_ratio(self.x_test)
        self.coeffs_test = gm.coefficients(self.y_test, ratio=self.ratio_test,
                                           ref=self.ref_test,
                                           orders=self.nn_orders_full)

        # defines the kernel
        if self.fixed_quantity_name == "energy" and \
                self.fixed_quantity_value < 70.1 and \
                self.fixed_quantity_value >= 1.:
            self.kernel = RBF(length_scale=self.ls,
                              length_scale_bounds=(self.ls_lower, self.ls_upper)) + \
                          WhiteKernel(1e-6, noise_level_bounds='fixed')
        else:
            self.kernel = RBF(length_scale=self.ls, \
                              length_scale_bounds=(self.ls_lower, self.ls_upper)) + \
                          WhiteKernel(1e-10, noise_level_bounds='fixed')

        # Define the GP
        self.gp = gm.ConjugateGaussianProcess(
            self.kernel, center=self.center, disp=self.disp, df=self.df,
            scale=self.std_est, n_restarts_optimizer=50, random_state=self.seed,
            sd=self.sd)

        # restricts coeffs and colors to only those orders desired for
        # evaluating statistical diagnostics
        self.nn_orders = self.orders_restricted
        self.coeffs = (self.coeffs.T[self.mask_restricted]).T
        self.coeffs_train = (self.coeffs_train.T[self.mask_restricted]).T
        self.coeffs_test = (self.coeffs_test.T[self.mask_restricted]).T

    def plot_coefficients(self, ax=None, whether_save=True):
        """
        Parameters
        ----------
        ax : Axes, optional
            Axes object for plotting. The default is None.
        whether_save : bool, optional
            Whether to save the figure. The default is True.

        Returns
        -------
        Figure with plot.
        """
        # optimizes the ConjugateGaussianProcess for the given parameters and extracts the
        # length scale
        self.gp.fit(self.X_train, self.coeffs_train)
        self.ls_true = np.exp(self.gp.kernel_.theta)
        # X_constraint = [self.x[i] for i in self.constraint[0]]
        # mask_constraint = np.reshape(~ np.isin(self.X_train, X_constraint), len(self.X_train))
        # self.pred, self.std = self.gp.predict(self.X,
        #                                 Xc = self.X_train[mask_constraint],
        #                                 y = self.coeffs_train[mask_constraint, :],
        #                                 return_std = True)
        self.pred, self.std = self.gp.predict(self.X, return_std=True)
        self.underlying_std = np.sqrt(self.gp.cov_factor_)

        self.underlying_std = np.sqrt(self.gp.cov_factor_)
        print("self.pred has shape " + str(np.shape(self.pred)))
        print("self.pred = " + str(self.pred))
        print("self.std has shape " + str(np.shape(self.std)))
        print("self.std = " + str(self.std))

        # plots the coefficients against the given input space
        if ax is None:
            fig, ax = plt.subplots(figsize=(3.2, 2.2))

        for i, n in enumerate(self.nn_orders_full[self.mask_restricted]):
            ax.fill_between(self.x, self.pred[:, i] + 2 * self.std,
                            self.pred[:, i] - 2 * self.std,
                            facecolor=self.light_colors[i], edgecolor=self.colors[i],
                            lw=edgewidth, alpha=1, zorder=5 * i - 4)
            ax.plot(self.x, self.pred[:, i], color=self.colors[i], ls='--', zorder=5 * i - 3)
            ax.plot(self.x, self.coeffs[:, i], color=self.colors[i], zorder=5 * i - 2)
            ax.plot(self.x_train, self.coeffs_train[:, i], color=self.colors[i],
                    ls='', marker='o',
                    # label=r'$c_{}$'.format(n),
                    zorder=5 * i - 1)

        # Format
        ax.axhline(2 * self.underlying_std, 0, 1, color=gray, zorder=-10, lw=1)
        ax.axhline(-2 * self.underlying_std, 0, 1, color=gray, zorder=-10, lw=1)
        ax.axhline(0, 0, 1, color=softblack, zorder=-10, lw=1)
        if np.max(self.x) < 1.1:
            ax.set_xticks(self.x_test, minor=True)
            ax.set_xticks([round(xx, 1) for xx in self.x_train])
        else:
            ax.set_xticks(self.x_test, minor=True)
            ax.set_xticks([round(xx, 0) for xx in self.x_train])
        ax.tick_params(which='minor', bottom=True, top=False)
        ax.set_xlabel(self.caption_coeffs)
        ax.set_yticks(ticks=[-2 * self.underlying_std, 2 * self.underlying_std])
        ax.set_yticklabels(labels=['{:.1f}'.format(-2 * self.underlying_std), '{:.1f}'.format(2 * self.underlying_std)])
        ax.set_yticks([-1 * self.underlying_std, self.underlying_std], minor=True)
        ax.legend(
            # ncol=2,
            borderpad=0.4,
            # labelspacing=0.5, columnspacing=1.3,
            borderaxespad=0.6,
            loc = 'best',
            title = self.title_coeffs).set_zorder(5 * i)

        # # takes constraint into account, if applicable
        # if self.constraint is not None and self.constraint[2] == self.x_quantity_name:
        #     dX = np.array([[self.x[i]] for i in self.constraint[0]])
        #     # std_interp = np.sqrt(np.diag(
        #     #     self.gp.cov(self.X) -
        #     #     self.gp.cov(self.X, dX) @ np.linalg.solve(self.gp.cov(dX, dX), self.gp.cov(dX, self.X))
        #     # ))
        #     _, std_interp = self.gp.predict(self.X,
        #                                     Xc=dX,
        #                                     y=np.array(self.constraint[1]),
        #                                     return_std=True)
        #
        #     ax.plot(self.x, 2 * std_interp, color='gray', ls='--', zorder=-10, lw=1)
        #     ax.plot(self.x, -2 * std_interp, color='gray', ls='--', zorder=-10, lw=1)

        # takes constraint into account, if applicable
        if self.constraint is not None and self.constraint[2] == self.x_quantity_name:
            dX = np.array([[self.x[i]] for i in self.constraint[0]])
            # std_interp = np.sqrt(np.diag(
            #     self.gp.cov(self.X) -
            #     self.gp.cov(self.X, dX) @ np.linalg.solve(self.gp.cov(dX, dX), self.gp.cov(dX, self.X))
            # ))
            _, std_interp = self.gp.predict(self.X,
                                            Xc=dX,
                                            y=np.array(self.constraint[1]),
                                            return_std=True)
            ax.plot(self.x, 2 * std_interp, color='gray', ls='--', zorder=-10, lw=1)
            ax.plot(self.x, -2 * std_interp, color='gray', ls='--', zorder=-10, lw=1)

        # draws length scales
        # ax.annotate("", xy=(np.min(self.x), -0.65 * 2 * self.underlying_std),
        #             xytext=(np.min(self.x) + self.ls, -0.65 * 2 * self.underlying_std),
        #             arrowprops=dict(arrowstyle="<->", capstyle='projecting', lw=1,
        #                             color='k', shrinkA = 0, shrinkB = 0), annotation_clip=False, zorder=5 * i)
        # ax.text(np.min(self.x) + self.ls + 0.2 * (np.max(self.x) - np.min(self.x)),
        #         -0.65 * 2 * self.underlying_std, r'$\ell_{\mathrm{guess}}$', fontsize=14,
        #         horizontalalignment='right', verticalalignment='center', zorder=5 * i)

        ax.annotate("", xy=(np.min(self.x), -0.9 * 2 * self.underlying_std),
                    xytext=(np.min(self.x) + self.ls_true, -0.9 * 2 * self.underlying_std),
                    arrowprops=dict(arrowstyle="<->", capstyle='projecting', lw=1,
                                    color='k', shrinkA = 0, shrinkB = 0), annotation_clip=False, zorder=5 * i)
        ax.text(np.min(self.x) + self.ls_true + 0.2 * (np.max(self.x) - np.min(self.x)),
                -0.9 * 2 * self.underlying_std, r'$\ell_{\mathrm{fit}}$', fontsize=14,
                horizontalalignment='right', verticalalignment='center', zorder=5 * i)

        # draws standard deviations
        # ax.annotate("", xy=(np.min(self.x) + 0.90 * (np.max(self.x) - np.min(self.x)), 0),
        #             xytext=(np.min(self.x) + 0.90 * (np.max(self.x) - np.min(self.x)),
        #                     -1. * self.std_est),
        #             arrowprops=dict(arrowstyle="<->", capstyle='projecting', lw=1,
        #                             color='k', shrinkA = 0, shrinkB = 0), annotation_clip=False, zorder=5 * i)
        # ax.text(np.min(self.x) + 0.90 * (np.max(self.x) - np.min(self.x)),
        #         -1.2 * self.std_est, r'$\sigma_{\mathrm{guess}}$', fontsize=14,
        #         horizontalalignment='center', verticalalignment='bottom', zorder=5 * i)

        ax.annotate("", xy=(np.min(self.x) + 0.74 * (np.max(self.x) - np.min(self.x)), 0),
                    xytext=(np.min(self.x) + 0.74 * (np.max(self.x) - np.min(self.x)),
                            -1. * self.underlying_std),
                    arrowprops=dict(arrowstyle="<->", capstyle='projecting', lw=1,
                                    color='k', shrinkA = 0, shrinkB = 0), annotation_clip=False, zorder=5 * i)
        ax.text(np.min(self.x) + 0.74 * (np.max(self.x) - np.min(self.x)),
                -1.2 * self.underlying_std, r'$\sigma_{\mathrm{fit}}$', fontsize=14,
                horizontalalignment='center', verticalalignment='bottom', zorder=5 * i)

        # saves figure
        if 'fig' in locals() and whether_save:
            fig.tight_layout()

            fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + \
                         '_' + 'interp_and_underlying_processes' + '_' + str(self.fixed_quantity_value) + str(
                        self.fixed_quantity_units) + \
                         '_' + self.scheme + '_' + self.scale + '_Q' + self.Q_param + '_' + self.vs_what + \
                         '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + \
                         self.train_pts_loc + '_' + self.p_param +
                         self.filename_addendum).replace('_0MeVlab_', '_'))

    def plot_md(self, ax=None, whether_save=True):
        """
        Parameters
        ----------
        ax : Axes, optional
            Axes object for plotting. The default is None.
        whether_save : bool, optional
            Whether to save the figure. The default is True.

        Returns
        -------
        Figure with plot.
        """
        try:
            # calculates and plots the squared Mahalanobis distance
            self.gp.kernel_

            # takes into account a constraint, if applicable
            if self.constraint is not None and self.constraint[2] == self.x_quantity_name:
                dX = np.array([[self.x[i]] for i in self.constraint[0]])
                self.mean, self.cov = self.gp.predict(self.X_test,
                                                      Xc=dX,
                                                      y=np.array(self.constraint[1]),
                                                      return_std=False,
                                                      return_cov=True)
            else:
                self.mean = self.gp.mean(self.X_test)
                self.cov = self.gp.cov(self.X_test)
            self.gr_dgn = gm.GraphicalDiagnostic(self.coeffs_test,
                                                 self.mean,
                                                 self.cov,
                                                 colors=self.colors,
                                                 gray=gray,
                                                 black=softblack)

            if ax is None:
                fig, ax = plt.subplots(figsize=(1.0, 2.2))

            self.gr_dgn.md_squared(type='box', trim=False, title=None,
                                   xlabel=r'$\mathrm{D}_{\mathrm{MD}}^2$', ax=ax,
                                   **{"size": 10})
            offset_xlabel(ax)
            plt.show()

            # saves figure
            if 'fig' in locals() and whether_save:
                fig.tight_layout();

                fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + \
                             '_' + 'md' + '_' + str(self.fixed_quantity_value) + str(self.fixed_quantity_units) + '_' + \
                             self.scheme + '_' + self.scale + '_Q' + self.Q_param + '_' + self.vs_what + \
                             '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + \
                             self.train_pts_loc + '_' + self.p_param +
                             self.filename_addendum).replace('_0MeVlab_', '_'))

        except:
            print("Error in calculating or plotting the Mahalanobis distance.")

    def plot_pc(self, ax=None, whether_save=True):
        """
        Parameters
        ----------
        ax : Axes, optional
            Axes object for plotting. The default is None.
        whether_save : bool, optional
            Whether to save the figure. The default is True.

        Returns
        -------
        Figure with plot.
        """
        try:
            # calculates and plots the pivoted Cholesky decomposition
            self.gp.kernel_

            # takes into account constraints, if applicable
            if self.constraint is not None and self.constraint[2] == self.x_quantity_name:
                dX = np.array([[self.x[i]] for i in self.constraint[0]])
                self.mean, self.cov = self.gp.predict(self.X_test,
                                                      Xc=dX,
                                                      y=np.array(self.constraint[1]),
                                                      return_std=False,
                                                      return_cov=True)
            else:
                self.mean = self.gp.mean(self.X_test)
                self.cov = self.gp.cov(self.X_test)
            self.gr_dgn = gm.GraphicalDiagnostic(self.coeffs_test,
                                                 self.mean,
                                                 self.cov,
                                                 colors=self.colors,
                                                 gray=gray,
                                                 black=softblack)

            with plt.rc_context({"text.usetex": True}):
                if ax is None:
                    fig, ax = plt.subplots(figsize=(3.2, 2.2))

                self.gr_dgn.pivoted_cholesky_errors(ax=ax, title=None)
                ax.set_xticks(np.arange(2, self.n_test_pts + 1, 2))
                ax.set_xticks(np.arange(1, self.n_test_pts + 1, 2), minor=True)
                ax.text(0.05, 0.95, r'$\mathrm{D}_{\mathrm{PC}}$', bbox=text_bbox,
                        transform=ax.transAxes, va='top', ha='left')

                # plots legend
                legend_handles = []
                for i, n in enumerate(self.nn_orders_full[self.mask_restricted]):
                    # legend_handles.append(Patch(color=self.colors[i], label=r'$c_{}$'.format(n)))
                    legend_handles.append(Line2D([0], [0], marker='o',
                                            color='w',
                                            label=r'$c_{}$'.format(n),
                                            markerfacecolor=self.colors[i],
                                            markersize=8))
                ax.legend(handles=legend_handles,
                          loc='center left',
                          bbox_to_anchor=(1, 0.5),
                          handletextpad=0.02,
                          borderpad=0.2)

                fig.tight_layout()
                plt.show()

                # saves figure
                if 'fig' in locals() and whether_save:
                    fig.tight_layout()

                    fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + \
                                 '_' + 'pc_vs_index' + '_' + str(self.fixed_quantity_value) + str(
                                self.fixed_quantity_units) + '_' + \
                                 self.scheme + '_' + self.scale + '_Q' + self.Q_param + '_' + self.vs_what + \
                                 '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + \
                                 self.train_pts_loc + '_' + self.p_param +
                                 self.filename_addendum).replace('_0MeVlab_', '_'))

        except:
            print("Error in calculating or plotting the pivoted Cholesky decomposition.")

    def plot_posterior_pdf(self, ax_joint=None, ax_marg_x=None,
                           ax_marg_y=None, whether_save=True):
        """
        Parameters
        ----------
        ax_joint : Axes, optional
            Joint axis. The default is None.
        ax_marg_x : Axes, optional
            Axis for marginalizing the y-coordinate. The default is None.
        ax_marg_y : Axes, optional
            Axis for marginalizing the x-coordinate. The default is None.
        whether_save : bool, optional
            Whether to save the figure. The default is True.

        Returns
        -------
        Figure with plot.
        """

        # functions for interpolating the ratio and reference scale in the TruncationGP
        def lambda_interp_f_ref(x_):
            X = np.ravel(x_)
            return self.interp_f_ref(X)

        def lambda_interp_f_ratio(x_, lambda_var):
            X = np.ravel(x_)
            return self.interp_f_ratio(X) * self.Lambda_b / lambda_var

        try:
            # creates the grid over which the posterior PDF will be plotted
            self.ls_vals = self.posteriorgrid.x_vals
            self.lambda_vals = self.posteriorgrid.y_vals

            # creates and fits the TruncationGP
            self.gp_post = gm.TruncationGP(self.kernel,
                                           ref=lambda_interp_f_ref,
                                           ratio=lambda_interp_f_ratio,
                                           center=self.center,
                                           disp=self.disp,
                                           df=self.df,
                                           scale=self.std_est,
                                           excluded=[0],
                                           ratio_kws={"lambda_var": self.Lambda_b})

            # takes account for the constraint, if applicable
            if self.constraint is not None and self.constraint[2] == self.x_quantity_name:
                self.gp_post.fit(self.X_train,
                                 self.y_train,
                                 orders=self.nn_orders_full,
                                 orders_eval=self.nn_orders,
                                 dX=np.array([[self.x[i]] for i in self.constraint[0]]),
                                 dy=[j for j in self.constraint[1]])
            else:
                self.gp_post.fit(self.X_train,
                                 self.y_train,
                                 orders=self.nn_orders_full,
                                 orders_eval=self.nn_orders)

            # evaluates the probability across the mesh
            self.ls_lambda_loglike = np.array([[
                self.gp_post.log_marginal_likelihood([ls_, ], orders_eval=self.nn_orders,
                                                     **{"lambda_var": lambda_})
                for ls_ in np.log(self.ls_vals)]
                for lambda_ in self.lambda_vals])

            # Makes sure that the values don't get too big or too small
            self.ls_lambda_like = np.exp(self.ls_lambda_loglike - np.max(self.ls_lambda_loglike))

            # Now compute the marginal distributions
            self.lambda_like = np.trapz(self.ls_lambda_like, x=self.ls_vals, axis=-1)
            self.ls_like = np.trapz(self.ls_lambda_like, x=self.lambda_vals, axis=0)

            # Normalize them
            self.lambda_like /= np.trapz(self.lambda_like, x=self.lambda_vals, axis=0)
            self.ls_like /= np.trapz(self.ls_like, x=self.ls_vals, axis=0)

            with plt.rc_context({"text.usetex": True, "text.latex.preview": True}):
                cmap_name = 'Blues'
                cmap = mpl.cm.get_cmap(cmap_name)

                # Setup axes
                if ax_joint == None and ax_marg_x == None and ax_marg_y == None:
                    fig, ax_joint, ax_marg_x, ax_marg_y = joint_plot(ratio=5, height=3.4)

                # Plot contour
                ax_joint.contour(self.ls_vals, self.lambda_vals, self.ls_lambda_like,
                                 levels=[np.exp(-0.5 * r ** 2) for r in np.arange(9, 0, -0.5)] + [0.999],
                                 cmap=cmap_name, vmin=-0.05, vmax=0.8, zorder=1)

                # Now plot the marginal distributions
                ax_marg_y.plot(self.lambda_like, self.lambda_vals, c=cmap(0.8), lw=1)
                ax_marg_y.fill_betweenx(self.lambda_vals, np.zeros_like(self.lambda_like),
                                        self.lambda_like, facecolor=cmap(0.2), lw=1)
                ax_marg_x.plot(self.ls_vals, self.ls_like, c=cmap(0.8), lw=1)
                ax_marg_x.fill_between(self.ls_vals, np.zeros_like(self.ls_vals),
                                       self.ls_like, facecolor=cmap(0.2), lw=1)

                # Formatting
                ax_joint.set_xlabel(r'$\ell$')
                ax_joint.set_ylabel(r'$\Lambda$')
                ax_joint.axvline(self.ls, 0, 1, c=gray, lw=1, zorder=0)
                ax_joint.axhline(self.Lambda_b, 0, 1, c=gray, lw=1, zorder=0)
                ax_joint.margins(x=0, y=0.)
                ax_joint.set_xlim(min(self.ls_vals), max(self.ls_vals))
                ax_joint.set_ylim(min(self.lambda_vals), max(self.lambda_vals))
                ax_marg_x.set_ylim(bottom=0);
                ax_marg_y.set_xlim(left=0);
                ax_joint.text(0.95, 0.95, r'pr$(\ell, \Lambda \,|\, \vec{\mathbf{y}}_k)$', ha='right', va='top',
                              transform=ax_joint.transAxes, bbox=text_bbox, fontsize=12)

                plt.show()

                if 'fig' in locals() and whether_save:
                    fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + \
                                 '_' + 'Lambda_ell_jointplot' + '_' + str(self.fixed_quantity_value) + str(
                                self.fixed_quantity_units) + '_' + \
                                 self.scheme + '_' + self.scale + '_Q' + self.Q_param + '_' + self.vs_what + \
                                 '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + \
                                 self.train_pts_loc + '_' + self.p_param +
                                 self.filename_addendum).replace('_0MeVlab_', '_'))

        except:
            print("Error in plotting the posterior PDF.")

    def plot_truncation_errors(self, online_data, residual_plot=True,
                               whether_save=True):
        """
        Parameters
        ----------
        online_data : array
            Summed order-by-order predictions for an observable.
        residual_plot : bool, optional
            Whether to plot the truncation error as residuals or as full sums.
        whether_save : bool, optional
            Whether to save the output. The default is True.

        Returns
        -------
        Figure (2)
            A figure with the order-by-order residuals plot and a figure with
            the order-by-order empirical coverage.
        """
        # sets up the data from PWA93 to which we'll compare
        self.online_data = online_data

        # functions for reference scale and dimensionless expansion parameter (ratio)
        def lambda_interp_f_ref(x_):
            X = np.ravel(x_)
            return self.interp_f_ref(X)

        def lambda_interp_f_ratio(x_, lambda_var):
            X = np.ravel(x_)
            return self.interp_f_ratio(X) * self.Lambda_b / lambda_var

        try:
            # creates the TruncationGP object
            self.gp_trunc = gm.TruncationGP(self.kernel,
                                            ref=lambda_interp_f_ref,
                                            ratio=lambda_interp_f_ratio,
                                            center=self.center,
                                            disp=self.disp,
                                            df=self.df,
                                            scale=self.std_est,
                                            excluded=self.excluded,
                                            ratio_kws={"lambda_var": self.Lambda_b})

            # fits the GP with or without a constraint
            if self.constraint is not None and self.constraint[2] == self.x_quantity_name:
                self.gp_trunc.fit(self.X_train, self.y_train,
                                  orders=self.nn_orders_full,
                                  # orders_eval=self.nn_orders,
                                  dX=np.array([[self.x[i]] for i in self.constraint[0]]),
                                  dy=[j for j in self.constraint[1]])
            else:
                self.gp_trunc.fit(self.X_train, self.y_train,
                                  orders=self.nn_orders_full,
                                  # orders_eval=self.nn_orders
                                  )

            # creates fig with two columns of axes
            fig, axes = plt.subplots(
                int(np.ceil(len(self.nn_orders_full[self.mask_restricted]) / 2)),
                2, sharex=True, sharey=False, figsize=(3.2, 4))
            # deletes extraneous axes to suit number of evaluated orders
            if 2 * np.ceil(len(self.nn_orders_full[self.mask_restricted]) / 2) > len(
                    self.nn_orders_full[self.mask_restricted]):
                fig.delaxes(axes[int(np.ceil(
                    len(self.nn_orders_full[self.mask_restricted]) / 2)) - 1, 1])

            for i, n in enumerate(self.nn_orders_full[self.mask_restricted]):
                # calculates the standard deviation of the truncation error
                _, self.std_trunc = self.gp_trunc.predict(self.X, order=n,
                                                          return_std=True, kind='trunc')

                # gets the "true" order-by-order data from online
                if self.fixed_quantity_name == "energy":
                    data_true = self.online_data[self.fixed_quantity_value, :]
                elif self.fixed_quantity_name == "angle":
                    if self.fixed_quantity_value == 0:
                        data_true = self.online_data
                    else:
                        data_true = self.online_data[:, self.fixed_quantity_value]

                for j in range(i, len(self.nn_orders_full[self.mask_restricted])):
                    ax = axes.ravel()[j]

                    # number of standard deviations around the dotted line to plot
                    # Why does this correspond to 67% confidence intervals?
                    std_coverage = 1

                    if residual_plot:
                        # calculates and plots the residuals
                        residual = data_true - (self.data[:, self.mask_restricted])[:, i]
                        ax.plot(self.x, residual, zorder=i - 4, c=self.colors[i])
                        ax.fill_between(self.x,
                                        residual + std_coverage * self.std_trunc,
                                        residual - std_coverage * self.std_trunc,
                                        zorder=i - 5,
                                        facecolor=self.light_colors[i],
                                        edgecolor=self.colors[i],
                                        lw=edgewidth)
                        ax.set_ylim(np.min(np.concatenate(
                            (residual + std_coverage * self.std_trunc, residual - std_coverage * self.std_trunc))),
                                    np.max(np.concatenate((residual + std_coverage * self.std_trunc,
                                                           residual - std_coverage * self.std_trunc))))

                    else:
                        # calculates and plots the true data
                        ax.plot(self.x,
                                (self.data[:, self.mask_restricted])[:, i],
                                zorder=i - 5, c=self.colors[i])
                        ax.fill_between(self.x,
                                        (self.data[:, self.mask_restricted])[:,
                                        i] + std_coverage * self.std_trunc,
                                        (self.data[:, self.mask_restricted])[:,
                                        i] - std_coverage * self.std_trunc,
                                        zorder=i - 5,
                                        facecolor=self.light_colors[i],
                                        edgecolor=self.colors[i],
                                        lw=edgewidth)
                        ax.set_ylim(np.min(np.concatenate(((self.data[:, self.mask_restricted])[:,
                                                           i] + std_coverage * self.std_trunc, (self.data[:, self.mask_restricted])[
                                                                                               :,
                                                                                               i] - std_coverage * self.std_trunc))),
                                    np.max(np.concatenate(((self.data[:, self.mask_restricted])[:,
                                                           i] + std_coverage * self.std_trunc, (self.data[:, self.mask_restricted])[
                                                                                               :,
                                                                                               i] - std_coverage * self.std_trunc))))

                    # # plots the testing points as vertical lines
                    # for line in self.x_test: ax.axvline(line, 0, 1, c = gray)

                ax = axes.ravel()[i]

                if residual_plot:
                    # plots a line at y = 0
                    ax.plot(self.x, np.zeros(len(self.x)), color=softblack, lw=1, ls='--')
                else:
                    # plots the true data
                    ax.plot(self.x, data_true, color=softblack, lw=1, ls='--')

                # formats x-axis labels and tick marks
                # ax.set_xlabel(self.caption_coeffs)
                ax.set_xticks([int(min(self.x) + (max(self.x) - min(self.x)) / 3),
                               int(min(self.x) + (max(self.x) - min(self.x)) / 3 * 2)])
                ax.set_xticks([tick for tick in self.x_test], minor=True)
            fig.supxlabel(self.caption_coeffs)
            plt.show()

            # saves
            if 'fig' in locals() and whether_save:
                # fig.suptitle(r'$\mathrm{' + self.observable_name + '\,(' + str(self.fixed_quantity_value) + '\,' + str(
                #     self.fixed_quantity_units) + ')\,' + \
                #              '\,for\,' + self.scheme + '\,' + self.scale + '}' + '\,(Q_{\mathrm{' + self.Q_param + \
                #              '}},\,\mathrm{' + self.p_param + '},\,\mathrm{' + self.vs_what + '})$', size=20)
                fig.tight_layout()

                if self.constraint is None:
                    fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + '_' +
                                 str(self.fixed_quantity_value) + str(self.fixed_quantity_units) +
                                 '_' + 'full_pred_truncation' + '_' + self.scheme + '_' +
                                 self.scale + '_Q' + self.Q_param + '_' + self.vs_what +
                                 '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' +
                                 self.train_pts_loc + '_' + self.p_param +
                                 self.filename_addendum).replace('_0MeVlab_', '_'))
                else:
                    fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + '_' +
                                 str(self.fixed_quantity_value) + str(self.fixed_quantity_units) +
                                 '_' + 'full_pred_truncation_constrained' + '_' + self.scheme + '_' +
                                 self.scale + '_Q' + self.Q_param + '_' + self.vs_what +
                                 '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' +
                                 self.train_pts_loc + '_' + self.p_param +
                                 self.filename_addendum).replace('_0MeVlab_', '_'))

            # creates interpolation function for the true and theory data
            data_interp = interp1d(self.x, self.data[:, self.mask_restricted].T)
            data_true_interp = interp1d(self.x, data_true)

            # calculates the covariance matrix and mean
            self.cov_wp = self.gp_trunc.cov(self.X_test, start=0, end=np.inf)
            self.mean_wp = self.gp_trunc.mean(self.X_test)

            # norms the residuals by factors of the ratio
            self.norm_residuals_wp = data_true_interp(self.X_test) - data_interp(self.X_test)
            denom = (np.tile(self.ratio_test,
                             (len(self.nn_orders_full[self.mask_restricted]), 1)).T) ** (
                                self.nn_orders_full[self.mask_restricted] + 1) * (np.sqrt(
                1 - np.tile(self.ratio_test,
                            (len(self.nn_orders_full[self.mask_restricted]), 1)) ** 2)).T
            self.norm_residuals_wp = self.norm_residuals_wp / (denom.T)[:, :, None]
            self.gr_dgn_wp = gm.GraphicalDiagnostic(self.norm_residuals_wp.T,
                                                    mean=self.mean_wp, cov=self.cov_wp,
                                                    colors=self.colors, gray=gray, black=softblack)

            fig, ax = plt.subplots(figsize=(3.4, 3.2))

            # creates the empirical coverage plot
            self.gr_dgn_wp.credible_interval(
                np.linspace(1e-5, 1, 100), band_perc=[0.68, 0.95], ax=ax,
                title="Empirical coverage (PWA93)\n" +
                      r'$\mathrm{' + self.observable_name + '\,(' + str(self.fixed_quantity_value) + '\,' + str(
                    self.fixed_quantity_units) + ')\,' + \
                      '\,for\,' + self.scheme + '\,' + self.scale + '}' + '\,(Q_{\mathrm{' + self.Q_param + \
                      '}},\,\mathrm{' + self.p_param + '},\,\mathrm{' + self.vs_what + '})$',
                xlabel=r'Credible Interval ($100\alpha\%$)',
                ylabel=r'Empirical Coverage ($\%$)\,(N = ' + str(len(self.X_test)) + r')')

            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_xticklabels([0, 20, 40, 60, 80, 100])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_yticklabels([0, 20, 40, 60, 80, 100])

            plt.show()

            # saves the figure
            if 'fig' in locals() and whether_save:
                fig.tight_layout()

                fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + '_' +
                             str(self.fixed_quantity_value) + str(self.fixed_quantity_units) +
                             '_' + 'truncation_error_empirical_coverage' + '_' + self.scheme + '_' +
                             self.scale + '_Q' + self.Q_param + '_' + self.vs_what +
                             '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' +
                             self.train_pts_loc + '_' + self.p_param +
                             self.filename_addendum).replace('_0MeVlab_', '_'))

        except:
            print("Error plotting the truncation errors.")

    def plot_credible_intervals(self, ax=None, whether_save=True):
        """
        Parameters
        ----------
        ax : Axes, optional
            Axes object for plotting. The default is None.
        whether_save : bool, optional
            Whether to save the figure. The default is True.

        Returns
        -------
        Figure with plot.
        """
        try:
            # calculates and plots credible intervals ("weather plots")
            self.gp.kernel_

            # takes account of constraints, if applicable
            if self.constraint is not None and self.constraint[2] == self.x_quantity_name:
                dX = np.array([[self.x[i]] for i in self.constraint[0]])
                self.mean, self.cov = self.gp.predict(self.X_test,
                                                      Xc=dX,
                                                      y=np.array(self.constraint[1]),
                                                      return_std=False,
                                                      return_cov=True)
            else:
                self.mean = self.gp.mean(self.X_test)
                self.cov = self.gp.cov(self.X_test)

            self.gr_dgn = gm.GraphicalDiagnostic(self.coeffs_test,
                                                 self.mean,
                                                 self.cov,
                                                 colors=self.colors,
                                                 gray=gray,
                                                 black=softblack)

            if ax is None:
                fig, ax = plt.subplots(figsize=(3.2, 2.2))

            self.gr_dgn.credible_interval(
                np.linspace(1e-5, 1, 100), band_perc=[0.68, 0.95], ax=ax, title=None,
                xlabel=r'Credible Interval ($100\alpha\%$)',
                # ylabel=r'Empirical Coverage ($\%$)\,(N = ' + str(len(self.X_test)) + r')')
                ylabel=r'Empirical Coverage ($\%$)')

            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_xticklabels([0, 20, 40, 60, 80, 100])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_yticklabels([0, 20, 40, 60, 80, 100])

            plt.show()

            # saves figure
            if 'fig' in locals() and whether_save:
                fig.tight_layout()

                fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + '_' +
                             str(self.fixed_quantity_value) + str(self.fixed_quantity_units) +
                             '_' + 'truncation_error_credible_intervals' + '_' + self.scheme + '_' +
                             self.scale + '_Q' + self.Q_param + '_' + self.vs_what +
                             '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' +
                             self.train_pts_loc + '_' + self.p_param +
                             self.filename_addendum).replace('_0MeVlab_', '_'))

        except:
            print("Error in plotting the credible intervals.")

    def plotzilla(self, whether_save=True):

        """
        Returns
        -------
        Figure with plot.
        """
        # using gridspec, plots the Mahalanobis distance, coefficient curves, credible
        # intervals, pivoted Cholesky, and Lambda-ell posterior pdf on one figure
        fig_main = plt.figure(figsize=(12, 10))

        gs = mpl.gridspec.GridSpec(ncols=30, nrows=24,
                                   wspace=200, hspace=400,
                                   figure=fig_main)

        ax_pdf_joint = fig_main.add_subplot(gs[2:12, 0:10])
        ax_pdf_x = fig_main.add_subplot(gs[0:2, 0:10])
        ax_pdf_y = fig_main.add_subplot(gs[2:12, 10:12])

        ax_md = fig_main.add_subplot(gs[0:24, 24:30])
        ax_coeff = fig_main.add_subplot(gs[0:12, 12:24])
        ax_ci = fig_main.add_subplot(gs[12:24, 0:12])
        ax_pc = fig_main.add_subplot(gs[12:24, 12:24])

        try:
            self.plot_coefficients(ax=ax_coeff, whether_save=True)
        except:
            print("Error in calculating or plotting the coefficient curves.")
        try:
            self.plot_md(ax=ax_md, whether_save=True)
        except:
            print("Error in calculating or plotting the Mahalanobis distance.")
        try:
            self.plot_pc(ax=ax_pc, whether_save=True)
        except:
            print("Error in calculating or plotting the pivoted Cholesky decomposition.")
        try:
            self.plot_credible_intervals(ax=ax_ci, whether_save=True)
        except:
            print("Error in calculating or plotting the credible intervals.")
        try:
            self.plot_posterior_pdf(ax_joint=ax_pdf_joint, ax_marg_x=ax_pdf_x,
                                    ax_marg_y=ax_pdf_y, whether_save=True)
        except:
            print("Error in calculating or plotting the posterior PDF.")

        # adds a title
        fig_main.suptitle(r'$\mathrm{' + self.observable_name + '\,(' + str(self.fixed_quantity_value) + '\,' + str(
            self.fixed_quantity_units) + ')\,' + \
                          '\,for\,' + self.scheme + '\,' + self.scale + '}' + '\,(Q_{\mathrm{' + self.Q_param + \
                          '}},\,\mathrm{' + self.p_param + '},\,\mathrm{' + self.vs_what + '})$', size=30)

        if whether_save:
            fig_main.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + \
                              '_' + 'plotzilla' + '_' + str(self.fixed_quantity_value) + str(
                        self.fixed_quantity_units) + '_' + \
                              self.scheme + '_' + self.scale + '_Q' + self.Q_param + '_' + self.vs_what + \
                              '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + \
                              self.train_pts_loc + '_' + self.p_param +
                              self.filename_addendum).replace('_0MeVlab_', '_'))

def interp_f_ratio_posterior(x_map, x_interp, p, Q_param, mpi_var, lambda_var):
    """
    Function for interpolating between the input space and the ratio across that input space.

    Parameters
    ----------
    x_map (array) : array of points onto which to map.
    x_interp (array) : array of points from which to map.
    p (array) : momentum/momenta for calculating the ratio (dimensionless expansion parameter).
    Q_param (str) : type of Q parametrization.
        Can be "smoothmax", "max", or "sum".
    mpi_var (float) : value of the (effective) pion mass (in MeV) for calculating the ratio.
    lambda_var (float) : value of the breakdown scale (in MeV) for calculating the ratio.
    """
    return interpn(x_interp, Q_approx(p, Q_param, Lambda_b=lambda_var, m_pi=mpi_var), x_map)

def ratio_fn_curvewise(X, p_grid_train, p_param, p_shape, Q_param, mpi_var, lambda_var, single_expansion = False):
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
    """
    p = np.array([])
    for pt in p_grid_train:
        try:
            p = np.append(p, p_approx(p_name = p_param, degrees = np.array([pt[0]]), prel = np.array([pt[1]])))
        except:
            p = np.append(p, p_approx(p_name=p_param, degrees=np.array([0]), prel=np.array([pt[0]])))

    return Q_approx(p = np.reshape(p, p_shape), Q_parametrization=Q_param, Lambda_b = lambda_var, m_pi = mpi_var, single_expansion=single_expansion)

def ratio_fn_posterior_const(X, p_shape, p_grid_train, Q):
    """
    Function for interpolating between the input space and the ratio across that input space.

    Parameters
    ----------
    X (array) : array of points onto which to map.
    p_shape (tuple) : shape into which to shape the array of p values.
    p_grid_train (array) : momentum/momenta for calculating the ratio (dimensionless expansion parameter).
    Q (float) : value of the ratio.
    """
    return Q


def make_likelihood_filename(FileNameObj,
        folder,
        observable_name,
        order_name,
        logpriors_names,
        random_vars_array,
):
    """
    Information for naming posterior pdf output files.

    Parameters
    ----------
    self (GSUMObj) : GSUM object with information on naming files.
    folder (str) : folder name.
    order_name (str) : abbreviation for the highest calculated order.
    logpriors_names (str list) : list of names for the log-priors added to the likelihood.
    random_vars_array (RandomVariable list) : list of RandomVariable objects.

    Returns
    ----------
    (str) : file name.
    """
    filename = (
            str(folder)
            + "/"
            + "posterior_pdf_curvewise"
            + "_"
            + str(observable_name)
            + "_"
            + str(FileNameObj.scheme)
            + "_"
            + str(FileNameObj.scale)
            + "_"
            + str(order_name)
            + "_"
            + "Q"
            + str(FileNameObj.Q_param)
            + "_"
            + str(FileNameObj.p_param)
            + "_"
            + str(FileNameObj.vs_what)
    )

    # for logprior in logpriors_names:
    #     filename += "_" + str(logprior)
    # for random_var in random_vars_array:
    #     filename += (
    #             "_"
    #             + str(random_var.name)
    #             + str(len(random_var.var))
    #             + "pts"
    #     )
    for (logprior, random_var) in zip(logpriors_names, random_vars_array):
        filename += "_" + str(random_var.name) + "_" + str(logprior) + '_' + str(len(random_var.var)) + "pts"

    return str(filename.replace("__", "_") + FileNameObj.filename_addendum + ".txt")

def calc_loglike_ray(mesh_cart,
                     batch_size,
                     log_likelihood,
                     gp_post,
                     log_likelihood_fn_kwargs,
                     ):
    """
    Calculates the log-likelihood for a set of inputs using a curvewise method.

    Parameters
    ----------
    mesh_cart (array) : Cartesian array of all possible ordered tuples of random variable meshes.
    batch_size (int) : batch size for Ray.
    log_likelihood (Ray) : Ray object with function for calculating the log-likelihood.
    gp_post (TruncationGP) : fitted Gaussian process object.
    log_likelihood_fn_kwargs (dict) : keyword arguments for log_likelihood.

    Returns
    ----------
    log_like (array) : log-likelihood generated by the function log_likelihood.
    """
    # calculates the likelihood using ray
    log_like_ids = []
    for i in range(0, len(mesh_cart), batch_size):
        batch = mesh_cart[i: i + batch_size]
        log_like_ids.append(log_likelihood.remote(gp_post,
                                                  # input_space,
                                                  # mom_pts,
                                                  batch,
                                                  # p_param,
                                                  # p_shape,
                                                  # Q_param
                                                  log_likelihood_fn_kwargs,
                                                  ))
    log_like = list(itertools.chain(*ray.get(log_like_ids)))

    return log_like
def add_logpriors(variables_array, obs_loglike):
    """
    Adds N log-priors to an N-dimensional array.

    Parameters
    ----------
    variables_array (RandomVariable list) : list of RandomVariable objects.
    obs_loglike (array) : log-likelihood to which to add the log-priors.

    Returns
    ----------
    obs_loglike (array) : log-likelihood after the log-priors have been added to it.
    """
    for i, logprior in enumerate([variable.logprior for variable in variables_array]):
        logprior_shape_tuple = (1,)
        for lst in range(len(variables_array) - 1, 0, -1):
            logprior_shape_tuple = (np.shape(obs_loglike)[(i + lst) % len(variables_array)],) + \
                                   logprior_shape_tuple
        obs_loglike += np.transpose(np.tile(logprior, logprior_shape_tuple),
                                    np.roll(np.arange(0, len(variables_array), dtype=int), i + 1))

    return obs_loglike

def marginalize_likelihoods(variables_array, like_list):
    """
    Marginalizes likelihoods into all possible 1- and 2-d posteriors.

    Parameters
    ----------
    variables_array (RandomVariable list) : list of RandomVariable objects.
    like_list (array) : list of likelihoods.
    order_num (int) : total number of orders in the evaluation.

    Returns
    ----------
    marg_post_array (array) : array of fully marginalized single-variable posteriors.
    joint_post_array (array) : array of fully marginalized joint posteriors.
    """
    marg_post_list = []
    joint_post_list = []

    for like_idx, like in enumerate(like_list):
        # creates the normalized fully marginalized posteriors
        for v, var in enumerate(variables_array):
            var_idx_array = np.arange(0, np.shape(variables_array)[0], 1, dtype=int)
            var_idx_array = var_idx_array[var_idx_array != v]
            var_idx_array = np.flip(var_idx_array)

            marg_post = np.copy(like)

            for idx in var_idx_array:
                marg_post = np.trapz(marg_post, x=variables_array[idx].var, axis=idx)

            marg_post /= np.trapz(marg_post, x=variables_array[v].var, axis=0)

            marg_post_list.append(list(marg_post))

        if np.shape(variables_array)[0] > 1:
            comb_array = []
            for ca in range(1, np.shape(variables_array)[0]):
                for ca_less in range(0, ca):
                    comb_array.append([ca, ca_less])
            comb_array = np.flip(np.array(comb_array), axis=1)
        else:
            comb_array = np.array([0, 0])

        for (v_norm, v_marg) in zip(comb_array,
                                    np.flip(np.array([np.arange(0, np.shape(variables_array)[0], 1, dtype=int)[
                                                          ~np.isin(
                                                              np.arange(0, np.shape(variables_array)[0], 1, dtype=int),
                                                              c)] for c in
                                                      comb_array]), axis=1)
                                    ):

            if like.ndim > 2:
                joint_post = np.trapz(like, x=variables_array[v_marg[0]].var, axis=v_marg[0])

                if like.ndim > 3:
                    for vmarg in v_marg[1:]:
                        joint_post = np.trapz(joint_post, x=variables_array[vmarg].var, axis=vmarg)
                joint_post /= np.trapz(np.trapz(joint_post,
                                                x=variables_array[v_norm[1]].var, axis=1),
                                       x=variables_array[v_norm[0]].var, axis=0
                                       )
            else:
                joint_post = like

            joint_post_list.append(joint_post)

    marg_post_array = np.reshape(marg_post_list, (len(variables_array), np.shape(like_list)[0]) + np.shape(marg_post_list)[1:], order='F')

    joint_post_array = np.array(joint_post_list)

    return marg_post_array, joint_post_array

@ray.remote
def log_likelihood(gp_fitted,
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
    return [gp_fitted.log_marginal_likelihood([pt[1 + n] for n in range(len(pt) - 2)],
                                              **{**log_likelihood_fn_kwargs,
                                                 **{"mpi_var": pt[-1],
                                                    "lambda_var": pt[0]}}

                                              ) for pt in mesh_points]

@ray.remote
def log_likelihood_const(gp_fitted,
                   mesh_points,
                   log_likelihood_fn_kwargs
                   ):
    """
    Function for interpolating calculating the log-likelihood for a fitted TrunctionTP object.
    Specifically, this is for cases with random variables (Q, ell_degrees, ell_tlab).
    Parameters
    ----------
    gp_fitted (TruncationTP) : Student t-distribution object from GSUM.
    mesh_points (array) : array over which evaluation takes place.
    log_likelihood_fn_kwargs (dict) : kwargs for evaluation.
    """
    return [gp_fitted.log_marginal_likelihood([pt[1 + n] for n in range(len(pt) - 1)],
                                              **{**log_likelihood_fn_kwargs,
                                                 **{"Q": pt[0]}}) for pt in mesh_points]

def plot_posteriors_curvewise(
                          light_colors,
                          nn_orders_array,
                          nn_orders_full_array,
                          excluded,
                          orders_labels_dict,
                          orders_names_dict,

                          nn_interaction,

                          center,
                          disp,
                          df,
                          std_est,

                          obs_data_grouped_list,
                          obs_name_grouped_list,
                          obs_labels_grouped_list,
                          mesh_cart_grouped_list,
                          t_lab,
                          t_lab_train_pts,
                          InputSpaceTlab,
                          LsTlab,
                          degrees,
                          degrees_train_pts,
                          InputSpaceDeg,
                          LsDeg,
                          variables_array,

                          mom_fn,
                          mom_fn_kwargs,

                          scaling_fn,
                          scaling_fn_kwargs,

                          ratio_fn,
                          ratio_fn_kwargs,

                          log_likelihood_fn,
                          log_likelihood_fn_kwargs,

                          orders=2,
                          FileName = None,
                          whether_plot_posteriors=True,
                          whether_plot_corner=True,
                          whether_use_data=True,
                          whether_save_data=True,
                          whether_save_plots=True,
                          ):
    """
    Calculates the log-likelihood for a set of inputs.

    Parameters
    ----------
    light_colors (array) : array of MatPlotLib colors for filling.
    nn_orders_array (int array) : array of orders for plotting data.
    nn_orders_full_array (int array) : array of orders corresponding to all possible orders of data.
    excluded (int list) : list of orders excluded from nn_orders_full_array, along with 0.
    orders_labels_dict (dict) : dictionary for linking order numbers and markdown strings for plotting.
    orders_names_dict (dict) : dictionary for linking order numbers and strings for file-naming.

    nn_interaction (str) : interaction for observables using in calculating E_to_p.
        Should be "np", "nn", or "pp".

    center (float) : initial guess for the mean of the distribution.
    disp (int) : initial guess for the standard deviation.
    df (int) : number of degrees of freedom.
    std_est (float) : estimate for the standard deviation.

    obs_data_grouped_list (array list) : list of arrays of observable data grouped together as decided by user.
    obs_name_grouped_list (str list) : list of strings for observables for file-naming as decided by user.
    obs_labels_grouped_list (str list) : list of markdown strings for observables for file-naming as decided by user.
    mesh_cart_grouped_list (array list) : list of arrays of Cartesian meshes for evaluating log-likelihood, grouped
        together as decided by user.
    t_lab (array) : array of lab-energy points for evaluation.
    t_lab_train_pts (array) : list of lab-energy training points for evaluation.
    InputSpaceTlab (InputSpaceBunch) : object encoding information about lab-energy input space
    LsTlab (LengthScale) : object encoding information about guess and bounds for lab-energy length scale.
    degrees (array) : array of scattering-angle points for evaluation.
    degrees_train_pts (array) : list of scattering-angle training points for evaluation.
    InputSpaceDeg : object encoding information about scattering-angle input space
    LsDeg (LengthScale) : object encoding information about guess and bounds for scattering-angle length scale.
    variables_array (RandomVariable array) : list of RandomVariable objects for each random variable.
        Must be in the order (Lambda_b, scattering-angle length scale, lab-energy length scale, mpi_eff).

    mom_fn (function) : function for converting from lab energy to relative momentum.
    mom_fn_kwargs (dict) : keyword arguments for mom_fn.

    scaling_fn (function) : function for scaling input space.
    scaling_fn_kwargs (dict) : keyword arguments for scaling_fn.

    ratio_fn (function) : function for evaluating the ratio (dimensionless expansion parameter, or Q).
    ratio_fn_kwargs (dict) : keyword arguments for ratio_fn.

    log_likelihood_fn (function) : function for evaluating the log-likelihood.
    log_likelihood_fn_kwargs (dict) : keyword arguments for log_likelihood_fn.

    orders (int) : number of orders to include in the calculations, starting from and including the highest allowed
        order and counting down.
        Default : 2
    FileName (FileNaming) : object encoding information for naming files.
        Default : None

    whether_plot_posteriors (bool) : whether to plot posteriors.
        Default : True
    whether_plot_corner : whether to plot corner plot.
        Default : True
    whether_use_data : whether to use already saved data for plotting.
        Default : True
    whether_save_data : whether to save data.
        Default : True
    whether_save_plots : whether to save plots.
        Default : True
    """

    # sets the number of orders and the corresponding colors
    order_num = int(orders)
    Lb_colors = light_colors[-1 * order_num:]

    # creates boolean array for treatment of the length scale (and any other variables) on an observable-by-
    # observable basis instead of a cross-observable basis
    marg_bool_array = np.array([v.marg_bool for v in variables_array])

    # initiates the ray kernel for utilizing all processors on the laptop
    ray.shutdown()
    ray.init()

    BATCH_SIZE = 100

    # list for appending log-likelihoods
    like_list = []

    for (obs_grouping, obs_name, mesh_cart_group) in zip(obs_data_grouped_list, obs_name_grouped_list, mesh_cart_grouped_list):
        for order_counter in range(1, order_num + 1):
            order = np.max(nn_orders_array) - order_num + order_counter

            try:
                # generates names for files and searches for whether they exist
                if not whether_use_data:
                    raise ValueError("You elected not to use saved data.")
                else:
                    # if they exist, they are read in, reshaped, and appended to like_list
                    like_list.append(np.reshape(
                        np.loadtxt(make_likelihood_filename(FileName,
                                                            "data",
                                                            obs_name,
                                                            orders_names_dict[order],
                                                            [variable.logprior_name for variable in variables_array],
                                                            variables_array,
                                                            )),
                        tuple([len(random_var.var) for random_var in variables_array[marg_bool_array]]),
                    ))

            except:
                # failing that, generates new data and saves it (if the user chooses)
                obs_loglike_sum = np.zeros(tuple(len(random_var.var) for random_var in variables_array[marg_bool_array]))

                for (obs_object, mesh_cart) in zip(obs_grouping, mesh_cart_group):
                    obs_data_full = obs_object.data

                    yref_type = obs_object.ref_type

                    if len(np.shape(obs_data_full)) == 2:
                        # 1D observables
                        if np.shape(obs_data_full)[1] == len(degrees):
                            # observables that depend only on scattering angle (of which none exist)
                            # sets kernel
                            kernel_posterior = RBF(length_scale=(LsDeg.ls_guess),
                                                   length_scale_bounds=(
                                                       (LsDeg.ls_bound_lower, LsDeg.ls_bound_upper))) + \
                                               WhiteKernel(1e-6, noise_level_bounds='fixed')

                            # converts the points in degrees to the current input space
                            degrees_input = InputSpaceDeg.input_space(**{"deg_input": degrees})
                            degrees_train_pts_input = InputSpaceDeg.input_space(**{"deg_input": degrees_train_pts})

                            # sieves the data
                            obs_data = np.reshape(
                                obs_data_full,
                                # (len(self.nn_orders_full), -1))
                                (len(nn_orders_full_array), -1))
                            obs_data_train = np.reshape(
                                obs_data_full[:, np.isin(degrees, degrees_train_pts)],
                                # (len(self.nn_orders_full), -1))
                                (len(nn_orders_full_array), -1))

                            # sets yref
                            if yref_type == "dimensionful":
                                yref = obs_data_train[-1]
                            elif yref_type == "dimensionless":
                                yref = np.ones((len(degrees_train_pts)))

                            # creates and fits the TruncationTP object
                            gp_post_obs = gm.TruncationTP(kernel_posterior,
                                                          ref=yref,
                                                          ratio=ratio_fn,
                                                          center=center,
                                                          disp=disp,
                                                          df=df,
                                                          scale=std_est,
                                                          excluded=excluded,
                                                          ratio_kws={**ratio_fn_kwargs,
                                                                     **{"p_shape": (len(degrees_train_pts))},
                                                                     **{"p_grid_train": degrees_train_pts[:, None]}
                                                                     }
                                                          )
                            # fits the TP to data
                            gp_post_obs.fit(scaling_fn(degrees_train_pts_input, **scaling_fn_kwargs)[:, None],
                                            (obs_data_train[:order, :]).T,
                                            orders = nn_orders_full_array[:order])

                            # puts important objects into ray objects
                            gp_post_ray = ray.put(gp_post_obs)

                            # calculates the posterior using ray
                            log_like = calc_loglike_ray(mesh_cart,
                                                        BATCH_SIZE,
                                                        log_likelihood_fn,
                                                        gp_post_ray,
                                                        log_likelihood_fn_kwargs={**log_likelihood_fn_kwargs,
                                                                                  **{"p_shape": (len(degrees_train_pts))},
                                                                                  **{"p_grid_train": degrees_train_pts[:, None]}
                                                                                  }
                                                        )
                            obs_loglike = np.reshape(log_like, tuple(
                                len(random_var.var) for random_var in variables_array))

                            # adds the log-priors to the log-likelihoods
                            obs_loglike = add_logpriors(variables_array, obs_loglike)
                            # makes sure that the values don't get too big or too small
                            obs_like = np.exp(obs_loglike - np.max(obs_loglike))
                            # marginalizes partially
                            for v, var in zip(np.flip(np.array(range(len(variables_array)))[~marg_bool_array]),
                                              np.flip(variables_array[~marg_bool_array])):
                                obs_like = np.trapz(obs_like, x=variables_array[v].var, axis=v)

                            # takes the log again to revert to the log-likelihood
                            obs_loglike_2d = np.log(obs_like)
                            obs_loglike_sum += obs_loglike_2d

                        elif np.shape(obs_data_full)[1] == len(t_lab):
                            # observables that depend only on scattering angle (e.g., total cross section, or SGT)
                            kernel_posterior = RBF(length_scale=(LsTlab.ls_guess),
                                                   length_scale_bounds=(
                                                       (LsTlab.ls_bound_lower, LsTlab.ls_bound_upper))) + \
                                               WhiteKernel(1e-6, noise_level_bounds='fixed')

                            # # converts the points in t_lab to the current input space
                            tlab_input = InputSpaceTlab.input_space(**{"E_lab": t_lab,
                                                                       "interaction": nn_interaction})
                            tlab_train_pts_input = InputSpaceTlab.input_space(**{"E_lab": t_lab_train_pts,
                                                                        "interaction": nn_interaction})

                            # converts points in t_lab to momentum
                            tlab_mom = mom_fn(t_lab, **mom_fn_kwargs)
                            tlab_train_pts_mom = mom_fn(t_lab_train_pts, **mom_fn_kwargs)

                            # sieves the data
                            obs_data = np.reshape(
                                obs_data_full,
                                # (len(self.nn_orders_full), -1))
                                (len(nn_orders_full_array), -1))
                            obs_data_train = np.reshape(
                                obs_data_full[:, np.isin(t_lab, t_lab_train_pts)],
                                # (len(self.nn_orders_full), -1))
                                (len(nn_orders_full_array), -1))

                            # sets yref
                            if yref_type == "dimensionful":
                                yref = obs_data_train[-1]
                            elif yref_type == "dimensionless":
                                yref = np.ones((len(t_lab_train_pts)))

                            # creates and fits the TruncationTP object
                            gp_post_obs = gm.TruncationTP(kernel_posterior,
                                                          ref=yref,
                                                          ratio=ratio_fn,
                                                          center=center,
                                                          disp=disp,
                                                          df=df,
                                                          scale=std_est,
                                                          excluded=excluded,
                                                          ratio_kws = {**ratio_fn_kwargs,
                                                                       **{"p_shape" : (len(tlab_train_pts_mom))},
                                                                       **{"p_grid_train" : tlab_train_pts_mom[:, None]}
                                                                       }
                                                          )
                            # fits the TP to data
                            gp_post_obs.fit(scaling_fn(tlab_train_pts_input, **scaling_fn_kwargs)[:, None],
                                            (obs_data_train[:order, :]).T,
                                            orders = nn_orders_full_array[:order])

                            # puts important objects into ray objects
                            gp_post_ray = ray.put(gp_post_obs)

                            # calculates the posterior using ray
                            log_like=calc_loglike_ray(mesh_cart,
                                                        BATCH_SIZE,
                                                        log_likelihood_fn,
                                                        gp_post_ray,
                                                        log_likelihood_fn_kwargs={**log_likelihood_fn_kwargs,
                                                                                  **{"p_shape" : (len(t_lab_train_pts))},
                                                                                  **{"p_grid_train" : tlab_train_pts_mom[:, None]}
                                                                                  }
                                                        )
                            obs_loglike = np.reshape(log_like, tuple(
                                len(random_var.var) for random_var in variables_array))

                            # adds the log-priors to the log-likelihoods
                            obs_loglike = add_logpriors(variables_array, obs_loglike)
                            # makes sure that the values don't get too big or too small
                            obs_like = np.exp(obs_loglike - np.max(obs_loglike))
                            # marginalizes partially
                            for v, var in zip(np.flip(np.array(range(len(variables_array)))[~marg_bool_array]),
                                              np.flip(variables_array[~marg_bool_array])):
                                obs_like = np.trapz(obs_like, x=variables_array[v].var, axis=v)
                            # takes the log again to revert to the log-likelihood
                            obs_loglike_2d = np.log(obs_like)
                            obs_loglike_sum += obs_loglike_2d

                    else:
                        # 2D observables
                        # sets kernel
                        kernel_posterior = RBF(length_scale=(LsDeg.ls_guess, LsTlab.ls_guess),
                                               length_scale_bounds=((LsDeg.ls_bound_lower, LsDeg.ls_bound_upper),
                                                                    (LsTlab.ls_bound_lower, LsTlab.ls_bound_upper))) + \
                                           WhiteKernel(1e-6, noise_level_bounds='fixed')

                        # converts points in t_lab to the current input space
                        tlab_input = InputSpaceTlab.input_space(**{"E_lab": t_lab,
                                                                    "interaction": nn_interaction})
                        tlab_train_pts_input = InputSpaceTlab.input_space(**{"E_lab": t_lab_train_pts,
                                                                        "interaction": nn_interaction})

                        # converts points in t_lab to momentum
                        tlab_mom = mom_fn(t_lab, **mom_fn_kwargs)
                        tlab_train_pts_mom = mom_fn(t_lab_train_pts, **mom_fn_kwargs)

                        # converts points in degrees to the current input space
                        degrees_input = InputSpaceDeg.input_space(**{"deg_input": degrees,
                                                                     "p_input" : tlab_mom})
                        degrees_train_pts_input = InputSpaceDeg.input_space(**{"deg_input": degrees_train_pts,
                                                                               "p_input" : tlab_train_pts_mom})

                        # creates a grid of training points in the 2D input space
                        if tlab_train_pts_input.ndim == 1 and degrees_train_pts_input.ndim == 1:
                            grid_train = scaling_fn(np.flip(np.array(list(itertools.product(tlab_train_pts_input, degrees_train_pts_input))), axis = 1),
                                            **scaling_fn_kwargs)
                        elif tlab_train_pts_input.ndim == 1 and degrees_train_pts_input.ndim != 1:
                            grid_train = scaling_fn(np.flip(
                                    np.array(
                                        [[np.tile(tlab_train_pts_input, (np.shape(degrees_train_pts_input)[0], 1)).flatten('F')[s], degrees_train_pts_input.flatten('F')[s]] for s in range(degrees_train_pts_input.size)]
                                    ),
                                axis=1),
                                **scaling_fn_kwargs)
                        elif tlab_train_pts_input.ndim != 1 and degrees_train_pts_input.ndim == 1:
                            # untested
                            grid_train = scaling_fn(np.flip(
                                np.array(
                                    [[tlab_train_pts_input.flatten('F')[s],
                                      np.tile(degrees_train_pts_input.flatten('F')[s], (np.shape(tlab_train_pts_input)[1], 1))] for s in range(tlab_train_pts_input.size)]
                                ),
                                axis=1),
                                **scaling_fn_kwargs)
                        else:
                            # untested
                            grid_train = scaling_fn(np.flip(
                                np.array(
                                    [[tlab_train_pts_input.flatten('F')[s],
                                      degrees_train_pts_input.flatten('F')[s]] for s in
                                     range(tlab_train_pts_input.size)]
                                ),
                                axis=1),
                                **scaling_fn_kwargs)

                        # sieves the data
                        obs_data = np.reshape(
                            obs_data_full,
                            (len(nn_orders_full_array), -1))
                        obs_data_train = np.reshape(
                            obs_data_full[:, np.isin(t_lab, t_lab_train_pts)][..., np.isin(degrees, degrees_train_pts)],
                            (len(nn_orders_full_array), -1))

                        # sets yref
                        if yref_type == "dimensionful":
                            yref = obs_data_train[-1]
                        elif yref_type == "dimensionless":
                            yref = np.ones((len(degrees_train_pts) * len(t_lab_train_pts)))

                        # # DELETE ASAP
                        # obs_data_train_ugly = np.swapaxes(np.swapaxes(np.tile(obs_data_grouped_list[0][0].data, (179, 1, 1)), 0, 1), 1, 2)
                        # obs_data_train_ugly = np.reshape(
                        #     obs_data_train_ugly[:, np.isin(t_lab, t_lab_train_pts)][..., np.isin(degrees, degrees_train_pts)],
                        #     (len(nn_orders_full_array), -1))
                        # yref = obs_data_train_ugly[-1] * (4 * np.pi)

                        # creates and fits the TruncationTP object
                        gp_post_obs = gm.TruncationTP(kernel_posterior,
                                                      ref=yref,
                                                      ratio=ratio_fn,
                                                      center=center,
                                                      disp=disp,
                                                      df=df,
                                                      scale=std_est,
                                                      excluded=excluded,
                                                      ratio_kws={**ratio_fn_kwargs,
                                                                 **{"p_shape" : (len(degrees_train_pts) * len(tlab_train_pts_mom))},
                                                                 **{"p_grid_train" : np.flip(np.array(list(itertools.product(tlab_train_pts_mom, degrees_train_pts))), axis = 1)}
                                                                 }

                                                      )
                        # fits the TP to data
                        gp_post_obs.fit(grid_train,
                                        (obs_data_train[:order, :]).T,
                                        # orders=self.nn_orders_full[:order])
                                        orders = nn_orders_full_array[:order])

                        # # takes account for the constraint, if applicable
                        # if obs_object.constraint is not None:
                        #     if obs_object.constraint[2] == 'angle':
                        #         print("We're invoking the constraint.")
                        #         gp_post_obs.fit(grid_train,
                        #                         (obs_data_train[:order, :]).T,
                        #                         orders=nn_orders_full_array[:order],
                        #                         dX=np.array([[degrees_input[i]] for i in obs_object.constraint[0]]),
                        #                         dy=[j for j in obs_object.constraint[1]])
                        #     # elif obs_object.constraint[2] == 'energy':
                        #     #     gp_post_obs.fit(grid_train,
                        #     #                     (obs_data_train[:order, :]).T,
                        #     #                     orders=nn_orders_full_array[:order],
                        #     #                     dX=np.array([[self.x[i]] for i in obs_object.constraint[0]]),
                        #     #                     dy=[j for j in obs_object.constraint[1]])
                        # else:
                        #     # fits the TP to data
                        #     gp_post_obs.fit(grid_train,
                        #                     (obs_data_train[:order, :]).T,
                        #                     # orders=self.nn_orders_full[:order])
                        #                     orders=nn_orders_full_array[:order])

                        # # puts important objects into ray objects
                        gp_post_ray = ray.put(gp_post_obs)

                        # calculates the posterior using ray
                        log_like = calc_loglike_ray(mesh_cart,
                                                    BATCH_SIZE,
                                                    log_likelihood_fn,
                                                    gp_post_ray,
                                                    log_likelihood_fn_kwargs={**log_likelihood_fn_kwargs,
                                                                              **{"p_shape" : (len(degrees_train_pts) * len(tlab_train_pts_mom))},
                                                                              **{"p_grid_train" : np.flip(np.array(list(itertools.product(tlab_train_pts_mom, degrees_train_pts))), axis = 1)}
                                                                              }
                                                    )
                        obs_loglike = np.reshape(log_like, tuple(
                            len(random_var.var) for random_var in variables_array))

                        # adds the log-priors to the log-likelihoods
                        obs_loglike = add_logpriors(variables_array, obs_loglike)
                        # makes sure that the values don't get too big or too small
                        obs_like = np.exp(obs_loglike - np.max(obs_loglike))
                        # marginalizes partially
                        for v, var in zip(np.flip(np.array(range(len(variables_array)))[~marg_bool_array]),
                                          np.flip(variables_array[~marg_bool_array])):
                            obs_like = np.trapz(obs_like, x=variables_array[v].var, axis=v)
                        # takes the log again to revert to the log-likelihood
                        obs_loglike_partmarg = np.log(obs_like)
                        obs_loglike_sum += obs_loglike_partmarg

                # makes sure that the values don't get too big or too small
                obs_like = np.exp(obs_loglike_sum - np.max(obs_loglike_sum))


                if whether_save_data:
                    # saves data, if the user chooses
                    np.savetxt(make_likelihood_filename(FileName,
                                                        "data",
                                                        obs_name,
                                                        orders_names_dict[order],
                                                        [variable.logprior_name for variable in variables_array],
                                                        variables_array,
                                                        ),
                               np.reshape(obs_like, (np.prod(
                                   [len(random_var.var) for random_var in variables_array[marg_bool_array]]))))

                like_list.append(obs_like)

    like_list = np.reshape(np.reshape(like_list, (np.shape(like_list)[0] // orders, orders) + np.shape(like_list)[1:], order = 'C'),
                          np.shape(like_list))

    if whether_plot_posteriors or whether_plot_corner:
        # calculates all joint and fully marginalized posterior pdfs
        marg_post_array, joint_post_array = marginalize_likelihoods(variables_array[marg_bool_array], like_list)

    if whether_plot_posteriors:
        # plots and saves all fully marginalized posterior pdfs
        for (variable, result) in zip(variables_array[marg_bool_array], marg_post_array):
            fig = plot_marg_posteriors(variable, result, obs_labels_grouped_list, Lb_colors, order_num,
                                       # self.nn_orders, self.orders_labels_dict, self, whether_save_plots, obs_name_grouped_list)
                                    nn_orders_array, orders_labels_dict)

            if whether_save_plots:
                # saves
                obs_name_corner_concat = ''.join(obs_name_grouped_list)
                fig.savefig(('figures/' + FileName.scheme + '_' + FileName.scale + '/' +
                             variable.name + '_posterior_pdf_curvewise' + '_' + obs_name_corner_concat +
                             '_' + FileName.scheme + '_' +
                             FileName.scale + '_Q' + FileName.Q_param + '_' + FileName.p_param + '_' +
                             InputSpaceDeg.name + 'x' + InputSpaceTlab.name +
                             FileName.filename_addendum).replace('_0MeVlab_', '_'))

    if whether_plot_corner:
        with plt.rc_context({"text.usetex": True}):
            # plots and saves all joint and fully marginalized posterior pdfs in the form of corner plots
            fig = plot_corner_posteriors('Blues',
                                         order_num,
                                         variables_array[marg_bool_array],
                                         marg_post_array,
                                         joint_post_array,
                                         FileName,
                                         obs_name_grouped_list,
                                         whether_save_plots,
                                         nn_orders_array,
                                         orders_labels_dict)

    # # creates a list of the values of the random variables corresponding to the
    # # point of highest probability in the posterior
    # indices_opt = np.where(like_list[-1] == np.amax(like_list[-1]))
    opt_vals_list = []
    # for idx, var in zip(indices_opt, [variable.var for variable in variables_array[marg_bool_array]]):
    #     opt_vals_list.append((var[idx])[0])
    # print("opt_vals_list = " + str(opt_vals_list))

    return opt_vals_list

def plot_posteriors_pointwise(
        light_colors,
        nn_orders_array,
        nn_orders_full_array,
        excluded,
        orders_labels_dict,

        obs_data_grouped_list,
        obs_name_grouped_list,
        obs_labels_grouped_list,
        t_lab,
        t_lab_train_pts,
        InputSpaceTlab,
        degrees,
        degrees_train_pts,
        InputSpaceDeg,
        variables_array,

        mom_fn_tlab,
        mom_fn_tlab_kwargs,

        mom_fn_degrees,
        mom_fn_degrees_kwargs,

        p_fn,
        p_fn_kwargs,

        ratio_fn,
        ratio_fn_kwargs,

        orders=2,
        FileName=None,
        whether_plot_posteriors=True,
        whether_save_plots=True,
):
    """
    Calculates the log-likelihood for a set of inputs using a pointwise method.

    Parameters
    ----------
    light_colors (array) : array of MatPlotLib colors for filling.
    nn_orders_array (int array) : array of orders for plotting data.
    nn_orders_full_array (int array) : array of orders corresponding to all possible orders of data.
    excluded (int list) : list of orders excluded from nn_orders_full_array, along with 0.
    orders_labels_dict (dict) : dictionary for linking order numbers and markdown strings for plotting.

    obs_data_grouped_list (array list) : list of arrays of observable data grouped together as decided by user.
    obs_name_grouped_list (str list) : list of strings for observables for file-naming as decided by user.
    obs_labels_grouped_list (str list) : list of markdown strings for observables for file-naming as decided by user.
    mesh_cart_grouped_list (array list) : list of arrays of Cartesian meshes for evaluating log-likelihood, grouped
        together as decided by user.
    t_lab (array) : array of lab-energy points for evaluation.
    t_lab_train_pts (array) : list of lab-energy training points for evaluation.
    InputSpaceTlab (InputSpaceBunch) : object encoding information about lab-energy input space
    degrees (array) : array of scattering-angle points for evaluation.
    degrees_train_pts (array) : list of scattering-angle training points for evaluation.
    InputSpaceDeg : object encoding information about scattering-angle input space
    variables_array (RandomVariable array) : list of RandomVariable objects for each random variable.
        Must be simply (Lambda_b).

    mom_fn_tlab (function) : function for converting from lab energy to relative momentum.
    mom_fn_tlab_kwargs (dict) : keyword arguments for mom_fn_tlab.

    mom_fn_degrees (function) : function for converting from scattering angle to relative momentum.
    mom_fn_degrees_kwargs (dict) : keyword arguments for mom_fn_degrees.

    p_fn (function) : function for converting from input spaces to relative momentum.
    p_fnp (dict) : keyword arguments for p_fn.

    ratio_fn (function) : function for evaluating the ratio (dimensionless expansion parameter, or Q).
    ratio_fn_kwargs (dict) : keyword arguments for ratio_fn.

    orders (int) : number of orders to include in the calculations, starting from and including the highest allowed
        order and counting down.
        Default : 2
    FileName (FileNaming) : object encoding information for naming files.
        Default : None

    whether_plot_posteriors (bool) : whether to plot posteriors.
        Default : True
    whether_save_plots : whether to save plots.
        Default : True
    """

    # sets the number of orders and the corresponding colors
    order_num = int(orders)
    Lb_colors = light_colors[-1 * order_num:]

    marg_post_list = []

    for (obs_grouping, obs_name) in zip(obs_data_grouped_list, obs_name_grouped_list):
        # generates names for files and searches for whether they exist
        for order_counter in range(1, order_num + 1):
            order = np.max(nn_orders_array) - order_num + order_counter

            # scale invariant: df = 0
            PointwiseModel = gm.TruncationPointwise(df=0, excluded=excluded)

            obs_post_sum = np.ones(
                tuple(len(random_var.var) for random_var in variables_array))

            for obs_object in obs_grouping:
                obs_data_full = obs_object.data
                yref_type = obs_object.ref_type

                if len(np.shape(obs_data_full)) == 2:
                    if np.shape(obs_data_full)[1] == len(degrees):
                        # converts the points in tlab_train_pts_mom and degrees_train_pts to momentum
                        tlab_train_pts_mom = mom_fn_tlab(**{**mom_fn_tlab_kwargs, **{"E_lab" : np.array([0])}})
                        degrees_train_pts_mom = mom_fn_degrees(**{**mom_fn_degrees_kwargs, **{"degrees": degrees_train_pts}})

                        # sieves the data
                        obs_data_train = np.reshape(
                            obs_data_full[:, np.isin(degrees, degrees_train_pts)],
                            (len(nn_orders_full_array), -1))

                        # sets yref
                        if yref_type == "dimensionful":
                            yref = obs_data_train[-1]
                        elif yref_type == "dimensionless":
                            yref = np.ones((len(degrees_train_pts)))

                        # calculates ratio for every training point and every value of Lambda_b
                        ratio_train = [ratio_fn(**{**ratio_fn_kwargs,
                                                   **{"p": np.reshape(p_fn(**{**p_fn_kwargs, **{
                                                      "prel" : tlab_train_pts_mom,
                                                      "degrees" : degrees_train_pts_mom
                                                   }}), len(degrees_train_pts)), "Lambda_b": Lb}})
                                       for Lb in variables_array[0].var]

                        # fits the TruncationPointwise object
                        pointwise_result, _, _ = compute_posterior_intervals(
                            PointwiseModel, obs_data_train, ratio_train, ref=yref,
                            orders=nn_orders_full_array,
                            max_idx=order - 1,
                            logprior=variables_array[0].logprior,
                            Lb=variables_array[0].var)

                        obs_post_sum *= pointwise_result

                    elif np.shape(obs_data_full)[1] == len(t_lab):
                        # converts the points in tlab_train_pts_mom and degrees_train_pts to momentum
                        tlab_train_pts_mom = mom_fn_tlab(**{**mom_fn_tlab_kwargs, **{"E_lab": t_lab_train_pts}})
                        degrees_train_pts_mom = mom_fn_degrees(
                            **{**mom_fn_degrees_kwargs, **{"degrees": np.array([0])}})

                        # sieves the data
                        obs_data_train = np.reshape(
                            obs_data_full[:, np.isin(t_lab, t_lab_train_pts)],
                            (len(nn_orders_full_array), -1))

                        # sets yref
                        if yref_type == "dimensionful":
                            yref = obs_data_train[-1]
                        elif yref_type == "dimensionless":
                            yref = np.ones((len(t_lab_train_pts)))

                        # calculates ratio for every training point and every value of Lambda_b
                        ratio_train = [ratio_fn(**{**ratio_fn_kwargs,
                                                   **{"p": np.reshape(p_fn(**{**p_fn_kwargs, **{
                                                       "prel": tlab_train_pts_mom,
                                                       "degrees": degrees_train_pts_mom
                                                   }}), len(t_lab_train_pts)), "Lambda_b": Lb}})
                                       for Lb in variables_array[0].var]

                        # fits the TruncationPointwise object
                        pointwise_result, _, _ = compute_posterior_intervals(
                            PointwiseModel, obs_data_train, ratio_train, ref=yref,
                            orders=nn_orders_full_array,
                            max_idx=order - 1,
                            logprior=variables_array[0].logprior,
                            Lb=variables_array[0].var)

                        obs_post_sum *= pointwise_result

                else:
                    # converts the points in tlab_train_pts_mom and degrees_train_pts to momentum
                    tlab_train_pts_mom = mom_fn_tlab(**{**mom_fn_tlab_kwargs, **{"E_lab": t_lab_train_pts}})
                    degrees_train_pts_mom = mom_fn_degrees(
                        **{**mom_fn_degrees_kwargs, **{"degrees": degrees_train_pts}})

                    # sieves data
                    obs_data_train = np.reshape(
                        obs_data_full[:, np.isin(t_lab, t_lab_train_pts)][
                            ..., np.isin(degrees, degrees_train_pts)],
                        (len(nn_orders_full_array), -1))

                    # sets yref
                    if yref_type == "dimensionful":
                        yref = obs_data_train[-1]
                    elif yref_type == "dimensionless":
                        yref = np.ones((len(degrees_train_pts) * len(t_lab_train_pts)))

                    # calculates ratio for every training point and every value of Lambda_b
                    ratio_train = [ratio_fn(**{**ratio_fn_kwargs,
                                               **{"p": np.reshape(p_fn(**{**p_fn_kwargs, **{
                                                   "prel": tlab_train_pts_mom,
                                                   "degrees": degrees_train_pts_mom
                                               }}), (len(degrees_train_pts) * len(t_lab_train_pts))), "Lambda_b": Lb}})
                                   for Lb in variables_array[0].var]


                    # fits the TruncationPointwise object
                    pointwise_result, _, _ = compute_posterior_intervals(
                        PointwiseModel, obs_data_train, ratio_train, ref=yref,
                        orders=nn_orders_full_array,
                        max_idx=order - 1,
                        logprior=variables_array[0].logprior, Lb=variables_array[0].var)

                    obs_post_sum *= pointwise_result

            # appends the normalized posterior
            marg_post_list.append(obs_post_sum / np.trapz(obs_post_sum, variables_array[0].var))

    marg_post_list = np.reshape(
        np.reshape(marg_post_list, (np.shape(marg_post_list)[0] // orders, orders) + np.shape(marg_post_list)[1:],
                   order='C'),
        np.shape(marg_post_list))

    # adds an extra dimension to comport with structure of existing code
    marg_post_list = marg_post_list[None, :]

    if whether_plot_posteriors:
        for (variable, result) in zip(variables_array, marg_post_list):
            # generates plots of posteriors for multiple observables and orders
            fig = plot_marg_posteriors(variable, result, obs_labels_grouped_list, Lb_colors, order_num,
                                       nn_orders_array, orders_labels_dict)

            # saves
            obs_name_corner_concat = ''.join(obs_name_grouped_list)
            if whether_save_plots:
                fig.savefig(('figures/' + FileName.scheme + '_' + FileName.scale + '/' +
                             variable.name + '_posterior_pdf_pointwise' + '_' + obs_name_corner_concat +
                             '_' + FileName.scheme + '_' +
                             FileName.scale + '_Q' + FileName.Q_param + '_' + FileName.p_param + '_' +
                             InputSpaceDeg.name + 'x' + InputSpaceTlab.name +
                             FileName.filename_addendum).replace('_0MeVlab_', '_'))

        # finds and prints the MAP value for Lambda_b
        indices_opt = np.where(marg_post_list[0, -1, :] == np.amax(marg_post_list[0, -1, :]))
        opt_vals_list = []
        for idx, var in zip(indices_opt, [variable.var for variable in variables_array]):
            opt_vals_list.append((var[idx])[0])

        print("opt_vals_list = " + str(opt_vals_list))
    return opt_vals_list

def scaling_fn(pts_array):
    try:
        pass
        # for pt_idx, pt in enumerate(pts_array):
        #     pts_array[pt_idx, :] = np.array([pts_array[pt_idx, 0],
        #                                      pts_array[pt_idx, 1] * 0.5])
        # for pt_idx, pt in enumerate(pts_array):
        #     pts_array[pt_idx, :] = np.array([pts_array[pt_idx, 0] * 25 / (2600 * (pts_array[pt_idx, 1])**(-0.79)),
        #                                      pts_array[pt_idx, 1]])
        # for pt_idx, pt in enumerate(pts_array):
        #     pts_array[pt_idx, :] = np.array([pts_array[pt_idx, 0] * 25 / (700 * (pts_array[pt_idx, 1]) ** (-0.58)),
        #                                      pts_array[pt_idx, 1]])
        # for pt_idx, pt in enumerate(pts_array):
        #     pts_array[pt_idx, :] = np.array([pts_array[pt_idx, 0] * 25 / (3200 * (pts_array[pt_idx, 1]) ** (-0.83)),
        #                                      pts_array[pt_idx, 1]])
        # for pt_idx, pt in enumerate(pts_array):
        #     pts_array[pt_idx, :] = np.array([pts_array[pt_idx, 0] * 25 / (3300 * (pts_array[pt_idx, 1]) ** (-0.83)),
        #                                      pts_array[pt_idx, 1]])
    except:
        pass
    return pts_array