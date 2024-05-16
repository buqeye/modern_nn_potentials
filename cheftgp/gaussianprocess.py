import os.path

import numpy as np
from scipy.interpolate import interp1d, interpn, griddata
from .utils import (
    versatile_train_test_split,
    versatile_train_test_split_nd,
    compute_posterior_intervals,
)
from .scattering import E_to_p
from .graphs import (
    offset_xlabel,
    joint_plot,
    setup_rc_params,
    plot_marg_posteriors,
    plot_corner_posteriors,
    softblack,
    gray,
    edgewidth,
    text_bbox,
)
from .eft import Q_approx, p_approx
import h5py
import ray
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import gsum as gm
from sklearn.gaussian_process.kernels import (
    RBF,
    WhiteKernel,
    NormalizedKernelMixin,
    Hyperparameter,
    _check_length_scale,
    ConstantKernel,
    Sum,
    Product,
    Exponentiation,
    StationaryKernelMixin,
    GenericKernelMixin,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone
from sklearn.utils.validation import _num_samples
from abc import ABCMeta, abstractmethod
import warnings
from inspect import signature
from scipy.spatial.distance import pdist, squareform, cdist
import itertools
import functools
from shapely.geometry import Polygon, Point

setup_rc_params()


class GPHyperparameters:
    def __init__(
        self,
        ls_class,
        center,
        ratio,
        nugget=1e-10,
        seed=None,
        df=np.inf,
        disp=0,
        scale=1,
        sd=None,
    ):
        """
        Information necessary for Gaussian process hyperparameters.

        Parameters
        ----------
        ls_class (LengthScale array) : LengthScale object with relevant information.
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
        self.ls_array = np.array([])
        self.ls_lower_array = np.array([])
        self.ls_upper_array = np.array([])
        self.whether_fit_array = np.array([])
        for lsc in ls_class:
            self.ls_array = np.append(self.ls_array, lsc.ls_guess)
            self.ls_lower_array = np.append(self.ls_lower_array, lsc.ls_bound_lower)
            self.ls_upper_array = np.append(self.ls_upper_array, lsc.ls_bound_upper)
            self.whether_fit_array = np.append(self.whether_fit_array, lsc.whether_fit)

        self.center = center
        self.ratio = ratio
        self.nugget = nugget
        self.seed = seed
        self.df = df
        self.disp = disp
        self.scale = scale
        self.sd = sd


class FileNaming:
    def __init__(
        self, Q_param, p_param, filename_addendum="", scheme="", scale="", vs_what=""
    ):
        """
        Information necessary to name files for output figures.

        Parameters
        ----------
        Q_param (str) : name of the Q parametrization.
        p_param (str) : name of the p parametrization.
        filename_addendum (str) : optional extra string to distinguish saved files.
            default : ""
        scheme (str) : potential regulator scheme.
            default : ""
        scale (str) : potential regulator scale.
            default : ""
        vs_what (str) : input space(s).
            default : ""
        """
        self.Q_param = Q_param
        self.p_param = p_param
        self.filename_addendum = filename_addendum
        self.scheme = scheme
        self.scale = scale
        self.vs_what = vs_what


class PosteriorBounds:
    def __init__(self, x_lower, x_upper, x_n, y_lower, y_upper, y_n):
        """
        Class for the boundaries of the 2D posterior PDF plot and the mesh on which it is plotted.
        """
        self.x_vals = np.linspace(x_lower, x_upper, x_n)
        self.y_vals = np.linspace(y_lower, y_upper, y_n)


class RandomVariable:
    def __init__(
        self,
        var,
        user_val,
        name,
        label,
        units,
        ticks,
        logprior,
        logprior_name,
        marg_bool=True,
    ):
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
    def __init__(
        self,
        orders_array,
        excluded,
        colors_array,
        lightcolors_array,
        orders_names_dict=None,
        orders_labels_dict=None,
    ):
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
        orders_labels_dict (dict): dictionary method linking the numerical indices (int)
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
        self.title = ""
        for piece in self.title_pieces:
            self.title += str(piece)
        return self.title


class ObservableBunch:
    def __init__(
        self,
        name,
        data,
        x_array,
        title,
        ref_type,
        nn_interaction,
        unit_string=None,
        constraint=None,
    ):
        """
        Class for an observable
        name (string) : (abbreviated) name for the observable
        data (array) : coefficient values at each order over the mesh
        x_array (array) : array(s) corresponding to the dimensions of the data.
            For example, for the differential cross section (DSG), this would be [energies, angles].
        title (string) : title for the coefficient plot
        ref_type (string) : tells whether the reference scale (to be divided out of the coefficient
            values) has dimension (e.g., the case of the cross section) or not (e.g., the case of the
            spin observables). Can only be "dimensionless" or "dimensionful".
        nn_interaction (str) : type of nucleon-nucleon interaction for which the observable is calculated.
            Can be "np", "nn", or "pp".
        unit_string (string) : string for units of observable. Default is None, but it should not be None if
            ref_type == 'dimensionful'.
        constraint (array or None): constraint on the values of the observable, including the
            name of the quantity for which the constraint applies.
            For dimensionful (i.e., cross-section) observables, should be None.
        """
        self.name = name
        self.data = data
        self.x_array = x_array
        self.title = title
        self.ref_type = ref_type
        self.nn_interaction = nn_interaction
        self.unit_string = unit_string
        self.constraint = constraint
        if (ref_type != "dimensionless") and (ref_type != "dimensionful"):
            raise Exception("ref_type must be dimensionless or dimensionful.")


class Interpolation:
    def __init__(self, x, y, kind="cubic"):
        """
        Class for an interpolater
        x (array) : x-coordinate data
        y (array) : y-coordinate data
        kind (string) : scipy.interpolate.interp1d interpolater 'kind'
        """
        self.x = x
        self.y = y
        self.kind = kind
        # self.f_interp = interp1d(self.x, self.y, kind=self.kind)
        self.f_interp = interp1d(self.x, self.y)


class TrainTestSplit:
    def __init__(
        self,
        name,
        n_train,
        n_test_inter,
        isclose_factor=0.01,
        offset_train_min_factor=0,
        offset_train_max_factor=0,
        xmin_train_factor=0,
        xmax_train_factor=1,
        offset_test_min_factor=0,
        offset_test_max_factor=0,
        xmin_test_factor=0,
        xmax_test_factor=1,
        train_at_ends=True,
        test_at_ends=False,
    ):
        """
        Class for an input space (i.e., x-coordinate)

        name (str) : (abbreviated) name for the combination of training and testing masks
        n_train (int array) : number of intervals into which to split x, with training points at the
            edges of each interval
        n_test_inter (int array) : number of subintervals into which to split the intervals between
            training points, with testing points at the edges of each subinterval
        isclose_factor (float or float array) : fraction of the total input space for the tolerance of making
            sure that training and testing points don't coincide
        offset_train_min_factor (float or float array) : fraction above the minimum of the input space where
            the first potential training point ought to go
        offset_train_max_factor (float or float array) : fraction below the maximum of the input space where
            the last potential training point ought to go
        xmin_train_factor (float or float array) : fraction of the input space below which there ought not to
            be training points
        xmax_train_factor (float or float array) : fraction of the input space above which there ought not to
            be training points
        offset_test_min_factor (float or float array) : fraction above the minimum of the input space where
            the first potential testing point ought to go
        offset_test_max_factor (float or float array) : fraction below the maximum of the input space where
            the last potential testing point ought to go
        xmin_test_factor (float or float array) : fraction of the input space below which there ought not to
            be testing points
        xmax_test_factor (float or float array) : fraction of the input space above which there ought not to
            be testing points
        train_at_ends (bool or bool array) : whether training points should be allowed at or near the
            endpoints of x
        test_at_ends (bool or bool array) : whether testing points should be allowed at or near the endpoints
            of x
        """
        self.name = name
        self.n_train = n_train
        self.n_test_inter = n_test_inter

        self.isclose_factor = isclose_factor * np.ones(len(self.n_train))
        self.offset_train_min_factor = offset_train_min_factor * np.ones(
            len(self.n_train)
        )
        self.offset_train_max_factor = offset_train_max_factor * np.ones(
            len(self.n_train)
        )
        self.xmin_train_factor = xmin_train_factor * np.ones(len(self.n_train))
        self.xmax_train_factor = xmax_train_factor * np.ones(len(self.n_train))
        self.offset_test_min_factor = offset_test_min_factor * np.ones(
            len(self.n_train)
        )
        self.offset_test_max_factor = offset_test_max_factor * np.ones(
            len(self.n_train)
        )
        self.xmin_test_factor = xmin_test_factor * np.ones(len(self.n_train))
        self.xmax_test_factor = xmax_test_factor * np.ones(len(self.n_train))
        self.train_at_ends = train_at_ends * np.ones(len(self.n_train), dtype=bool)
        self.test_at_ends = test_at_ends * np.ones(len(self.n_train), dtype=bool)

    def make_masks(self, x, y):
        """Returns the training and testing points in the input space and the corresponding
        (interpolated) data values after calculating the actual values for xmin, xmax, and
        offsets using the corresponding factors and the input space

        Parameters
        ----------
        x (ND array) : input space
        y (oD x ND array) : data points at each input space value, with N>1 dimensions for N dimensions and
            o orders
        """
        self.x = x
        self.y = y

        # calculates the actual value for each offset, xmin, and xmax
        self.offset_train_min = self.offset_train_min_factor * (
            np.amax(self.x, axis=tuple(range(self.x.ndim - 1)))
            - np.amin(self.x, axis=tuple(range(self.x.ndim - 1)))
        )
        self.offset_train_max = self.offset_train_max_factor * (
            np.amax(self.x, axis=tuple(range(self.x.ndim - 1)))
            - np.amin(self.x, axis=tuple(range(self.x.ndim - 1)))
        )
        self.xmin_train = np.amin(
            self.x, axis=tuple(range(self.x.ndim - 1))
        ) + self.xmin_train_factor * (
            np.amax(self.x, axis=tuple(range(self.x.ndim - 1)))
            - np.amin(self.x, axis=tuple(range(self.x.ndim - 1)))
        )
        self.xmax_train = np.amin(
            self.x, axis=tuple(range(self.x.ndim - 1))
        ) + self.xmax_train_factor * (
            np.amax(self.x, axis=tuple(range(self.x.ndim - 1)))
            - np.amin(self.x, axis=tuple(range(self.x.ndim - 1)))
        )
        self.offset_test_min = self.offset_test_min_factor * (
            np.amax(self.x, axis=tuple(range(self.x.ndim - 1)))
            - np.amin(self.x, axis=tuple(range(self.x.ndim - 1)))
        )
        self.offset_test_max = self.offset_test_max_factor * (
            np.amax(self.x, axis=tuple(range(self.x.ndim - 1)))
            - np.amin(self.x, axis=tuple(range(self.x.ndim - 1)))
        )
        self.xmin_test = np.amin(
            self.x, axis=tuple(range(self.x.ndim - 1))
        ) + self.xmin_test_factor * (
            np.amax(self.x, axis=tuple(range(self.x.ndim - 1)))
            - np.amin(self.x, axis=tuple(range(self.x.ndim - 1)))
        )
        self.xmax_test = np.amin(
            self.x, axis=tuple(range(self.x.ndim - 1))
        ) + self.xmax_test_factor * (
            np.amax(self.x, axis=tuple(range(self.x.ndim - 1)))
            - np.amin(self.x, axis=tuple(range(self.x.ndim - 1)))
        )

        # creates the x and y training and testing points
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
        ) = versatile_train_test_split_nd(self)

        return self.x_train, self.x_test, self.y_train, self.y_test


class ScaleSchemeBunch:
    # os.path.join(os.path.abspath(__file__), os.pardir)
    def __init__(
        self,
        file_name,
        orders_full,
        cmaps_str,
        potential_string,
        cutoff_string,
        dir_path="",
    ):
        """
        Information relevant to a particular scheme (regulator choice) and scale (cutoff choice).

        Parameters
        ----------
        file_name (str) : name for files that includes information about the scale and scheme.
        orders_full (int array) : array with the full range of orders for that scheme and scale.
        cmaps_str (str array) : array of matplotlib cmap objects' names corresponding to each order of coefficient.
        potential_string (str) : name of potential (scheme).
        cutoff_string (str) : name of cutoff (scale).
        dir_path (str) : path to directory where data is stored on each scale/scheme combination.
            default : ""
        """
        self.file_name = file_name
        self.orders_full = orders_full
        self.cmaps_str = cmaps_str
        self.potential_string = potential_string
        self.cutoff_string = cutoff_string
        self.name = self.potential_string + self.cutoff_string
        self.dir_path = dir_path

        self.full_path = self.dir_path + self.file_name

        # gets matplotlib cmaps and extracts colors and light colors
        self.cmaps = [plt.get_cmap(name) for name in self.cmaps_str]
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
    def __init__(
        self,
        name,
        ls_guess_factor,
        ls_bound_lower_factor,
        ls_bound_upper_factor,
        whether_fit=True,
    ):
        """
        Class for setting a guess for the Gaussian process correlation length scale and its
        bounds.

        Parameters
        ----------
        name (str) : name for the class instance.
        ls_guess_factor (float array) : ls_guess_factor * total length of the input space = initial guess for length scale.
        ls_bound_lower_factor (float array) : ls_bound_lower_factor * total length of the input space = lower bound for length scale fitting.
        ls_bound_upper_factor (float array) : ls_bound_upper_factor * total length of the input space = upper bound for length scale fitting.
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


class NSKernelParam:
    def __init__(self, param_guess, param_bounds):
        """
        Class for setting the arguments and bounds of a nonstationary Gaussian process kernel..

        Parameters
        ----------
        param_guess (float array) : initial guess for the parameter's value.
        param_bounds (float array) : bounds on fitting procedure for each parameter.
        """
        self.param_guess = param_guess
        self.param_bounds = param_bounds


class GSUMDiagnostics:
    def __init__(
        self,
        schemescale,
        observable,
        inputspace,
        traintestsplit,
        gphyperparameters,
        orderinfo,
        filenaming,
        x_quantity,
        warping_fn=None,
        warping_fn_kwargs=None,
    ):
        """
        Class for everything involving Jordan Melendez's GSUM library for observables that
        can be plotted against angle.

        Parameters
        ----------
        schemscale (ScaleSchemeBunch) : potential (with cutoff) being considered.
        observable (ObservableBunch) : observable being plotted.
        inputspace (InputSpaceBunch) : input space against which the observable is plotted.
        traintestsplit (TrainTestSplit) : training and testing masks.
        gphyperparameters (GPHyperparameters) : parameters for fitted Gaussian process.
        orderinfo (OrderInfo) : information about the EFT orders and their colors.
        filenaming (FileNaming) : strings for naming the save files.
        x_quantity (list) : information about the default quantities against which the observable is plotted, along with
            values at which it may be fixed.
            Of the form [name of the quantity (str), values at which the observable is considered (1D array),
            values at which the observable can be considered (1D array), units of the quantity (str)] * N,
            for an observable of N dimensions.
        warping_fn (callable) : function for distorting the input space.
        warping_fn_kwargs (dict) : keyword arguments for warping_fn.
        """
        # information on the scheme and scale (i.e., the potential and cutoff)
        self.schemescale = schemescale
        self.scheme = self.schemescale.potential_string
        self.scale = self.schemescale.cutoff_string

        # information on the observable
        self.observable = observable
        self.observable_name = self.observable.name
        self.observable_label = self.observable.title
        self.data_raw = self.observable.data
        self.data = self.observable.data
        self.ref_type = self.observable.ref_type
        self.nn_interaction = self.observable.nn_interaction
        self.observable_units = self.observable.unit_string
        self.constraint = self.observable.constraint

        # information on the quantity(ies) against which the observable is plotted by default
        self.x_quantity_name = []
        self.x_quantity_array = []
        self.x_quantity_full = []
        self.x_quantity_units = []
        for xq in x_quantity:
            self.x_quantity_name.append(xq[0])
            self.x_quantity_array.append(xq[1])
            self.x_quantity_full.append(xq[2])
            self.x_quantity_units.append(xq[3])
        # counts the number of dimensions in the input space that are not fixed at one value
        self.x_quantity_num = 0
        for xq in self.x_quantity_array:
            if np.shape(xq)[0] != 1:
                self.x_quantity_num += 1

        # information on the input space
        self.vs_what = np.array([])
        self.x = np.array([])
        self.X = np.array([])
        self.caption_coeffs = np.array([])
        self.title_coeffs = np.array([])
        for isp_idx, isp in enumerate(inputspace):
            self.vs_what = np.append(self.vs_what, isp.name)
            self.caption_coeffs = np.append(self.caption_coeffs, isp.caption)
            self.title_coeffs = np.append(self.title_coeffs, isp.title)

        # calculates the input space array
        if len(inputspace) == 1:
            try:
                self.x_full = gm.cartesian(
                    *[
                        isp.input_space(
                            **{
                                "deg_input": self.x_quantity_full[1],
                                "p_input": E_to_p(
                                    self.x_quantity_full[0],
                                    interaction=self.nn_interaction,
                                ),
                                "E_lab": self.x_quantity_full[0],
                                "interaction": self.nn_interaction,
                            }
                        )
                        for isp in inputspace
                    ]
                )
                self.x = gm.cartesian(
                    *[
                        isp.input_space(
                            **{
                                "deg_input": self.x_quantity_array[1],
                                "p_input": E_to_p(
                                    self.x_quantity_array[0],
                                    interaction=self.nn_interaction,
                                ),
                                "E_lab": self.x_quantity_array[0],
                                "interaction": self.nn_interaction,
                            }
                        )
                        for isp in inputspace
                    ]
                )
            except:
                self.x_full = gm.cartesian(
                    *[
                        isp.input_space(
                            **{
                                "p_input": E_to_p(
                                    self.x_quantity_full[0],
                                    interaction=self.nn_interaction,
                                ),
                                "E_lab": self.x_quantity_full[0],
                                "interaction": self.nn_interaction,
                            }
                        )
                        for isp in inputspace
                    ]
                )
                self.x = gm.cartesian(
                    *[
                        isp.input_space(
                            **{
                                "p_input": E_to_p(
                                    self.x_quantity_array[0],
                                    interaction=self.nn_interaction,
                                ),
                                "E_lab": self.x_quantity_array[0],
                                "interaction": self.nn_interaction,
                            }
                        )
                        for isp in inputspace
                    ]
                )
        elif len(inputspace) == 2:
            try:
                self.x_full = create_pairs(
                    *[
                        isp.input_space(
                            **{
                                "deg_input": self.x_quantity_full[1],
                                "p_input": E_to_p(
                                    self.x_quantity_full[0],
                                    interaction=self.nn_interaction,
                                ),
                                "E_lab": self.x_quantity_full[0],
                                "interaction": self.nn_interaction,
                            }
                        )
                        for isp in inputspace
                    ]
                )
                self.x = create_pairs(
                    *[
                        isp.input_space(
                            **{
                                "deg_input": self.x_quantity_array[1],
                                "p_input": E_to_p(
                                    self.x_quantity_array[0],
                                    interaction=self.nn_interaction,
                                ),
                                "E_lab": self.x_quantity_array[0],
                                "interaction": self.nn_interaction,
                            }
                        )
                        for isp in inputspace
                    ]
                )
            except:
                self.x_full = create_pairs(
                    *[
                        isp.input_space(
                            **{
                                "p_input": E_to_p(
                                    self.x_quantity_full[0],
                                    interaction=self.nn_interaction,
                                ),
                                "E_lab": self.x_quantity_full[0],
                                "interaction": self.nn_interaction,
                            }
                        )
                        for isp in inputspace
                    ]
                )
                self.x = create_pairs(
                    *[
                        isp.input_space(
                            **{
                                "p_input": E_to_p(
                                    self.x_quantity_array[0],
                                    interaction=self.nn_interaction,
                                ),
                                "E_lab": self.x_quantity_array[0],
                                "interaction": self.nn_interaction,
                            }
                        )
                        for isp in inputspace
                    ]
                )

        self.X = self.x[..., None]

        # reshapes the input array
        self.x = np.reshape(
            self.x,
            tuple(len(xq) for xq in self.x_quantity_array if len(xq) > 1)
            + (self.x_quantity_num,),
        )
        self.X = np.reshape(
            self.X,
            tuple(len(xq) for xq in self.x_quantity_array if len(xq) > 1)
            + (self.x_quantity_num,)
            + (1,),
        )

        # information on the train/test split
        self.traintestsplit = traintestsplit

        # creates a mask for treating the observable as fixed in at least one dimension
        mymask = functools.reduce(
            np.multiply,
            np.ix_(
                *[
                    np.isin(xqfull, xqval).astype(int)
                    for (xqfull, xqval) in zip(
                        self.x_quantity_full, self.x_quantity_array
                    )
                ]
            ),
        )
        mymask_tiled = np.tile(
            mymask, (np.shape(self.data)[0],) + (1,) * (self.data.ndim - 1)
        )
        # masks the data
        self.data = self.data[mymask_tiled.astype(bool)]
        self.data = np.reshape(
            self.data,
            (np.shape(mymask_tiled)[0],)
            + tuple([len(arr) for arr in self.x_quantity_array if len(arr) > 1]),
        )

        # applies a warping function that distorts the input space
        if warping_fn is None:
            pass
        else:
            self.x = warping_fn(self.x, **warping_fn_kwargs)

        # information on the train/test split
        self.train_pts_loc = self.traintestsplit.name
        # creates the train/test split
        self.traintestsplit.make_masks(self.x, self.data)
        self.x_train = self.traintestsplit.x_train
        self.X_train = self.x_train
        self.n_train_pts = np.shape(self.x_train)[0]
        self.x_test = self.traintestsplit.x_test
        self.X_test = self.x_test
        self.n_test_pts = np.shape(self.x_test)[0]
        self.y_train = self.traintestsplit.y_train
        self.y_test = self.traintestsplit.y_test

        # information on the GP hyperparameters
        self.gphyperparameters = gphyperparameters
        self.ls_array = self.gphyperparameters.ls_array
        self.ls_lower = self.gphyperparameters.ls_lower_array
        self.ls_upper = self.gphyperparameters.ls_upper_array
        self.whether_fit_array = self.gphyperparameters.whether_fit_array
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

        # names and labels for the orders under test. Defaults to a familiar scheme if none is specified.
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
            self.orders_labels_dict = {
                6: r"N$^{4}$LO$^{+}$",
                5: r"N$^{4}$LO",
                4: r"N$^{3}$LO",
                3: r"N$^{2}$LO",
                2: r"NLO",
            }
        else:
            self.orders_labels_dict = self.orderinfo.orders_labels_dict

        # information for naming the file
        self.filenaming = filenaming
        self.Q_param = self.filenaming.Q_param
        self.p_param = self.filenaming.p_param
        self.filename_addendum = self.filenaming.filename_addendum

        # sets the reference scale
        if self.ref_type == "dimensionless":
            self.ref = np.ones(np.shape(self.data)[1:])
            self.ref_train = np.ones(np.shape(self.x_train)[0])
            self.ref_test = np.ones(np.shape(self.x_test)[0])

        elif self.ref_type == "dimensionful":
            self.ref = self.data[-1, ...]
            if mymask.ndim <= np.squeeze(self.x).ndim:
                self.ref_train = np.squeeze(
                    griddata(
                        self.x[mymask.astype(bool)],
                        np.reshape(self.ref, (np.prod(np.shape(self.ref)),)),
                        self.x_train,
                    )
                )
                self.ref_test = np.squeeze(
                    griddata(
                        self.x[mymask.astype(bool)],
                        np.reshape(self.ref, (np.prod(np.shape(self.ref)),)),
                        self.x_test,
                    )
                )

            if mymask.ndim > np.squeeze(self.x).ndim:
                self.ref_train = np.squeeze(
                    griddata(
                        self.x,
                        np.reshape(self.ref, (np.prod(np.shape(self.ref)),)),
                        self.x_train,
                    )
                )
                self.ref_test = np.squeeze(
                    griddata(
                        self.x,
                        np.reshape(self.ref, (np.prod(np.shape(self.ref)),)),
                        self.x_test,
                    )
                )

        # sets the ratio
        if mymask.ndim <= np.squeeze(self.x).ndim:
            self.ratio = np.reshape(
                self.ratio[mymask.astype(bool)], np.shape(self.data)[1:]
            )
            self.ratio_train = np.squeeze(
                griddata(
                    self.x[mymask.astype(bool)],
                    np.reshape(self.ratio, (np.prod(np.shape(self.ratio)),)),
                    self.x_train,
                )
            )

            self.ratio_test = np.squeeze(
                griddata(
                    self.x[mymask.astype(bool)],
                    np.reshape(self.ratio, (np.prod(np.shape(self.ratio)),)),
                    self.x_test,
                )
            )

        elif mymask.ndim > np.squeeze(self.x).ndim:
            self.ratio = np.reshape(self.ratio, np.shape(self.data)[1:])
            self.ratio_train = np.squeeze(
                griddata(
                    self.x,
                    np.reshape(self.ratio, (np.prod(np.shape(self.ratio)),)),
                    self.x_train,
                )
            )

            self.ratio_test = np.squeeze(
                griddata(
                    self.x,
                    np.reshape(self.ratio, (np.prod(np.shape(self.ratio)),)),
                    self.x_test,
                )
            )

        # extracts the coefficients
        self.coeffs = gm.coefficients(
            np.reshape(
                self.data,
                (np.shape(self.data)[0],) + (np.prod(np.shape(self.data)[1:]),),
            ).T,
            ratio=np.reshape(self.ratio, (np.prod(np.shape(self.ratio)),)),
            ref=np.reshape(self.ref, (np.prod(np.shape(self.ratio)),)),
            orders=self.nn_orders_full,
        )
        self.coeffs_train = gm.coefficients(
            self.y_train.T,
            ratio=self.ratio_train,
            ref=self.ref_train,
            orders=self.nn_orders_full,
        )
        self.coeffs_test = gm.coefficients(
            self.y_test.T,
            ratio=self.ratio_test,
            ref=self.ref_test,
            orders=self.nn_orders_full,
        )

        # defines the kernel
        self.kernel = RBF(
            length_scale=self.ls_array,
            length_scale_bounds=np.array(
                [[lsl, lsu] for (lsl, lsu) in zip(self.ls_lower, self.ls_upper)]
            ),
        ) + WhiteKernel(1e-6, noise_level_bounds="fixed")

        # defines the Gaussian process (GP) object
        self.gp = gm.ConjugateGaussianProcess(
            self.kernel,
            center=self.center,
            disp=self.disp,
            df=self.df,
            scale=self.std_est,
            n_restarts_optimizer=50,
            random_state=self.seed,
            sd=self.sd,
        )

        # restricts coeffs and colors to only those orders desired for evaluating statistical diagnostics
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

        # predicts the GP over specified x values, with error bars
        if np.squeeze(self.x).ndim == 1:
            self.pred, self.std = self.gp.predict(self.x, return_std=True)
        else:
            self.pred, self.std = self.gp.predict(self.X_test, return_std=True)
        self.underlying_std = np.sqrt(self.gp.cov_factor_)

        if np.shape(self.x)[-1] == 1:
            # plots the coefficients against the given input space in 1D
            if ax is None:
                fig, ax = plt.subplots(figsize=(3.2, 2.2))

            for i, n in enumerate(self.nn_orders_full[self.mask_restricted]):
                # print(np.shape(np.squeeze(self.x)))
                # print(np.shape(self.pred[:, i]))
                # print(np.shape(self.std))
                ax.fill_between(
                    np.squeeze(self.x),
                    self.pred[:, i] + 2 * self.std,
                    self.pred[:, i] - 2 * self.std,
                    facecolor=self.light_colors[i],
                    edgecolor=self.colors[i],
                    lw=edgewidth,
                    alpha=1,
                    zorder=5 * i - 4,
                )
                ax.plot(
                    np.squeeze(self.x),
                    self.pred[:, i],
                    color=self.colors[i],
                    ls="--",
                    zorder=5 * i - 3,
                )
                ax.plot(
                    np.squeeze(self.x),
                    self.coeffs[:, i],
                    color=self.colors[i],
                    zorder=5 * i - 2,
                )
                ax.plot(
                    self.x_train,
                    self.coeffs_train[:, i],
                    color=self.colors[i],
                    ls="",
                    marker="o",
                    # label=r'$c_{}$'.format(n),
                    zorder=5 * i - 1,
                )

            # Format
            ax.axhline(2 * self.underlying_std, 0, 1, color=gray, zorder=-10, lw=1)
            ax.axhline(-2 * self.underlying_std, 0, 1, color=gray, zorder=-10, lw=1)
            ax.axhline(0, 0, 1, color=softblack, zorder=-10, lw=1)
            if np.max(np.squeeze(self.x)) < 1.1:
                ax.set_xticks(np.squeeze(self.x_test), minor=True)
                ax.set_xticks([round(xx, 1) for xx in np.squeeze(self.x_train)])
            else:
                ax.set_xticks(np.squeeze(self.x_test), minor=True)
                ax.set_xticks([round(xx, 0) for xx in np.squeeze(self.x_train)])
            ax.tick_params(which="minor", bottom=True, top=False)
            ax.set_xlabel(self.caption_coeffs[0])
            ax.set_yticks(ticks=[-2 * self.underlying_std, 2 * self.underlying_std])
            ax.set_yticklabels(
                labels=[
                    "{:.1f}".format(-2 * self.underlying_std),
                    "{:.1f}".format(2 * self.underlying_std),
                ]
            )
            ax.set_yticks([-1 * self.underlying_std, self.underlying_std], minor=True)
            ax.legend(
                # ncol=2,
                borderpad=0.4,
                # labelspacing=0.5, columnspacing=1.3,
                borderaxespad=0.6,
                loc="best",
                title=self.title_coeffs[0],
            ).set_zorder(5 * i)

            # takes constraint into account, if applicable
            if (
                self.constraint is not None
                and np.any(
                    [self.constraint[-1] == name for name in self.x_quantity_name]
                )
                and np.shape(
                    np.array(self.x_quantity_array)[
                        np.array(
                            [
                                self.constraint[-1] == name
                                for name in self.x_quantity_name
                            ]
                        )
                    ][0]
                )[0]
                != 1
                and np.shape(self.x)[-1] == 1
            ):
                dX = np.array([[np.squeeze(self.x)[i]] for i in self.constraint[0]])
                # std_interp = np.sqrt(np.diag(
                #     self.gp.cov(self.X) -
                #     self.gp.cov(self.X, dX) @ np.linalg.solve(self.gp.cov(dX, dX), self.gp.cov(dX, self.X))
                # ))
                # print(np.shape(self.x))
                # print(np.shape(dX))
                # print(np.shape(np.array(self.constraint[1])))
                _, std_interp = self.gp.predict(
                    self.x, Xc=dX, y=np.array(self.constraint[1]), return_std=True
                )
                ax.plot(
                    np.squeeze(self.x),
                    2 * std_interp,
                    color="gray",
                    ls="--",
                    zorder=-10,
                    lw=1,
                )
                ax.plot(
                    np.squeeze(self.x),
                    -2 * std_interp,
                    color="gray",
                    ls="--",
                    zorder=-10,
                    lw=1,
                )

            # draws length scales
            # ax.annotate("", xy=(np.min(self.x), -0.65 * 2 * self.underlying_std),
            #             xytext=(np.min(self.x) + self.ls, -0.65 * 2 * self.underlying_std),
            #             arrowprops=dict(arrowstyle="<->", capstyle='projecting', lw=1,
            #                             color='k', shrinkA = 0, shrinkB = 0), annotation_clip=False, zorder=5 * i)
            # ax.text(np.min(self.x) + self.ls + 0.2 * (np.max(self.x) - np.min(self.x)),
            #         -0.65 * 2 * self.underlying_std, r'$\ell_{\mathrm{guess}}$', fontsize=14,
            #         horizontalalignment='right', verticalalignment='center', zorder=5 * i)

            ax.annotate(
                "",
                xy=(np.min(self.x), -0.9 * 2 * self.underlying_std),
                xytext=(np.min(self.x) + self.ls_true, -0.9 * 2 * self.underlying_std),
                arrowprops=dict(
                    arrowstyle="<->",
                    capstyle="projecting",
                    lw=1,
                    color="k",
                    shrinkA=0,
                    shrinkB=0,
                ),
                annotation_clip=False,
                zorder=5 * i,
            )
            ax.text(
                np.min(self.x) + self.ls_true + 0.2 * (np.max(self.x) - np.min(self.x)),
                -0.9 * 2 * self.underlying_std,
                r"$\ell_{\mathrm{fit}}$",
                fontsize=14,
                horizontalalignment="right",
                verticalalignment="center",
                zorder=5 * i,
            )

            # draws standard deviations
            # ax.annotate("", xy=(np.min(self.x) + 0.90 * (np.max(self.x) - np.min(self.x)), 0),
            #             xytext=(np.min(self.x) + 0.90 * (np.max(self.x) - np.min(self.x)),
            #                     -1. * self.std_est),
            #             arrowprops=dict(arrowstyle="<->", capstyle='projecting', lw=1,
            #                             color='k', shrinkA = 0, shrinkB = 0), annotation_clip=False, zorder=5 * i)
            # ax.text(np.min(self.x) + 0.90 * (np.max(self.x) - np.min(self.x)),
            #         -1.2 * self.std_est, r'$\sigma_{\mathrm{guess}}$', fontsize=14,
            #         horizontalalignment='center', verticalalignment='bottom', zorder=5 * i)

            ax.annotate(
                "",
                xy=(np.min(self.x) + 0.74 * (np.max(self.x) - np.min(self.x)), 0),
                xytext=(
                    np.min(self.x) + 0.74 * (np.max(self.x) - np.min(self.x)),
                    -1.0 * self.underlying_std,
                ),
                arrowprops=dict(
                    arrowstyle="<->",
                    capstyle="projecting",
                    lw=1,
                    color="k",
                    shrinkA=0,
                    shrinkB=0,
                ),
                annotation_clip=False,
                zorder=5 * i,
            )
            ax.text(
                np.min(self.x) + 0.74 * (np.max(self.x) - np.min(self.x)),
                -1.2 * self.underlying_std,
                r"$\sigma_{\mathrm{fit}}$",
                fontsize=14,
                horizontalalignment="center",
                verticalalignment="bottom",
                zorder=5 * i,
            )

        elif np.shape(self.x)[-1] == 2:
            # plots the coefficients against the given input space in 2D
            fig, ax_array = plt.subplots(
                np.shape(self.coeffs)[-1],
                1,
                figsize=(3.2, np.shape(self.coeffs)[-1] * 2.2),
            )

            x_train_scatter, y_train_scatter = self.x_train.T
            x_test_scatter, y_test_scatter = self.x_test.T

            for i, n in enumerate(self.orders_restricted):
                ax_array[i].contourf(
                    self.x[..., 0],
                    self.x[..., 1],
                    np.reshape(self.coeffs[..., i], np.shape(self.x)[:-1]),
                    cmap=self.schemescale.cmaps_str[i],
                )
                ax_array[i].scatter(x_train_scatter, y_train_scatter, c="black", s=12)
                ax_array[i].scatter(x_test_scatter, y_test_scatter, c="gray", s=2)

                ax_array[i].set_xlim(
                    np.amin(self.x[..., 0])
                    - 0.03 * (np.amax(self.x[..., 0]) - np.amin(self.x[..., 0])),
                    np.amax(self.x[..., 0])
                    + 0.03 * (np.amax(self.x[..., 0]) - np.amin(self.x[..., 0])),
                )
                ax_array[i].set_ylim(
                    np.amin(self.x[..., 1])
                    - 0.03 * (np.amax(self.x[..., 1]) - np.amin(self.x[..., 1])),
                    np.amax(self.x[..., 1])
                    + 0.03 * (np.amax(self.x[..., 1]) - np.amin(self.x[..., 1])),
                )

                # plots the length scale
                ax_array[i].arrow(
                    (1.06 - 1) / (2 * 1.06),
                    (1.06 - 1) / (2 * 1.06),
                    self.ls_true[0]
                    / 1.06
                    / (np.amax(self.x[..., 0]) - np.amin(self.x[..., 0])),
                    0,
                    facecolor="black",
                    head_length=0.05,
                    shape="left",
                    width=0.01,
                    head_width=0.05,
                    length_includes_head=True,
                    transform=ax_array[i].transAxes,
                )
                ax_array[i].arrow(
                    (1.06 - 1) / (2 * 1.06),
                    (1.06 - 1) / (2 * 1.06),
                    0,
                    self.ls_true[1]
                    / 1.06
                    / (np.amax(self.x[..., 1]) - np.amin(self.x[..., 1])),
                    facecolor="black",
                    head_length=0.05,
                    shape="right",
                    width=0.01,
                    head_width=0.05,
                    length_includes_head=True,
                    transform=ax_array[i].transAxes,
                )

                ax_array[i].set_ylabel(self.caption_coeffs[1])

                ax_array[i].legend(title=r"$c_{}$".format(n), loc="upper right")

            ax_array[i].set_xlabel(self.caption_coeffs[0])
        # saves figure
        if "fig" in locals() and whether_save:
            fig.tight_layout()

            fig.savefig(
                (
                    "figures/"
                    + self.scheme
                    + "_"
                    + self.scale
                    + "/"
                    + self.observable_name
                    + "_"
                    + "interp_and_underlying_processes"
                    + "_"
                    + str(self.fixed_quantity_value)
                    + str(self.fixed_quantity_units)
                    + "_"
                    + self.scheme
                    + "_"
                    + self.scale
                    + "_Q"
                    + self.Q_param
                    + "_"
                    + self.vs_what
                    + "_"
                    + str(self.n_train_pts)
                    + "_"
                    + str(self.n_test_pts)
                    + "_"
                    + self.train_pts_loc
                    + "_"
                    + self.p_param
                    + self.filename_addendum
                ).replace("_0MeVlab_", "_")
            )

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
            if (
                self.constraint is not None
                and np.any(
                    [self.constraint[-1] == name for name in self.x_quantity_name]
                )
                and np.shape(
                    np.array(self.x_quantity_array)[
                        np.array(
                            [
                                self.constraint[-1] == name
                                for name in self.x_quantity_name
                            ]
                        )
                    ][0]
                )[0]
                != 1
                and np.shape(self.x)[-1] == 1
            ):
                dX = np.array([[np.squeeze(self.x)[i]] for i in self.constraint[0]])
                self.mean, self.cov = self.gp.predict(
                    self.x_test,
                    Xc=dX,
                    y=np.array(self.constraint[1]),
                    return_std=False,
                    return_cov=True,
                )
            else:
                self.mean = self.gp.mean(self.X_test)
                self.cov = self.gp.cov(self.X_test)
            self.gr_dgn = gm.GraphicalDiagnostic(
                self.coeffs_test,
                self.mean,
                self.cov,
                colors=self.colors,
                gray=gray,
                black=softblack,
            )

            if ax is None:
                fig, ax = plt.subplots(figsize=(1.0, 2.2))

            self.gr_dgn.md_squared(
                type="box",
                trim=False,
                title=None,
                xlabel=r"$\mathrm{D}_{\mathrm{MD}}^2$",
                ax=ax,
                **{"size": 10}
            )
            offset_xlabel(ax)
            plt.show()

            # saves figure
            if "fig" in locals() and whether_save:
                fig.tight_layout()

                fig.savefig(
                    (
                        "figures/"
                        + self.scheme
                        + "_"
                        + self.scale
                        + "/"
                        + self.observable_name
                        + "_"
                        + "md"
                        + "_"
                        + str(self.fixed_quantity_value)
                        + str(self.fixed_quantity_units)
                        + "_"
                        + self.scheme
                        + "_"
                        + self.scale
                        + "_Q"
                        + self.Q_param
                        + "_"
                        + self.vs_what
                        + "_"
                        + str(self.n_train_pts)
                        + "_"
                        + str(self.n_test_pts)
                        + "_"
                        + self.train_pts_loc
                        + "_"
                        + self.p_param
                        + self.filename_addendum
                    ).replace("_0MeVlab_", "_")
                )

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
            if (
                self.constraint is not None
                and np.any(
                    [self.constraint[-1] == name for name in self.x_quantity_name]
                )
                and np.shape(
                    np.array(self.x_quantity_array)[
                        np.array(
                            [
                                self.constraint[-1] == name
                                for name in self.x_quantity_name
                            ]
                        )
                    ][0]
                )[0]
                != 1
                and np.shape(self.x)[-1] == 1
            ):
                dX = np.array([[np.squeeze(self.x[i])] for i in self.constraint[0]])
                self.mean, self.cov = self.gp.predict(
                    self.x_test,
                    Xc=dX,
                    y=np.array(self.constraint[1]),
                    return_std=False,
                    return_cov=True,
                )
            else:
                self.mean = self.gp.mean(self.X_test)
                self.cov = self.gp.cov(self.X_test)
            self.gr_dgn = gm.GraphicalDiagnostic(
                self.coeffs_test,
                self.mean,
                self.cov,
                colors=self.colors,
                gray=gray,
                black=softblack,
            )

            with plt.rc_context({"text.usetex": True}):
                if ax is None:
                    fig, ax = plt.subplots(figsize=(3.2, 2.2))

                self.gr_dgn.pivoted_cholesky_errors(ax=ax, title=None)
                ax.set_xticks(np.arange(2, self.n_test_pts + 1, 2))
                ax.set_xticks(np.arange(1, self.n_test_pts + 1, 2), minor=True)
                ax.text(
                    0.05,
                    0.95,
                    r"$\mathrm{D}_{\mathrm{PC}}$",
                    bbox=text_bbox,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                )

                # plots legend
                legend_handles = []
                for i, n in enumerate(self.nn_orders_full[self.mask_restricted]):
                    # legend_handles.append(Patch(color=self.colors[i], label=r'$c_{}$'.format(n)))
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            label=r"$c_{}$".format(n),
                            markerfacecolor=self.colors[i],
                            markersize=8,
                        )
                    )
                ax.legend(
                    handles=legend_handles,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    handletextpad=0.02,
                    borderpad=0.2,
                )

                fig.tight_layout()
                plt.show()

                # saves figure
                if "fig" in locals() and whether_save:
                    fig.tight_layout()

                    fig.savefig(
                        (
                            "figures/"
                            + self.scheme
                            + "_"
                            + self.scale
                            + "/"
                            + self.observable_name
                            + "_"
                            + "pc_vs_index"
                            + "_"
                            + str(self.fixed_quantity_value)
                            + str(self.fixed_quantity_units)
                            + "_"
                            + self.scheme
                            + "_"
                            + self.scale
                            + "_Q"
                            + self.Q_param
                            + "_"
                            + self.vs_what
                            + "_"
                            + str(self.n_train_pts)
                            + "_"
                            + str(self.n_test_pts)
                            + "_"
                            + self.train_pts_loc
                            + "_"
                            + self.p_param
                            + self.filename_addendum
                        ).replace("_0MeVlab_", "_")
                    )

        except:
            print(
                "Error in calculating or plotting the pivoted Cholesky decomposition."
            )

    def plot_posterior_pdf(
        self, ax_joint=None, ax_marg_x=None, ax_marg_y=None, whether_save=True
    ):
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
        # this is deprecated

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
            self.gp_post = gm.TruncationGP(
                self.kernel,
                ref=lambda_interp_f_ref,
                ratio=lambda_interp_f_ratio,
                center=self.center,
                disp=self.disp,
                df=self.df,
                scale=self.std_est,
                excluded=[0],
                ratio_kws={"lambda_var": self.Lambda_b},
            )

            # takes account for the constraint, if applicable
            if (
                self.constraint is not None
                and self.constraint[2] == self.x_quantity_name
            ):
                self.gp_post.fit(
                    self.X_train,
                    self.y_train,
                    orders=self.nn_orders_full,
                    orders_eval=self.nn_orders,
                    dX=np.array([[self.x[i]] for i in self.constraint[0]]),
                    dy=[j for j in self.constraint[1]],
                )
            else:
                self.gp_post.fit(
                    self.X_train,
                    self.y_train,
                    orders=self.nn_orders_full,
                    orders_eval=self.nn_orders,
                )

            # evaluates the probability across the mesh
            self.ls_lambda_loglike = np.array(
                [
                    [
                        self.gp_post.log_marginal_likelihood(
                            [
                                ls_,
                            ],
                            orders_eval=self.nn_orders,
                            **{"lambda_var": lambda_}
                        )
                        for ls_ in np.log(self.ls_vals)
                    ]
                    for lambda_ in self.lambda_vals
                ]
            )

            # Makes sure that the values don't get too big or too small
            self.ls_lambda_like = np.exp(
                self.ls_lambda_loglike - np.max(self.ls_lambda_loglike)
            )

            # Now compute the marginal distributions
            self.lambda_like = np.trapz(self.ls_lambda_like, x=self.ls_vals, axis=-1)
            self.ls_like = np.trapz(self.ls_lambda_like, x=self.lambda_vals, axis=0)

            # Normalize them
            self.lambda_like /= np.trapz(self.lambda_like, x=self.lambda_vals, axis=0)
            self.ls_like /= np.trapz(self.ls_like, x=self.ls_vals, axis=0)

            with plt.rc_context({"text.usetex": True, "text.latex.preview": True}):
                cmap_name = "Blues"
                cmap = mpl.cm.get_cmap(cmap_name)

                # Setup axes
                if ax_joint == None and ax_marg_x == None and ax_marg_y == None:
                    fig, ax_joint, ax_marg_x, ax_marg_y = joint_plot(
                        ratio=5, height=3.4
                    )

                # Plot contour
                ax_joint.contour(
                    self.ls_vals,
                    self.lambda_vals,
                    self.ls_lambda_like,
                    levels=[np.exp(-0.5 * r**2) for r in np.arange(9, 0, -0.5)]
                    + [0.999],
                    cmap=cmap_name,
                    vmin=-0.05,
                    vmax=0.8,
                    zorder=1,
                )

                # Now plot the marginal distributions
                ax_marg_y.plot(self.lambda_like, self.lambda_vals, c=cmap(0.8), lw=1)
                ax_marg_y.fill_betweenx(
                    self.lambda_vals,
                    np.zeros_like(self.lambda_like),
                    self.lambda_like,
                    facecolor=cmap(0.2),
                    lw=1,
                )
                ax_marg_x.plot(self.ls_vals, self.ls_like, c=cmap(0.8), lw=1)
                ax_marg_x.fill_between(
                    self.ls_vals,
                    np.zeros_like(self.ls_vals),
                    self.ls_like,
                    facecolor=cmap(0.2),
                    lw=1,
                )

                # Formatting
                ax_joint.set_xlabel(r"$\ell$")
                ax_joint.set_ylabel(r"$\Lambda$")
                ax_joint.axvline(self.ls, 0, 1, c=gray, lw=1, zorder=0)
                ax_joint.axhline(self.Lambda_b, 0, 1, c=gray, lw=1, zorder=0)
                ax_joint.margins(x=0, y=0.0)
                ax_joint.set_xlim(min(self.ls_vals), max(self.ls_vals))
                ax_joint.set_ylim(min(self.lambda_vals), max(self.lambda_vals))
                ax_marg_x.set_ylim(bottom=0)
                ax_marg_y.set_xlim(left=0)
                ax_joint.text(
                    0.95,
                    0.95,
                    r"pr$(\ell, \Lambda \,|\, \vec{\mathbf{y}}_k)$",
                    ha="right",
                    va="top",
                    transform=ax_joint.transAxes,
                    bbox=text_bbox,
                    fontsize=12,
                )

                plt.show()

                if "fig" in locals() and whether_save:
                    fig.savefig(
                        (
                            "figures/"
                            + self.scheme
                            + "_"
                            + self.scale
                            + "/"
                            + self.observable_name
                            + "_"
                            + "Lambda_ell_jointplot"
                            + "_"
                            + str(self.fixed_quantity_value)
                            + str(self.fixed_quantity_units)
                            + "_"
                            + self.scheme
                            + "_"
                            + self.scale
                            + "_Q"
                            + self.Q_param
                            + "_"
                            + self.vs_what
                            + "_"
                            + str(self.n_train_pts)
                            + "_"
                            + str(self.n_test_pts)
                            + "_"
                            + self.train_pts_loc
                            + "_"
                            + self.p_param
                            + self.filename_addendum
                        ).replace("_0MeVlab_", "_")
                    )

        except:
            print("Error in plotting the posterior PDF.")

    def plot_truncation_errors(
        self, online_data, residual_plot=True, whether_save=True
    ):
        """
        Plots the experimental vs. theoretical (with error bars) observable values, or the residuals of these two
        quantities; and the corresponding empirical coverage ("weather") plot.

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
        print("self.online_data = " + str(self.online_data))

        # functions for reference scale and dimensionless expansion parameter (ratio)
        # def lambda_interp_f_ref(x_):
        #     X = np.ravel(x_)
        #     return self.interp_f_ref(X)
        #
        # def lambda_interp_f_ratio(x_, lambda_var):
        #     X = np.ravel(x_)
        #     return self.interp_f_ratio(X) * self.Lambda_b / lambda_var

        def interp_f_ratio(x_interp):
            X = np.reshape(x_interp, (np.prod(np.shape(x_interp)[:-1]),))
            return griddata(self.x, self.ratio, X)

        def interp_f_ref(x_interp):
            X = np.reshape(x_interp, (np.prod(np.shape(x_interp)[:-1]),))
            return griddata(self.x, self.ref, X)

        # try:
        # creates the TruncationGP object
        print("self.ratio has shape " + str(np.shape(self.ratio)))
        self.gp_trunc = gm.TruncationGP(
            self.kernel,
            ref=interp_f_ref,
            ratio=interp_f_ratio,
            center=self.center,
            disp=self.disp,
            df=self.df,
            scale=self.std_est,
            excluded=self.excluded,
            ratio_kws={},
        )

        # fits the GP with or without a constraint
        # if self.constraint is not None and self.constraint[2] == self.x_quantity_name:
        #     self.gp_trunc.fit(self.X_train, self.y_train,
        #                       orders=self.nn_orders_full,
        #                       # orders_eval=self.nn_orders,
        #                       dX=np.array([[self.x[i]] for i in self.constraint[0]]),
        #                       dy=[j for j in self.constraint[1]])
        # else:
        print("self.x_train = " + str(self.x_train))
        print("self.y_train = " + str(self.y_train))
        self.gp_trunc.fit(
            self.x_train,
            self.y_train.T,
            orders=self.nn_orders_full,
            # orders_eval=self.nn_orders
        )

        # creates fig with two columns of axes
        fig, axes = plt.subplots(
            int(np.ceil(len(self.nn_orders_full[self.mask_restricted]) / 2)),
            2,
            sharex=True,
            sharey=True,
            figsize=(3.2, 4),
        )
        # deletes extraneous axes to suit number of evaluated orders
        if 2 * np.ceil(len(self.nn_orders_full[self.mask_restricted]) / 2) > len(
            self.nn_orders_full[self.mask_restricted]
        ):
            fig.delaxes(
                axes[
                    int(np.ceil(len(self.nn_orders_full[self.mask_restricted]) / 2))
                    - 1,
                    1,
                ]
            )

        for i, n in enumerate(self.nn_orders_full[self.mask_restricted]):
            # calculates the standard deviation of the truncation error
            # print("self.X has shape " + str(np.shape(self.X)))
            _, self.std_trunc = self.gp_trunc.predict(
                self.x, order=n, return_std=True, kind="trunc"
            )
            print("self.std_trunc = " + str(self.std_trunc))
            if i == 0:
                std_trunc0 = self.std_trunc

            # gets the "true" order-by-order data from online
            # if self.fixed_quantity_name == "energy":
            #     data_true = self.online_data[self.fixed_quantity_value, :]
            # elif self.fixed_quantity_name == "angle":
            #     if self.fixed_quantity_value == 0:
            #         data_true = self.online_data
            #     else:
            #         data_true = self.online_data[:, self.fixed_quantity_value]
            print("self.online_data has shape " + str(np.shape(self.online_data)))
            data_true = self.online_data
            print("data_true has shape " + str(np.shape(data_true)))

            for j in range(i, len(self.nn_orders_full[self.mask_restricted])):
                ax = axes.ravel()[j]

                # number of standard deviations around the dotted line to plot
                # 0.5 corresponds to 68% confidence intervals, and 1 to 95%
                std_coverage = 0.5

                if residual_plot:
                    # calculates and plots the residuals
                    print("self.data has shape " + str(np.shape(self.data)))
                    print(
                        "self.mask_restricted has shape "
                        + str(np.shape(self.mask_restricted))
                    )
                    # residual = data_true - (self.data[:, self.mask_restricted])[:, i]
                    residual = data_true - (self.data[self.mask_restricted, :])[i, :]
                    # print("residual = " + str(residual))
                    ax.plot(
                        np.squeeze(self.x), residual, zorder=i - 4, c=self.colors[i]
                    )
                    ax.fill_between(
                        np.squeeze(self.x),
                        residual + std_coverage * self.std_trunc,
                        residual - std_coverage * self.std_trunc,
                        zorder=i - 5,
                        facecolor=self.light_colors[i],
                        edgecolor=self.colors[i],
                        lw=edgewidth,
                    )
                    # ax.set_ylim(np.min(np.concatenate(
                    #     (residual + std_coverage * self.std_trunc, residual - std_coverage * self.std_trunc))),
                    #             np.max(np.concatenate((residual + std_coverage * self.std_trunc,
                    #                                    residual - std_coverage * self.std_trunc))))
                    ax.set_ylim(
                        np.min(
                            np.concatenate(
                                (
                                    data_true
                                    - (self.data[self.mask_restricted, :])[1, :]
                                    + std_coverage * std_trunc0 / 2,
                                    data_true
                                    - (self.data[self.mask_restricted, :])[1, :]
                                    - std_coverage * std_trunc0 / 2,
                                )
                            )
                        ),
                        np.max(
                            np.concatenate(
                                (
                                    data_true
                                    - (self.data[self.mask_restricted, :])[1, :]
                                    + std_coverage * std_trunc0 / 2,
                                    data_true
                                    - (self.data[self.mask_restricted, :])[1, :]
                                    - std_coverage * std_trunc0 / 2,
                                )
                            )
                        ),
                    )

                else:
                    # calculates and plots the true data
                    ax.plot(
                        np.squeeze(self.x),
                        (self.data[self.mask_restricted, :])[i, :],
                        zorder=i - 5,
                        c=self.colors[i],
                    )
                    ax.fill_between(
                        np.squeeze(self.x),
                        (self.data[self.mask_restricted, :])[i, :]
                        + std_coverage * self.std_trunc,
                        (self.data[self.mask_restricted, :])[i, :]
                        - std_coverage * self.std_trunc,
                        zorder=i - 5,
                        facecolor=self.light_colors[i],
                        edgecolor=self.colors[i],
                        lw=edgewidth,
                    )
                    ax.set_ylim(
                        np.min(
                            np.concatenate(
                                (
                                    (self.data[self.mask_restricted, :])[:, i]
                                    + std_coverage * self.std_trunc,
                                    (self.data[self.mask_restricted, :])[i, :]
                                    - std_coverage * self.std_trunc,
                                )
                            )
                        ),
                        np.max(
                            np.concatenate(
                                (
                                    (self.data[self.mask_restricted, :])[:, i]
                                    + std_coverage * self.std_trunc,
                                    (self.data[self.mask_restricted, :])[i, :]
                                    - std_coverage * self.std_trunc,
                                )
                            )
                        ),
                    )

                # # plots the testing points as vertical lines
                # for line in self.x_test: ax.axvline(line, 0, 1, c = gray)

            ax = axes.ravel()[i]

            if residual_plot:
                # plots a line at y = 0
                ax.plot(self.x, np.zeros(len(self.x)), color=softblack, lw=1, ls="--")
            else:
                # plots the true data
                ax.plot(self.x, data_true, color=softblack, lw=1, ls="--")

            # formats x-axis labels and tick marks
            # ax.set_xlabel(self.caption_coeffs)
            ax.set_xticks([int(tick) for tick in np.squeeze(self.x_train)])
            ax.set_xticks([tick for tick in np.squeeze(self.x_test)], minor=True)
        fig.supxlabel(self.caption_coeffs[0], fontsize=12)
        fig.supylabel(
            r"$["
            + self.observable_label
            + "]_{\mathrm{res}}$ ("
            + self.observable_units
            + ")",
            fontsize=12,
        )
        plt.show()

        # saves
        if "fig" in locals() and whether_save:
            # fig.suptitle(r'$\mathrm{' + self.observable_name + '\,(' + str(self.fixed_quantity_value) + '\,' + str(
            #     self.fixed_quantity_units) + ')\,' + \
            #              '\,for\,' + self.scheme + '\,' + self.scale + '}' + '\,(Q_{\mathrm{' + self.Q_param + \
            #              '}},\,\mathrm{' + self.p_param + '},\,\mathrm{' + self.vs_what + '})$', size=20)
            fig.tight_layout()

            if self.constraint is None:
                fig.savefig(
                    (
                        "figures/"
                        + self.scheme
                        + "_"
                        + self.scale
                        + "/"
                        + self.observable_name
                        + "_"
                        + str(self.fixed_quantity_value)
                        + str(self.fixed_quantity_units)
                        + "_"
                        + "full_pred_truncation"
                        + "_"
                        + self.scheme
                        + "_"
                        + self.scale
                        + "_Q"
                        + self.Q_param
                        + "_"
                        + self.vs_what
                        + "_"
                        + str(self.n_train_pts)
                        + "_"
                        + str(self.n_test_pts)
                        + "_"
                        + self.train_pts_loc
                        + "_"
                        + self.p_param
                        + self.filename_addendum
                    ).replace("_0MeVlab_", "_")
                )
            else:
                fig.savefig(
                    (
                        "figures/"
                        + self.scheme
                        + "_"
                        + self.scale
                        + "/"
                        + self.observable_name
                        + "_"
                        + str(self.fixed_quantity_value)
                        + str(self.fixed_quantity_units)
                        + "_"
                        + "full_pred_truncation_constrained"
                        + "_"
                        + self.scheme
                        + "_"
                        + self.scale
                        + "_Q"
                        + self.Q_param
                        + "_"
                        + self.vs_what
                        + "_"
                        + str(self.n_train_pts)
                        + "_"
                        + str(self.n_test_pts)
                        + "_"
                        + self.train_pts_loc
                        + "_"
                        + self.p_param
                        + self.filename_addendum
                    ).replace("_0MeVlab_", "_")
                )

        # creates interpolation function for the true and theory data
        # data_interp = interp1d(self.x, self.data[self.mask_restricted, :].T)
        # data_true_interp = interp1d(self.x, data_true)
        # data_interp = interp1d(self.x, self.data[self.mask_restricted, :].T)
        # data_true_interp = interp1d(self.x, data_true)

        # calculates the covariance matrix and mean
        self.cov_wp = self.gp_trunc.cov(self.x_test, start=0, end=np.inf)
        self.mean_wp = self.gp_trunc.mean(self.x_test)

        # norms the residuals by factors of the ratio
        # self.norm_residuals_wp = data_true_interp(self.X_test) - data_interp(self.X_test)
        print("self.x has shape " + str(np.shape(self.x)))
        print("self.x_test has shape " + str(np.shape(self.x_test)))
        print("data_true has shape " + str(np.shape(data_true)))
        print("self.data has shape " + str(np.shape(self.data)))
        print("self.mask_restricted has shape " + str(np.shape(self.mask_restricted)))
        # print(np.shape(griddata(self.x, data_true, self.x_test)))
        # print(np.shape(griddata(self.x, self.data, self.x_test)))
        self.norm_residuals_wp = np.array([])
        for i in range(len(self.nn_orders_full[self.mask_restricted])):
            self.norm_residuals_wp = np.append(
                self.norm_residuals_wp,
                griddata(self.x, data_true, self.x_test)
                - griddata(
                    self.x, self.data[self.mask_restricted, :][i, :], self.x_test
                ),
            )
        print(
            "self.norm_residuals_wp has shape " + str(np.shape(self.norm_residuals_wp))
        )
        self.norm_residuals_wp = np.reshape(
            self.norm_residuals_wp,
            (len(self.nn_orders_full[self.mask_restricted]),)
            + (np.shape(self.x_test)[0],),
        )
        denom = (
            np.tile(
                self.ratio_test, (len(self.nn_orders_full[self.mask_restricted]), 1)
            ).T
        ) ** (self.nn_orders_full[self.mask_restricted] + 1) * (
            np.sqrt(
                1
                - np.tile(
                    self.ratio_test, (len(self.nn_orders_full[self.mask_restricted]), 1)
                )
                ** 2
            )
        ).T
        self.norm_residuals_wp = self.norm_residuals_wp / (denom.T)
        self.gr_dgn_wp = gm.GraphicalDiagnostic(
            self.norm_residuals_wp.T,
            mean=self.mean_wp,
            cov=self.cov_wp,
            colors=self.colors,
            gray=gray,
            black=softblack,
        )

        fig, ax = plt.subplots(figsize=(3.4, 3.2))

        # creates the empirical coverage plot
        self.gr_dgn_wp.credible_interval(
            np.linspace(1e-5, 1, 100),
            band_perc=[0.68, 0.95],
            ax=ax,
            # title="Empirical coverage (PWA93)\n" +
            #       r'$\mathrm{' + self.observable_name + '\,(' + str(self.fixed_quantity_value) + '\,' + str(
            #     self.fixed_quantity_units) + ')\,' + \
            #       '\,for\,' + self.scheme + '\,' + self.scale + '}' + '\,(Q_{\mathrm{' + self.Q_param + \
            #       '}},\,\mathrm{' + self.p_param + '},\,\mathrm{' + self.vs_what + '})$',
            xlabel=r"Credible Interval ($100\alpha\%$)",
            # ylabel=r'Empirical Coverage ($\%$)\,(N = ' + str(len(self.X_test)) + r')')
            ylabel=r"Empirical Coverage ($\%$)",
        )

        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xticklabels([0, 20, 40, 60, 80, 100])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticklabels([0, 20, 40, 60, 80, 100])

        plt.show()

        # saves the figure
        if "fig" in locals() and whether_save:
            fig.tight_layout()

            fig.savefig(
                (
                    "figures/"
                    + self.scheme
                    + "_"
                    + self.scale
                    + "/"
                    + self.observable_name
                    + "_"
                    + str(self.fixed_quantity_value)
                    + str(self.fixed_quantity_units)
                    + "_"
                    + "truncation_error_empirical_coverage"
                    + "_"
                    + self.scheme
                    + "_"
                    + self.scale
                    + "_Q"
                    + self.Q_param
                    + "_"
                    + self.vs_what
                    + "_"
                    + str(self.n_train_pts)
                    + "_"
                    + str(self.n_test_pts)
                    + "_"
                    + self.train_pts_loc
                    + "_"
                    + self.p_param
                    + self.filename_addendum
                ).replace("_0MeVlab_", "_")
            )

        # except:
        #     print("Error plotting the truncation errors.")

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
            if (
                self.constraint is not None
                and self.constraint[2] == self.x_quantity_name
            ):
                dX = np.array([[self.x[i]] for i in self.constraint[0]])
                self.mean, self.cov = self.gp.predict(
                    self.X_test,
                    Xc=dX,
                    y=np.array(self.constraint[1]),
                    return_std=False,
                    return_cov=True,
                )
            else:
                self.mean = self.gp.mean(self.X_test)
                self.cov = self.gp.cov(self.X_test)

            self.gr_dgn = gm.GraphicalDiagnostic(
                self.coeffs_test,
                self.mean,
                self.cov,
                colors=self.colors,
                gray=gray,
                black=softblack,
            )

            if ax is None:
                fig, ax = plt.subplots(figsize=(3.2, 2.2))

            self.gr_dgn.credible_interval(
                np.linspace(1e-5, 1, 100),
                band_perc=[0.68, 0.95],
                ax=ax,
                title=None,
                xlabel=r"Credible Interval ($100\alpha\%$)",
                # ylabel=r'Empirical Coverage ($\%$)\,(N = ' + str(len(self.X_test)) + r')')
                ylabel=r"Empirical Coverage ($\%$)",
            )

            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_xticklabels([0, 20, 40, 60, 80, 100])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_yticklabels([0, 20, 40, 60, 80, 100])

            plt.show()

            # saves figure
            if "fig" in locals() and whether_save:
                fig.tight_layout()

                fig.savefig(
                    (
                        "figures/"
                        + self.scheme
                        + "_"
                        + self.scale
                        + "/"
                        + self.observable_name
                        + "_"
                        + str(self.fixed_quantity_value)
                        + str(self.fixed_quantity_units)
                        + "_"
                        + "truncation_error_credible_intervals"
                        + "_"
                        + self.scheme
                        + "_"
                        + self.scale
                        + "_Q"
                        + self.Q_param
                        + "_"
                        + self.vs_what
                        + "_"
                        + str(self.n_train_pts)
                        + "_"
                        + str(self.n_test_pts)
                        + "_"
                        + self.train_pts_loc
                        + "_"
                        + self.p_param
                        + self.filename_addendum
                    ).replace("_0MeVlab_", "_")
                )

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

        gs = mpl.gridspec.GridSpec(
            ncols=30, nrows=24, wspace=200, hspace=400, figure=fig_main
        )

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
            print(
                "Error in calculating or plotting the pivoted Cholesky decomposition."
            )
        try:
            self.plot_credible_intervals(ax=ax_ci, whether_save=True)
        except:
            print("Error in calculating or plotting the credible intervals.")
        try:
            self.plot_posterior_pdf(
                ax_joint=ax_pdf_joint,
                ax_marg_x=ax_pdf_x,
                ax_marg_y=ax_pdf_y,
                whether_save=True,
            )
        except:
            print("Error in calculating or plotting the posterior PDF.")

        # adds a title
        fig_main.suptitle(
            r"$\mathrm{"
            + self.observable_name
            + "\,("
            + str(self.fixed_quantity_value)
            + "\,"
            + str(self.fixed_quantity_units)
            + ")\,"
            + "\,for\,"
            + self.scheme
            + "\,"
            + self.scale
            + "}"
            + "\,(Q_{\mathrm{"
            + self.Q_param
            + "}},\,\mathrm{"
            + self.p_param
            + "},\,\mathrm{"
            + self.vs_what
            + "})$",
            size=30,
        )

        if whether_save:
            fig_main.savefig(
                (
                    "figures/"
                    + self.scheme
                    + "_"
                    + self.scale
                    + "/"
                    + self.observable_name
                    + "_"
                    + "plotzilla"
                    + "_"
                    + str(self.fixed_quantity_value)
                    + str(self.fixed_quantity_units)
                    + "_"
                    + self.scheme
                    + "_"
                    + self.scale
                    + "_Q"
                    + self.Q_param
                    + "_"
                    + self.vs_what
                    + "_"
                    + str(self.n_train_pts)
                    + "_"
                    + str(self.n_test_pts)
                    + "_"
                    + self.train_pts_loc
                    + "_"
                    + self.p_param
                    + self.filename_addendum
                ).replace("_0MeVlab_", "_")
            )


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
    return interpn(
        x_interp, Q_approx(p, Q_param, Lambda_b=lambda_var, m_pi=mpi_var), x_map
    )


def ratio_fn_curvewise(
    X,
    p_grid_train,
    p_param,
    p_shape,
    Q_param,
    mpi_var,
    lambda_var,
    single_expansion=False,
):
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
    # print("p_grid_train in ratio_fn = " + str(p_grid_train))
    p = np.array([])
    for pt in p_grid_train:
        # print("pt = " + str(pt))
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

    return Q_approx(
        p=np.reshape(p, p_shape),
        Q_parametrization=Q_param,
        Lambda_b=lambda_var,
        m_pi=mpi_var,
        single_expansion=single_expansion,
    )


def ratio_fn_constant(X, p_shape, p_grid_train, Q):
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


def make_likelihood_filename(
    FileNameObj,
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
    FileNameObj (FileNaming) : FileNaming object with information on naming files.
    folder (str) : folder name.
    observable_name (str) : observable name.
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

    # for logprior, random_var in zip(logpriors_names, random_vars_array):
    #     filename += (
    #         "_"
    #         + str(random_var.name)
    #         + "_"
    #         + str(logprior)
    #         + "_"
    #         + str(len(random_var.var))
    #         + "pts"
    #     )

    print(filename)

    return str(filename.replace("__", "_") + FileNameObj.filename_addendum + ".txt")


def calc_loglike_ray(
    mesh_cart,
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
        batch = mesh_cart[i : i + batch_size]
        log_like_ids.append(
            log_likelihood.remote(
                gp_post,
                batch,
                log_likelihood_fn_kwargs,
            )
        )
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
            logprior_shape_tuple = (
                np.shape(obs_loglike)[(i + lst) % len(variables_array)],
            ) + logprior_shape_tuple
        obs_loglike += np.transpose(
            np.tile(logprior, logprior_shape_tuple),
            np.roll(np.arange(0, len(variables_array), dtype=int), i + 1),
        )

    return obs_loglike


def marginalize_likelihoods(variables_array, like_list):
    """
    Marginalizes likelihoods into all possible 1- and 2-d posteriors.

    Parameters
    ----------
    variables_array (RandomVariable NumPu array) : list of RandomVariable objects.
    like_list (array) : list of likelihoods.

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
            # creates an array of indices for marginalization
            var_idx_array = np.arange(0, np.shape(variables_array)[0], 1, dtype=int)
            var_idx_array = var_idx_array[var_idx_array != v]
            var_idx_array = np.flip(var_idx_array)

            marg_post = np.copy(like)

            # marginalizes by integrating over all indices but one
            for idx in var_idx_array:
                marg_post = np.trapz(marg_post, x=variables_array[idx].var, axis=idx)

            # normalizes the marginalized distributions
            marg_post /= np.trapz(marg_post, x=variables_array[v].var, axis=0)

            # adds marginalized and normalized posterior to list
            marg_post_list.append(list(marg_post))

        # creates an array of arrays of indices over which to marginalize for the joint posteriors
        if np.shape(variables_array)[0] > 1:
            comb_array = []
            for ca in range(1, np.shape(variables_array)[0]):
                for ca_less in range(0, ca):
                    comb_array.append([ca, ca_less])
            comb_array = np.flip(np.array(comb_array), axis=1)
        else:
            comb_array = np.array([0, 0])

        # marginalizes and normalizes the joint posteriors
        for v_norm, v_marg in zip(
            comb_array,
            np.flip(
                np.array(
                    [
                        np.arange(0, np.shape(variables_array)[0], 1, dtype=int)[
                            ~np.isin(
                                np.arange(
                                    0, np.shape(variables_array)[0], 1, dtype=int
                                ),
                                c,
                            )
                        ]
                        for c in comb_array
                    ]
                ),
                axis=1,
            ),
        ):
            if like.ndim > 2:
                joint_post = np.trapz(
                    like, x=variables_array[v_marg[0]].var, axis=v_marg[0]
                )

                if like.ndim > 3:
                    for vmarg in v_marg[1:]:
                        joint_post = np.trapz(
                            joint_post, x=variables_array[vmarg].var, axis=vmarg
                        )
                joint_post /= np.trapz(
                    np.trapz(joint_post, x=variables_array[v_norm[1]].var, axis=1),
                    x=variables_array[v_norm[0]].var,
                    axis=0,
                )
            else:
                joint_post = like

            # appends the result to a list
            joint_post_list.append(joint_post)

    # reshapes the fully marginalized posterior list
    marg_post_array = np.reshape(
        marg_post_list,
        (len(variables_array), np.shape(like_list)[0]) + np.shape(marg_post_list)[1:],
        order="F",
    )

    joint_post_array = np.array(joint_post_list)

    return marg_post_array, joint_post_array


@ray.remote
def log_likelihood(gp_fitted, mesh_points, log_likelihood_fn_kwargs):
    """
    Calculates the log-likelihood for a set of inputs.

    Parameters
    ----------
    gp_fitted (TruncationGP) : fitted Gaussian process object.
    mesh_points (float array) : array of tuples of random variables at which to evaluate the log-likelihood.
        Must be in the order (lambda_var, all length scales, mpi_var).
    log_likelihood_fn_kwargs (dict) : keyword arguments for log_likelihood.
    """
    return [
        gp_fitted.log_marginal_likelihood(
            [pt[1 + n] for n in range(len(pt) - 2)],
            **{**log_likelihood_fn_kwargs, **{"mpi_var": pt[-1], "lambda_var": pt[0]}}
        )
        for pt in mesh_points
    ]


@ray.remote
def log_likelihood_const(gp_fitted, mesh_points, log_likelihood_fn_kwargs):
    """
    Function for interpolating calculating the log-likelihood for a fitted TrunctionTP object.
    Specifically, this is for cases with random variables (Q, ell_degrees).
    Parameters
    ----------
    gp_fitted (TruncationTP) : Student t-distribution object from GSUM.
    mesh_points (array) : array over which evaluation takes place.
    log_likelihood_fn_kwargs (dict) : kwargs for evaluation.
    """
    return [
        gp_fitted.log_marginal_likelihood(
            [pt[1 + n] for n in range(len(pt) - 1)],
            **{**log_likelihood_fn_kwargs, **{"Q": pt[0]}}
        )
        for pt in mesh_points
    ]


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
    degrees,
    degrees_train_pts,
    InputSpaceDeg,
    length_scale_list,
    cbar_list,
    variables_array,
    mom_fn,
    mom_fn_kwargs,
    ratio_fn,
    ratio_fn_kwargs,
    log_likelihood_fn,
    log_likelihood_fn_kwargs,
    warping_fn=None,
    warping_fn_kwargs=None,
    cbar_fn=None,
    cbar_fn_kwargs=None,
    scaling_fn=None,
    scaling_fn_kwargs=None,
    orders=2,
    FileName=None,
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
    degrees (array) : array of scattering-angle points for evaluation.
    degrees_train_pts (array) : list of scattering-angle training points for evaluation.
    InputSpaceDeg : object encoding information about scattering-angle input space
    length_scale_list (NSKernelParam list) : list of initial guesses and bounds for the NSRBF parameters.
    variables_array (RandomVariable array) : list of RandomVariable objects for each random variable.
        Must be in the order (Lambda_b, scattering-angle length scale, lab-energy length scale, mpi_eff).

    mom_fn (function) : function for converting from lab energy to relative momentum.
    mom_fn_kwargs (dict) : keyword arguments for mom_fn.

    warping_fn (function) : function for scaling input space.
    warping_fn_kwargs (dict) : keyword arguments for warping_fn.

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
    whether_plot_corner (bool) : whether to plot corner plot.
        Default : True
    whether_use_data (bool) : whether to use already saved data for plotting.
        Default : True
    whether_save_data (bool) : whether to save data.
        Default : True
    whether_save_plots (bool) : whether to save plots.
        Default : True
    """

    # sets the number of orders and the corresponding colors
    order_num = int(orders)
    Lb_colors = light_colors[-1 * order_num :]

    # creates boolean array for treatment of the length scale (and any other variables) on an observable-by-
    # observable basis instead of a cross-observable basis
    marg_bool_array = np.array([v.marg_bool for v in variables_array])

    # initiates the ray kernel for utilizing all processors on the laptop
    ray.shutdown()
    ray.init()

    # batch size for Ray
    BATCH_SIZE = 100

    # list for appending log-likelihoods
    like_list = []

    # sorts out case when warping_fn = None
    if warping_fn is None:
        warping_fn = lambda warp: warp
        warping_fn_kwargs = {}

    for obs_grouping, obs_name, mesh_cart_group in zip(
        obs_data_grouped_list, obs_name_grouped_list, mesh_cart_grouped_list
    ):
        # loops through the orders of interest
        for order_counter in range(1, order_num + 1):
            order = np.max(nn_orders_array) - order_num + order_counter

            try:
                # generates names for files and searches for whether they exist
                if not whether_use_data:
                    raise ValueError("You elected not to use saved data.")
                else:
                    # if they exist, they are read in, reshaped, and appended to like_list
                    like_list.append(
                        np.reshape(
                            np.loadtxt(
                                make_likelihood_filename(
                                    FileName,
                                    "data",
                                    obs_name,
                                    orders_names_dict[order],
                                    [
                                        variable.logprior_name
                                        for variable in variables_array
                                    ],
                                    variables_array,
                                )
                            ),
                            tuple(
                                [
                                    len(random_var.var)
                                    for random_var in variables_array[marg_bool_array]
                                ]
                            ),
                        )
                    )

            except:
                # failing that, generates new data and saves it (if the user chooses)
                obs_loglike_sum = np.zeros(
                    tuple(
                        len(random_var.var)
                        for random_var in variables_array[marg_bool_array]
                    )
                )

                for obs_object, mesh_cart in zip(obs_grouping, mesh_cart_group):
                    # observable data
                    obs_data_full = obs_object.data

                    # sets yref depending on whether the observable is dimensionful or dimensionless
                    yref_type = obs_object.ref_type

                    # initializes kernel
                    kernel_posterior = NSRBF(
                        length_scale=tuple(
                            [LS.param_guess for LS in length_scale_list]
                        ),
                        length_scale_bounds=tuple(
                            [tuple(LS.param_bounds) for LS in length_scale_list]
                        ),
                        length_scale_fn=scaling_fn,
                        length_scale_fn_kwargs=scaling_fn_kwargs,
                        cbar=tuple(
                            [CB.param_guess for CB in cbar_list]
                        ),
                        cbar_bounds=tuple(
                            [tuple(CB.param_bounds) for CB in cbar_list]
                        ),
                        cbar_fn=cbar_fn,
                        cbar_fn_kwargs=cbar_fn_kwargs,
                    ) + NSWhiteKernel(1e-6, noise_level_bounds="fixed")

                    if len(np.shape(obs_data_full)) == 2:
                        # 1D observables
                        if np.shape(obs_data_full)[1] == len(degrees):
                            # observables that depend only on scattering angle (of which none exist)
                            # doesn't quite work since it needs a momentum

                            # converts the points in degrees to the current input space
                            degrees_input = InputSpaceDeg.input_space(
                                **{"deg_input": degrees}
                            )
                            degrees_train_pts_input = InputSpaceDeg.input_space(
                                **{"deg_input": degrees_train_pts}
                            )

                            # warps the input space
                            input_space_warped = warping_fn(
                                np.reshape(
                                    gm.cartesian(
                                        *[
                                            degrees_input,
                                        ]
                                    ),
                                    (len(degrees), 1),
                                ),
                                **warping_fn_kwargs
                            )

                            # creates grids for training points and the corresponding momenta at those points
                            grid_train = degrees_train_pts_input
                            p_grid_train = degrees_train_pts

                            p_grid_train = p_grid_train[
                                [
                                    (
                                        pt >= np.min(input_space_warped[:, 0])
                                        and pt <= np.max(input_space_warped[:, 0])
                                    )
                                    for pt in grid_train
                                ]
                            ][:, None]
                            grid_train = grid_train[
                                [
                                    (
                                        pt >= np.min(input_space_warped[:, 0])
                                        and pt <= np.max(input_space_warped[:, 0])
                                    )
                                    for pt in grid_train
                                ]
                            ][:, None]

                            # training data
                            obs_data_train = np.array([])
                            for norder in obs_data_full:
                                obs_data_train = np.append(
                                    obs_data_train,
                                    griddata(
                                        np.reshape(
                                            input_space_warped,
                                            (
                                                np.prod(
                                                    np.shape(input_space_warped)[0:-1]
                                                ),
                                            )
                                            + (np.shape(input_space_warped)[-1],),
                                        ),
                                        np.reshape(norder, np.prod(np.shape(norder))),
                                        grid_train,
                                    ),
                                )
                            obs_data_train = np.reshape(
                                obs_data_train,
                                (np.shape(obs_data_full)[0],)
                                + (np.shape(grid_train)[0],),
                            )

                            # sets yref
                            if yref_type == "dimensionful":
                                yref = obs_data_train[-1]
                            elif yref_type == "dimensionless":
                                # yref = np.ones((len(degrees_train_pts)))
                                yref = np.ones((np.shape(grid_train)[0],))

                            # creates and fits the TruncationTP object
                            gp_post_obs = gm.TruncationTP(
                                kernel_posterior,
                                ref=yref,
                                ratio=ratio_fn,
                                center=center,
                                disp=disp,
                                df=df,
                                scale=std_est,
                                excluded=excluded,
                                ratio_kws={
                                    **ratio_fn_kwargs,
                                    **{"p_shape": np.shape(p_grid_train)[:-1]},
                                    **{"p_grid_train": p_grid_train},
                                },
                            )

                            # sets important quantities within TruncationTP
                            gp_post_obs.X_train_ = grid_train
                            gp_post_obs.y_train_ = (obs_data_train[:order, :]).T
                            gp_post_obs.orders_ = nn_orders_full_array[:order]

                            # makes important objects into ray objects
                            gp_post_ray = ray.put(gp_post_obs)

                            # calculates the posterior using ray
                            log_like = calc_loglike_ray(
                                mesh_cart,
                                BATCH_SIZE,
                                log_likelihood_fn,
                                gp_post_ray,
                                log_likelihood_fn_kwargs={
                                    **log_likelihood_fn_kwargs,
                                    # **{"p_shape": (len(degrees_train_pts))},
                                    # **{"p_grid_train": degrees_train_pts[:, None]}
                                    **{"p_shape": np.shape(p_grid_train)[:-1]},
                                    **{"p_grid_train": p_grid_train},
                                },
                            )
                            obs_loglike = np.reshape(
                                log_like,
                                tuple(
                                    len(random_var.var)
                                    for random_var in variables_array
                                ),
                            )

                            # adds the log-priors to the log-likelihoods
                            obs_loglike = add_logpriors(variables_array, obs_loglike)
                            # makes sure that the values don't get too big or too small
                            obs_like = np.exp(obs_loglike - np.max(obs_loglike))
                            # marginalizes partially
                            for v, var in zip(
                                np.flip(
                                    np.array(range(len(variables_array)))[
                                        ~marg_bool_array
                                    ]
                                ),
                                np.flip(variables_array[~marg_bool_array]),
                            ):
                                obs_like = np.trapz(
                                    obs_like, x=variables_array[v].var, axis=v
                                )

                            # takes the log again to revert to the log-likelihood
                            obs_loglike_2d = np.log(obs_like)
                            obs_loglike_sum += obs_loglike_2d

                        elif np.shape(obs_data_full)[1] == len(t_lab):
                            # observables that depend only on scattering angle (e.g., total cross section, or SGT)

                            # converts the points in t_lab to the current input space
                            tlab_input = InputSpaceTlab.input_space(
                                **{"E_lab": t_lab, "interaction": nn_interaction}
                            )
                            tlab_train_pts_input = InputSpaceTlab.input_space(
                                **{
                                    "E_lab": t_lab_train_pts,
                                    "interaction": nn_interaction,
                                }
                            )

                            # converts points in t_lab to momentum
                            tlab_mom = mom_fn(t_lab, **mom_fn_kwargs)
                            tlab_train_pts_mom = mom_fn(
                                t_lab_train_pts, **mom_fn_kwargs
                            )

                            # warps the input space
                            input_space_warped = warping_fn(
                                np.reshape(
                                    gm.cartesian(
                                        *[
                                            tlab_input,
                                        ]
                                    ),
                                    (len(t_lab), 1),
                                ),
                                **warping_fn_kwargs
                            )

                            # creates grids for training points and the corresponding momenta at those points
                            grid_train = tlab_train_pts_input
                            p_grid_train = tlab_train_pts_mom

                            p_grid_train = p_grid_train[
                                [
                                    (
                                        pt >= np.min(input_space_warped[:, 0])
                                        and pt <= np.max(input_space_warped[:, 0])
                                    )
                                    for pt in grid_train
                                ]
                            ][:, None]
                            grid_train = grid_train[
                                [
                                    (
                                        pt >= np.min(input_space_warped[:, 0])
                                        and pt <= np.max(input_space_warped[:, 0])
                                    )
                                    for pt in grid_train
                                ]
                            ][:, None]

                            # training data
                            obs_data_train = np.array([])
                            for norder in obs_data_full:
                                obs_data_train = np.append(
                                    obs_data_train,
                                    griddata(
                                        np.reshape(
                                            input_space_warped,
                                            (
                                                np.prod(
                                                    np.shape(input_space_warped)[0:-1]
                                                ),
                                            )
                                            + (np.shape(input_space_warped)[-1],),
                                        ),
                                        np.reshape(norder, np.prod(np.shape(norder))),
                                        grid_train,
                                    ),
                                )
                            obs_data_train = np.reshape(
                                obs_data_train,
                                (np.shape(obs_data_full)[0],)
                                + (np.shape(grid_train)[0],),
                            )

                            # sets yref
                            if yref_type == "dimensionful":
                                yref = obs_data_train[-1]
                            elif yref_type == "dimensionless":
                                # yref = np.ones((len(t_lab_train_pts)))
                                yref = np.ones((np.shape(grid_train)[0],))

                            # creates and fits the TruncationTP object
                            gp_post_obs = gm.TruncationTP(
                                kernel_posterior,
                                ref=yref,
                                ratio=ratio_fn,
                                center=center,
                                disp=disp,
                                df=df,
                                scale=std_est,
                                excluded=excluded,
                                ratio_kws={
                                    **ratio_fn_kwargs,
                                    **{"p_shape": np.shape(p_grid_train)[:-1]},
                                    **{"p_grid_train": p_grid_train},
                                },
                            )

                            # sets important quantities within TruncationTP
                            gp_post_obs.X_train_ = grid_train
                            gp_post_obs.y_train_ = (obs_data_train[:order, :]).T
                            gp_post_obs.orders_ = nn_orders_full_array[:order]

                            # puts important objects into ray objects
                            gp_post_ray = ray.put(gp_post_obs)

                            # calculates the posterior using ray
                            log_like = calc_loglike_ray(
                                mesh_cart,
                                BATCH_SIZE,
                                log_likelihood_fn,
                                gp_post_ray,
                                log_likelihood_fn_kwargs={
                                    **log_likelihood_fn_kwargs,
                                    # **{"p_shape" : (len(t_lab_train_pts))},
                                    # **{"p_grid_train" : tlab_train_pts_mom[:, None]}
                                    **{"p_shape": np.shape(p_grid_train)[:-1]},
                                    **{"p_grid_train": p_grid_train},
                                },
                            )
                            obs_loglike = np.reshape(
                                log_like,
                                tuple(
                                    len(random_var.var)
                                    for random_var in variables_array
                                ),
                            )

                            # adds the log-priors to the log-likelihoods
                            obs_loglike = add_logpriors(variables_array, obs_loglike)
                            # makes sure that the values don't get too big or too small
                            obs_like = np.exp(obs_loglike - np.max(obs_loglike))
                            # marginalizes partially
                            for v, var in zip(
                                np.flip(
                                    np.array(range(len(variables_array)))[
                                        ~marg_bool_array
                                    ]
                                ),
                                np.flip(variables_array[~marg_bool_array]),
                            ):
                                obs_like = np.trapz(
                                    obs_like, x=variables_array[v].var, axis=v
                                )
                            # takes the log again to revert to the log-likelihood
                            obs_loglike_2d = np.log(obs_like)
                            obs_loglike_sum += obs_loglike_2d

                    else:
                        # 2D observables

                        # converts points in t_lab to the current input space
                        tlab_input = InputSpaceTlab.input_space(
                            **{"E_lab": t_lab, "interaction": nn_interaction}
                        )
                        tlab_train_pts_input = InputSpaceTlab.input_space(
                            **{"E_lab": t_lab_train_pts, "interaction": nn_interaction}
                        )

                        # converts points in t_lab to momentum
                        tlab_mom = mom_fn(t_lab, **mom_fn_kwargs)
                        tlab_train_pts_mom = mom_fn(t_lab_train_pts, **mom_fn_kwargs)

                        # converts points in degrees to the current input space
                        degrees_input = InputSpaceDeg.input_space(
                            **{"deg_input": degrees, "p_input": tlab_mom}
                        )
                        degrees_train_pts_input = InputSpaceDeg.input_space(
                            **{
                                "deg_input": degrees_train_pts,
                                "p_input": tlab_train_pts_mom,
                            }
                        )

                        # warps the input space
                        try:
                            input_space_warped = warping_fn(
                                create_pairs(tlab_input, degrees_input),
                                **warping_fn_kwargs
                            )
                        except:
                            input_space_warped = warping_fn(
                                np.reshape(
                                    gm.cartesian(*[tlab_input, degrees_input]),
                                    (len(t_lab), len(degrees), 2),
                                ),
                                **warping_fn_kwargs
                            )

                        warped_poly = Polygon(
                            np.concatenate(
                                [
                                    input_space_warped[0, :, ...],
                                    input_space_warped[:, -1, ...],
                                    input_space_warped[-1, :, ...],
                                    np.flip(input_space_warped[:, 0, ...], axis=0),
                                ]
                            )
                        )

                        # creates grids for training points and the corresponding momenta at those points
                        try:
                            grid_train = create_pairs(
                                tlab_train_pts_input, degrees_train_pts_input
                            )
                        except:
                            grid_train = gm.cartesian(
                                *[tlab_train_pts_input, degrees_train_pts_input]
                            )
                        grid_train = np.reshape(
                            grid_train,
                            (np.prod(np.shape(grid_train)[:-1]),)
                            + (np.shape(grid_train)[-1],),
                        )

                        p_grid_train = gm.cartesian(
                            *[tlab_train_pts_mom, degrees_train_pts]
                        )
                        p_grid_train = p_grid_train[
                            [
                                warped_poly.buffer(0.001).contains(Point(pt))
                                for pt in grid_train
                            ],
                            ...,
                        ]

                        grid_train = grid_train[
                            [
                                warped_poly.buffer(0.001).contains(Point(pt))
                                for pt in grid_train
                            ],
                            ...,
                        ]

                        # training data
                        obs_data_train = np.array([])
                        for norder in obs_data_full:
                            obs_data_train = np.append(
                                obs_data_train,
                                griddata(
                                    np.reshape(
                                        input_space_warped,
                                        (np.prod(np.shape(input_space_warped)[0:-1]),)
                                        + (np.shape(input_space_warped)[-1],),
                                    ),
                                    np.reshape(norder, np.prod(np.shape(norder))),
                                    grid_train,
                                ),
                            )
                        obs_data_train = np.reshape(
                            obs_data_train,
                            (np.shape(obs_data_full)[0],) + (np.shape(grid_train)[0],),
                        )

                        # sets yref
                        if yref_type == "dimensionful":
                            yref = obs_data_train[-1]
                        elif yref_type == "dimensionless":
                            yref = np.ones((np.shape(grid_train)[0],))

                        # creates and fits the TruncationTP object
                        gp_post_obs = gm.TruncationTP(
                            kernel_posterior,
                            ref=yref,
                            ratio=ratio_fn,
                            center=center,
                            disp=disp,
                            df=df,
                            scale=std_est,
                            excluded=excluded,
                            # ratio_kws={**ratio_fn_kwargs,
                            #            **{"p_shape" : (len(degrees_train_pts) * len(tlab_train_pts_mom))},
                            #            **{"p_grid_train" : np.flip(np.array(list(itertools.product(tlab_train_pts_mom, degrees_train_pts))), axis = 1)}
                            #            }
                            ratio_kws={
                                **ratio_fn_kwargs,
                                **{"p_shape": np.shape(p_grid_train)[:-1]},
                                **{"p_grid_train": p_grid_train},
                            },
                        )

                        # sets important quantities within TruncationTP
                        gp_post_obs.X_train_ = grid_train
                        gp_post_obs.y_train_ = (obs_data_train[:order, :]).T
                        gp_post_obs.orders_ = nn_orders_full_array[:order]

                        # puts important objects into ray objects
                        gp_post_ray = ray.put(gp_post_obs)

                        # calculates the posterior using ray
                        log_like = calc_loglike_ray(
                            mesh_cart,
                            BATCH_SIZE,
                            log_likelihood_fn,
                            gp_post_ray,
                            log_likelihood_fn_kwargs={
                                **log_likelihood_fn_kwargs,
                                **{"p_shape": np.shape(p_grid_train)[:-1]},
                                **{"p_grid_train": p_grid_train},
                            },
                        )
                        obs_loglike = np.reshape(
                            log_like,
                            tuple(
                                len(random_var.var) for random_var in variables_array
                            ),
                        )

                        # adds the log-priors to the log-likelihoods
                        obs_loglike = add_logpriors(variables_array, obs_loglike)
                        # makes sure that the values don't get too big or too small
                        obs_like = np.exp(obs_loglike - np.max(obs_loglike))
                        # marginalizes partially
                        for v, var in zip(
                            np.flip(
                                np.array(range(len(variables_array)))[~marg_bool_array]
                            ),
                            np.flip(variables_array[~marg_bool_array]),
                        ):
                            obs_like = np.trapz(
                                obs_like, x=variables_array[v].var, axis=v
                            )
                        # takes the log again to revert to the log-likelihood
                        obs_loglike_partmarg = np.log(obs_like)
                        obs_loglike_sum += obs_loglike_partmarg

                # makes sure that the values don't get too big or too small
                obs_like = np.exp(obs_loglike_sum - np.max(obs_loglike_sum))

                if whether_save_data:
                    # saves data, if the user chooses
                    np.savetxt(
                        make_likelihood_filename(
                            FileName,
                            "data",
                            obs_name,
                            orders_names_dict[order],
                            [variable.logprior_name for variable in variables_array],
                            variables_array,
                        ),
                        np.reshape(
                            obs_like,
                            (
                                np.prod(
                                    [
                                        len(random_var.var)
                                        for random_var in variables_array[
                                            marg_bool_array
                                        ]
                                    ]
                                )
                            ),
                        ),
                    )

                like_list.append(obs_like)

    like_list = np.reshape(
        np.reshape(
            like_list,
            (np.shape(like_list)[0] // orders, orders) + np.shape(like_list)[1:],
            order="C",
        ),
        np.shape(like_list),
    )

    if whether_plot_posteriors or whether_plot_corner:
        # calculates all joint and fully marginalized posterior pdfs
        marg_post_array, joint_post_array = marginalize_likelihoods(
            variables_array[marg_bool_array], like_list
        )

    # array of stats (MAP, mean, and stddev)
    fit_stats_array = np.array([])

    if whether_plot_posteriors:
        # plots and saves all fully marginalized posterior pdfs
        for variable, result in zip(variables_array[marg_bool_array], marg_post_array):
            fig, fit_stats = plot_marg_posteriors(
                variable,
                result,
                obs_labels_grouped_list,
                Lb_colors,
                order_num,
                # self.nn_orders, self.orders_labels_dict, self, whether_save_plots, obs_name_grouped_list)
                nn_orders_array,
                orders_labels_dict,
            )

            fit_stats_array = np.append(fit_stats_array, fit_stats)

            if whether_save_plots:
                # saves
                obs_name_corner_concat = "".join(obs_name_grouped_list)
                fig.savefig(
                    (
                        "figures/"
                        + FileName.scheme
                        + "_"
                        + FileName.scale
                        + "/"
                        + variable.name
                        + "_posterior_pdf_curvewise"
                        + "_"
                        + obs_name_corner_concat
                        + "_"
                        + FileName.scheme
                        + "_"
                        + FileName.scale
                        + "_"
                        + "Q"
                        + FileName.Q_param
                        + "_"
                        + FileName.p_param
                        + "_"
                        + InputSpaceDeg.name
                        + "x"
                        + InputSpaceTlab.name
                        + FileName.filename_addendum
                    ).replace("_0MeVlab_", "_")
                )

    if whether_plot_corner:
        with plt.rc_context({"text.usetex": True}):
            # plots and saves all joint and fully marginalized posterior pdfs in the form of corner plots
            fig = plot_corner_posteriors(
                variables_array[marg_bool_array],
                marg_post_array,
                joint_post_array,
                obs_name_grouped_list,
                "Blues",
                order_num,
                nn_orders_array,
                orders_labels_dict,
                FileName,
                whether_save_plots,
            )

    return fit_stats_array


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
    Lb_colors = light_colors[-1 * order_num :]

    marg_post_list = []

    for obs_grouping, obs_name in zip(obs_data_grouped_list, obs_name_grouped_list):
        # generates names for files and searches for whether they exist
        for order_counter in range(1, order_num + 1):
            order = np.max(nn_orders_array) - order_num + order_counter

            # scale invariant: df = 0
            PointwiseModel = gm.TruncationPointwise(df=0, excluded=excluded)

            obs_post_sum = np.ones(
                tuple(len(random_var.var) for random_var in variables_array)
            )

            for obs_object in obs_grouping:
                obs_data_full = obs_object.data
                yref_type = obs_object.ref_type

                if len(np.shape(obs_data_full)) == 2:
                    if np.shape(obs_data_full)[1] == len(degrees):
                        # converts the points in tlab_train_pts_mom and degrees_train_pts to momentum
                        tlab_train_pts_mom = mom_fn_tlab(
                            **{**mom_fn_tlab_kwargs, **{"E_lab": np.array([0])}}
                        )
                        degrees_train_pts_mom = mom_fn_degrees(
                            **{
                                **mom_fn_degrees_kwargs,
                                **{"degrees": degrees_train_pts},
                            }
                        )

                        # sieves the data
                        obs_data_train = np.reshape(
                            obs_data_full[:, np.isin(degrees, degrees_train_pts)],
                            (len(nn_orders_full_array), -1),
                        )

                        # sets yref
                        if yref_type == "dimensionful":
                            yref = obs_data_train[-1]
                        elif yref_type == "dimensionless":
                            yref = np.ones((len(degrees_train_pts)))

                        # calculates ratio for every training point and every value of Lambda_b
                        ratio_train = [
                            ratio_fn(
                                **{
                                    **ratio_fn_kwargs,
                                    **{
                                        "p": np.reshape(
                                            p_fn(
                                                **{
                                                    **p_fn_kwargs,
                                                    **{
                                                        "prel": tlab_train_pts_mom,
                                                        "degrees": degrees_train_pts_mom,
                                                    },
                                                }
                                            ),
                                            len(degrees_train_pts),
                                        ),
                                        "Lambda_b": Lb,
                                    },
                                }
                            )
                            for Lb in variables_array[0].var
                        ]

                        # fits the TruncationPointwise object
                        pointwise_result, _, _ = compute_posterior_intervals(
                            PointwiseModel,
                            obs_data_train,
                            ratio_train,
                            ref=yref,
                            orders=nn_orders_full_array,
                            max_idx=order - 1,
                            logprior=variables_array[0].logprior,
                            Lb=variables_array[0].var,
                        )

                        obs_post_sum *= pointwise_result

                    elif np.shape(obs_data_full)[1] == len(t_lab):
                        # converts the points in tlab_train_pts_mom and degrees_train_pts to momentum
                        tlab_train_pts_mom = mom_fn_tlab(
                            **{**mom_fn_tlab_kwargs, **{"E_lab": t_lab_train_pts}}
                        )
                        degrees_train_pts_mom = mom_fn_degrees(
                            **{**mom_fn_degrees_kwargs, **{"degrees": np.array([0])}}
                        )

                        # sieves the data
                        obs_data_train = np.reshape(
                            obs_data_full[:, np.isin(t_lab, t_lab_train_pts)],
                            (len(nn_orders_full_array), -1),
                        )

                        # sets yref
                        if yref_type == "dimensionful":
                            yref = obs_data_train[-1]
                        elif yref_type == "dimensionless":
                            yref = np.ones((len(t_lab_train_pts)))

                        # calculates ratio for every training point and every value of Lambda_b
                        ratio_train = [
                            ratio_fn(
                                **{
                                    **ratio_fn_kwargs,
                                    **{
                                        "p": np.reshape(
                                            p_fn(
                                                **{
                                                    **p_fn_kwargs,
                                                    **{
                                                        "prel": tlab_train_pts_mom,
                                                        "degrees": degrees_train_pts_mom,
                                                    },
                                                }
                                            ),
                                            len(t_lab_train_pts),
                                        ),
                                        "Lambda_b": Lb,
                                    },
                                }
                            )
                            for Lb in variables_array[0].var
                        ]

                        # fits the TruncationPointwise object
                        pointwise_result, _, _ = compute_posterior_intervals(
                            PointwiseModel,
                            obs_data_train,
                            ratio_train,
                            ref=yref,
                            orders=nn_orders_full_array,
                            max_idx=order - 1,
                            logprior=variables_array[0].logprior,
                            Lb=variables_array[0].var,
                        )

                        obs_post_sum *= pointwise_result

                else:
                    # converts the points in tlab_train_pts_mom and degrees_train_pts to momentum
                    tlab_train_pts_mom = mom_fn_tlab(
                        **{**mom_fn_tlab_kwargs, **{"E_lab": t_lab_train_pts}}
                    )
                    degrees_train_pts_mom = mom_fn_degrees(
                        **{**mom_fn_degrees_kwargs, **{"degrees": degrees_train_pts}}
                    )

                    # sieves data
                    obs_data_train = np.reshape(
                        obs_data_full[:, np.isin(t_lab, t_lab_train_pts)][
                            ..., np.isin(degrees, degrees_train_pts)
                        ],
                        (len(nn_orders_full_array), -1),
                    )

                    # sets yref
                    if yref_type == "dimensionful":
                        yref = obs_data_train[-1]
                    elif yref_type == "dimensionless":
                        yref = np.ones((len(degrees_train_pts) * len(t_lab_train_pts)))

                    # calculates ratio for every training point and every value of Lambda_b
                    ratio_train = [
                        ratio_fn(
                            **{
                                **ratio_fn_kwargs,
                                **{
                                    "p": np.reshape(
                                        p_fn(
                                            **{
                                                **p_fn_kwargs,
                                                **{
                                                    "prel": tlab_train_pts_mom,
                                                    "degrees": degrees_train_pts_mom,
                                                },
                                            }
                                        ),
                                        (len(degrees_train_pts) * len(t_lab_train_pts)),
                                    ),
                                    "Lambda_b": Lb,
                                },
                            }
                        )
                        for Lb in variables_array[0].var
                    ]

                    # fits the TruncationPointwise object
                    pointwise_result, _, _ = compute_posterior_intervals(
                        PointwiseModel,
                        obs_data_train,
                        ratio_train,
                        ref=yref,
                        orders=nn_orders_full_array,
                        max_idx=order - 1,
                        logprior=variables_array[0].logprior,
                        Lb=variables_array[0].var,
                    )

                    obs_post_sum *= pointwise_result

            # appends the normalized posterior
            marg_post_list.append(
                obs_post_sum / np.trapz(obs_post_sum, variables_array[0].var)
            )

    marg_post_list = np.reshape(
        np.reshape(
            marg_post_list,
            (np.shape(marg_post_list)[0] // orders, orders)
            + np.shape(marg_post_list)[1:],
            order="C",
        ),
        np.shape(marg_post_list),
    )

    # adds an extra dimension to comport with structure of existing code
    marg_post_list = marg_post_list[None, :]

    # array of stats (MAP, mean, and stddev)
    fit_stats_array = np.array([])

    if whether_plot_posteriors:
        for variable, result in zip(variables_array, marg_post_list):
            # generates plots of posteriors for multiple observables and orders
            fig, fit_stats = plot_marg_posteriors(
                variable,
                result,
                obs_labels_grouped_list,
                Lb_colors,
                order_num,
                nn_orders_array,
                orders_labels_dict,
            )

            fit_stats_array = np.append(fit_stats_array, fit_stats)

            # saves
            obs_name_corner_concat = "".join(obs_name_grouped_list)
            if whether_save_plots:
                fig.savefig(
                    (
                        "figures/"
                        + FileName.scheme
                        + "_"
                        + FileName.scale
                        + "/"
                        + variable.name
                        + "_posterior_pdf_pointwise"
                        + "_"
                        + obs_name_corner_concat
                        + "_"
                        + FileName.scheme
                        + "_"
                        + FileName.scale
                        + "_Q"
                        + FileName.Q_param
                        + "_"
                        + FileName.p_param
                        + "_"
                        + InputSpaceDeg.name
                        + "x"
                        + InputSpaceTlab.name
                        + FileName.filename_addendum
                    ).replace("_0MeVlab_", "_")
                )

        # finds and prints the MAP value for Lambda_b
        indices_opt = np.where(
            marg_post_list[0, -1, :] == np.amax(marg_post_list[0, -1, :])
        )
        opt_vals_list = []
        for idx, var in zip(
            indices_opt, [variable.var for variable in variables_array]
        ):
            opt_vals_list.append((var[idx])[0])

        print("opt_vals_list = " + str(opt_vals_list))
    return fit_stats_array


class NontationaryKernelMixin:
    """Mixin for kernels which are stationary: k(X, Y)= f(X-Y).

    .. versionadded:: 0.18
    """

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return False


class NSKernel(metaclass=ABCMeta):
    """Base class for all nonstationary kernels.

    .. versionadded:: 0.18
    """

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = dict()

        # introspect the constructor arguments to find the model parameters
        # to represent
        cls = self.__class__
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        init_sign = signature(init)
        args, varargs = [], []
        for parameter in init_sign.parameters.values():
            if parameter.kind != parameter.VAR_KEYWORD and parameter.name != "self":
                args.append(parameter.name)
            if parameter.kind == parameter.VAR_POSITIONAL:
                varargs.append(parameter.name)

        if len(varargs) != 0:
            raise RuntimeError(
                "scikit-learn kernels should always "
                "specify their parameters in the signature"
                " of their __init__ (no varargs)."
                " %s doesn't follow this convention." % (cls,)
            )
        for arg in args:
            params[arg] = getattr(self, arg)

        return params

    def set_params(self, **params):
        """Set the parameters of this kernel.

        The method works on simple kernels as well as on nested kernels.
        The latter have parameters of the form ``<component>__<parameter>``
        so that it's possible to update each component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            split = key.split("__", 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError(
                        "Invalid parameter %s for kernel %s. "
                        "Check the list of available parameters "
                        "with `kernel.get_params().keys()`." % (name, self)
                    )
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError(
                        "Invalid parameter %s for kernel %s. "
                        "Check the list of available parameters "
                        "with `kernel.get_params().keys()`."
                        % (key, self.__class__.__name__)
                    )
                setattr(self, key, value)
        return self

    def clone_with_theta(self, theta):
        """Returns a clone of self with given hyperparameters theta.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The hyperparameters
        """
        cloned = clone(self)
        cloned.theta = theta
        return cloned

    @property
    def n_dims(self):
        """Returns the number of non-fixed hyperparameters of the kernel."""
        return self.theta.shape[0]

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter specifications."""
        r = [
            getattr(self, attr)
            for attr in dir(self)
            if attr.startswith("hyperparameter_")
        ]
        return r

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        theta = []
        params = self.get_params()
        for hyperparameter in self.hyperparameters:
            if not hyperparameter.fixed:
                theta.append(params[hyperparameter.name])
        if len(theta) > 0:
            return np.log(np.hstack(theta))
        else:
            return np.array([])

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        params = self.get_params()
        i = 0
        for hyperparameter in self.hyperparameters:
            if hyperparameter.fixed:
                continue
            if hyperparameter.n_elements > 1:
                # vector-valued parameter
                params[hyperparameter.name] = np.exp(
                    theta[i : i + hyperparameter.n_elements]
                )
                i += hyperparameter.n_elements
            else:
                params[hyperparameter.name] = np.exp(theta[i])
                i += 1

        if i != len(theta):
            raise ValueError(
                "theta has not the correct number of entries."
                " Should be %d; given are %d" % (i, len(theta))
            )
        self.set_params(**params)

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        bounds = [
            hyperparameter.bounds
            for hyperparameter in self.hyperparameters
            if not hyperparameter.fixed
        ]
        if len(bounds) > 0:
            return np.log(np.vstack(bounds))
        else:
            return np.array([])

    def __add__(self, b):
        if not isinstance(b, NSKernel):
            return Sum(self, ConstantKernel(b))
        return Sum(self, b)

    def __radd__(self, b):
        if not isinstance(b, NSKernel):
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        if not isinstance(b, NSKernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    def __rmul__(self, b):
        if not isinstance(b, NSKernel):
            return Product(ConstantKernel(b), self)
        return Product(b, self)

    def __pow__(self, b):
        return Exponentiation(self, b)

    def __eq__(self, b):
        if type(self) != type(b):
            return False
        params_a = self.get_params()
        params_b = b.get_params()
        for key in set(list(params_a.keys()) + list(params_b.keys())):
            if np.any(params_a.get(key, None) != params_b.get(key, None)):
                return False
        return True

    def __repr__(self):
        return "{0}({1})".format(
            self.__class__.__name__, ", ".join(map("{0:.3g}".format, self.theta))
        )

    @abstractmethod
    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the kernel."""

    @abstractmethod
    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """

    @abstractmethod
    def is_stationary(self):
        """Returns whether the kernel is stationary."""

    @property
    def requires_vector_input(self):
        """Returns whether the kernel is defined on fixed-length feature
        vectors or generic objects. Defaults to True for backward
        compatibility."""
        return True

    def _check_bounds_params(self):
        """Called after fitting to warn if bounds may have been too tight."""
        list_close = np.isclose(self.bounds, np.atleast_2d(self.theta).T)
        idx = 0
        for hyp in self.hyperparameters:
            if hyp.fixed:
                continue
            for dim in range(hyp.n_elements):
                if list_close[idx, 0]:
                    warnings.warn(
                        "The optimal value found for "
                        "dimension %s of parameter %s is "
                        "close to the specified lower "
                        "bound %s. Decreasing the bound and"
                        " calling fit again may find a "
                        "better value." % (dim, hyp.name, hyp.bounds[dim][0]),
                        ConvergenceWarning,
                    )
                elif list_close[idx, 1]:
                    warnings.warn(
                        "The optimal value found for "
                        "dimension %s of parameter %s is "
                        "close to the specified upper "
                        "bound %s. Increasing the bound and"
                        " calling fit again may find a "
                        "better value." % (dim, hyp.name, hyp.bounds[dim][1]),
                        ConvergenceWarning,
                    )
                idx += 1


class NSRBF(NontationaryKernelMixin, NormalizedKernelMixin, NSKernel):
    """Nonstationary radial basis function kernel (aka squared-exponential kernel).

    The NSRBF kernel is a nonstationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length scale
    parameter :math:`l>0`, which can either be a scalar (isotropic variant
    of the kernel) or a vector with the same number of dimensions as the inputs
    X (anisotropic variant of the kernel). The kernel is given by:

    .. math::
        k(x_i, x_j) = \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)

    where :math:`l` is the length scale of the kernel and
    :math:`d(\\cdot,\\cdot)` is the Euclidean distance.
    For advice on how to set the length scale parameter, see e.g. [1]_.

    Unlike the standard scikit-learn RBF, the NSRBF can handle length scales that
    depend functionally on the input space.

    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth.
    See [2]_, Chapter 4, Section 4.2, for further details of the RBF kernel.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    length_scale_fn (callable or None) : function for generating the length scale from the input space.
        Default : None

    length_scale_fn_kwargs (dict or None) : keyword arguments for that function.
        Default : None

    References
    ----------
    .. [1] `David Duvenaud (2014). "The Kernel Cookbook:
        Advice on Covariance functions".
        <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_

    .. [2] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = 1.0 * RBF(1.0)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9866...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8354..., 0.03228..., 0.1322...],
           [0.7906..., 0.0652..., 0.1441...]])
    """

    def __init__(
        self,
        length_scale=1.0,
        length_scale_bounds=(1e-5, 1e5),
        length_scale_fn=None,
        length_scale_fn_kwargs=None,
        cbar=1.0,
        cbar_bounds=(1e-5, 1e5),
        cbar_fn=None,
        cbar_fn_kwargs=None,
    ):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

        self.length_scale_fn = length_scale_fn
        self.length_scale_fn_kwargs = length_scale_fn_kwargs

        if cbar is not None:
            self.cbar = cbar
        else:
            self.cbar = 1.0
        self.cbar_bounds = cbar_bounds

        self.cbar_fn = cbar_fn
        self.cbar_fn_kwargs = cbar_fn_kwargs

    @property
    def anisotropic_cbar(self):
        return np.iterable(self.cbar) and len(self.cbar) > 1

    @property
    def hyperparameter_cbar(self):
        if self.anisotropic_cbar:
            # print("The variance is anisotropic.")
            return Hyperparameter(
                "cbar",
                "numeric",
                self.cbar_bounds,
                len(self.cbar),
            )
        return Hyperparameter("cbar", "numeric", self.cbar_bounds)

    @property
    def anisotropic_length_scale(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1
    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic_length_scale:
            # print("The length scale is anisotropic.")
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)

        if self.length_scale_fn_kwargs is None:
            length_scale_fn_kwargs = {}
        else:
            length_scale_fn_kwargs = self.length_scale_fn_kwargs

        if self.cbar_fn_kwargs is None:
            cbar_fn_kwargs = {}
        else:
            cbar_fn_kwargs = self.cbar_fn_kwargs

        if Y is None:
            if callable(self.length_scale_fn):
                # if length scale(s) is/are nonstationary, calculate them
                length_scale_fn_kwargs = {
                    **length_scale_fn_kwargs,
                    **{"ls_array": self.length_scale},
                }
                length_scale = self.length_scale_fn(X, **length_scale_fn_kwargs)
            else:
                # otherwise, behave like a stationary kernel
                length_scale = self.length_scale

            if callable(self.cbar_fn):
                # if length scale(s) is/are nonstationary, calculate them
                cbar_fn_kwargs = {
                    **cbar_fn_kwargs,
                    **{"cbar_array": self.cbar},
                }
                cbar = self.cbar_fn(X, **cbar_fn_kwargs)
            else:
                # otherwise, behave like a stationary kernel
                cbar = self.cbar

        if Y is None:
            dists = pdist(X / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
            K = K * cbar**(2)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")

            if callable(self.length_scale_fn):
                length_scale_X = self.length_scale_fn(X, **self.length_scale_fn_kwargs)
                length_scale_Y = self.length_scale_fn(Y, **self.length_scale_fn_kwargs)
            else:
                length_scale_X = length_scale
                length_scale_Y = length_scale

            dists = cdist(X / length_scale_X, Y / length_scale_Y, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)

            if callable(self.cbar_fn):
                cbar_X = self.cbar_fn(X, **self.cbar_fn_kwargs)
                cbar_Y = self.cbar_fn(Y, **self.cbar_fn_kwargs)
            else:
                cbar_X = cbar
                cbar_Y = cbar

            K = cbar_X * cbar_Y * K

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                    length_scale**2
                )
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
            )
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0]
            )


class NSWhiteKernel(StationaryKernelMixin, GenericKernelMixin, NSKernel):
    """White kernel.

    The main use-case of this kernel is as part of a sum-kernel where it
    explains the noise of the signal as independently and identically
    normally-distributed. The parameter noise_level equals the variance of this
    noise.

    .. math::
        k(x_1, x_2) = noise\\_level \\text{ if } x_i == x_j \\text{ else } 0


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    noise_level : float, default=1.0
        Parameter controlling the noise level (variance)

    noise_level_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'noise_level'.
        If set to "fixed", 'noise_level' cannot be changed during
        hyperparameter tuning.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel(noise_level=0.5)
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3680...
    >>> gpr.predict(X[:2,:], return_std=True)
    (array([653.0..., 592.1... ]), array([316.6..., 316.6...]))
    """

    def __init__(self, noise_level=1.0, noise_level_bounds=(1e-5, 1e5)):
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds

    @property
    def hyperparameter_noise_level(self):
        return Hyperparameter("noise_level", "numeric", self.noise_level_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_X, n_features) or list of object,\
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
            optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = self.noise_level * np.eye(_num_samples(X))
            if eval_gradient:
                if not self.hyperparameter_noise_level.fixed:
                    return (
                        K,
                        self.noise_level * np.eye(_num_samples(X))[:, :, np.newaxis],
                    )
                else:
                    return K, np.empty((_num_samples(X), _num_samples(X), 0))
            else:
                return K
        else:
            return np.zeros((_num_samples(X), _num_samples(Y)))

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return np.full(
            _num_samples(X), self.noise_level, dtype=np.array(self.noise_level).dtype
        )

    def __repr__(self):
        return "{0}(noise_level={1:.3g})".format(
            self.__class__.__name__, self.noise_level
        )


def ns_check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.shape(X)[-1] != np.shape(length_scale)[-1]:
        raise ValueError("X and length must have same final dimension.")
    return length_scale


def create_pairs(arr1, arr2):
    """Creates pairs from arr1 and arr2

    Parameters
    ----------
    arr1 : float array of shape (m).
    arr2 : float array of shape (m, n).

    Returns
    -------
    pairs : float array of pairs of arr1 and arr2 of shape (m, n).
    """
    m, n = arr1.shape[0], arr2.shape[-1]
    pairs = np.empty((m, n, 2))
    for i in range(m):
        for j in range(n):
            pairs[i, j, 0] = arr1[
                i
            ]  # Assign the first array to the first dimension of the pairs
            try:
                pairs[i, j, 1] = arr2[
                    i, j
                ]  # Assign the second array to the second dimension of the pairs
            except:
                pairs[i, j, 1] = arr2[j]
    return pairs
