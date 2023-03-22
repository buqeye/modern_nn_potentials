import numpy as np
from scipy.interpolate import interp1d
from .utils import versatile_train_test_split
import h5py
class GPHyperparameters:
    def __init__(self, ls_class, center, ratio, nugget=1e-10, seed=None, df=np.inf,
                 disp=0, scale=1, sd=None):
        """
        Class for the hyperparameters of a Gaussian process.
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
    def __init__(self, scheme, scale, Q_param, p_param, filename_addendum=""):
        """
        scheme (str) : name of the scheme
        scale (str) : name of the scale
        Q_param (str) : name of the Q parametrization
        p_param (str) : name of the p parametrization
        filename_addendum (str) : optional extra string
        """
        self.scheme = scheme
        self.scale = scale
        self.Q_param = Q_param
        self.p_param = p_param
        self.filename_addendum = filename_addendum


class PosteriorBounds:
    def __init__(self, x_lower, x_upper, x_n, y_lower, y_upper, y_n):
        """
        Class for the boundaries of the 2D posterior PDF plot and the mesh on which it is plotted.
        """
        self.x_vals = np.linspace(x_lower, x_upper, x_n)
        self.y_vals = np.linspace(y_lower, y_upper, y_n)


class RandomVariable:
    def __init__(self, var, name, label, units, ticks):
        self.var = var
        self.name = name
        self.label = label
        self.units = units
        self.ticks = ticks


class OrderInfo:
    def __init__(self, orders_array, orders_mask, colors_array, lightcolors_array,
                 orders_restricted=[], mask_restricted=[], orders_names_dict=None,
                 orders_labels_dict=None):
        """
        Class for the number of orders under consideration and the color for each.

        Parameters
        ----------
        orders_array (array): list of orders at which the potential CAN BE evaluated
        orders_mask (array): boolean mask corresponding to orders_array
        colors_array (array): list of colors corresponding to each order
        lightcolors_array (array): list of lighter versions of colors_array
        orders_restricted (array): list of orders at which the potential WILL BE evaluated
            Set to orders_array if no value is given.
        mask_restricted (array): boolean mask corresponding to orders_restricted
            Set to orders_mask if no value is given.
        orders_names_dict (dict): dictionary method linking the numerical indices (int)
            of EFT orders and their corresponding abbreviations (str)
        orders_names_dict (dict): dictionary method linking the numerical indices (int)
            of EFT orders and their corresponding math-mode-formatted labels (str)
        """
        self.orders_full = orders_array
        self.mask_full = orders_mask
        self.colors_array = colors_array
        self.lightcolors_array = lightcolors_array

        if orders_restricted == []:
            self.orders_restricted = self.orders_full
        else:
            self.orders_restricted = orders_restricted
        if mask_restricted == []:
            self.mask_restricted = self.mask_full
        else:
            self.mask_restricted = mask_restricted

        self.orders_names_dict = orders_names_dict
        self.orders_labels_dict = orders_labels_dict


class InputSpaceBunch:
    """
    Class for an input space (i.e., x-coordinate)
    name (string) : (abbreviated) name for the input space
    input_space (array) : x-coordinate mesh points for evaluation
    mom (array) : momenta for the purpose of calculating the ratio
    caption (string) : caption for the x-axis of the coefficient plots for that input space
    tick_marks (array) : major tick marks for the x-axis of the coefficient plots
    title_pieces (array) : information to be concatenated into the coefficient plot's title
    """

    def __init__(self, name, input_space, mom, caption, title_pieces):
        self.name = name
        self.input_space = input_space
        # self.mom = mom * np.ones(len(input_space))
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
    """
    Class for an observable
    name (string) : (abbreviated) name for the observable
    data (array) : coefficient values at each order over the mesh
    energies (array) : energies at which the observable will be evaluated (None for observables
        plotted against energy)
    title (string) : title for the coefficient plot
    ref_type (string) : tells whether the reference scale (to be divided out of the coefficient
        values) has dimension (e.g., the case of the cross section) or not (e.g., the case of the
    spin observables). Can only be "dimensionless" or "dimensionful".
    constraint (array or None): constraint on the values of the observable, including the
        name of the quantity for which the constraint applies.
        For dimensionful (i.e., cross-section) observables, should be None.
    """

    def __init__(self, name, data, energies, angles, title, ref_type, constraint=None):
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
    """
    Class for an interpolater
    x (array) : x-coordinate data
    y (array) : y-coordinate data
    kind (string) : scipy.interpolate.interp1d interpolater 'kind'
    """

    def __init__(self, x, y, kind='cubic'):
        self.x = x
        self.y = y
        self.kind = kind
        self.f_interp = interp1d(self.x, self.y, kind=self.kind)


class TrainTestSplit:
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

    def __init__(self, name, n_train, n_test_inter, isclose_factor=0.01, \
                 offset_train_min_factor=0, offset_train_max_factor=0, \
                 xmin_train_factor=0, xmax_train_factor=1, \
                 offset_test_min_factor=0, offset_test_max_factor=0, \
                 xmin_test_factor=0, xmax_test_factor=1, \
                 train_at_ends=True, test_at_ends=False):
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
        # print(self.x)
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
            versatile_train_test_split(self.interp_obj, \
                                       self.n_train, n_test_inter=self.n_test_inter, \
                                       isclose_factor=self.isclose_factor, \
                                       offset_train_min=self.offset_train_min, \
                                       offset_train_max=self.offset_train_max, \
                                       xmin_train=self.xmin_train, xmax_train=self.xmax_train, \
                                       offset_test_min=self.offset_test_min, \
                                       offset_test_max=self.offset_test_max, \
                                       xmin_test=self.xmin_test, xmax_test=self.xmax_test, \
                                       train_at_ends=self.train_at_ends, test_at_ends=self.test_at_ends)

        return self.x_train, self.x_test, self.y_train, self.y_test


class ScaleSchemeBunch:
    def __init__(self, file_name, orders_full, cmaps, potential_string, cutoff_string,
                 dir_path="./"):
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
        response = h5py.File(self.full_path, "r")
        obs_data = np.array(response[observable_string][:])
        response.close()
        return obs_data


class LengthScale:
    """
    Class for setting a guess for the Gaussian process correlation length scale and its
    bounds
    x (array) : x-coordinate data
    ls_guess_factor (float) : fraction of the total input space length for the initial
        length scale guess
    ls_bound_lower_factor (float) : fraction of the initial length scale guess for the lower
        bound of fitting
    ls_bound_upper_factor (float) : fraction of the initial length scale guess for the upper
        bound of fitting
    whether_fit (bool) : should the fit procedure be performed?
    """

    def __init__(self, name, ls_guess_factor, ls_bound_lower_factor,
                 ls_bound_upper_factor, whether_fit=True):
        self.name = name
        self.ls_guess_factor = ls_guess_factor
        self.ls_bound_lower_factor = ls_bound_lower_factor
        self.ls_bound_upper_factor = ls_bound_upper_factor
        self.whether_fit = whether_fit

    def make_guess(self, x):
        self.ls_guess = (np.max(x) - np.min(x)) * self.ls_guess_factor

        if self.whether_fit:
            self.ls_bound_lower = self.ls_bound_lower_factor * self.ls_guess
            self.ls_bound_upper = self.ls_bound_upper_factor * self.ls_guess
        else:
            self.ls_bound_lower = self.ls_guess.copy()
            self.ls_bound_upper = self.ls_guess.copy()