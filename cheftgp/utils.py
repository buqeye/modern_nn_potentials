import numpy as np
import decimal
import gsum as gm
import urllib
import tables
import scipy
from shapely.geometry import Polygon, Point


def correlation_coefficient(x, y, pdf):
    """
    Calculates the correlation coefficient of a 2-d posterior pdf.

    Parameters
    ----------
    x (array) : 1-d array of x-axis mesh points.
    y (array) : 2-d array of y-axis mesh points.
    pdf (array) : array of probabilities with the dimensions (len(x), len(y)).
    """
    # normalizes the pdf
    pdf /= np.trapz(np.trapz(pdf, x=y, axis=0), x=x, axis=0)

    # finds the maximum value
    pdf_max = np.amax(pdf)

    # figures out the x and y coordinates of the max
    x_max = x[np.argwhere(pdf == pdf_max)[0, 1]]
    y_max = y[np.argwhere(pdf == pdf_max)[0, 0]]

    # finds variance in x and y
    sigma_x_sq = np.trapz(
        np.trapz(np.tile((x - x_max) ** 2, (len(y), 1)) * pdf,
                 x=x, axis=1),
        x=y, axis=0)
    sigma_y_sq = np.trapz(
        np.trapz(np.tile(np.reshape((y - y_max) ** 2, (len(y), 1)), (1, len(x))) * pdf,
                 x=y, axis=0),
        x=x, axis=0)

    # finds sigmaxy
    sigma_xy_sq = np.trapz(
        np.trapz(np.tile(np.reshape(y - y_max, (len(y), 1)), (1, len(x))) * \
                 np.tile(x - x_max, (len(y), 1)) * pdf,
                 x=x, axis=1),
        x=y, axis=0)

    # finds the correlation coefficient
    corr_coeff = sigma_xy_sq / (np.sqrt(sigma_x_sq) * np.sqrt(sigma_y_sq))

    return corr_coeff


def mean_and_stddev(x, pdf):
    """
    Returns the mean and standard deviation of a 1-d posterior pdf.

    Parameters
    ----------
    x (array) : array of x-axis mesh points.
    pdf (array) : array of probabilities of dimension (len(x)).
    """
    # normalizes the pdf
    pdf /= np.trapz(pdf, x=x, axis=0)

    # finds the maximum value
    pdf_max = np.amax(pdf)

    # figures out the x coordinate of the max
    x_max = x[np.argwhere(pdf == pdf_max)][0]

    # finds the mean
    mean = np.trapz(x * pdf, x=x, axis=0)

    # finds the standard deviation
    sigma_x = np.sqrt(np.trapz((x - x_max) ** 2 * pdf, x=x, axis=0))

    return mean, sigma_x


def sig_figs(number, n_figs):
    """
    Parameters
    ----------
    number (float) : a number.
    n_figs (int) : number of significant figures.

    Returns
    -------
    number_string (str) : number with n_figs significant figures
    """
    # formats the number as a string
    number_string = np.format_float_positional(
        np.float64(
            np.format_float_scientific(
                number, precision=n_figs - 1)))

    # eliminates any unncessary zeros and decimal points
    if ((np.float64(number_string) > 10 ** (n_figs - 1)) and (number_string[-1] == '.')):
        number_string = number_string[:-1]
        return np.int(number_string)
    else:
        return np.float64(number_string)

def round_to_same_digits(number, comparand):
    """
    Parameters
    ----------
    number (float) : a number.
    comparand (float) : another number.

    Returns
    -------
    number with the same number of digits past the decimal point as comparand.
    """
    if decimal.Decimal(str(comparand)).as_tuple().exponent == 0:
        return int(number)
    else:
        return np.around(number, decimals = decimal.Decimal(str(comparand)).as_tuple().exponent)

def compute_posterior_intervals(model, data, ratios, ref, orders, max_idx, logprior, Lb):
    """
    Calculates a likelihood for the breakdown scale using a pointwise (uncorrelated) GP model.

    Parameters
    ----------
    model (TruncationPointwise) : model to be fit.
    data (array) : data for the observable of interest.
    ratios (array) : 1-d mesh of dimensionless expansion parameter values across the region of fitting.
    ref (array) : 1-d mesh of reference scale values across the region of fitting.
    orders (int list) : list of orders for which fitting data is available
    max_idx (int) : highest order for calculation.
    logprior (array) : 1-d log-prior to add to calculated likelihood pdf.
    Lb (array) : 1-d array of values for Lambda_b (breakdown scale) mesh (in MeV).

    Returns
    -------
    posterior (array) : 1-d array of probability values at each value of Lb.
    bounds (array) : array of dimension (2, 2) with upper and lower bounds of 68% and 95% confidence intervals.
    median (float) : median value of posterior.
    """
    model.fit(data[:max_idx+1].T, ratio=ratios[0], ref=ref, orders=orders[:max_idx+1])
    log_like = np.array([model.log_likelihood(ratio=ratio) for ratio in ratios])
    log_like += logprior
    posterior = np.exp(log_like - np.max(log_like))
    posterior /= np.trapz(posterior, x=Lb)  # Normalize

    bounds = np.zeros((2,2))
    for i, p in enumerate([0.68, 0.95]):
        bounds[i] = gm.hpd_pdf(pdf=posterior, alpha=p, x=Lb)

    median = gm.median_pdf(pdf=posterior, x=Lb)
    return posterior, bounds, median

def find_nearest_val(array, value):
    """
    Finds the value in array closest to value and returns that entry.

    Parameters
    ----------
    array (array) : 1-d array.
    value (float) : number.

    Returns
    -------
    (float) entry in array closest to value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_idx(array, value):
    """
    Finds the value in array closest to value and returns that entry.

    Parameters
    ----------
    array (array) : 1-d array.
    value (float) : number.

    Returns
    -------
    idx (int) : index of the entry in array closest to value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def mask_mapper(array_from, array_to, mask_from):
    """
    Converts from one mask to another by mapping the entries of the first to the nearest-in-
    value entries in the second.

    Parameters
    ----------
    array_from (float array) : 1-d array.
    array_to (float array) : 1-d array.
    mask_from (bool array) : 1-d array of boolean values for array_from.

    Returns
    -------
    (bool array) : indices to be applied to array_to to get as close as possible in value to array_from[mask_from].
    """
    mask_array = [( np.argwhere(array_to == find_nearest_val(array_to, i)) ) for i in array_from[mask_from]]
    mask = np.zeros(len(array_from))
    for i in range(len(mask_array)):
        mask[mask_array[i]] = 1
    return np.array(mask.astype(int), dtype = bool)


def versatile_train_test_split(interp_obj, n_train, n_test_inter=1, isclose_factor=0.01, \
                               offset_train_min=0, offset_train_max=0, xmin_train=None, xmax_train=None, \
                               offset_test_min=0, offset_test_max=0, xmin_test=None, xmax_test=None, \
                               train_at_ends=True, test_at_ends=False):
    """
    Returns the training and testing points in the input space and the corresponding
    (interpolated) data values

    Parameters
    ----------
    interp_obj (InterpObj) : function generated with scipy.interpolate.interp1d(x, y), plus
        x and y
    n_train (int) : number of intervals into which to split x, with training points at the
        edges of each interval
    n_test_inter (int) : number of subintervals into which to split the intervals between
        training points, with testing points at the edges of each subinterval
    isclose_factor (float) : fraction of the total input space for the tolerance of making
        sure that training and testing points don't coincide
    offset_train_min (float) : value above the minimum of the input space where the first
        potential training point ought to go
    offset_train_max (float) : value below the maximum of the input space where the last
        potential training point ought to go
    xmin_train (float) : minimum value within the input space below which there ought not to
        be training points
    xmax_train (float) : maximum value within the input space above which there ought not to
        be training points
    offset_test_min (float) : value above the minimum of the input space where the first
        potential testing point ought to go
    offset_test_max (float) : value below the maximum of the input space where the last
        potential testing point ought to go
    xmin_test (float) : minimum value within the input space below which there ought not to
        be testing points
    xmax_test (float) : maximum value within the input space above which there ought not to
        be testing points
    train_at_ends (bool) : whether training points should be allowed at or near the
        endpoints of x
    test_at_ends (bool) : whether testing points should be allowed at or near the endpoints
        of x
    """
    # gets information from the InterpObj
    x = interp_obj.x
    y = interp_obj.y
    kind_interp = interp_obj.kind
    f_interp = interp_obj.f_interp

    # creates initial sets of training and testing x points
    x_train = np.linspace(np.min(x) + offset_train_min, np.max(x) - offset_train_max, \
                          n_train + 1)
    x_test = np.linspace(np.min(x) + offset_test_min, np.max(x) - offset_test_max, \
                         n_train * n_test_inter + 1)

    # sets the xmin and xmax values to the minima and maxima, respectively, of the
    # input space if no other value is given
    if xmin_train == None: xmin_train = np.min(x);
    if xmax_train == None: xmax_train = np.max(x);
    if xmin_test == None: xmin_test = np.min(x);
    if xmax_test == None: xmax_test = np.max(x);

    # eliminates, using a mask, all values for the training and testing x points outside of
    # x
    x_train = x_train[np.invert([(x_train[i] < np.min(x) or x_train[i] > np.max(x)) \
                                 for i in range(len(x_train))])]
    x_test = x_test[np.invert([(x_test[i] < np.min(x) or x_test[i] > np.max(x)) \
                               for i in range(len(x_test))])]

    # eliminates, using a mask, all values for the training and testing x points outside of
    # the bounds specified by xmin and xmax
    x_train = x_train[np.invert([(x_train[i] < xmin_train or x_train[i] > xmax_train) \
                                 for i in range(len(x_train))])]
    x_test = x_test[np.invert([(x_test[i] < xmin_test or x_test[i] > xmax_test) \
                               for i in range(len(x_test))])]

    # eliminates, using a mask, all values in the testing x points that are close enough
    # (within some tolerance) to any value in the training x points
    mask_filter_array = [[np.isclose(x_test[i], x_train[j], \
                                     atol=isclose_factor * (np.max(x) - np.min(x))) \
                          for i in range(len(x_test))] for j in range(len(x_train))]
    mask_filter_list = np.invert(np.sum(mask_filter_array, axis=0, dtype=bool))
    x_test = x_test[mask_filter_list]

    # evaluates the interpolater at the training and testing x points
    y_train = f_interp(x_train)
    y_test = f_interp(x_test)

    # eliminates training and/or testing points if they lie at the edges of the input space
    if not train_at_ends:
        if np.isclose(x_train[0], x[0], atol=isclose_factor * (np.max(x) - np.min(x))):
            x_train = x_train[1:]
            if y_train.ndim == 3:
                y_train = y_train[:, :, 1:]
            elif y_train.ndim == 2:
                y_train = y_train[:, 1:]
        if np.isclose(x_train[-1], x[-1], atol=isclose_factor * (np.max(x) - np.min(x))):
            x_train = x_train[:-1]
            if y_train.ndim == 3:
                y_train = y_train[:, :, :-1]
            elif y_train.ndim == 2:
                y_train = y_train[:, :-1]
    if not test_at_ends:
        if np.isclose(x_test[0], x[0], atol=isclose_factor * (np.max(x) - np.min(x))):
            x_test = x_test[1:]
            if y_test.ndim == 3:
                y_test = y_test[:, :, 1:]
            elif y_test.ndim == 2:
                y_test = y_test[:, 1:]
        if np.isclose(x_test[-1], x[-1], atol=isclose_factor * (np.max(x) - np.min(x))):
            x_test = x_test[:-1]
            if y_test.ndim == 3:
                y_test = y_test[:, :, :-1]
            elif y_test.ndim == 2:
                y_test = y_test[:, :-1]

    return x_train, x_test, y_train, y_test

# def versatile_train_test_split_nd(x, y, n_train, n_test_inter=1, isclose_factor=0.01, \
#                                offset_train_min=0, offset_train_max=0, xmin_train=None, xmax_train=None, \
#                                offset_test_min=0, offset_test_max=0, xmin_test=None, xmax_test=None, \
#                                train_at_ends=True, test_at_ends=False):
def versatile_train_test_split_nd(tts):
    """
    Returns the training and testing points in the input space and the corresponding
    (interpolated) data values

    Parameters
    ----------
    interp_obj (InterpObj) : function generated with scipy.interpolate.interp1d(x, y), plus
        x and y
    n_train (int array) : number of intervals into which to split x, with training points at the
        edges of each interval
    n_test_inter (int array) : number of subintervals into which to split the intervals between
        training points, with testing points at the edges of each subinterval
    isclose_factor (float array) : fraction of the total input space for the tolerance of making
        sure that training and testing points don't coincide
    offset_train_min (float array) : value above the minimum of the input space where the first
        potential training point ought to go
    offset_train_max (float array) : value below the maximum of the input space where the last
        potential training point ought to go
    xmin_train (float array) : minimum value within the input space below which there ought not to
        be training points
    xmax_train (float array) : maximum value within the input space above which there ought not to
        be training points
    offset_test_min (float array) : value above the minimum of the input space where the first
        potential testing point ought to go
    offset_test_max (float array) : value below the maximum of the input space where the last
        potential testing point ought to go
    xmin_test (float array) : minimum value within the input space below which there ought not to
        be testing points
    xmax_test (float array) : maximum value within the input space above which there ought not to
        be testing points
    train_at_ends (bool array) : whether training points should be allowed at or near the
        endpoints of x
    test_at_ends (bool array) : whether testing points should be allowed at or near the endpoints
        of x
    """
    # creates initial sets of training and testing x points
    x_train = gm.cartesian(*[np.linspace(np.amin(tts.x, axis = tuple(range(tts.x.ndim - 1)))[idx] + tts.offset_train_min[idx],
                          np.amax(tts.x, axis = tuple(range(tts.x.ndim - 1)))[idx] - tts.offset_train_max[idx],
                          tts.n_train[idx] + 1) for idx in range(tts.y.ndim - 1)])
    x_test = gm.cartesian(
        *[np.linspace(np.amin(tts.x, axis=tuple(range(tts.x.ndim - 1)))[idx] + tts.offset_test_min[idx],
                      np.amax(tts.x, axis=tuple(range(tts.x.ndim - 1)))[idx] - tts.offset_test_max[idx],
                      tts.n_train[idx] * tts.n_test_inter[idx] + 1) for idx in range(tts.y.ndim - 1)])

    # eliminates, using a mask, all values for the training and testing x points outside of...
    if np.shape(tts.x)[-1] == 2:
        # ... the quadrilateral described by the boundaries of the input space
        warped_poly = Polygon(np.concatenate([
            tts.x[0, :, ...],
            tts.x[:, -1, ...],
            tts.x[-1, :, ...],
            np.flip(tts.x[:, 0, ...], axis=0),
        ]))

        x_train = x_train[[warped_poly.buffer(0.001).contains(Point(pt)) for pt in x_train], ...]
        x_test = x_test[[warped_poly.buffer(0.001).contains(Point(pt)) for pt in x_test], ...]

    elif np.shape(tts.x)[-1] == 1:
        # ... the range encompassed by the max. and min. of the input space
        x_train = x_train[
                           [(pt >= np.min(tts.x[:, 0]) and pt <= np.max(tts.x[:, 0])) for pt in x_train], ...][:,
                       None]
        x_test = x_test[
                          [(pt >= np.min(tts.x[:, 0]) and pt <= np.max(tts.x[:, 0])) for pt in x_test], ...][:,
                      None]

    # eliminates, using a mask, all values for the training and testing x points outside of
    # the bounds specified by xmin and xmax
    x_train = x_train[np.prod(np.invert(np.less(x_train, tts.xmin_train) + \
                                        np.greater(x_train, tts.xmax_train)),
                              axis=-1).astype(bool)]
    x_test = x_test[np.prod(np.invert(np.less(x_test, tts.xmin_test) + \
                                      np.greater(x_test, tts.xmax_test)),
                            axis=-1).astype(bool)]

    # eliminates, using a mask, all values in the testing x points that are close enough
    # (within some tolerance) to any value in the training x points
    mask_filter_array = np.array([[np.isclose(x_test_tuple, x_train_tuple,
                                     rtol=tts.isclose_factor)
                          for x_test_tuple in x_test]
                         for x_train_tuple in x_train])

    mask_filter_array = np.invert(mask_filter_array)
    mask_filter_array = np.prod(mask_filter_array, axis=0, dtype=bool)
    mask_filter_array = np.sum(mask_filter_array, axis=-1, dtype=bool)
    x_test = x_test[mask_filter_array]

    # evaluates the interpolater at the training and testing x points
    y_train = np.array([])
    y_test = np.array([])
    for norder in tts.y:
        y_train = np.append(y_train, scipy.interpolate.griddata(
                                             np.reshape(tts.x, (np.prod(np.shape(tts.x)[0:-1]), ) + (np.shape(tts.x)[-1], )),
                                             np.reshape(norder, np.prod(np.shape(norder))),
                                             x_train)
                            )
        y_test = np.append(y_test, scipy.interpolate.griddata(
            np.reshape(tts.x, (np.prod(np.shape(tts.x)[0:-1]),) + (np.shape(tts.x)[-1],)),
            np.reshape(norder, np.prod(np.shape(norder))),
            x_test)
                            )
    y_train = np.reshape(y_train, (np.shape(tts.y)[0], ) + (np.shape(x_train)[0], ))
    y_test = np.reshape(y_test, (np.shape(tts.y)[0], ) + (np.shape(x_test)[0], ))

    return x_train, x_test, y_train, y_test

def get_nn_online_data():
    # We get the NN data from a separate place in our github respository.
    nn_online_pot = "pwa93"
    nn_online_url = "https://github.com/buqeye/buqeyebox/blob/master/nn_scattering/NN-online-Observables.h5?raw=true"
    nno_response = urllib.request.urlopen(nn_online_url)
    nn_online_file = tables.open_file(
        "nn_online_example.h5",
        driver="H5FD_CORE",
        driver_core_image=nno_response.read(),
        driver_core_backing_store=0,
    )
    SGT_nn_online = nn_online_file.get_node("/" + nn_online_pot + "/SGT").read()
    DSG_nn_online = nn_online_file.get_node("/" + nn_online_pot + "/DSG").read()[:, :-1]
    AY_nn_online = nn_online_file.get_node("/" + nn_online_pot + "/PB").read()[:, :-1]
    A_nn_online = nn_online_file.get_node("/" + nn_online_pot + "/A").read()[:, :-1]
    D_nn_online = nn_online_file.get_node("/" + nn_online_pot + "/D").read()[:, :-1]
    AXX_nn_online = nn_online_file.get_node("/" + nn_online_pot + "/AXX").read()[:, :-1]
    AYY_nn_online = nn_online_file.get_node("/" + nn_online_pot + "/AYY").read()[:, :-1]

    # creates a dictionary that links the NN online data for each observable to the
    # eventual predictions for that observable by a given potential scheme and scale
    online_data_dict = {
        "SGT": SGT_nn_online,
        "DSG": DSG_nn_online,
        "AY": AY_nn_online,
        "A": A_nn_online,
        "D": D_nn_online,
        "AXX": AXX_nn_online,
        "AYY": AYY_nn_online,
    }

    return online_data_dict