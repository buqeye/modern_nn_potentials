import numpy as np
import decimal
import gsum as gm

def correlation_coefficient(x, y, pdf):
    # normalizes the pdf
    pdf /= np.trapz(np.trapz(pdf, x=y, axis=0), x=x, axis=0)
    # print("pdf = " + str(pdf))

    # finds the maximum value
    pdf_max = np.amax(pdf)
    # print("pdf_max = " + str(pdf_max))

    # figures out the x and y coordinates of the max
    # print(np.argwhere(pdf == pdf_max))
    x_max = x[np.argwhere(pdf == pdf_max)[0, 1]]
    # print("x_max = " + str(x_max))
    y_max = y[np.argwhere(pdf == pdf_max)[0, 0]]
    # print("y_max = " + str(y_max))

    # finds variance in x and y
    sigma_x_sq = np.trapz(
        np.trapz(np.tile((x - x_max) ** 2, (len(y), 1)) * pdf,
                 x=x, axis=1),
        x=y, axis=0)
    # print("sigma_x_sq = " + str(sigma_x_sq))
    #     sigma_x_sq = np.trapz(pdf @ (x - x_max)**2, x = y, axis = 0)
    #     print("sigma_x_sq = " + str(sigma_x_sq))
    sigma_y_sq = np.trapz(
        np.trapz(np.tile(np.reshape((y - y_max) ** 2, (len(y), 1)), (1, len(x))) * pdf,
                 x=y, axis=0),
        x=x, axis=0)
    #     print("sigma_y_sq = " + str(sigma_y_sq))
    #     sigma_y_sq = np.trapz((y - y_max)**2 @ pdf, x = x, axis = 0)
    # print("sigma_y_sq = " + str(sigma_y_sq))

    # finds sigmaxy
    sigma_xy_sq = np.trapz(
        np.trapz(np.tile(np.reshape(y - y_max, (len(y), 1)), (1, len(x))) * \
                 np.tile(x - x_max, (len(y), 1)) * pdf,
                 x=x, axis=1),
        x=y, axis=0)
    # print("sigma_xy_sq = " + str(sigma_xy_sq))
    #     sigma_xy_sq = (y - y_max) @ pdf @ (x - x_max)
    #     print("sigma_xy_sq = " + str(sigma_xy_sq))

    # finds the correlation coefficient
    corr_coeff = sigma_xy_sq / (np.sqrt(sigma_x_sq) * np.sqrt(sigma_y_sq))
    # print(corr_coeff)

    return corr_coeff


def mean_and_stddev(x, pdf):
    # normalizes the pdf
    pdf /= np.trapz(pdf, x=x, axis=0)
    # print("pdf = " + str(pdf))

    # finds the maximum value
    pdf_max = np.amax(pdf)
    # print("pdf_max = " + str(pdf_max))

    # figures out the x coordinate of the max
    # print(np.argwhere(pdf == pdf_max))
    x_max = x[np.argwhere(pdf == pdf_max)][0]
    # print("x_max = " + str(x_max))

    # finds the mean
    mean = np.trapz(x * pdf, x=x, axis=0)
    # print("mean = " + str(mean))

    # finds the standard deviation
    sigma_x = np.sqrt(np.trapz((x - x_max) ** 2 * pdf, x=x, axis=0))
    # print("sigma_x = " + str(sigma_x))

    return mean, sigma_x


def sig_figs(number, n_figs):
    """
    Parameters
    ----------
    number : float
        A number.
    n_figs : int
        Number of significant figures.

    Returns
    -------
    None.

    """
    # formats the number as a string
    number_string = np.format_float_positional(
        np.float64(
            np.format_float_scientific(
                number, precision=n_figs - 1)))
    # print("number_string = " + number_string)

    # eliminates any unncessary zeros and decimal points
    # while((np.float64(number_string) > 10**(n_figs - 1)) and ((number_string[-1] == '0') or (number_string[-1] == '.'))):
    #     number_string = number_string[:-1]
    if ((np.float64(number_string) > 10 ** (n_figs - 1)) and (number_string[-1] == '.')):
        number_string = number_string[:-1]
        # print("We chopped off the decimal point.")
        return np.int(number_string)
    else:
        return np.float64(number_string)




def round_to_same_digits(number, comparand):
    # print("We called the function correctly.")
    # print(str(comparand))
    # print(str(decimal.Decimal(str(comparand)).as_tuple().exponent))
    if decimal.Decimal(str(comparand)).as_tuple().exponent == 0:
        return int(number)
    else:
        return np.around(number, decimals = decimal.Decimal(str(comparand)).as_tuple().exponent)

def compute_posterior_intervals(model, data, ratios, ref, orders, max_idx, logprior, Lb):
    # print("We're about to fit.")
    # print("data has shape " + str(np.shape(data[:max_idx+1].T)))
    # print("ratio has shape " + str(np.shape(ratios[0])))
    # print("ref has shape " + str(np.shape(ref)))
    # print("orders has shape " + str(np.shape(orders[:max_idx+1])))
    model.fit(data[:max_idx+1].T, ratio=ratios[0], ref=ref, orders=orders[:max_idx+1])
    # raise ValueError("something")
    log_like = np.array([model.log_likelihood(ratio=ratio) for ratio in ratios])
    log_like += logprior
    posterior = np.exp(log_like - np.max(log_like))
    posterior /= np.trapz(posterior, x=Lb)  # Normalize

    bounds = np.zeros((2,2))
    for i, p in enumerate([0.68, 0.95]):
        # bounds[i] = gm.hpd_pdf(pdf=posterior, alpha=p, x=Lb, disp=False)
        bounds[i] = gm.hpd_pdf(pdf=posterior, alpha=p, x=Lb)

    median = gm.median_pdf(pdf=posterior, x=Lb)
    return posterior, bounds, median

def find_nearest_val(array, value):
    """
    Finds the value in array closest to value and returns that entry.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_idx(array, value):
    """
    Finds the value in array closest to value and returns that entry.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def mask_mapper(array_from, array_to, mask_from):
    """
    Converts from one mask to another by mapping the entries of the first to the nearest-in-
    value entries in the second.
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