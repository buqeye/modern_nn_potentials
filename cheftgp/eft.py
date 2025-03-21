import numpy as np
import math


def Q_approx(p, Q_parametrization, Lambda_b, m_pi=138, single_expansion=False):
    """
    Returns the dimensionless expansion parameter Q.

    Parameters
    ----------
    p (float or array) : momentum (in MeV)
    Q_parametrization (str) : can be "smax", "max", "sum", or "rawsum"
    Lambda_b (float) : value for the cutoff (in MeV)
    m_pi (float) : value for the pion mass (in MeV)
        default : 138
    single_expansion (bool) : whether the soft scale should take into account only p
        default : False
    """
    if single_expansion:
        # for expansions with the momentum p as the only soft scale
        m_pi = 0

    if Q_parametrization == "smax":
        # Interpolate to smooth the transition from m_pi to p with a ratio
        # of polynomials
        n = 8
        q = (m_pi**n + p**n) / (m_pi ** (n - 1) + p ** (n - 1)) / Lambda_b
        return np.array(q, dtype = np.float64)

    elif Q_parametrization == "max":
        # Transition from m_pi to p with a maximum function
        try:
            q = max(p, m_pi) / Lambda_b
        except:
            q = (
                np.reshape(
                    [max(p_val, m_pi) for p_val in np.array(p).flatten()], np.shape(p)
                )
                / Lambda_b
            )
        return q

    elif Q_parametrization == "sum":
        # Transition from m_pi to p with a simple sum and a scaling factor (k_sum) as a function of mpi
        q = (p + m_pi) / (m_pi + Lambda_b)
        return q

    elif Q_parametrization == "rawsum":
        # Transition from m_pi to p with a simple sum
        q = (p + m_pi) / Lambda_b
        return q


def Qsum_to_Qsmoothmax(m_pi):
    """
    Converts the denominator of the dimensionless expansion parameter from the Qsmax to the Qsum prescription, based on
    a rough (linear) empirical formula, in terms of the value of m_pi in the numerator.

    Parameters
    ----------
    m_pi (float or array) : pion mass value(s) (in MeV).
    """
    return (m_pi + 750) / 600
    # return 1.7


def p_approx(p_name, prel, degrees):
    """
    Returns the dimensionless expansion parameter Q.

    Parameters
    ----------
    p_name (str): name for the parametrization of the momentum
    prel (float or array): relative momentum for the interaction (in MeV)
    degrees (float or array): degrees
    """
    if p_name == "Qofprel" or p_name == "pprel":
        return np.tile(np.array(prel), (len(degrees), 1))

    elif p_name == "Qofqcm" or p_name == "pqcm":
        return np.array([np.array(deg_to_qcm(prel, d)) for d in degrees])

    elif p_name == "Qofpq" or p_name == "psmax":
        return np.array(
            [[softmax_mom(p, deg_to_qcm(p, d)) for p in prel] for d in degrees]
        )
    elif p_name == "Qofpqmax" or p_name == "pmax":
        return np.array([[max(p, deg_to_qcm(p, d)) for p in prel] for d in degrees])


def deg_fn(deg_input, **kwargs):
    """
    Converts degrees to degrees.

    Parameters
    ----------
    deg_input (float or array) : angle measure value(s) (in degrees).
    """
    return deg_input


def neg_cos(deg_input, **kwargs):
    """
    Converts degrees to the negative of the cosine.

    Parameters
    ----------
    deg_input (float or array) : angle measure value(s) (in degrees).
    """
    return -1 * np.cos(np.radians(deg_input))


def deg_to_qcm(p_input, deg_input, **kwargs):
    """
    Returns the center-of-momentum momentum transfer q in MeV (shape: p_input x deg_input).

    Parameters
    ----------
    p_input (float array) : relative momentum given in MeV.
    deg_input (float array) : angle measure given in degrees
    """
    try:
        return np.array(
            [
                np.array(p_input * np.sqrt(2 * (1 - np.cos(np.radians(d)))))
                for d in deg_input
            ]
        ).T
    except:
        return p_input * np.sqrt(2 * (1 - np.cos(np.radians(deg_input))))


def deg_to_qcm2(p_input, deg_input, **kwargs):
    """
    Returns the center-of-momentum momentum transfer q squared, in MeV^2 (shape: p_input x deg_input).

    Parameters
    ----------
    p_input (float) : relative momentum given in MeV.
    deg_input (float) : angle measure given in degrees.
    """
    try:
        return np.array(
            [
                np.array((p_input * np.sqrt(2 * (1 - np.cos(np.radians(d))))) ** (2))
                for d in deg_input
            ]
        ).T
    except:
        return (p_input * np.sqrt(2 * (1 - np.cos(np.radians(deg_input))))) ** (2)


def Elab_fn(E_lab, **kwargs):
    """
    Converts lab energy to lab energy.

    Parameters
    ----------
    E_lab (float or array) : lab energy value(s) (in MeV).
    """
    return E_lab


def sin_thing(deg_input, **kwargs):
    """
    Converts degrees to a rather jury-rigged functon of the inverse of hyperbolic tangent.

    Parameters
    ----------
    deg_input (float or array) : angle measure value(s) (in degrees).
    """
    return 0.6 * (1.6 + np.arctanh(np.radians(deg_input - 90) / 1.7))


def softmax_mom(p, q, n=5):
    """
    Two-place softmax function.

    Parameters
    ----------
    p (float) : one interpolant.
    q (float) : another interpolant.
    n (float) : scaling parameter.
        default : 5
    """
    return 1 / n * math.log(1.01 ** (n * p) + 1.01 ** (n * q), 1.01)


def Lb_logprior(Lambda_b):
    """
    Uniform log-prior for the breakdown scale (in MeV).
    Similar to Melendez et al., Eq. (31)
    """
    return np.where((0 <= Lambda_b) & (Lambda_b <= 4000), 0, -np.inf)
    # return np.where((200 <= Lambda_b) & (Lambda_b <= 1000), 0, -np.inf)


def mpieff_logprior(m_pi):
    """
    Uniform log-prior for the effective pion mass (in MeV).
    Similar to Melendez et al., Eq. (31)
    """
    # return np.where((50 <= m_pi) & (m_pi <= 300), np.log(1. / m_pi), -np.inf)
    return np.where((0 <= m_pi) & (m_pi <= 4000), 0, -np.inf)
