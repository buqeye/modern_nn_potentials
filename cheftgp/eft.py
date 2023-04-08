import numpy as np
import math


def Q_approx(p, Q_parametrization, Lambda_b, m_pi=138,
             single_expansion=False):
    """
    Returns the dimensionless expansion parameter Q.

    Parameters
    ----------
    p (float or array) : momentum (in MeV)
    Q_parametrization (str) : can be "smoothmax", "max", or "sum"
    Lambda_b (float) : value for the cutoff (in MeV)
    """
    if single_expansion:
        m_pi = 0

    if Q_parametrization == "smoothmax":
        # Interpolate to smooth the transition from m_pi to p with a ratio
        # of polynomials
        n = 8
        q = (m_pi ** n + p ** n) / (m_pi ** (n - 1) + p ** (n - 1)) / Lambda_b
        return q

    elif Q_parametrization == "max":
        # Transition from m_pi to p with a maximum function
        try:
            q = [max(P, m_pi) / Lambda_b for P in p]
        except:
            q = max(p, m_pi) / Lambda_b
        return q

    elif Q_parametrization == "sum":
        # Transition from m_pi to p with a simple sum
        q = (p + m_pi) / (Qsum_to_Qsmoothmax(m_pi) * Lambda_b)
        # q = (p + m_pi) / Lambda_b
        return q

def Qsum_to_Qsmoothmax(m_pi):
    return m_pi / 320 + 1


def p_approx(p_name, prel, degrees):
    """
    Returns the dimensionless expansion parameter Q.

    Parameters
    ----------
    p_name (str): name for the parametrization of the momentum
    prel (float): relative momentum for the interaction (in MeV)
    degrees (float array): degrees
    """

    if p_name == "Qofprel":
        try:
            return np.array(prel * np.ones(len(degrees)))
        except:
            return np.array(prel)

    elif p_name == "Qofqcm":
        return np.array(deg_to_qcm(prel, degrees))

    elif p_name == "Qofpq":
        return np.array([softmax_mom(prel, q)
                         for q in deg_to_qcm(prel, degrees)])


def deg_fn(deg_input, **kwargs):
    return deg_input


def neg_cos(deg_input, **kwargs):
    return -1 * np.cos(np.radians(deg_input))


def deg_to_qcm(p_input, deg_input, **kwargs):
    """
    Returns the center-of-momentum momentum transfer q in MeV.

    Parameters
    ----------
    p_rel       = float
                  relative momentum given in MeV.
    degrees     = number
                  angle measure given in degrees
    """
    return p_input * np.sqrt(2 * (1 - np.cos(np.radians(deg_input))))


def deg_to_qcm2(p_input, deg_input, **kwargs):
    """
    Returns the center-of-momentum momentum transfer q squared, in MeV^2.

    Parameters
    ----------
    p_rel       = float
                  relative momentum given in MeV.
    degrees     = number
                  angle measure given in degrees
    """
    return (p_input * np.sqrt(2 * (1 - np.cos(np.radians(deg_input))))) ** (2)


def Elab_fn(E_lab, **kwargs):
    return E_lab


def sin_thing(deg, **kwargs):
    return np.array([np.sin(np.radians(d)) if d <= 90 else 2 - np.sin(np.radians(d)) for d in deg])


def softmax_mom(p, q, n=5):
    """
    Two-place softmax function.

    Parameters
    ----------
    p       = float
            one interpolant.
    q       = float
            another interpolant.
    n       = float
            scaling parameter.
    """
    return 1 / n * math.log(1.01 ** (n * p) + 1.01 ** (n * q), 1.01)

def Lb_logprior(Lambda_b):
    """Melendez et al., Eq. (31)"""
    # return np.where((300 <= Lambda_b) & (Lambda_b <= 1500), np.log(1. / Lambda_b), -np.inf)
    return np.where((300 <= Lambda_b) & (Lambda_b <= 1500), 0, -np.inf)

def mpieff_logprior(m_pi):
    """Melendez et al., Eq. (31)"""
    # return np.where((50 <= m_pi) & (m_pi <= 300), np.log(1. / m_pi), -np.inf)
    return np.where((10 <= m_pi) & (m_pi <= 500), 0, -np.inf)