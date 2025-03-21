import numpy as np
from .constants import *


def E_to_p(E_lab, interaction, **kwargs):
    """
    Returns p (in MeV).

    Parameters
    ----------
    E_lab (float) : lab energy (in MeV).
    interaction (str) : abbreviation for the interaction.
        options: "pp", "nn", "np"
    """

    if interaction == "pp":
        m1, m2 = m_p, m_p
    if interaction == "nn":
        m1, m2 = m_n, m_n
    if interaction == "np":
        m1, m2 = m_n, m_p
    p_rel = np.sqrt(
        E_lab * m2**2 * (E_lab + 2 * m1) / ((m1 + m2) ** 2 + 2 * m2 * E_lab)
    )
    return p_rel


def mom_fn_degrees(degrees):
    """
    Returns momentum (in MeV) given degrees.

    Parameters
    ----------
    degrees (any) : scattering angle measure (in degrees)
    """
    return degrees
