def Q_approx(p, Q_parametrization, Lambda_b, m_pi=138, interaction='np',
             single_expansion=False):
    """
    Returns the dimensionless expansion parameter Q.

    Parameters
    ----------
    p (float or array) : momentum (in MeV)
    Q_parametrization (str) : can be "smoothmax", "max", or "sum"
    Lambda_b (float) : value for the cutoff (in MeV)
    interaction (str) : can be "np", "nn", or "pp"
    """
    if single_expansion:
        m_pi = 0
    # else:
    #     m_pi = 200  # Set to 0 to just return p/Lambda_b

    #     p = E_to_p(E, interaction)

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