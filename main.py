import math
import warnings
from itertools import product

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psychrolib
import scipy
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from numba import jit
from pythermalcomfort.models import (
    athb,
    cooling_effect,
)
from pythermalcomfort.psychrometrics import p_sat_torr
from pythermalcomfort.utilities import check_standard_compliance_array
from scipy import optimize
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score, mean_absolute_error

mpl.use("TkAgg")

warnings.filterwarnings("ignore")


psychrolib.SetUnitSystem(psychrolib.SI)


palette_tp = [
    "#06A6EE",
    "#31CAA8",
    "#FF412C",
]
palette_tp = [
    "#4F96FF",
    "#60E693",
    "#FF362B",
]
palette_tp = [
    "#C40025",
    "#008D3D",
    "#0067B2",
]
palette_tsv = [
    "#8DA2D8",
    "#A9D4F5",
    "#D1EAFA",
    "#D2E8BF",
    "#F0BECB",
    "#DE7B6A",
    "#A65558",
]

# sns.palplot(palette_tsv)
# sns.palplot(palette_tp)
# sns.palplot(palette_primary)

percentiles_to_show = [0.025, 0.25, 0.5, 0.75, 0.975]

palette_primary = [
    "#FFBA22",
    "#FF2380",
    "#6600AB",
    "#FF7BB3",
    "#003CAB",
    "#83DFCB",
    "#C20800",
    "#C57FFF",
    "#FF8D80",
    "#6ACAF5",
    "#006B46",
    "#FF6756",
    "#9F29FF",
    "#8001D8",
    "#FF4F99",
    "#006BDB",
    "#5AD5B9",
    "#E8281E",
    "#B254FF",
    "#149678",
    "#FFE146",
]

sns.set_context("paper")
mpl.rcParams["figure.figsize"] = [7.0, 3.5]
sns.set_style(
    "whitegrid",
    {
        "grid.color": ".85",
        "grid.linewidth": "1",
        "grid.linestyle": "--",
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
    },
)

c_gold = "#FDB515"
c_brown = "#6C3302"
c_blue = "#3B7EA1"
c_green = "#53626F"

plt.rc("axes.spines", top=False, right=False, left=False)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 8.8

applicability_limits = {
    "ta": [10, 30],
    "tr": [10, 40],
    "vel_r": [0, 1],
    "clo": [0, 1.5],
    "met": [1, 4],
    "thermal_sensation": [-3.5, 3.5],
    "pmv": [-3.49999, 3.5],
    "pmv_ce": [-3.49999, 3.5],
    "rh": [0, 100],
    "pa": [0, 2700],
}

var_names = {
    "ta": r"$t_{db}$",
    "tr": r"$\overline{t_{r}}$",
    "top": r"$t_{o}$",
    "vel_r": r"$V$",
    "rh": r"RH",
    "clo": r"$I_{cl}$",
    "met": r"$M$",
    "thermal_sensation": "Thermal Sensation Vote (TSV)",
    "thermal_sensation_round": "Thermal Sensation Vote (TSV)",
    "thermal_preference": "Thermal Preference Vote (TPV)",
    "age": "Age (year)",
    "ht": "Height (m)",
    "wt": "Weight (kg)",
    "t_mot_isd": r"$t_{ormt}$" + r" $(^{\circ}$C)",
    "pmv": r"PMV",
    "pmv_round": r"PMV",
    "lr_hb_pmv": r"PMV$_{hb}$",
    "lr_hb_pmv_ce": r"PMV$_{CE,hb}$",
    "lr_hb_pmv_set": r"PMV$_{SET,hb}$",
    "lr_hb_pmv_gagge": r"PMV$_{Gagge,hb}$",
    "lr_hb_athb": r"ATHB$_{hb}$",
    "pmv_ce_round": r"PMV$_{CE}$",
    "pmv_ce": r"PMV$_{CE}$",
    "pmv_set": r"PMV$_{SET}$",
    "pmv_set_round": r"PMV$_{SET}$",
    "pmv_gagge": r"PMV$_{Gagge}$",
    "pmv_gagge_round": r"PMV$_{Gagge}$",
    "pmv_toby": r"PMV$_{Toby}$",
    "athb_round": r"ATHB",
    "athb": r"ATHB",
}

var_units = {
    "ta": r"$^{\circ}$C",
    "tr": r"$^{\circ}$C",
    "vel_r": r"m/s",
    "rh": r"%",
    "clo": r"clo",
    "met": r"met",
}

models_to_test = [
    "pmv",
    "pmv_ce",
]


@jit(nopython=True)
def pmv_ppd_optimized(tdb, tr, vr, rh, met, clo, wme):
    pa = rh * 10 * math.exp(16.6536 - 4030.183 / (tdb + 235))

    icl = 0.155 * clo  # thermal insulation of the clothing in M2K/W
    m = met * 58.15  # metabolic rate in W/M2
    w = wme * 58.15  # external work in W/M2
    mw = m - w  # internal heat production in the human body
    # calculation of the clothing area factor
    if icl <= 0.078:
        f_cl = 1 + (1.29 * icl)  # ratio of surface clothed body over nude body
    else:
        f_cl = 1.05 + (0.645 * icl)

    # heat transfer coefficient by forced convection
    hcf = 12.1 * math.sqrt(vr)
    hc = hcf  # initialize variable
    taa = tdb + 273
    tra = tr + 273
    t_cla = taa + (35.5 - tdb) / (3.5 * icl + 0.1)

    p1 = icl * f_cl
    p2 = p1 * 3.96
    p3 = p1 * 100
    p4 = p1 * taa
    p5 = (308.7 - 0.028 * mw) + (p2 * (tra / 100.0) ** 4)
    xn = t_cla / 100
    xf = t_cla / 50
    eps = 0.00015

    n = 0
    while abs(xn - xf) > eps:
        xf = (xf + xn) / 2
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        if hcf > hcn:
            hc = hcf
        else:
            hc = hcn
        xn = (p5 + p4 * hc - p2 * xf**4) / (100 + p3 * hc)
        n += 1
        if n > 150:
            raise StopIteration("Max iterations exceeded")

    tcl = 100 * xn - 273

    # heat loss diff. through skin
    hl1 = 3.05 * 0.001 * (5733 - (6.99 * mw) - pa)
    # heat loss by sweating
    if mw > 58.15:
        hl2 = 0.42 * (mw - 58.15)
    else:
        hl2 = 0
    # latent respiration heat loss
    hl3 = 1.7 * 0.00001 * m * (5867 - pa)
    # dry respiration heat loss
    hl4 = 0.0014 * m * (34 - tdb)
    # heat loss by radiation
    hl5 = 3.96 * f_cl * (xn**4 - (tra / 100.0) ** 4)
    # heat loss by convection
    hl6 = f_cl * hc * (tcl - tdb)

    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    _pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    heat_loss = mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6

    return _pmv, heat_loss


def pmv_ppd(tdb, tr, vr, rh, met, clo, wme=0, standard="ISO", **kwargs):
    """Returns Predicted Mean Vote (`PMV`_) and Predicted Percentage of
    Dissatisfied ( `PPD`_) calculated in accordance to main thermal comfort
    Standards. The PMV is an index that predicts the mean value of the thermal
    sensation votes (self-reported perceptions) of a large group of people on a
    sensation scale expressed from –3 to +3 corresponding to the categories:
    cold, cool, slightly cool, neutral, slightly warm, warm, and hot. [1]_

    While the PMV equation is the same for both the ISO and ASHRAE standards, in the
    ASHRAE 55 PMV equation, the SET is used to calculate the cooling effect first,
    this is then subtracted from both the air and mean radiant temperatures, and the
    differences are used as input to the PMV model, while the airspeed is set to 0.1m/s.
    Please read more in the Note below.

    Parameters
    ----------
    tdb : float or array-like
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : float or array-like
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    vr : float or array-like
        relative air speed, default in [m/s] in [fps] if `units` = 'IP'

        Note: vr is the relative air speed caused by body movement and not the air
        speed measured by the air speed sensor. The relative air speed is the sum of the
        average air speed measured by the sensor plus the activity-generated air speed
        (Vag). Where Vag is the activity-generated air speed caused by motion of
        individual body parts. vr can be calculated using the function
        :py:meth:`pythermalcomfort.utilities.v_relative`.
    rh : float or array-like
        relative humidity, [%]
    met : float or array-like
        metabolic rate, [met]
    clo : float or array-like
        clothing insulation, [clo]

        Note: The activity as well as the air speed modify the insulation characteristics
        of the clothing and the adjacent air layer. Consequently, the ISO 7730 states that
        the clothing insulation shall be corrected [2]_. The ASHRAE 55 Standard corrects
        for the effect of the body movement for met equal or higher than 1.2 met using
        the equation clo = Icl × (0.6 + 0.4/met) The dynamic clothing insulation, clo,
        can be calculated using the function
        :py:meth:`pythermalcomfort.utilities.clo_dynamic`.
    wme : float or array-like
        external work, [met] default 0
    standard : {"ISO", "ASHRAE"}
        comfort standard used for calculation

        - If "ISO", then the ISO Equation is used
        - If "ASHRAE", then the ASHRAE Equation is used

        Note: While the PMV equation is the same for both the ISO and ASHRAE standards,
        the ASHRAE Standard Use of the PMV model is limited to air speeds below 0.10
        m/s (20 fpm).
        When air speeds exceed 0.10 m/s (20 fpm), the comfort zone boundaries are
        adjusted based on the SET model.
        This change was indroduced by the `Addendum C to Standard 55-2020`_

    Other Parameters
    ----------------
    units : {'SI', 'IP'}
        select the SI (International System of Units) or the IP (Imperial Units) system.
    limit_inputs : boolean default True
        By default, if the inputs are outsude the standard applicability limits the
        function returns nan. If False returns pmv and ppd values even if input values are
        outside the applicability limits of the model.

        The ASHRAE 55 2020 limits are 10 < tdb [°C] < 40, 10 < tr [°C] < 40,
        0 < vr [m/s] < 2, 1 < met [met] < 4, and 0 < clo [clo] < 1.5.
        The ISO 7730 2005 limits are 10 < tdb [°C] < 30, 10 < tr [°C] < 40,
        0 < vr [m/s] < 1, 0.8 < met [met] < 4, 0 < clo [clo] < 2, and -2 < PMV < 2.
    airspeed_control : boolean default True
        This only applies if standard = "ASHRAE". By default it is assumed that the
        occupant has control over the airspeed. In this case the ASHRAE 55 Standard does
        not imposes any airspeed limits. On the other hand, if the occupant has no control
        over the airspeed the ASHRAE 55 imposes an upper limit for v which varies as a
        function of the operative temperature, for more information please consult the
        Standard.

    Returns
    -------
    pmv : float or array-like
        Predicted Mean Vote
    ppd : float or array-like
        Predicted Percentage of Dissatisfied occupants, [%]

    Notes
    -----
    You can use this function to calculate the `PMV`_ and `PPD`_ in accordance with
    either the ASHRAE 55 2020 Standard [1]_ or the ISO 7730 Standard [2]_.

    .. _PMV: https://en.wikipedia.org/wiki/Thermal_comfort#PMV/PPD_method
    .. _PPD: https://en.wikipedia.org/wiki/Thermal_comfort#PMV/PPD_method
    .. _Addendum C to Standard 55-2020: https://www.ashrae.org/file%20library/technical%20resources/standards%20and%20guidelines/standards%20addenda/55_2020_c_20210430.pdf

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import pmv_ppd
        >>> from pythermalcomfort.utilities import v_relative, clo_dynamic
        >>> tdb = 25
        >>> tr = 25
        >>> rh = 50
        >>> v = 0.1
        >>> met = 1.4
        >>> clo = 0.5
        >>> # calculate relative air speed
        >>> v_r = v_relative(v=v, met=met)
        >>> # calculate dynamic clothing
        >>> clo_d = clo_dynamic(clo=clo, met=met)
        >>> results = pmv_ppd(tdb=tdb, tr=tr, vr=v_r, rh=rh, met=met, clo=clo_d)
        >>> print(results)
        {'pmv': 0.06, 'ppd': 5.1}
        >>> print(results["pmv"])
        -0.06
        >>> # you can also pass an array-like of inputs
        >>> results = pmv_ppd(tdb=[22, 25], tr=tr, vr=v_r, rh=rh, met=met, clo=clo_d)
        >>> print(results)
        {'pmv': array([-0.47,  0.06]), 'ppd': array([9.6, 5.1])}

    Raises
    ------
    StopIteration
        Raised if the number of iterations exceeds the threshold
    ValueError
        The 'standard' function input parameter can only be 'ISO' or 'ASHRAE'
    """
    default_kwargs = {"units": "SI", "limit_inputs": True, "airspeed_control": True}
    kwargs = {**default_kwargs, **kwargs}

    tdb = np.array(tdb)
    tr = np.array(tr)
    vr = np.array(vr)
    met = np.array(met)
    clo = np.array(clo)
    wme = np.array(wme)

    standard = standard.lower()
    if standard not in ["iso", "ashrae"]:
        raise ValueError(
            "PMV calculations can only be performed in compliance with ISO or ASHRAE "
            "Standards"
        )

    (
        tdb_valid,
        tr_valid,
        v_valid,
        met_valid,
        clo_valid,
    ) = check_standard_compliance_array(
        standard,
        tdb=tdb,
        tr=tr,
        v=vr,
        met=met,
        clo=clo,
        airspeed_control=kwargs["airspeed_control"],
    )

    # if v_r is higher than 0.1 follow methodology ASHRAE Appendix H, H3
    ce = 0.0
    if standard == "ashrae":
        ce = np.where(
            vr >= 0.1,
            np.vectorize(cooling_effect, cache=True)(tdb, tr, vr, rh, met, clo, wme),
            0.0,
        )

    tdb = tdb - ce
    tr = tr - ce
    vr = np.where(ce > 0, 0.1, vr)

    (pmv_array, heat_losses) = np.vectorize(pmv_ppd_optimized, cache=True)(
        tdb, tr, vr, rh, met, clo, wme
    )

    ppd_array = 100.0 - 95.0 * np.exp(
        -0.03353 * np.power(pmv_array, 4.0) - 0.2179 * np.power(pmv_array, 2.0)
    )

    # Checks that inputs are within the bounds accepted by the model if not return nan
    if kwargs["limit_inputs"]:

        all_valid = ~(
            np.isnan(tdb_valid)
            | np.isnan(tr_valid)
            | np.isnan(v_valid)
            | np.isnan(met_valid)
            | np.isnan(clo_valid)
        )
        pmv_array = np.where(all_valid, pmv_array, np.nan)
        ppd_array = np.where(all_valid, ppd_array, np.nan)

    return {
        "pmv": np.around(pmv_array, 2),
        "ppd": np.around(ppd_array, 1),
        "heat loss": np.around(heat_losses, 2),
    }


def pmv(tdb, tr, vr, rh, met, clo, wme=0, standard="ISO", **kwargs):
    """Returns Predicted Mean Vote (`PMV`_) calculated in accordance to main
    thermal comfort Standards. The PMV is an index that predicts the mean value
    of the thermal sensation votes (self-reported perceptions) of a large group
    of people on a sensation scale expressed from –3 to +3 corresponding to the
    categories: cold, cool, slightly cool, neutral, slightly warm, warm, and hot. [1]_

    While the PMV equation is the same for both the ISO and ASHRAE standards, in the
    ASHRAE 55 PMV equation, the SET is used to calculate the cooling effect first,
    this is then subtracted from both the air and mean radiant temperatures, and the
    differences are used as input to the PMV model, while the airspeed is set to 0.1m/s.
    Please read more in the Note below.

    Parameters
    ----------
    tdb : float or array-like
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : float or array-like
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    vr : float or array-like
        relative air speed, default in [m/s] in [fps] if `units` = 'IP'

        Note: vr is the relative air speed caused by body movement and not the air
        speed measured by the air speed sensor. The relative air speed is the sum of the
        average air speed measured by the sensor plus the activity-generated air speed
        (Vag). Where Vag is the activity-generated air speed caused by motion of
        individual body parts. vr can be calculated using the function
        :py:meth:`pythermalcomfort.utilities.v_relative`.
    rh : float or array-like
        relative humidity, [%]
    met : float or array-like
        metabolic rate, [met]
    clo : float or array-like
        clothing insulation, [clo]

        Note: The activity as well as the air speed modify the insulation characteristics
        of the clothing and the adjacent air layer. Consequently, the ISO 7730 states that
        the clothing insulation shall be corrected [2]_. The ASHRAE 55 Standard corrects
        for the effect of the body movement for met equal or higher than 1.2 met using
        the equation clo = Icl × (0.6 + 0.4/met) The dynamic clothing insulation, clo,
        can be calculated using the function
        :py:meth:`pythermalcomfort.utilities.clo_dynamic`.
    wme : float or array-like
        external work, [met] default 0
    standard : {"ISO", "ASHRAE"}
        comfort standard used for calculation

        - If "ISO", then the ISO Equation is used
        - If "ASHRAE", then the ASHRAE Equation is used

        Note: While the PMV equation is the same for both the ISO and ASHRAE standards,
        the ASHRAE Standard Use of the PMV model is limited to air speeds below 0.10
        m/s (20 fpm).
        When air speeds exceed 0.10 m/s (20 fpm), the comfort zone boundaries are
        adjusted based on the SET model.
        This change was indroduced by the `Addendum C to Standard 55-2020`_

    Other Parameters
    ----------------
    units : {'SI', 'IP'}
        select the SI (International System of Units) or the IP (Imperial Units) system.
    limit_inputs : boolean default True
        By default, if the inputs are outsude the standard applicability limits the
        function returns nan. If False returns pmv and ppd values even if input values are
        outside the applicability limits of the model.

        The ASHRAE 55 2020 limits are 10 < tdb [°C] < 40, 10 < tr [°C] < 40,
        0 < vr [m/s] < 2, 1 < met [met] < 4, and 0 < clo [clo] < 1.5.
        The ISO 7730 2005 limits are 10 < tdb [°C] < 30, 10 < tr [°C] < 40,
        0 < vr [m/s] < 1, 0.8 < met [met] < 4, 0 < clo [clo] < 2, and -2 < PMV < 2.
    airspeed_control : boolean default True
        This only applies if standard = "ASHRAE". By default it is assumed that the
        occupant has control over the airspeed. In this case the ASHRAE 55 Standard does
        not impose any airspeed limits. On the other hand, if the occupant has no control
        over the airspeed the ASHRAE 55 imposes an upper limit for v which varies as a
        function of the operative temperature, for more information please consult the
        Standard.

    Returns
    -------
    pmv : float or array-like
        Predicted Mean Vote

    Notes
    -----
    You can use this function to calculate the `PMV`_ [1]_ [2]_.

    .. _PMV: https://en.wikipedia.org/wiki/Thermal_comfort#PMV/PPD_method
    .. _Addendum C to Standard 55-2020: https://www.ashrae.org/file%20library/technical%20resources/standards%20and%20guidelines/standards%20addenda/55_2020_c_20210430.pdf

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import pmv
        >>> from pythermalcomfort.utilities import v_relative, clo_dynamic
        >>> tdb = 25
        >>> tr = 25
        >>> rh = 50
        >>> v = 0.1
        >>> met = 1.4
        >>> clo = 0.5
        >>> # calculate relative air speed
        >>> v_r = v_relative(v=v, met=met)
        >>> # calculate dynamic clothing
        >>> clo_d = clo_dynamic(clo=clo, met=met)
        >>> results = pmv(tdb=tdb, tr=tr, vr=v_r, rh=rh, met=met, clo=clo_d)
        >>> print(results)
        0.06
        >>> # you can also pass an array-like of inputs
        >>> results = pmv(tdb=[22, 25], tr=tr, vr=v_r, rh=rh, met=met, clo=clo_d)
        >>> print(results)
        array([-0.47,  0.06])
    """
    default_kwargs = {"units": "SI", "limit_inputs": True, "airspeed_control": True}
    kwargs = {**default_kwargs, **kwargs}

    return pmv_ppd(tdb, tr, vr, rh, met, clo, wme, standard, **kwargs)["pmv"]


@jit(nopython=True)
def two_nodes_optimized(
    tdb,
    tr,
    v,
    met,
    clo,
    vapor_pressure,
    wme,
    body_surface_area,
    p_atmospheric,
    body_position,
    calculate_ce=False,
    max_skin_blood_flow=90,
    max_sweating=500,
    w_max=False,
):
    # Initial variables as defined in the ASHRAE 55-2020
    air_speed = max(v, 0.1)
    k_clo = 0.25
    body_weight = 70  # body weight in kg
    met_factor = 58.2  # met conversion factor
    sbc = 0.000000056697  # Stefan-Boltzmann constant (W/m2K4)
    c_sw = 170  # driving coefficient for regulatory sweating
    c_dil = 120  # driving coefficient for vasodilation ashrae says 50 see page 9.19
    c_str = 0.5  # driving coefficient for vasoconstriction

    temp_skin_neutral = 33.7
    temp_core_neutral = 36.8
    alfa = 0.1
    temp_body_neutral = alfa * temp_skin_neutral + (1 - alfa) * temp_core_neutral
    skin_blood_flow_neutral = 6.3

    t_skin = temp_skin_neutral
    t_core = temp_core_neutral
    m_bl = skin_blood_flow_neutral

    # initialize some variables
    e_skin = 0.1 * met  # total evaporative heat loss, W
    q_sensible = 0  # total sensible heat loss, W
    w = 0  # skin wettedness
    _set = 0  # standard effective temperature
    e_rsw = 0  # heat lost by vaporization sweat
    e_diff = 0  # vapor diffusion through skin
    e_max = 0  # maximum evaporative capacity
    m_rsw = 0  # regulatory sweating
    q_res = 0  # heat loss due to respiration
    et = 0  # effective temperature
    e_req = 0  # evaporative heat loss required for tmp regulation
    r_ea = 0
    r_ecl = 0
    c_res = 0  # convective heat loss respiration

    pressure_in_atmospheres = p_atmospheric / 101325
    length_time_simulation = 60  # length time simulation
    n_simulation = 0

    r_clo = 0.155 * clo  # thermal resistance of clothing, C M^2 /W
    f_a_cl = 1.0 + 0.15 * clo  # increase in body surface area due to clothing
    lr = 2.2 / pressure_in_atmospheres  # Lewis ratio
    rm = (met - wme) * met_factor  # metabolic rate
    m = met * met_factor  # metabolic rate

    e_comfort = 0.42 * (rm - met_factor)  # evaporative heat loss during comfort
    if e_comfort < 0:
        e_comfort = 0

    i_cl = 1.0  # permeation efficiency of water vapour naked skin
    if clo > 0:
        i_cl = 0.45  # permeation efficiency of water vapour through the clothing layer

    if not w_max:  # if the user did not pass a value of w_max
        w_max = 0.38 * pow(air_speed, -0.29)  # critical skin wettedness naked
        if clo > 0:
            w_max = 0.59 * pow(air_speed, -0.08)  # critical skin wettedness clothed

    # h_cc corrected convective heat transfer coefficient
    h_cc = 3.0 * pow(pressure_in_atmospheres, 0.53)
    # h_fc forced convective heat transfer coefficient, W/(m2 °C)
    h_fc = 8.600001 * pow((air_speed * pressure_in_atmospheres), 0.53)
    h_cc = max(h_cc, h_fc)
    if not calculate_ce and met > 0.85:
        h_c_met = 5.66 * (met - 0.85) ** 0.39
        h_cc = max(h_cc, h_c_met)

    h_r = 4.7  # linearized radiative heat transfer coefficient
    h_t = h_r + h_cc  # sum of convective and radiant heat transfer coefficient W/(m2*K)
    r_a = 1.0 / (f_a_cl * h_t)  # resistance of air layer to dry heat
    t_op = (h_r * tr + h_cc * tdb) / h_t  # operative temperature

    t_body = alfa * t_skin + (1 - alfa) * t_core  # mean body temperature, °C

    # respiration
    q_res = 0.0023 * m * (44.0 - vapor_pressure)  # latent heat loss due to respiration
    c_res = 0.0014 * m * (34.0 - tdb)  # sensible convective heat loss respiration

    while n_simulation < length_time_simulation:

        n_simulation += 1

        iteration_limit = 150  # for following while loop
        # t_cl temperature of the outer surface of clothing
        t_cl = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo)  # initial guess
        n_iterations = 0
        tc_converged = False

        while not tc_converged:

            # 0.95 is the clothing emissivity from ASHRAE fundamentals Ch. 9.7 Eq. 35
            if body_position == "sitting":
                # 0.7 ratio between radiation area of the body and the body area
                h_r = 4.0 * 0.95 * sbc * ((t_cl + tr) / 2.0 + 273.15) ** 3.0 * 0.7
            else:  # if standing
                # 0.73 ratio between radiation area of the body and the body area
                h_r = 4.0 * 0.95 * sbc * ((t_cl + tr) / 2.0 + 273.15) ** 3.0 * 0.73
            h_t = h_r + h_cc
            r_a = 1.0 / (f_a_cl * h_t)
            t_op = (h_r * tr + h_cc * tdb) / h_t
            t_cl_new = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo)
            if abs(t_cl_new - t_cl) <= 0.01:
                tc_converged = True
            t_cl = t_cl_new
            n_iterations += 1

            if n_iterations > iteration_limit:
                raise StopIteration("Max iterations exceeded")

        q_sensible = (t_skin - t_op) / (r_a + r_clo)  # total sensible heat loss, W
        # hf_cs rate of energy transport between core and skin, W
        # 5.28 is the average body tissue conductance in W/(m2 C)
        # 1.163 is the thermal capacity of blood in Wh/(L C)
        hf_cs = (t_core - t_skin) * (5.28 + 1.163 * m_bl)
        s_core = m - hf_cs - q_res - c_res - wme  # rate of energy storage in the core
        s_skin = hf_cs - q_sensible - e_skin  # rate of energy storage in the skin
        tc_sk = 0.97 * alfa * body_weight  # thermal capacity skin
        tc_cr = 0.97 * (1 - alfa) * body_weight  # thermal capacity core
        d_t_sk = (s_skin * body_surface_area) / (
            tc_sk * 60.0
        )  # rate of change skin temperature °C per minute
        d_t_cr = (
            s_core * body_surface_area / (tc_cr * 60.0)
        )  # rate of change core temperature °C per minute
        t_skin = t_skin + d_t_sk
        t_core = t_core + d_t_cr
        t_body = alfa * t_skin + (1 - alfa) * t_core
        # sk_sig thermoregulatory control signal from the skin
        sk_sig = t_skin - temp_skin_neutral
        warm_sk = (sk_sig > 0) * sk_sig  # vasodilation signal
        colds = ((-1.0 * sk_sig) > 0) * (-1.0 * sk_sig)  # vasoconstriction signal
        # c_reg_sig thermoregulatory control signal from the skin, °C
        c_reg_sig = t_core - temp_core_neutral
        # c_warm vasodilation signal
        c_warm = (c_reg_sig > 0) * c_reg_sig
        # c_cold vasoconstriction signal
        c_cold = ((-1.0 * c_reg_sig) > 0) * (-1.0 * c_reg_sig)
        # bd_sig thermoregulatory control signal from the body
        bd_sig = t_body - temp_body_neutral
        warm_b = (bd_sig > 0) * bd_sig
        m_bl = (skin_blood_flow_neutral + c_dil * c_warm) / (1 + c_str * colds)
        if m_bl > max_skin_blood_flow:
            m_bl = max_skin_blood_flow
        if m_bl < 0.5:
            m_bl = 0.5
        m_rsw = c_sw * warm_b * math.exp(warm_sk / 10.7)  # regulatory sweating
        if m_rsw > max_sweating:
            m_rsw = max_sweating
        e_rsw = 0.68 * m_rsw  # heat lost by vaporization sweat
        r_ea = 1.0 / (lr * f_a_cl * h_cc)  # evaporative resistance air layer
        r_ecl = r_clo / (lr * i_cl)
        e_req = (
            rm - q_res - c_res - q_sensible
        )  # evaporative heat loss required for tmp regulation
        e_max = (math.exp(18.6686 - 4030.183 / (t_skin + 235.0)) - vapor_pressure) / (
            r_ea + r_ecl
        )
        p_rsw = e_rsw / e_max  # ratio heat loss sweating to max heat loss sweating
        w = 0.06 + 0.94 * p_rsw  # skin wetness
        e_diff = w * e_max - e_rsw  # vapor diffusion through skin
        if w > w_max:
            w = w_max
            p_rsw = w_max / 0.94
            e_rsw = p_rsw * e_max
            e_diff = 0.06 * (1.0 - p_rsw) * e_max
        if e_max < 0:
            e_diff = 0
            e_rsw = 0
            w = w_max
        e_skin = (
            e_rsw + e_diff
        )  # total evaporative heat loss sweating and vapor diffusion
        m_rsw = (
            e_rsw / 0.68
        )  # back calculating the mass of regulatory sweating as a function of e_rsw
        met_shivering = 19.4 * colds * c_cold  # met shivering W/m2
        m = rm + met_shivering
        alfa = 0.0417737 + 0.7451833 / (m_bl + 0.585417)

    q_skin = q_sensible + e_skin  # total heat loss from skin, W
    # p_s_sk saturation vapour pressure of water of the skin
    p_s_sk = math.exp(18.6686 - 4030.183 / (t_skin + 235.0))

    # standard environment - where _s at end of the variable names stands for standard
    h_r_s = h_r  # standard environment radiative heat transfer coefficient

    h_c_s = 3.0 * pow(pressure_in_atmospheres, 0.53)
    if not calculate_ce and met > 0.85:
        h_c_met = 5.66 * (met - 0.85) ** 0.39
        h_c_s = max(h_c_s, h_c_met)
    if h_c_s < 3.0:
        h_c_s = 3.0

    h_t_s = (
        h_c_s + h_r_s
    )  # sum of convective and radiant heat transfer coefficient W/(m2*K)
    r_clo_s = (
        1.52 / ((met - wme / met_factor) + 0.6944) - 0.1835
    )  # thermal resistance of clothing, °C M^2 /W
    r_cl_s = 0.155 * r_clo_s  # thermal insulation of the clothing in M2K/W
    f_a_cl_s = 1.0 + k_clo * r_clo_s  # increase in body surface area due to clothing
    f_cl_s = 1.0 / (
        1.0 + 0.155 * f_a_cl_s * h_t_s * r_clo_s
    )  # ratio of surface clothed body over nude body
    i_m_s = 0.45  # permeation efficiency of water vapour through the clothing layer
    i_cl_s = (
        i_m_s * h_c_s / h_t_s * (1 - f_cl_s) / (h_c_s / h_t_s - f_cl_s * i_m_s)
    )  # clothing vapor permeation efficiency
    r_a_s = 1.0 / (f_a_cl_s * h_t_s)  # resistance of air layer to dry heat
    r_ea_s = 1.0 / (lr * f_a_cl_s * h_c_s)
    r_ecl_s = r_cl_s / (lr * i_cl_s)
    h_d_s = 1.0 / (r_a_s + r_cl_s)
    h_e_s = 1.0 / (r_ea_s + r_ecl_s)

    # calculate Standard Effective Temperature (SET)
    delta = 0.0001
    dx = 100.0
    set_old = round(t_skin - q_skin / h_d_s, 2)
    while abs(dx) > 0.01:
        err_1 = (
            q_skin
            - h_d_s * (t_skin - set_old)
            - w
            * h_e_s
            * (p_s_sk - 0.5 * (math.exp(18.6686 - 4030.183 / (set_old + 235.0))))
        )
        err_2 = (
            q_skin
            - h_d_s * (t_skin - (set_old + delta))
            - w
            * h_e_s
            * (
                p_s_sk
                - 0.5 * (math.exp(18.6686 - 4030.183 / (set_old + delta + 235.0)))
            )
        )
        _set = set_old - delta * err_1 / (err_2 - err_1)
        dx = _set - set_old
        set_old = _set

    # calculate Effective Temperature (ET)
    h_d = 1 / (r_a + r_clo)
    h_e = 1 / (r_ea + r_ecl)
    et_old = t_skin - q_skin / h_d
    delta = 0.0001
    dx = 100.0
    while abs(dx) > 0.01:
        err_1 = (
            q_skin
            - h_d * (t_skin - et_old)
            - w
            * h_e
            * (p_s_sk - 0.5 * (math.exp(18.6686 - 4030.183 / (et_old + 235.0))))
        )
        err_2 = (
            q_skin
            - h_d * (t_skin - (et_old + delta))
            - w
            * h_e
            * (p_s_sk - 0.5 * (math.exp(18.6686 - 4030.183 / (et_old + delta + 235.0))))
        )
        et = et_old - delta * err_1 / (err_2 - err_1)
        dx = et - et_old
        et_old = et

    tbm_l = (0.194 / 58.15) * rm + 36.301  # lower limit for evaporative regulation
    tbm_h = (0.347 / 58.15) * rm + 36.669  # upper limit for evaporative regulation

    t_sens = 0.4685 * (t_body - tbm_l)  # predicted thermal sensation
    if (t_body >= tbm_l) & (t_body < tbm_h):
        t_sens = w_max * 4.7 * (t_body - tbm_l) / (tbm_h - tbm_l)
    elif t_body >= tbm_h:
        t_sens = w_max * 4.7 + 0.4685 * (t_body - tbm_h)

    disc = (
        4.7 * (e_rsw - e_comfort) / (e_max * w_max - e_comfort - e_diff)
    )  # predicted thermal discomfort
    if disc <= 0:
        disc = t_sens

    # PMV Gagge
    pmv_gagge = (0.303 * math.exp(-0.036 * m) + 0.028) * (e_req - e_comfort - e_diff)

    # PMV SET
    dry_set = h_d_s * (t_skin - _set)
    e_req_set = rm - c_res - q_res - dry_set
    pmv_set = (0.303 * math.exp(-0.036 * m) + 0.028) * (e_req_set - e_comfort - e_diff)

    # Predicted  Percent  Satisfied  With  the  Level  of  Air  Movement
    ps = 100 * (1.13 * (t_op**0.5) - 0.24 * t_op + 2.7 * (v**0.5) - 0.99 * v)

    return (
        _set,
        e_skin,
        e_rsw,
        e_max,
        q_sensible,
        q_skin,
        q_res,
        t_core,
        t_skin,
        m_bl,
        m_rsw,
        w,
        w_max,
        et,
        pmv_gagge,
        pmv_set,
        disc,
        e_req - e_comfort - e_diff,
    )


def two_nodes(
    tdb,
    tr,
    v,
    rh,
    met,
    clo,
    wme=0,
    body_surface_area=1.8258,
    p_atmospheric=101325,
    body_position="standing",
    max_skin_blood_flow=90,
    **kwargs,
):
    """Two-node model of human temperature regulation Gagge et al. (1986).

    [10]_ This model it can be used to calculate a variety of indices,
    including:

    * Gagge's version of Fanger's Predicted Mean Vote (PMV). This function uses the Fanger's PMV equations but it replaces the heat loss and gain terms with those calculated by the two node model developed by Gagge et al. (1986) [10]_.

    * PMV SET and the predicted thermal sensation based on SET [10]_. This function is similar in all aspects to the :py:meth:`pythermalcomfort.models.pmv_gagge` however, it uses the :py:meth:`pythermalcomfort.models.set` equation to calculate the dry heat loss by convection.

    * Thermal discomfort (DISC) as the relative thermoregulatory strain necessary to restore a state of comfort and thermal equilibrium by sweating [10]_. DISC is described numerically as: comfortable and pleasant (0), slightly uncomfortable but acceptable (1), uncomfortable and unpleasant (2), very uncomfortable (3), limited tolerance (4), and intolerable (S). The range of each category is ± 0.5 numerically. In the cold, the classical negative category descriptions used for Fanger's PMV apply [10]_.

    * Heat gains and losses via convection, radiation and conduction.

    * The Standard Effective Temperature (SET)

    * The New Effective Temperature (ET)

    * The Predicted  Thermal  Sensation  (TSENS)

    * The Predicted  Percent  Dissatisfied  Due  to  Draft  (PD)

    * Predicted  Percent  Satisfied  With  the  Level  of  Air  Movement"   (PS)

    Parameters
    ----------
    tdb : float or array-like
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : float or array-like
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    v : float or array-like
        air speed, default in [m/s] in [fps] if `units` = 'IP'
    rh : float or array-like
        relative humidity, [%]
    met : float or array-like
        metabolic rate, [met]
    clo : float or array-like
        clothing insulation, [clo]
    wme : float or array-like
        external work, [met] default 0
    body_surface_area : float
        body surface area, default value 1.8258 [m2] in [ft2] if `units` = 'IP'

        The body surface area can be calculated using the function
        :py:meth:`pythermalcomfort.utilities.body_surface_area`.
    p_atmospheric : float
        atmospheric pressure, default value 101325 [Pa] in [atm] if `units` = 'IP'
    body_position: str default="standing" or array-like
        select either "sitting" or "standing"
    max_skin_blood_flow : float
        maximum blood flow from the core to the skin, [kg/h/m2] default 80

    Other Parameters
    ----------------
    round: boolean, default True
        if True rounds output values, if False it does not round them
    max_sweating : float
        Maximum rate at which regulatory sweat is generated, [kg/h/m2]
    w_max : float
        Maximum skin wettedness (w) adimensional. Ranges from 0 and 1.

    Returns
    -------
    e_skin : float or array-like
        Total rate of evaporative heat loss from skin, [W/m2]. Equal to e_rsw + e_diff
    e_rsw : float or array-like
        Rate of evaporative heat loss from sweat evaporation, [W/m2]
    e_diff : float or array-like
        Rate of evaporative heat loss from moisture diffused through the skin, [W/m2]
    e_max : float or array-like
        Maximum rate of evaporative heat loss from skin, [W/m2]
    q_sensible : float or array-like
        Sensible heat loss from skin, [W/m2]
    q_skin : float or array-like
        Total rate of heat loss from skin, [W/m2]. Equal to q_sensible + e_skin
    q_res : float or array-like
        Total rate of heat loss through respiration, [W/m2]
    t_core : float or array-like
        Core temperature, [°C]
    t_skin : float or array-like
        Skin temperature, [°C]
    m_bl : float or array-like
        Skin blood flow, [kg/h/m2]
    m_rsw : float or array-like
        Rate at which regulatory sweat is generated, [kg/h/m2]
    w : float or array-like
        Skin wettedness, adimensional. Ranges from 0 and 1.
    w_max : float or array-like
        Skin wettedness (w) practical upper limit, adimensional. Ranges from 0 and 1.
    set : float or array-like
        Standard Effective Temperature (SET)
    et : float or array-like
        New Effective Temperature (ET)
    pmv_gagge : float or array-like
        PMV Gagge
    pmv_set : float or array-like
        PMV SET
    pd : float or array-like
        Predicted  Percent  Dissatisfied  Due  to  Draft"
    ps : float or array-like
        Predicted  Percent  Satisfied  With  the  Level  of  Air  Movement
    disc : float or array-like
        Thermal discomfort
    t_sens : float or array-like
        Predicted  Thermal  Sensation

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import two_nodes
        >>> print(two_nodes(tdb=25, tr=25, v=0.3, rh=50, met=1.2, clo=0.5))
        {'e_skin': 15.8, 'e_rsw': 6.5, 'e_diff': 9.3, ... }
        >>> print(two_nodes(tdb=[25, 25], tr=25, v=0.3, rh=50, met=1.2, clo=0.5))
        {'e_skin': array([15.8, 15.8]), 'e_rsw': array([6.5, 6.5]), ... }
    """
    default_kwargs = {
        "round": True,
        "calculate_ce": False,
        "max_sweating": 500,
        "w_max": False,
    }
    kwargs = {**default_kwargs, **kwargs}

    tdb = np.array(tdb)
    tr = np.array(tr)
    v = np.array(v)
    rh = np.array(rh)
    met = np.array(met)
    clo = np.array(clo)
    wme = np.array(wme)
    body_position = np.array(body_position)

    vapor_pressure = rh * p_sat_torr(tdb) / 100

    (
        _set,
        e_skin,
        e_rsw,
        e_max,
        q_sensible,
        q_skin,
        q_res,
        t_core,
        t_skin,
        m_bl,
        m_rsw,
        w,
        w_max,
        et,
        pmv_gagge,
        pmv_set,
        disc,
        heat_loss,
    ) = np.vectorize(two_nodes_optimized, cache=True)(
        tdb=tdb,
        tr=tr,
        v=v,
        met=met,
        clo=clo,
        vapor_pressure=vapor_pressure,
        wme=wme,
        body_surface_area=body_surface_area,
        p_atmospheric=p_atmospheric,
        body_position=body_position,
        calculate_ce=kwargs["calculate_ce"],
        max_skin_blood_flow=max_skin_blood_flow,
        max_sweating=kwargs["max_sweating"],
        w_max=kwargs["w_max"],
    )

    output = {
        "e_skin": e_skin,
        "e_rsw": e_rsw,
        "e_max": e_max,
        "q_sensible": q_sensible,
        "q_skin": q_skin,
        "q_res": q_res,
        "t_core": t_core,
        "t_skin": t_skin,
        "m_bl": m_bl,
        "m_rsw": m_rsw,
        "w": w,
        "w_max": w_max,
        "_set": _set,
        "et": et,
        "pmv_gagge": pmv_gagge,
        "pmv_set": pmv_set,
        "disc": disc,
        "heat loss": heat_loss,
    }

    for key in output.keys():
        # round the results if needed
        if kwargs["round"]:
            output[key] = np.around(output[key], 1)

    return output


def save_var_latex(key, value, units=False, round_var=False):
    import csv

    dict_var = {}

    file_path = "Manuscript/src/mydata.dat"

    try:
        with open(file_path, newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                dict_var[row[0]] = row[1]
    except FileNotFoundError:
        pass

    if round_var:
        value = round(value, round_var)

    if units:
        dict_var[key] = f"\\qty{{{value}}}{{{units}}}"
    else:
        dict_var[key] = f"\\num{{{value}}}"

    with open(file_path, "w") as f:
        for key in dict_var.keys():
            f.write(f"{key},{dict_var[key]}\n")


def importing_filtering_processing(load_preprocessed=False):

    df_ = pd.read_csv(
        r"./Data/db_measurements_v2.1.0.csv.gz",
        compression="gzip",
        dtype={
            "subject_id": str,
            "air_movement_acceptability": str,
            "air_movement_preference": str,
        },
    )

    save_var_latex("entries_db_all", df_.shape[0])

    # entries without ta, rh, v, clo, met
    df_valid_input = df_.dropna(subset=["ta", "vel_r", "rh", "met", "clo"])
    df_valid_input_no_tr = df_valid_input.dropna(subset=["tr"]).shape[0]
    save_var_latex(
        "entries_db_valid",
        int(100 - df_valid_input.shape[0] / df_.shape[0] * 100),
        "\\percent",
    )
    save_var_latex(
        "entries_db_valid_no_tr",
        int(100 - df_valid_input_no_tr / df_valid_input.shape[0] * 100),
        "\\percent",
    )

    if load_preprocessed:
        return pd.read_pickle(r"./Data/db_analysis.pkl.gz", compression="gzip")

    pa_arr = []
    for i, row in df_.iterrows():
        pa_arr.append(psychrolib.GetVapPresFromRelHum(row["ta"], row["rh"] / 100))

    df_["pa"] = pa_arr

    # remove entries outside the Standards' applicability limits
    for key in applicability_limits.keys():
        if "pmv" in key:
            continue
        df_ = df_[
            (df_[key] >= applicability_limits[key][0])
            & (df_[key] <= applicability_limits[key][1])
        ]

    results_pmv_ppd = pmv_ppd(
        tdb=df_.ta,
        tr=df_.tr,
        vr=df_.vel_r,
        rh=df_.rh,
        met=df_.met,
        clo=df_.clo_d,
        standard="iso",
        limit_inputs=False,
    )

    df_["pmv"] = results_pmv_ppd["pmv"]
    df_["ppd"] = results_pmv_ppd["ppd"]

    results_pmv_ppd = pmv_ppd(
        tdb=df_.ta,
        tr=df_.tr,
        vr=df_.vel_r,
        rh=df_.rh,
        met=df_.met,
        clo=df_.clo_d,
        standard="ashrae",
        limit_inputs=False,
    )

    df_["pmv_ce"] = results_pmv_ppd["pmv"]
    df_["ppd_ce"] = results_pmv_ppd["ppd"]

    for key in applicability_limits.keys():
        if "pmv" in key:
            df_ = df_[
                (df_[key] >= applicability_limits[key][0])
                & (df_[key] <= applicability_limits[key][1])
            ]

    # calculate rounded variables and differences
    for model_ in models_to_test:
        rounded_col = f"{model_}_round"
        diff_col = f"diff_ts_{model_}"
        df_[rounded_col] = df_[model_].round()
        df_[diff_col] = df_[["thermal_sensation", model_]].diff(axis=1)[model_]

    df_["thermal_sensation_round"] = df_["thermal_sensation"].round()
    df_["thermal_sensation_round - pmv_ce_round"] = (
        df_["thermal_sensation"] - df_["pmv_ce_round"]
    )

    save_var_latex("entries_db_used", df_.shape[0])
    save_var_latex("entries_db_filtered_by_limit_inputs", df_valid_input.shape[0] - df_.shape[0])

    df_meta = pd.read_csv("./Data/db_metadata.csv")
    df_ = pd.merge(df_, df_meta, on="building_id", how="left")

    df_.to_pickle(r"./Data/db_analysis.pkl.gz", compression="gzip")

    return df_


def bar_chart(
    data,
    ind="tsv",
    show_per=True,
    figletter=False,
    variables=["pmv_round", "pmv_ce_round"],
):
    if data.vel_r.min() != 0:
        f, axs = plt.subplots(
            1, 2, sharey=True, constrained_layout=True, figsize=(8.0, 4.1)
        )
    else:
        f, axs = plt.subplots(
            1, 2, sharey=True, constrained_layout=True, figsize=(8.0, 4)
        )

    for ix, model in enumerate(variables):
        if ind == "pmv":
            _df = (
                data.groupby(["thermal_sensation_round", model])[model]
                .count()
                .unstack("thermal_sensation_round")
            )
            x = model
            x_label = "PMV"
            axs[ix].set(xlabel=var_names[model], ylabel="Percentage [%]")
            # conside the special case I am only including data with thermal_sensation = 0
            if _df.columns == [0.0]:
                for index in _df.index:
                    if index in _df.columns:
                        continue
                    _df[index] = 0
                _df = _df[_df.index.sort_values()]

        else:
            _df = (
                data.groupby(["thermal_sensation_round", model])[
                    "thermal_sensation_round"
                ]
                .count()
                .unstack(model)
            )
            x = "thermal_sensation_round"
            x_label = "thermal_sensation"
            axs[ix].set(xlabel=x_label, ylabel="Percentage [%]")
            if data.vel_r.min() == 0:
                axs[ix].set_title(var_names[model], y=0.9)
        df_total = _df.sum(axis=1)
        df_rel = _df.div(df_total, 0) * 100
        for col in df_rel.index:
            if col in df_rel.columns:
                continue
            else:
                df_rel[col] = 0
        df_rel = df_rel.reindex(sorted(df_rel.columns), axis=1)
        colors = [
            (33 / 255, 102 / 255, 172 / 255),
            (103 / 255, 169 / 255, 207 / 255),
            (209 / 255, 229 / 255, 240 / 255),
            (153 / 255, 213 / 255, 148 / 255),
            (253 / 255, 219 / 255, 199 / 255),
            (239 / 255, 138 / 255, 98 / 255),
            (178 / 255, 24 / 255, 43 / 255),
        ]
        df_plot = df_rel.reset_index()
        df_plot[x] = pd.to_numeric(df_plot[x], downcast="integer")
        cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
        hist = df_plot.plot(
            x=x,
            kind="bar",
            stacked=True,
            mark_right=True,
            width=0.95,
            rot=0,
            legend=False,
            ax=axs[ix],
            colormap=cmap1,
        )
        if ind == "pmv":
            axs[ix].set(xlabel=var_names[model], ylabel="Percentage [%]")
        else:
            axs[ix].set(xlabel=x_label, ylabel="Percentage [%]")
            if data.vel_r.min() == 0:
                axs[ix].set_title(var_names[model], y=1.1)
                axs[ix].set_xticklabels("")
                axs[ix].set_xlabel("")
            else:
                axs[ix].set_xticklabels(
                    [
                        "Cold",
                        "Cool",
                        "Sl. Cool",
                        "Neutral",
                        "Sl. Warm",
                        "Warm",
                        "Hot",
                    ],
                    Fontsize=9,
                )
        sns.despine(ax=axs[ix], left=True, bottom=True)

        # show accuracy
        df_acc = df_rel[df_rel.index.isin(df_rel.columns)]
        df_acc = df_acc[df_acc.index]
        diagonal = pd.Series(np.diag(df_acc), index=df_acc.index)

        axs[ix].grid(axis="x")

        for ix_s, value in enumerate(diagonal):
            if value != value:
                value = 0
            axs[ix].text(
                ix_s, 110, f"{value:.0f}%", va="center", ha="center", fontsize=9
            )

        # show surveys counts
        values = data.groupby([x])[x].count()
        for ix_s, value in enumerate(values):
            axs[ix].text(
                ix_s, 105, f"{value:.0f}", va="center", ha="center", fontsize=9
            )

        # add percentages
        if show_per:
            for index, row in df_rel.fillna(0).reset_index(drop=True).iterrows():
                cum_sum, el = 0, 0
                for ixe, el in enumerate(row):
                    if el > 7:
                        axs[ix].text(
                            index,
                            cum_sum + el / 2,
                            f"{el:.0f}%",
                            va="center",
                            ha="center",
                        )
                    cum_sum += el

    # if data.vel_r.min() != 0:
    # sm = plt.cm.ScalarMappable(cmap=cmap1, norm=plt.Normalize(vmin=-3.5, vmax=+3.5))
    # cmap = mpl.cm.rainbow
    # bounds = np.linspace(-3.5, 3.5, 8)
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # sm = plt.cm.get_cmap("rainbow", 5)
    # cbar = plt.colorbar(
    #     mpl.cm.ScalarMappable(norm=norm, cmap=cmap1),
    #     ticks=np.linspace(-3, 3, 7),
    #     ax=axs,
    #     orientation="horizontal",
    #     aspect=70,
    # )
    # cbar.ax.set_xticklabels(
    #     [
    #         "-3",
    #         "-2",
    #         "-1",
    #         "0",
    #         "1",
    #         "2",
    #         "3",
    #     ]
    # )
    # cbar.outline.set_visible(False)
    # cbar.set_label("PMV")

    if figletter:
        plt.gcf().text(0.05, 0.95, f"{figletter})", weight="bold")

    plt.savefig(f"./Manuscript/src/figures/bar_plot_{ind}_Vmin_{data.vel_r.min()}.pdf")


def scatter_plot_flip_x(data):
    f, axs = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True)
    sns.regplot(
        data=data,
        x="thermal_sensation",
        y="pmv",
        ax=axs[0],
        scatter_kws={"s": 5, "alpha": 0.5, "color": "lightgray"},
    )
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x=data["thermal_sensation"], y=data["pmv"]
    )
    print("thermal_sensation x-axis:", slope, intercept, r_value, p_value, std_err)
    sns.regplot(
        data=data,
        y="thermal_sensation",
        x="pmv",
        ax=axs[1],
        scatter_kws={"s": 5, "alpha": 0.5, "color": "lightgray"},
    )
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x=data["pmv"], y=data["thermal_sensation"]
    )
    print("ISO x-axis:", slope, intercept, r_value, p_value, std_err)


def scatter_plot(data, ind="tsv", x_jitter=0):
    f, axs = plt.subplots(1, 2, constrained_layout=True)

    for ix, model in enumerate(["pmv", "pmv_ce"]):
        if ind == "pmv":
            sns.regplot(
                data=df, x=df[model], y="thermal_sensation", ax=axs[ix], x_jitter=0.1
            )
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                y=df["thermal_sensation"], x=df[model]
            )
        else:
            axs[ix].scatter(
                # data=data,
                y=data[model],
                x=data["thermal_sensation"],
                alpha=0.5,
                s=5,
                c="lightgray",
            )
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x=data["thermal_sensation"], y=data[model]
            )

        # mean absolute error
        r2 = r2_score(data["thermal_sensation"], data[model])
        mae = mean_absolute_error(data["TSV"], data[model])

        axs[ix].set(ylim=(-3.5, 3.5))

        axs[ix].text(
            0.5,
            0.85,
            f"{var_names[model]}={slope:.2}*TSV{intercept:.2}\n"
            + r"R$^2$"
            + f"={r_value**2:.2}, MAE={mae:.2}",
            transform=axs[ix].transAxes,
            ha="center",
            va="center",
        )

        color = "#53626F"
        if model == "pmv":
            color = "#3B7EA1"

        axs[ix].plot(data["TSV"], intercept + data["TSV"] * slope, color=color)
        axs[ix].set(ylabel=var_names[model])
        axs[ix].set_xticks(
            np.arange(-3, 4, step=1),
        )
        axs[ix].set_xticklabels(
            [
                "Cold",
                "Cool",
                "Sl. Cool",
                "Neutral",
                "Sl. Warm",
                "Warm",
                "Hot",
            ],
            Fontsize=9,
        )
        sns.despine(bottom=True, left=True)
        axs[ix].grid(axis="x")

    plt.tight_layout()
    plt.savefig("./Manuscript/src/figures/scatter_tsv_pmv.pdf")


def plot_error_prediction(data):
    f, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(8.0, 6))

    _df = (
        data[["thermal_sensation_round", "diff_ts_pmv", "diff_ts_pmv_ce"]]
        .set_index("thermal_sensation_round")
        .stack()
        .reset_index()
    )
    _df.columns = ["TSV", "model", "delta"]
    _df["model"] = _df["model"].map(
        {"diff_ts_pmv": "PMV", "diff_ts_pmv_ce": r"PMV$_{CE}$"}
    )
    _df["TSV"] = pd.to_numeric(_df["TSV"], downcast="integer")
    sns.violinplot(
        data=_df,
        x="TSV",
        y="delta",
        size="data",
        split=True,
        hue="model",
        inner="quartile",
        palette=["#3B7EA1", "#53626F"],
    )
    acceptable_error = 0.5
    axs.fill_between(
        [-0.5, len(_df["TSV"].unique()) - 0.5],
        acceptable_error,
        -acceptable_error,
        color="k",
        alpha=0.3,
        edgecolor="b",
        linewidth=0.0,
    )
    axs.set(yticks=([-6, -4, -2, -0.5, 0, 0.5, 2, 4]))

    axs.set_xticklabels(
        [
            "Cold",
            "Cool",
            "Sl. Cool",
            "Neutral",
            "Sl. Warm",
            "Warm",
            "Hot",
        ],
    )
    for label in axs.xaxis.get_majorticklabels():
        label.set_y(+0.05)

    axs.set(ylabel="PMV - TSV")
    sns.despine(bottom=True, left=True)
    # plt.legend(frameon=False, loc=1)
    # leg = axs.legend()
    # leg.get_frame().set_edgecolor("b")
    # leg.get_frame().set_linewidth(0.0)
    axs.legend(
        handles=[
            patches.Patch(color="#3B7EA1", label="PMV"),
            patches.Patch(color="#53626F", label="PMV$_{CE}$"),
        ],
        frameon=False,
        loc=1,
    )

    # t-test
    for ix, tsv_vote in enumerate(_df["TSV"].sort_values().unique()):
        sample_1 = _df[(_df["TSV"] == tsv_vote) & (_df["model"] == "PMV")]["delta"]
        sample_2 = _df[(_df["TSV"] == tsv_vote) & (_df["model"] == "PMV$_{CE}$")][
            "delta"
        ]
        p = round(stats.ttest_ind(sample_1, sample_2).pvalue, 3)
        if p < 0.01:
            text_p = r"$p$ < 0.01"
        elif p <= 0.05:
            text_p = r"$p$ = " + str(p)
        else:
            text_p = r"$p$ = " + str(round(p, 1))
        if ix < 3:
            axs.text(ix, -2.5 - ix / 2, text_p, ha="center", va="center")
        else:
            axs.text(ix, 5.2 - ix / 2, text_p, ha="center", va="center")
        perc_1 = round(
            (sample_1.abs() <= acceptable_error).sum() / sample_1.shape[0] * 100
        )
        perc_2 = round(
            (sample_2.abs() <= acceptable_error).sum() / sample_1.shape[0] * 100
        )
        perc_1_1 = round((sample_1.abs() <= 1).sum() / sample_1.shape[0] * 100)
        perc_2_2 = round((sample_2.abs() <= 1).sum() / sample_1.shape[0] * 100)
        y_text = 1.5
        rad = 0.5
        if ix < 2 or ix > 3:
            axs.text(ix + 0.15, 0, f"{perc_2}%", va="center")
            axs.text(ix - 0.15, 0, f"{perc_1}%", ha="right", va="center")
        if ix == 2 or ix == 3:
            if ix == 2:
                y_text = -1.5
                rad = -0.5
            # axs.text(ix + 0.1, y_text, f"{perc_2}%", va="center")
            # axs.text(ix - 0.1, y_text, f"{perc_1}%", ha="right", va="center")
            axs.annotate(
                f"{perc_1}%",
                xy=(ix - 0.3, 0),
                xycoords="data",
                textcoords="data",
                xytext=(ix - 0.1, y_text),
                va="center",
                ha="right",
                arrowprops=dict(
                    arrowstyle="->", connectionstyle=f"arc3,rad={rad}", fc="k", ec="k"
                ),
            )
            axs.annotate(
                f"{perc_2}%",
                xy=(ix + 0.3, 0),
                xycoords="data",
                textcoords="data",
                xytext=(ix + 0.1, y_text),
                va="center",
                arrowprops=dict(
                    arrowstyle="->", connectionstyle=f"arc3,rad={-rad}", fc="k", ec="k"
                ),
            )

    plt.savefig(
        f"./Manuscript/src/figures/prediction_error_Vmin_{data.vel_r.min()}.pdf",
    )


def plot_distribution_variable():
    f, axs = plt.subplots(1, 6, constrained_layout=True, figsize=(7, 3))

    for ix, var in enumerate(["ta", "tr", "rh", "vel_r", "clo", "met"]):
        sns.boxenplot(
            y=var,
            data=df,
            ax=axs[ix],
            color=c_gold,
            showfliers=False,
            linewidth=0.5,
            k_depth="proportion",
        )
        axs[ix].set(
            ylabel="",
            xlabel=f"{var_names[var]} ({var_units[var]})",
        )
        if var == "ta" or var == "tr":
            axs[ix].set(yticks=(np.linspace(10, 34, 5)), ylim=(10, 34))
        if var == "rh":
            axs[ix].set(
                yticks=(np.linspace(10, 90, 5)),
                ylim=(10, 90),
            )
        if var == "vel_r":
            axs[ix].set(
                yticks=(np.linspace(0, 1, 5)),
                ylim=(0, 1),
            )
        if var == "clo":
            axs[ix].set(
                yticks=(np.linspace(0.1, 1.6, 4)),
                ylim=(0.1, 1.6),
            )
        if var == "met":
            axs[ix].set(
                yticks=(np.linspace(0.8, 2.8, 5)),
                ylim=(0.8, 2.8),
            )

    sns.despine(bottom=True, left=True)
    plt.savefig("./Manuscript/src/figures/dist_input_data.pdf")
    plt.show()

    desc = df[["ta", "tr", "rh", "vel_r", "clo", "met"]].describe(
        percentiles=percentiles_to_show
    )

    save_var_latex("ta_95_perc_min", desc["ta"]["2.5%"], "\\celsius", round_var=2)
    save_var_latex("rh_95_perc_max", desc["rh"]["97.5%"], "\\percent", round_var=2)
    save_var_latex("v_95_perc_max", desc["vel_r"]["97.5%"], "\\m\\per\\s", round_var=2)
    save_var_latex("v_median", desc["vel_r"]["50%"], "\\m\\per\\s", round_var=2)

    save_var_latex("entries_with_sex_info", df["gender"].dropna().shape[0])

    r2 = r2_score(df.ta, df.tr)

    df["const"] = 1
    f, axs = plt.subplots(1, 4, figsize=(8, 3))
    for ix, var in enumerate(["age", "ht", "wt", "t_mot_isd"]):
        sns.boxenplot(
            y=var,
            data=df,
            ax=axs[ix],
            color=c_gold,
            showfliers=False,
            linewidth=0.5,
            k_depth="proportion",
        )
        axs[ix].set(xlabel=var_names[var], xticks=[], ylabel="")
        if var == "age":
            axs[ix].set(
                yticks=(np.linspace(10, 100, 4)),
                ylim=(10, 100),
            )
        elif var == "ht":
            axs[ix].set(
                yticks=(np.linspace(1, 2, 5)),
                ylim=(1, 2),
            ),
        elif var == "wt":
            axs[ix].set(
                yticks=(np.linspace(40, 130, 4)),
                ylim=(38, 130),
            )
        else:
            axs[ix].set(
                yticks=(np.linspace(-28, 32, 5)),
                ylim=(-28, 32),
            )
    sns.despine(bottom=True, left=True)
    plt.tight_layout()
    plt.savefig("./Manuscript/src/figures/dist_other_data.pdf")
    plt.show()

    # filter data with age between 20 and 35
    df_age_size = df[(df["age"] >= 20) & (df["age"] <= 35)].shape[0]
    save_var_latex(
        "percentage_age_20_35",
        int(df_age_size / df["age"].dropna().shape[0] * 100),
        "\\percent",
    )
    # filter data with age over 60
    df_age_size = df[(df["age"] >= 60)].shape[0]
    save_var_latex(
        "percentage_age_over_60",
        df_age_size / df["age"].dropna().shape[0] * 100,
        "\\percent",
        1,
    )
    # filter data with running mean between 10 and 25
    df_size = df[(df["t_mot_isd"] >= 10) & (df["t_mot_isd"] <= 25)].shape[0]
    save_var_latex(
        "running_mean_10_25",
        df_size / df["t_mot_isd"].dropna().shape[0] * 100,
        "\\percent",
        1,
    )


def analyse_studies_with_three_speed_measurements(
    data_: pd.DataFrame = None,
    air_speed_limit: float = 0.2,
    temperature_limit: float = 24.0,
):
    # [['contributor', 'publication', 'region', 'country', 'city']]
    df_three_heights = data_[data_["vel_r"] >= air_speed_limit].dropna(subset="vel_l")
    df_three_heights.replace(
        {
            "https://doi.org/10.1016/j.buildenv.2010.05.024": "Zhang, Y. et al. (2010).",
            "https://doi.org/10.1016/j.buildenv.2012.09.009": "Zhang, Y., Chen, H., Meng, Q. (2013)",
            "https://doi.org/10.1016/j.buildenv.2018.01.018": "Tartarini, F., Cooper, P., Fleming, R. (2018)",
        },
        inplace=True,
    )
    df_three_heights["publication"] = (
        df_three_heights["publication"].str.split(")").str[0] + ")"
    )
    df_three_heights["publication"] = df_three_heights["publication"].str.replace(
        "&", r"and"
    )
    df_three_heights["publication"] = df_three_heights["publication"].str.replace(
        "Benton, C. C. and Brager, G. S. Advanced Customer Technology Test (ACT2)",
        "Benton, C. C. and Brager, G. S. ACT2",
    )
    df_three_heights["publication"] = df_three_heights["publication"].str.replace(
        "Benton, C. et al. Advanced Customer Technology Test (ACT2)",
        "Benton, C. et al. ACT2",
    )
    df_three_heights.loc[df_three_heights["publication"].isna(), "publication"] = (
        df_three_heights.loc[df_three_heights["publication"].isna(), "contributor"]
    )
    df_three_heights["publication"] = df_three_heights["publication"].str.replace(
        "Fred Baumann",
        "Bauman, F.",
    )
    df_three_heights["publication"] = df_three_heights["publication"].str.replace(
        "Schiller, G. E. (1990)",
        "Schiller (Brager), G. (1990)",
    )
    df_three_heights["publication"] = df_three_heights["publication"].str.replace(
        "Donnini, G. et al (1996)",
        "Donnini, G. et al. (1996)",
    )

    df_publications = (
        df_three_heights.groupby(["publication"])["vel_r"]
        .count()
        .sort_values(ascending=False)
    )
    df_publications = df_publications[df_publications >= 5]
    df_publications = df_publications.reset_index()

    df_publications[r"\% Total"] = (
        df_publications["vel_r"] / df_publications["vel_r"].sum() * 100
    )
    # df_publications.loc["Total"] = df_publications.sum(numeric_only=True)
    # df_publications.replace({np.nan: "Total"}, inplace=True)
    df_publications[["publication", "vel_r", r"\% Total"]].rename(
        columns={"vel_r": "Count", "publication": "Publication"}
    ).to_latex(
        "./Manuscript/src/tables/publications_with_three_heights.tex",
        float_format="{:.0f}".format,
        index=False,
    )

    authors_to_show = df_publications["publication"].tolist()

    df_delta_v = df_three_heights["vel_r"] - df_three_heights[
        ["vel_h", "vel_m", "vel_l"]
    ].mean(axis=1)
    save_var_latex("delta_v_mean", df_delta_v.mean(), "\\m\\per\\s", 2)

    # plt.figure()
    # sns.scatterplot(
    #     x="ta",
    #     y="vel_r",
    #     data=df_three_heights[
    #         df_three_heights["publication"].isin(
    #             [
    #                 "Cena, K., and de Dear, R. J. (1999)",
    #                 "de Dear, R.J. and Fountain, M.E. (1994)",'Zhang, Y., Chen, H., Meng, Q. (2013)', 'Zhang, Y. et al. (2010)'
    #             ]
    #         )
    #     ],
    #     hue="publication",
    #     # hue="met",
    #     palette="Set1",
    # )
    # plt.show()
    #
    # plt.figure()
    # sns.boxenplot(
    #     x="publication",
    #     y="vel_r",
    #     data=df_three_heights[
    #         df_three_heights["publication"].isin(
    #             [
    #                 "Cena, K., and de Dear, R. J. (1999)",
    #                 "de Dear, R.J. and Fountain, M.E. (1994)",'Zhang, Y., Chen, H., Meng, Q. (2013)', 'Zhang, Y. et al. (2010)'
    #             ]
    #         )
    #     ],
    #     # hue="publication",
    #     # hue="met",
    #     palette="Set1",
    # )
    # plt.show()

    df_three_heights[
        # df_three_heights["publication"].isin(
        #     [
        #         "Zhang, Y., Chen, H., Meng, Q. (2013)",
        #         "Zhang, Y. et al. (2010)",
        #         "Tartarini, F., Cooper, P., Fleming, R. (2018)",
        #     ]
        # )
        # &
        # (
                df_three_heights["ta"] > temperature_limit
        # )
    ].to_csv("data_three_heights.csv")

    plt.close("all")
    df_plot = df_three_heights.loc[
        df_three_heights["publication"].isin(authors_to_show),
        ["vel_h", "vel_m", "vel_l", "vel_r", "publication"],
    ]
    df_plot.publication = df_plot.publication.astype("category")
    df_plot.publication = df_plot.publication.cat.set_categories(authors_to_show)
    df_plot = df_plot.sort_values(["publication"])
    f, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 8))
    df_stacked = (
        df_plot[["vel_h", "vel_m", "vel_l", "vel_r", "publication"]]
        .set_index("publication")
        .stack()
    )
    df_stacked.index = df_stacked.index.map(" ".join)
    df_stacked = df_stacked.reset_index()
    df_stacked.columns = ["publication", "value"]
    sns.boxenplot(
        df_stacked,
        y="publication",
        x="value",
        ax=ax,
        palette=["#FC9514", "#FDB515", "#FCCF14", c_green],
        showfliers=False,
    )
    y_labels = ax.get_yticklabels()
    new_y_labels = []
    index_change = 1
    for ix, label in enumerate(y_labels):
        text = label.get_text()
        if "vel_m" in text:
            text = text.replace("vel_m", "      Chest height")
            index_change += 3
        else:
            text = text.replace("vel_h", "vel_Head height")
            text = text.replace("vel_l", "vel_Ankle height")
            text = text.replace("vel_r", "vel_Relative wind speed")
            text = text.split("vel_")[1]
        label.set_text(text)
        new_y_labels.append(label)
    ax.set_yticklabels(new_y_labels)
    ax.set(xlabel="Air speed (m/s)", ylabel="")

    # ax2 = plt.axes([0, 0.07, 1, 0.929], facecolor=(1, 1, 1, 0))
    y_value = -0.5
    for ix in range(0, len(authors_to_show) + 1):
        ax.axhline(
            y_value,
            0,
            1,
            color="gray",
            linewidth=0.25,
        )
        y_value += 4

    # ax.grid(False)
    plt.savefig("./Manuscript/src/figures/boxenplot_wind_speed_three_heights.pdf")
    plt.show()

    # df_perc = (
    #     (
    #         df.groupby(["country", "contributor"])["record_id"]
    #         .count()
    #         .sort_values(ascending=False)
    #         / df.shape[0]
    #         * 100
    #     )
    #     .to_frame()
    #     .rename(columns={"record_id": "percentage"})
    #     .round()
    # )
    # df_count = (
    #     (
    #         df.groupby(["country", "contributor"])["record_id"]
    #         .count()
    #         .sort_values(ascending=False)
    #     )
    #     .to_frame()
    #     .rename(columns={"record_id": "count"})
    #     .round(1)
    # )
    # df_sources = pd.concat([df_count, df_perc], axis=1).head(10)
    # df_sources.loc["Total"] = df_sources.sum()
    # print(df_sources.reset_index().to_markdown())
    #
    # top_contributors = (
    #     df.groupby(["country", "contributor"])["record_id"]
    #     .count()
    #     .sort_values(ascending=False)
    #     .head(10)
    #     .reset_index()["contributor"]
    #     .to_list()
    # )
    #
    # df_top = df[df["contributor"].isin(top_contributors)]
    #
    # sns.boxenplot(df_top, y="contributor", x="vel_r")
    # plt.tight_layout()
    # plt.show()
    #
    # # only those with v > 0.2
    # df_top_vel = df[df["vel_r"] > 0.2]
    #
    # df_perc = (
    #     (
    #         df_top_vel.groupby(["country", "contributor"])["record_id"]
    #         .count()
    #         .sort_values(ascending=False)
    #         / df_top_vel.shape[0]
    #         * 100
    #     )
    #     .to_frame()
    #     .rename(columns={"record_id": "percentage"})
    #     .round()
    # )
    # df_count = (
    #     (
    #         df_top_vel.groupby(["country", "contributor"])["record_id"]
    #         .count()
    #         .sort_values(ascending=False)
    #     )
    #     .to_frame()
    #     .rename(columns={"record_id": "count"})
    #     .round(1)
    # )
    # df_sources = pd.concat([df_count, df_perc], axis=1).head(10)
    # df_sources.loc["Total"] = df_sources.sum()
    # print(df_sources.reset_index().to_markdown())
    #
    # top_contributors = (
    #     df_top_vel.groupby(["country", "contributor"])["record_id"]
    #     .count()
    #     .sort_values(ascending=False)
    #     .head(10)
    #     .reset_index()["contributor"]
    #     .to_list()
    # )
    #
    # df_top = df_top_vel[df_top_vel["contributor"].isin(top_contributors)]
    #
    # sns.boxenplot(df_top, y="contributor", x="vel_r")
    # plt.tight_layout()
    # plt.show()
    #
    # df_top_vel.groupby("contributor")[
    #     ["vel_r", "vel_h", "vel_m", "vel_l"]
    # ].count().sort_values(by="vel_h", ascending=False)
    # df_top_vel.dropna(subset="vel_l")


def plot_bubble_models_vs_tsv(
    data_: pd.DataFrame() = None,
    fig_name: str = "./Manuscript/src/figures/bubble_models_vs_tsv.pdf",
    fig_letters: list[str] = ["a", "b"],
) -> None:
    data_plot = data_.copy()

    plt.close("all")
    # Scatter thermal_sensation vs pmv prediction
    f, axs = plt.subplots(
        1, len(models_to_test), sharex=True, sharey=True, constrained_layout=True
    )
    axs = axs.flatten()

    for ix, model in enumerate(models_to_test):
        # sns.regplot(x="thermal_sensation", y=pmv, data=df,ax=axs[ix], scatter_kws={"s":2, "alpha":0.3}, line_kws={"color":"k"})
        df_plot = data_plot.copy()
        df_plot["ts_binned"] = pd.cut(
            data_plot["thermal_sensation"],
            np.arange(-3.75, 4.25, 0.5),
        )

        df_plot["y_binned"] = pd.cut(data_plot[model], np.arange(-3.75, 4.25, 0.5))
        df_plot = df_plot.groupby(["ts_binned", "y_binned"]).size()
        axs[ix].scatter(
            pd.IntervalIndex(df_plot.index.get_level_values("ts_binned")).mid,
            pd.IntervalIndex(df_plot.index.get_level_values("y_binned")).mid,
            s=df_plot / 20,
            alpha=0.8,
            c=c_gold,
        )
        p = sns.regplot(
            x="thermal_sensation",
            y=model,
            data=data_plot,
            ax=axs[ix],
            ci=None,
            line_kws={"color": "k", "linewidth": 2},
            scatter=False,
            lowess=True,
        )
        print(model)
        print(
            round(
                scipy.stats.linregress(
                    x=p.get_lines()[0].get_xdata(), y=p.get_lines()[0].get_ydata()
                ).intercept,
                2,
            )
        )
        axs[ix].plot(
            [-3, 3],
            [-3, 3],
            c=c_brown,
            ls="--",
        )
        axs[ix].set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5))
        axs[ix].text(2, 2.2, "Identity line", rotation=41)
        axs[ix].axvline(0, c="darkgray", ls="--")
        axs[ix].axhline(0, c="darkgray", ls="--")
        axs[ix].grid(None)
        axs[ix].set(title=var_names[model], ylabel="", xlabel="")

    axs[0].set(ylabel="PMV value")

    f.supxlabel(var_names["thermal_sensation"])
    ax2 = plt.axes([0, 0, 1, 1], facecolor=(1, 1, 1, 0))
    ax2.text(0.025, 0.95, fig_letters[0], fontsize=16, fontweight="bold")
    ax2.text(0.55, 0.95, fig_letters[1], fontsize=16, fontweight="bold")
    ax2.axis("off")
    ax2.grid(False)
    plt.savefig(fig_name, dpi=300)
    plt.show()


def plot_bar_tp_by_ts(data_: pd.DataFrame() = None) -> None:
    x_var, y_var = "thermal_sensation_round", "thermal_preference"
    save_var_latex(
        f"entries_with_tp",
        data_["thermal_preference"].value_counts().sum(),
    )

    df_count = data_.groupby(x_var)[[y_var]].count()

    save_var_latex(
        f"perc_tsv_neutral",
        int(
            data_[x_var]
            .value_counts(normalize=True)
            .to_frame()
            .query("index == 0")
            .values[0][0]
            * 100
        ),
        "\\percent",
    )

    save_var_latex(
        f"perc_tsv_hot",
        int(
            data_[x_var]
            .value_counts(normalize=True)
            .to_frame()
            .query("index == 3")
            .values[0][0]
            * 100
        ),
        "\\percent",
    )
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, wspace=0.15)
    ax1 = fig.add_subplot(gs[0, :-1])
    df_plot = data_.groupby(x_var)[y_var].value_counts(normalize=True) * 100
    df_plot.unstack(y_var).plot.barh(
        stacked=True, color=palette_tp, ax=ax1, linewidth=0, width=0.85
    )
    for ix, row in df_plot.unstack(y_var).round(0).reset_index(drop=True).iterrows():
        x_shift = 0
        print(ix)
        for el in row:
            print(el)
            if el > 3:
                ax1.text(
                    x_shift + el / 2,
                    ix,
                    f"{int(el)}%",
                    c="white",
                    ha="center",
                    va="center",
                )
            x_shift += el
    ax1.set(xlabel="Percentage (%)", ylabel=var_names[x_var])
    ax1.legend(
        bbox_to_anchor=(0.5, 0.975),
        loc="lower center",
        borderaxespad=0,
        frameon=False,
        ncol=3,
    )
    ax1.grid(None)
    ax1.set_yticklabels(
        [
            "Cold",
            "Cool",
            "Sl. Cool",
            "Neutral",
            "Sl. Warm",
            "Warm",
            "Hot",
        ],
    )
    for ix, row in df_count.reset_index().iterrows():
        ax1.text(102, ix, int(row[y_var]), va="center", ha="left")

    ax2 = fig.add_subplot(gs[0, -1])
    data_.groupby(x_var)[x_var].count().plot.bar(
        color=palette_tsv, ax=ax2, linewidth=0, width=0.85
    )
    ax2.set(
        ylabel="",
        xlabel=var_names[x_var],
        title="Number of votes",
        yticks=(np.linspace(0, 20000, 5)),
    )
    ax2.set(
        yticklabels=(f"{int(x)} K" for x in ax2.get_yticks() / 1000),
    )
    ax2.set_xticklabels(
        [
            "Cold",
            "Cool",
            "Sl. Cool",
            "Neutral",
            "Sl. Warm",
            "Warm",
            "Hot",
        ],
    )
    ax2.yaxis.tick_right()
    ax2.grid(None)
    plt.savefig(f"./Manuscript/src/figures/bar_plot_tp_by_ts.pdf")
    plt.show()


def plot_stacked_bar_predictions_ts(
    data_: pd.DataFrame = None,
    v_min: float = 0.0,
    fig_name: str = None,
    fig_letters: list[str] = ["a", "b"],
    fig_letters_height: float = 0.86,
) -> None:
    plt.close("all")

    if fig_name is None:
        fig_name = f"bar_stacked_model_accuracy_{v_min}"
    models = models_to_test

    # Stacked boxplot
    f, axs = plt.subplots(1, len(models), sharex=True, sharey=True)
    axs = axs.flatten()

    for ix, pmv in enumerate(models):
        var = f"{pmv}_round"
        df_v = df[df["vel_r"] >= v_min]
        df_plot = (
            df_v.groupby("thermal_sensation_round")[var]
            .value_counts(normalize=True)
            .unstack(var)
        )
        if len(df_plot.columns) != 7:
            for x in range(-3, 4):
                if x in df_plot.columns:
                    continue
                else:
                    df_plot[x] = np.nan
        df_plot = df_plot[df_plot.columns.sort_values()]
        df_plot *= 100
        df_plot.plot.bar(
            stacked=True, color=palette_tsv, ax=axs[ix], rot=0, linewidth=0, width=0.85
        )
        accuracy = round(
            accuracy_score(df_v[var].fillna(999), df_v["thermal_sensation_round"]) * 100
        )
        axs[ix].set(xlabel="")
        title = " - All data" if v_min == 0 else f" - $V$  $\geq$ {v_min}"
        axs[ix].set_title(f"Overall accuracy {var_names[pmv]}={accuracy}% {title}")
        handles, labels = axs[ix].get_legend_handles_labels()
        axs[ix].get_legend().remove()
        axs[ix].grid(False)

        df_match = df_plot.stack().reset_index()
        df_match = df_match[df_match["thermal_sensation_round"] == df_match[var]]
        for x in axs[ix].get_xticklabels():
            try:
                match = df_match[df_match[var] == float(x._text)][0].values[0]
                axs[ix].text(
                    x._x,
                    102,
                    f"{int(match)}%",
                    va="center",
                    ha="center",
                )
            except IndexError:
                axs[ix].text(
                    x._x,
                    102,
                    f"0 %",
                    va="center",
                    ha="center",
                )
        axs[ix].set_xticklabels(
            [
                "Cold",
                "Cool",
                "Sl. Cool",
                "Neutral",
                "Sl. Warm",
                "Warm",
                "Hot",
            ],
        )
        axs[ix].tick_params(axis="both", which="major", labelsize=8)
        # axs[ix].tick_params(axis='both', which='minor', labelsize=8)

    left = 0.08
    wspace = 0.05
    if v_min == 0:
        plt.subplots_adjust(left=left, right=1, bottom=0.07, top=0.85, wspace=wspace)
        cax = plt.axes([0, 0.95, 1, 0.05])
        cax.axis("off")
        cax.legend(
            handles,
            [
                "Cold",
                "Cool",
                "Sl. Cool",
                "Neutral",
                "Sl. Warm",
                "Warm",
                "Hot",
            ],
            frameon=False,
            loc="upper center",
            ncol=7,
        )

    else:
        plt.subplots_adjust(left=left, right=1, bottom=0.15, top=0.93, wspace=wspace)
        f.supxlabel(var_names["thermal_sensation"])

    axs[0].set(ylabel="Percentage (%)")

    ax2 = plt.axes([0, 0, 1, 1], facecolor=(1, 1, 1, 0))
    ax2.text(0.032, fig_letters_height, fig_letters[0], fontsize=16, fontweight="bold")
    ax2.text(0.55, fig_letters_height, fig_letters[1], fontsize=16, fontweight="bold")
    ax2.axis("off")
    ax2.grid(False)
    plt.savefig(f"./Manuscript/src/figures/{fig_name}.pdf")
    plt.show()


def plot_stacked_bar_predictions_model():
    plt.close("all")

    # Stacked boxplot
    f, axs = plt.subplots(
        1, len(models_to_test), sharex=True, sharey=True, constrained_layout=True
    )
    axs = axs.flatten()

    for ix, pmv in enumerate(models_to_test):
        var = f"{pmv}_round"
        df_plot = (
            df.groupby(var)["thermal_sensation_round"]
            .value_counts(normalize=True)
            .unstack("thermal_sensation_round")
        )
        df_counts = df.groupby(var)["thermal_sensation_round"].count()
        if len(df_plot.index) != 7:
            for x in range(-3, 4):
                if x in df_plot.index:
                    continue
                else:
                    df_plot = pd.concat(
                        [
                            df_plot,
                            pd.DataFrame(np.nan, index=[x], columns=df_plot.columns),
                        ],
                    )
        if len(df_plot.columns) != 7:
            for x in range(-3, 4):
                if x in df_plot.columns:
                    continue
                else:
                    df_plot[x] = np.nan
        df_plot = df_plot[df_plot.columns.sort_values()]
        df_plot = df_plot.sort_index()
        df_plot.plot.bar(stacked=True, color=palette_tsv, ax=axs[ix], rot=0)
        axs[ix].set(title=var_names[pmv], xlabel="")
        handles, labels = axs[ix].get_legend_handles_labels()
        axs[ix].get_legend().remove()
        df_match = df_plot.stack()
        df_match = df_match[
            df_match.index.get_level_values(0) == df_match.index.get_level_values(1)
        ]
        df_match = df_match.to_frame().reset_index()
        for x in axs[ix].get_xticklabels():
            try:
                match = df_match[df_match["thermal_sensation_round"] == float(x._text)][
                    0
                ].values[0]
                count = df_counts[df_counts.index == float(x._text)].values[0]
                axs[ix].text(
                    x._x,
                    0.5,
                    f"{int(match * 100)}% - #{count}",
                    va="center",
                    ha="center",
                    size=10,
                    rotation=90,
                )
            except IndexError:
                axs[ix].text(
                    x._x,
                    0.5,
                    f"0 %",
                    va="center",
                    ha="center",
                    size=10,
                    rotation=90,
                )

    plt.subplots_adjust(left=0.05, right=1, bottom=0.2, top=0.85)
    cax = plt.axes([0, 0.95, 1, 0.05])
    cax.axis("off")

    cax.legend(
        handles,
        labels,
        frameon=False,
        # mode="expand",
        # bbox_to_anchor=(0, 1.1, 1, 0.2),
        loc="upper center",
        ncol=7,
    )
    f.supxlabel("Model prediction")
    plt.savefig(f"./Manuscript/src/figures/bar_stacked_model_accuracy_model.pdf")


def plot_stacked_bar_predictions_tp():
    plt.close("all")

    # Stacked boxplot
    f, axs = plt.subplots(
        1, len(models_to_test), sharex=True, sharey=True, constrained_layout=True
    )
    axs = axs.flatten()

    for ix, model in enumerate(models_to_test):
        df_plot = df.copy()
        df_plot[model] = pd.cut(
            df_plot[model],
            [-3.5, -0.5, 0.5, 3.5],
            labels=["warmer", "no change", "cooler"],
        )
        df_plot = (
            df_plot.groupby("thermal_preference")[model]
            .value_counts(normalize=True)
            .unstack()
        )
        # if len(df_plot.columns) != 7:
        #     for x in range(-3, 4):
        #         if x in df_plot.columns:
        #             continue
        #         else:
        #             df_plot[x] = np.nan
        df_plot = df_plot[df_plot.columns.sort_values(ascending=False)]
        df_plot.plot.bar(stacked=True, color=palette_tp, ax=axs[ix])
        axs[ix].set(title=var_names[model], xlabel="")
        handles, labels = axs[ix].get_legend_handles_labels()
        axs[ix].get_legend().remove()
        df_match = df_plot.stack().reset_index()
        df_match = df_match[df_match["thermal_preference"] == df_match["level_1"]]
        for x in axs[ix].get_xticklabels():
            try:
                match = df_match[df_match["level_1"] == x._text][0].values[0]
                axs[ix].text(
                    x._x,
                    0.5,
                    f"{int(match * 100)}%",
                    va="center",
                    ha="center",
                    size=10,
                    rotation=90,
                )
            except IndexError:
                pass

    plt.subplots_adjust(left=0.05, right=1, bottom=0.2, top=0.85)
    cax = plt.axes([0, 0.95, 1, 0.05])
    cax.axis("off")

    cax.legend(
        handles,
        labels,
        frameon=False,
        # mode="expand",
        # bbox_to_anchor=(0, 1.1, 1, 0.2),
        loc="upper center",
        ncol=7,
    )
    f.supxlabel(var_names["thermal_preference"])
    plt.savefig(f"./Manuscript/src/figures/bar_stacked_model_accuracy_tp.pdf")


def plot_bias_distribution_whole_db(
    hb_models=False,
    data_: pd.DataFrame = None,
    fig_name: str = "hist_discrepancies",
    fig_letters: list[str] = ["a)", "b)", "c)", "d)"],
    fig_size: tuple[int, int] = (7, 4.5),
):

    models = models_to_test
    if hb_models:
        models = [f"lr_hb_{x}" for x in models_to_test[:-1]]
        fig_name = f"{fig_name}_hb"

    # plot bias distribution
    f, axes = plt.subplots(
        2,
        len(models),
        sharex=True,
        sharey="row",
        constrained_layout=True,
        figsize=fig_size,
    )

    for row, v in enumerate([0, 0.2]):
        axs = axes[row, :]
        for ix, model in enumerate(models):
            df_plot = data_.loc[data_["vel_r"] >= v, f"diff_ts_{model}"]
            interval = 0.5
            bins_plot = np.arange(-3, 3, interval / 2)
            axs[ix].hist(df_plot, bins=bins_plot, color=c_gold)
            axs[ix].hist(
                df_plot[(df_plot >= -interval) & (df_plot < interval)],
                bins=bins_plot,
                color=c_brown,
            )
            y_label = "Number of data points" if ix == 0 else ""
            title = var_names[model]
            title += f" - All data points" if row == 0 else r" - V $\geq$" + f" {v} m/s"
            axs[ix].set(title=title, ylabel=y_label, xlabel="", xlim=(-3, 3))
            mpl.pyplot.locator_params(axis="y", nbins=3)
            stats = {
                # "mean": df_plot.mean().round(2),
                "median": df_plot.median().round(2),
                # "sd": df_plot.std().round(2),
                # "sd": df_plot.std().round(2),
                # "p_skew": skewtest(df_plot, nan_policy="omit").pvalue,
            }
            for var in stats.keys():
                save_var_latex(f"bias_{var}_{model}_{v}", stats[var])
            y_text = 3500
            if row == 1:
                y_text = 1200
                axs[row].set_yticks([0, 500, 1000])
            axs[ix].text(
                2.3,
                y_text,
                f"Median = {stats['median']}\nIQR = {df_plot.quantile(.25).round(2)},"
                f" {df_plot.quantile(.75).round(2)}",
                va="center",
                ha="center",
            )
            axs[ix].grid(axis="x")
            axs[ix].axvline(0, c="k", ls="--")

    f.supxlabel(r"Difference between PMV$_i$ and TSV$_i$")
    ax2 = plt.axes([0, 0, 1, 1], facecolor=(1, 1, 1, 0))
    ax2.text(0.025, 0.95, fig_letters[0], fontsize=16, fontweight="bold")
    ax2.text(0.55, 0.95, fig_letters[1], fontsize=16, fontweight="bold")
    ax2.text(0.025, 0.5, fig_letters[2], fontsize=16, fontweight="bold")
    ax2.text(0.55, 0.5, fig_letters[3], fontsize=16, fontweight="bold")
    ax2.axis("off")
    ax2.grid(False)
    plt.savefig(f"./Manuscript/src/figures/{fig_name}.pdf")
    plt.show()

    # plot bias distribution only for entries with three air speeds
    f, axs = plt.subplots(
        1,
        len(models),
        sharex=True,
        sharey="row",
        constrained_layout=True,
        figsize=(7, 2.5),
    )

    for ix, model in enumerate(models):
        df_plot = data_[data_["vel_r"] >= 0.2].dropna(subset=["vel_l"])[
            f"diff_ts_{model}"
        ]
        interval = 0.5
        bins_plot = np.arange(-3, 3, interval / 2)
        axs[ix].hist(df_plot, bins=bins_plot, color=c_gold)
        axs[ix].hist(
            df_plot[(df_plot >= -interval) & (df_plot < interval)],
            bins=bins_plot,
            color=c_brown,
        )
        y_label = "Number of data points" if ix == 0 else ""
        title = var_names[model]
        title += f" - V $\geq$" + f" {v} m/s and three heights"
        axs[ix].set(title=title, ylabel=y_label, xlabel="", xlim=(-3, 3))
        mpl.pyplot.locator_params(axis="y", nbins=3)
        stats = {
            "median": df_plot.median().round(2),
        }
        y_text = 150
        axs[ix].text(
            2.3,
            y_text,
            f"Median = {stats['median']}\nIQR = {df_plot.quantile(.25).round(2)},"
            f" {df_plot.quantile(.75).round(2)}",
            va="center",
            ha="center",
        )
        axs[ix].grid(axis="x")
        axs[ix].axvline(0, c="k", ls="--")

    f.supxlabel(r"Difference between PMV$_i$ and TSV$_i$")
    ax2 = plt.axes([0, 0, 1, 1], facecolor=(1, 1, 1, 0))
    ax2.text(0.025, 0.9, fig_letters[0], fontsize=16, fontweight="bold")
    ax2.text(0.55, 0.9, fig_letters[1], fontsize=16, fontweight="bold")
    ax2.axis("off")
    ax2.grid(False)
    plt.savefig(f"./Manuscript/src/figures/hist_discrepancies_three_heights.pdf")
    plt.show()

    # plot bias distribution only for entries with three air speeds
    f, axs = plt.subplots(
        1,
        len(models),
        sharex=True,
        sharey="row",
        constrained_layout=True,
        figsize=(7, 2.5),
    )

    for ix, model in enumerate(models):
        df_plot = pd.read_csv("data_three_heights.csv")[
            f"diff_ts_{model}"
        ]
        interval = 0.5
        bins_plot = np.arange(-3, 3, interval / 2)
        axs[ix].hist(df_plot, bins=bins_plot, color=c_gold)
        axs[ix].hist(
            df_plot[(df_plot >= -interval) & (df_plot < interval)],
            bins=bins_plot,
            color=c_brown,
        )
        y_label = "Number of data points" if ix == 0 else ""
        title = var_names[model]
        title += f" - V $\geq$" + f" 0.2 m/s and three heights"
        axs[ix].set(title=title, ylabel=y_label, xlabel="", xlim=(-3, 3))
        mpl.pyplot.locator_params(axis="y", nbins=3)
        stats = {
            "median": df_plot.median().round(2),
        }
        y_text = 50
        axs[ix].text(
            2.3,
            y_text,
            f"Median = {stats['median']}\nIQR = {df_plot.quantile(.25).round(2)},"
            f" {df_plot.quantile(.75).round(2)}",
            va="center",
            ha="center",
        )
        axs[ix].grid(axis="x")
        axs[ix].axvline(0, c="k", ls="--")

    f.supxlabel(r"Difference between PMV$_i$ and TSV$_i$")
    ax2 = plt.axes([0, 0, 1, 1], facecolor=(1, 1, 1, 0))
    ax2.text(0.025, 0.9, fig_letters[0], fontsize=16, fontweight="bold")
    ax2.text(0.55, 0.9, fig_letters[1], fontsize=16, fontweight="bold")
    ax2.axis("off")
    ax2.grid(False)
    plt.savefig(f"./Manuscript/src/figures/hist_discrepancies_three_heights_filtered.pdf")
    plt.show()


def plot_bias_distribution_by(variable="building_id"):
    # plot bias by building
    # plt.close("all")
    for ix, model in enumerate(models_to_test):
        color = palette_primary[ix]
        f, axs = plt.subplots(1, 1, constrained_layout=True)
        sns.violinplot(
            x=variable,
            y=f"diff_ts_{model}",
            data=df,
            ax=axs,
            color=color,
            scale="count",
        )
        counts = df.groupby(variable)["const"].count()
        axs.axhline(-0.5, c="r")
        axs.axhline(+0.5, c="r")

        labels = [x._text for x in axs.get_xticklabels()]
        for i, label in enumerate(labels):
            axs.text(i - 0.25, -2, label[:20], va="center", ha="center", rotation=90)
            axs.text(i - 0.25, 2, counts[label], va="center", ha="center", rotation=90)

        axs.set(
            ylabel=variable,
            ylim=(-2, 2),
            xlabel="",
            xticklabels="",
        )
        plt.suptitle(model)
        plt.savefig(f"./Manuscript/src/figures/bias_by_{variable}_{model}.pdf")


def plot_bias_distribution_by_building():
    # plot bias by building
    # plt.close("all")
    for ix, model in enumerate(models_to_test):
        color = palette_primary[ix]
        f, axs = plt.subplots(1, 1, constrained_layout=True)
        sns.violinplot(
            x="building_id",
            y=f"diff_ts_{model}",
            data=df,
            ax=axs,
            color=color,
            scale="count",
        )
        if model == "pmv":
            good_buildings = df.groupby("building_id")[f"diff_ts_{model}"].median()
            good_buildings = good_buildings[good_buildings.between(-0.5, 0.5)].index
        axs.axhline(-0.5, c="r")
        axs.axhline(+0.5, c="r")

        axs.set(
            ylabel="building id",
            ylim=(-2, 2),
            xlabel="",
            xticklabels="",
        )
        plt.suptitle(model)
        plt.savefig(f"./Manuscript/src/figures/bias_buildings.pdf")


def plot_bias_distribution_by_contributor():
    # plot bias by contributor
    # plt.close("all")
    for ix, model in enumerate(models_to_test):
        color = palette_primary[ix]
        f, axs = plt.subplots(1, 1, constrained_layout=True)
        sns.violinplot(
            x="contributor",
            y=f"diff_ts_{model}",
            data=df,
            ax=axs,
            color=color,
            scale="count",
        )
        if model == "pmv":
            good_buildings = df.groupby("building_id")[f"diff_ts_{model}"].median()
            good_buildings = good_buildings[good_buildings.between(-0.5, 0.5)].index
        axs.axhline(-0.5, c="r")
        axs.axhline(+0.5, c="r")

        contributors = [x._text.split(" ")[1] for x in axs.get_xticklabels()]
        for i, contributor in enumerate(contributors):
            axs.text(i - 0.25, -2, contributor, va="center", ha="center", rotation=90)

        axs.set(
            ylabel="building id",
            ylim=(-2, 2),
            xlabel="",
            xticklabels="",
        )
        plt.suptitle(model)
        plt.savefig(f"./Manuscript/src/figures/bias_contributors.pdf")


def plot_bias_distribution_by_variable_binned():
    variables = [
        "ta",
        # "tr",
        # "top",
        # "t_mot_isd",
        "vel_r",
        "rh",
        "clo",
        "met",
        # "thermal_sensation",
        # "thermal_preference",
        # "pmv",
    ]

    bins = {
        # vel_r 0.0058333333333333 0.45
        # rh 21.27 71.5
        # clo 0.3000000000000001 1.3099999999999998
        # met 1.0 1.877133105802048
        "ta": [18.5, 21.0, 24.0, 27.0, 29.0],
        # "tr": np.arange(18, 31.5, 2),
        # "top": np.arange(17.5, 30.5, 2),
        "vel_r": [0.0, 0.15, 0.3, 0.46],
        "rh": [21, 40, 60, 72],
        "hr": np.arange(0, 20, 2),
        "clo": [0.3, 0.5, 0.7, 0.9, 1.1, 1.31],
        "met": [1.0, 1.2, 1.4, 1.6, 1.9],
        "thermal_sensation": np.arange(-3.5, 4.5, 1),
        "thermal_preference": np.arange(-3.5, 4.5, 1),
        "pmv": np.arange(-3.5, 4.5, 1),
    }

    # filter_good_buildings = False
    # plt.close("all")

    df_analysis = df.copy()
    df_analysis.loc[df_analysis.vel_r == 0, "vel_r"] = 0.0000001

    f, axs = plt.subplots(5, 1, figsize=(7, 9))
    axs = axs.flatten()
    for i, var in enumerate(variables):
        # plot bias distribution
        ax = axs[i]

        variable_to_split = "pmv"

        df_plot = pd.DataFrame()
        for ix, model in enumerate(models_to_test):
            df_model = (
                df_analysis[[var, f"diff_ts_{model}", variable_to_split]]
                .copy()
                .dropna()
            )
            df_model["model"] = model
            df_model.rename(columns={f"diff_ts_{model}": "diff_ts"}, inplace=True)
            df_plot = pd.concat(
                [
                    df_plot,
                    df_model,
                ]
            )
        df_plot = df_plot.loc[:, ~df_plot.columns.duplicated()].copy()
        # exclude categories with very little data
        if var != "thermal_preference":
            print(
                var,
                df_plot[var].quantile(percentiles_to_show[0]),
                df_plot[var].quantile(percentiles_to_show[-1]),
            )

            df_plot[var] = pd.cut(df_plot[var], bins=bins[var])

        print(df_plot.groupby([var, "model"])["diff_ts"].median())

        print(df_plot[df_plot.index.duplicated()])
        sns.boxenplot(
            x=var,
            y="diff_ts",
            data=df_plot.reset_index(),
            ax=ax,
            hue="model",
            palette=["#9dad33", "#00b0da"],
            linewidth=0.5,
            showfliers=False,
            alpha=0.8,
        )
        ax.get_legend().remove()
        ax.axhline(-0.5, c="r", ls="--", lw=0.75)
        ax.axhline(+0.5, c="r", ls="--", lw=0.75)

        # if "preference" not in var:
        #     x_labels = []
        #     for x in pd.IntervalIndex(sorted(df_plot[var].cat.categories.unique())).mid:
        #         if ("ta" == var) or ("rh" == var):
        #             x_labels.append(int(x))
        #         else:
        #             x_labels.append(round(x, 2))
        #
        #     ax.set(
        #         xticklabels=x_labels,
        #     )
        ax.set(
            xlabel=f"{var_names[var].split(' ')[-1]} ({var_units[var]})",
            ylim=(-2, 2),
            ylabel=r"PMV$_i$ - TSV$_i$",
            yticks=(np.linspace(-2, 2, 5)),
        )

    handles, labels = axs[ix].get_legend_handles_labels()
    f.legend(
        handles=handles,
        labels=[var_names[x] for x in labels],
        bbox_to_anchor=(0.5, 1),
        loc="upper center",
        # borderaxespad=0,
        frameon=False,
        ncol=3,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
    plt.savefig(f"./Manuscript/src/figures/bias_models.pdf")


def table_f1_scores():
    conditions_to_report = [
        ("vel_r", 0),
        ("vel_r", 0.2),
        ("good_vel", None),
        ("pmv", 1.5),
    ]
    df_final = pd.DataFrame()
    for condition in conditions_to_report:
        results_f1 = {}
        if condition[0] == "vel_r":
            df_table = df[df[condition[0]] >= condition[1]]
        elif condition[0] == "pmv":
            df_table = df[(-condition[1] <= df["pmv"]) & (df["pmv"] <= condition[1])]
            df_table = df_table[
                (-condition[1] <= df_table["thermal_sensation"])
                & (df_table["thermal_sensation"] <= condition[1])
            ]
        elif condition[0] == "good_vel":
            df_table = df[df["vel_r"] >= 0.2]
            df_table = df_table.dropna(subset=["vel_l"])
        for model in models_to_test:
            df_analysis = (
                df_table[[f"{model}_round", "thermal_sensation_round"]].copy().dropna()
            )
            x = df_analysis[f"{model}_round"]
            y = df_analysis[f"thermal_sensation_round"]
            results_f1[model] = {}
            for type in ["micro", "macro", "weighted"]:
                results_f1[model][type] = f1_score(y, x, average=type)
        df_f1 = pd.DataFrame.from_dict(results_f1)
        df_f1 = df_f1.rename(columns=var_names)
        df_f1["condition"] = f"{condition}"
        df_final = pd.concat([df_final, df_f1])
    print(df_final.to_markdown())
    df_final.reset_index(inplace=True)
    df_final = df_final.rename(columns={"index": "F1 score"})
    df_final.to_latex(
        "./Manuscript/src/tables/f1.tex",
        float_format="{:.2f}".format,
        index=False,
    )
    with open("./Manuscript/src/tables/f1.tex", "r") as f:
        file = f.read()
        file = file.replace("lrrl", "lccc")
        file = file.replace("condition", "Dataset")
        file = file.replace("('vel_r', 0)", "\multirow{3}{*}{All data}", 1)
        file = file.replace(
            "('vel_r', 0)",
            "",
        )
        file = file.replace(r"micro", r"\specialrule{.01em}{.05em}{.05em} micro")
        file = file.replace(r"\specialrule{.01em}{.05em}{.05em}", r"", 1)
        file = file.replace(
            r"('vel_r', 0.2)", r"\multirow{3}{*}{\ac{v} $\geq$ \qty{0.2}{\m\per\s}}", 1
        )
        file = file.replace(
            "('vel_r', 0.2)",
            "",
        )
        file = file.replace(
            r"('pmv', 1.5)",
            r"\multirow{3}{*}{$\lvert \textrm{PMV}\lvert \leq 1.5$ and $\lvert \textrm{TSV}\lvert \leq 1.5$}",
            1,
        )
        file = file.replace(
            "('pmv', 1.5)",
            "",
        )
        file = file.replace(
            "('good_vel', None)",
            r"\multirow{3}{*}{\ac{v} $\geq$ \qty{0.2}{\m\per\s} at three heights}",
            1,
        )
        file = file.replace(
            "('good_vel', None)",
            "",
        )
    with open("./Manuscript/src/tables/f1.tex", "w") as f:
        f.write(file)

    results_f1 = {}
    for model in [f"lr_hb_{x}" for x in models_to_test[:-1]]:
        df_analysis = df[[f"{model}_round", "thermal_sensation_round"]].copy().dropna()
        x = df_analysis[f"{model}_round"]
        y = df_analysis[f"thermal_sensation_round"]
        results_f1[model] = {}
        for type in ["micro", "macro", "weighted"]:
            results_f1[model][type] = f1_score(y, x, average=type)
    df_f1 = pd.DataFrame.from_dict(results_f1)
    print(df_f1.to_markdown())


def pmv_jiayu_fed(tdb, tr, vr, rh, met, clo, wme=0):

    pa = rh * 10 * math.exp(16.6536 - 4030.183 / (tdb + 235))

    icl = 0.155 * clo  # thermal insulation of the clothing in M2K/W
    m = met * 58.15  # metabolic rate in W/M2
    w = wme * 58.15  # external work in W/M2
    mw = m - w  # internal heat production in the human body
    # calculation of the clothing area factor
    if icl <= 0.078:
        f_cl = 1 + (1.29 * icl)  # ratio of surface clothed body over nude body
    else:
        f_cl = 1.05 + (0.645 * icl)

    # heat transfer coefficient by forced convection
    hcf = 12.1 * math.sqrt(vr)
    hc = hcf  # initialize variable
    taa = tdb + 273
    tra = tr + 273
    t_cla = taa + (35.5 - tdb) / (3.5 * icl + 0.1)

    p1 = icl * f_cl
    p2 = p1 * 3.96
    p3 = p1 * 100
    p4 = p1 * taa
    p5 = (308.7 - 0.028 * mw) + (p2 * (tra / 100.0) ** 4)
    xn = t_cla / 100
    xf = t_cla / 50
    eps = 0.00015

    n = 0
    while abs(xn - xf) > eps:
        xf = (xf + xn) / 2
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        if hcf > hcn:
            hc = hcf
        else:
            hc = hcn
        xn = (p5 + p4 * hc - p2 * xf**4) / (100 + p3 * hc)
        n += 1
        if n > 150:
            raise StopIteration("Max iterations exceeded")

    tcl = 100 * xn - 273

    # heat loss diff. through skin
    hl1 = 3.05 * 0.001 * (5733 - (6.99 * mw) - pa)
    # heat loss by sweating
    if mw > 58.15:
        hl2 = 0.42 * (mw - 58.15)
    else:
        hl2 = 0
    # latent respiration heat loss
    hl3 = 1.7 * 0.00001 * m * (5867 - pa)
    # dry respiration heat loss
    hl4 = 0.0014 * m * (34 - tdb)
    # heat loss by radiation
    hl5 = 3.96 * f_cl * (xn**4 - (tra / 100.0) ** 4)
    # heat loss by convection
    hl6 = f_cl * hc * (tcl - tdb)

    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    return mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6

    plt.close("all")
    y_array = []
    l_array = []
    for t in np.arange(20, 40, 0.1):
        y = -8.471 + 0.33 * t
        met = 0.98
        l = pmv_jiayu_fed(t, t, 0.1, 50, met, 0.6)
        y_array.append(y)
        l_array.append(l)
        # print(t, l)
    plt.plot(l_array, y_array, c="g")
    y_array = []
    l_array = []
    for t in np.arange(20, 40, 0.1):
        y = -3.643 + 0.175 * t
        met = 1.56
        l = pmv_jiayu_fed(t, t, 0.2, 50, met, 0.6)
        pmv = l * (0.303 * math.exp(-0.036 * met * 58.12) + 0.028)
        pmv = l * (0.31 * math.exp(-0.04 * met * 58.12) + 0.028)
        y_array.append(y)
        l_array.append(l)
        # print(t, l)
    plt.plot(l_array, y_array, c="gray")
    # plt.plot([-30, 30], [-2, +2])

    [x for x in df.columns if "_hb" in x]
    plt.scatter(y=df["thermal_sensation"], x=df["pmv_hb"], c="gray")
    plt.figure()
    sns.regplot(
        y=df["pmv_hb"],
        x=df["thermal_sensation"],
        data=df,
        scatter_kws={"s": 5, "alpha": 0.5, "color": "lightgray"},
        lowess=True,
    )
    plt.tight_layout()

    # logistic regression
    from sklearn.linear_model import LogisticRegression

    df_reg = df[["pmv_hb", "met", "clo", "thermal_sensation_round"]].dropna()
    clf = LogisticRegression(random_state=0).fit(
        df_reg[["pmv_hb", "met", "clo"]], df_reg["thermal_sensation_round"]
    )
    clf.predict([[0, 1, 0.6]])


def compare_pmv_disc():
    def generate_t_rh_combinations(t_range, rh_range):
        all_combinations = list(product(t_range, rh_range))
        return pd.DataFrame(all_combinations, columns=["t", "rh"])

    def heatmap(
        data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs
    ):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (M, N).
        row_labels
            A list or array of length M with the labels for the rows.
        col_labels
            A list or array of length N with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if ax is None:
            ax = plt.gca()

        if cbar_kw is None:
            cbar_kw = {}

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
        ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(
            ax.get_xticklabels(),
            rotation=-90,
            ha="center",
            va="center",
            rotation_mode="anchor",
        )

        # Turn spines off and create white grid.
        ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    def annotate_heatmap(
        im,
        data=None,
        valfmt="{x:.2f}",
        textcolors=("black", "white"),
        threshold=None,
        **textkw,
    ):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.0

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center", verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = mpl.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts

    df = generate_t_rh_combinations(np.arange(26, 36, 1), np.arange(0, 110, 10))

    wind_speeds = [0.1, 0.6]
    f, axs = plt.subplots(len(wind_speeds), 3, constrained_layout=True, figsize=(7, 4))

    for ix, v in enumerate(wind_speeds):
        results_two_node = two_nodes(
            tdb=df["t"],
            tr=df["t"],
            v=v,
            rh=df["rh"],
            met=1.2,
            clo=0.5,
            limit_inputs=False,
        )
        df["disc"] = results_two_node["disc"]
        results_pmv_ce = pmv_ppd(
            tdb=df["t"],
            tr=df["t"],
            vr=v,
            rh=df["rh"],
            met=1.2,
            clo=0.5,
            limit_inputs=False,
            standard="ashrae",
        )
        df["pmv_ce"] = results_pmv_ce["pmv"]
        results_pmv_iso = pmv_ppd(
            tdb=df["t"],
            tr=df["t"],
            vr=v,
            rh=df["rh"],
            met=1.2,
            clo=0.5,
            limit_inputs=False,
            standard="iso",
        )
        df["pmv_iso"] = results_pmv_iso["pmv"]
        pivot_disc = df.pivot(index="rh", columns="t", values="disc").sort_index(
            ascending=False
        )
        im = heatmap(
            pivot_disc.values,
            pivot_disc.index,
            pivot_disc.columns,
            ax=axs[ix][0],
            cmap=mpl.colormaps["magma"].resampled(5),
            cbarlabel="DISC",
            vmin=0,
            vmax=5,
        )
        pivot_pmv_ce = df.pivot(index="rh", columns="t", values="pmv_ce").sort_index(
            ascending=False
        )
        im = heatmap(
            pivot_pmv_ce.values,
            pivot_pmv_ce.index,
            pivot_pmv_ce.columns,
            ax=axs[ix][1],
            cmap=mpl.colormaps["magma"].resampled(6),
            cbarlabel="PMV CE",
            vmin=0,
            vmax=3,
        )
        pivot_pmv_iso = df.pivot(index="rh", columns="t", values="pmv_iso").sort_index(
            ascending=False
        )
        im = heatmap(
            pivot_pmv_iso.values,
            pivot_pmv_iso.index,
            pivot_pmv_iso.columns,
            ax=axs[ix][2],
            cmap=mpl.colormaps["magma"].resampled(6),
            cbarlabel="PMV ISO",
            vmin=0,
            vmax=3,
        )
        # texts = annotate_heatmap(im, valfmt="{x:.1f} t")
        # sns.heatmap(pivot, annot=False, ax=axs[ix], fmt=".1f")
        axs[ix][0].set_title(f"{v=} m/s")
        axs[ix][0].grid()
        axs[ix][1].grid()
        axs[ix][2].grid()
    plt.savefig(f"./Manuscript/src/figures/pmv_vs_disc.pdf")
    plt.show()


def compare_pmv_pmv_ce_comfort_region():
    met = 1.2
    clo = 0.5
    wme = 0
    results = []
    for v in [0.2, 0.4]:
        for model in ["iso", "ashrae"]:
            for pmv_limit in [-0.5, 0.5]:
                for rh in np.arange(0, 110, 10):

                    def function(x):
                        return (
                            pmv(
                                x,
                                x,
                                vr=v,
                                rh=rh,
                                met=met,
                                clo=clo,
                                wme=wme,
                                round=False,
                                standard=model,
                                limit_inputs=False,
                            )
                            - pmv_limit
                        )

                    temp = optimize.brentq(function, 10, 40)
                    results.append(
                        {
                            "rh": rh,
                            "temp": temp,
                            "pmv_limit": pmv_limit,
                            "model": model,
                            "v": v,
                        }
                    )

    df = pd.DataFrame(results)

    f, axs = plt.subplots(len(df["v"].unique()), 1, figsize=(7, 6), sharex=True)
    for ix, v in enumerate(df["v"].unique()):
        df_v = df[df["v"] == v]
        for model in ["iso", "ashrae"]:
            color = "#9dad33" if model == "iso" else "#00b0da"
            label = "PMV" if model == "iso" else r"PMV$_{CE}$"
            df_model = df_v[df_v["model"] == model]
            t1 = df_model[df_model["pmv_limit"] == -0.5]
            t2 = df_model[df_model["pmv_limit"] == 0.5]
            axs[ix].fill_betweenx(
                t1["rh"], t1["temp"], t2["temp"], color=color, alpha=0.5, label=label
            )
            rh_50 = df_model[df_model["rh"] == 50].values
            v_offset = -20 if model == "iso" else 20
            for val in rh_50:
                axs[ix].plot([val[1]], [val[0]], "o", c="gray")
                axs[ix].annotate(
                    f"{val[1]:.1f}",
                    (val[1], val[0]),
                    textcoords="offset points",
                    xytext=(0, v_offset),
                    arrowprops=dict(
                        arrowstyle="->", facecolor="black", color="gray", lw=0.5
                    ),
                    ha="center",
                    va="center",
                )
        axs[ix].set(
            ylabel="Relative Humidity (%)",
            xlabel="Temperature (°C)",
            title=r"$V$" + f" = {v} m/s",
            ylim=(0, 100),
            yticks=np.arange(0, 110, 25),
        )
        axs[ix].legend(frameon=False)
    axs[0].set(xlabel="")
    axs[1].get_legend().remove()
    axs[0].legend(
        bbox_to_anchor=(0.77, 0.85), loc="center left", borderaxespad=0, frameon=False
    )
    plt.tight_layout()
    plt.savefig(f"./Manuscript/src/figures/pmv_comfort_regions.pdf")
    plt.show()
    # r = pmv(tdb=temp, tr=temp, vr=v, rh=rh, met=met, clo=clo, wme=wme, round=False)


if __name__ == "__main__":

    plt.close("all")

if __name__ == "__plot__":

    # filter data outside standard applicability limits
    df = importing_filtering_processing(load_preprocessed=True)

    analyse_studies_with_three_speed_measurements(data_=df.copy())

    compare_pmv_pmv_ce_comfort_region()

    # Figure 1 and 2
    plot_distribution_variable()

    # Figure 3
    plot_bar_tp_by_ts(data_=df.copy())

    # plot model accuracy using bar chart
    plot_stacked_bar_predictions_ts(data_=df.copy(), v_min=0, fig_letters=["a)", "b)"])
    plot_stacked_bar_predictions_ts(
        data_=df.copy(), v_min=0.2, fig_letters=["c)", "d)"], fig_letters_height=0.95
    )
    # only for entries with velocity >= 0.2 m/s and three heights
    plot_stacked_bar_predictions_ts(
        data_=df[df["vel_r"] >= 0.2].dropna(subset="vel_l"),
        v_min=0.2,
        fig_name="bar_stacked_model_accuracy_0.2_three_heights",
        fig_letters=["a)", "b)"],
        fig_letters_height=0.95,
    )
    # # only for entries with velocity >= 0.2 m/s and three heights
    # plot_stacked_bar_predictions_ts(
    #     data_=pd.read_csv("data_three_heights.csv"),
    #     v_min=0.2,
    #     fig_name="bar_stacked_model_accuracy_0.2_three_heights_filtered",
    #     fig_letters=["a)", "b)"],
    #     fig_letters_height=0.95,
    # )

    # print Markdown table of f1-scores
    table_f1_scores()

    # plot model results vs TSV
    plot_bubble_models_vs_tsv(
        data_=df.copy(),
        fig_letters=["a)", "b)"],
    )
    plot_bubble_models_vs_tsv(
        data_=df[df["vel_r"] >= 0.2].dropna(subset="vel_l"),
        fig_name="./Manuscript/src/figures/bubble_models_vs_tsv_three_heights.pdf",
        fig_letters=["a)", "b)"],
    )
    # plot_bubble_models_vs_tsv(
    #     data_=pd.read_csv("data_three_heights.csv"),
    #     fig_name="./Manuscript/src/figures/bubble_models_vs_tsv_three_heights_filtered.pdf",
    #     fig_letters=["a)", "b)"],
    # )

    # plot bias distribution
    plot_bias_distribution_whole_db(data_=df.copy())
    plot_bias_distribution_whole_db(data_=df.copy(), fig_size=(5, 5), fig_name="bias_whole_db_graphical_abstract")
    # plot_bias_distribution_whole_db(hb_models=True)

    # # plot bias by building
    # plot_bias_distribution_by_building()
    # plot_bias_distribution_by(variable="building_id")
    # plot_bias_distribution_by(variable="contributor")
    # plot_bias_distribution_by(variable="region")
    # plot_bias_distribution_by(variable="climate")
    # plot_bias_distribution_by(variable="building_type")
    # plot_bias_distribution_by(variable="cooling_type")
    # plot_bias_distribution_by(variable="country")

    # # plot bias by contributor
    # plot_bias_distribution_by_contributor()

    # plot bias by each variable
    plot_bias_distribution_by_variable_binned()

    compare_pmv_disc()

    plt.close("all")
    f, axs = plt.subplots(1, 5, constrained_layout=True, sharey=True, sharex=True)
    for ix, model in enumerate(models_to_test):
        sns.regplot(
            x=model,
            y="set",
            data=df,
            scatter_kws={"s": 5, "alpha": 0.5, "color": "lightgray"},
            # lowess=True,
            ax=axs[ix],
        )
    plt.savefig(f"./Manuscript/src/figures/scatter_set_vs_models.pdf")

    plt.close("all")
    f, axs = plt.subplots(1, 5, constrained_layout=True, sharey=True, sharex=True)
    for ix, model in enumerate(models_to_test):
        sns.regplot(
            x=f"{model}_hb",
            y="thermal_sensation",
            data=df,
            scatter_kws={"s": 5, "alpha": 0.5, "color": "lightgray"},
            ax=axs[ix],
            # lowess=True,
            # y_partial="met",
            ci=None,
        )
    plt.savefig(f"./Manuscript/src/figures/scatter_tsv_vs_hb.pdf")

    plt.close("all")
    # plot_bias_distribution_by_variable_binned()

    all_combinations = list(
        product(np.arange(26, 30, 2), np.arange(10, 90, 10), np.arange(0.2, 0.8, 0.2))
    )
    df_comb = pd.DataFrame(all_combinations, columns=["t", "rh", "v"])
    df_comb["iso"] = pmv(
        df_comb.t, df_comb.t, df_comb.v, df_comb.rh, 1.2, 0.5, standard="ISO"
    )
    df_comb["ash"] = pmv(
        df_comb.t, df_comb.t, df_comb.v, df_comb.rh, 1.2, 0.5, standard="ashrae"
    )
    df_comb["iso-ash"] = df_comb["iso"] - df_comb["ash"]
    for v in df_comb.v.unique():
        df_v = df_comb[df_comb.v == v]
        f, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(4, 7))
        iso = df_v.pivot(index="rh", columns="t", values="iso").sort_index(
            ascending=False
        )
        sns.heatmap(iso, annot=True, ax=axs[0], fmt=".1f", vmin=-1, vmax=1)
        axs[0].set_title("iso")
        ash = df_v.pivot(index="rh", columns="t", values="ash").sort_index(
            ascending=False
        )
        sns.heatmap(ash, annot=True, ax=axs[1], fmt=".1f", vmin=-1, vmax=1)
        axs[1].set_title("ash")
        ia = df_v.pivot(index="rh", columns="t", values="iso-ash").sort_index(
            ascending=False
        )
        sns.heatmap(ia, annot=True, ax=axs[2], fmt=".1f", vmin=-1, vmax=1)
        axs[2].set_title("iso-ash")
        f.suptitle(f"{v=} m/s")
        plt.show()

        # # Create plot specification.
        # p = (
        #     ggplot({"x": df_v.t, "y1": df_v.iso, "y2": df_v.ash})
        #     + ggsize(500, 250)
        #     + geom_violin(
        #         aes("x", "y1", fill="..quantile.."),
        #         show_half=-1,
        #         scale="count",
        #         # quantiles=[0.02, 0.25, 0.5, 0.75, 0.98],
        #         quantile_lines=True,
        #     )
        #     + geom_violin(
        #         aes("x", "y2", fill="..quantile.."),
        #         show_half=1,
        #         scale="count",
        #         # quantiles=[0.02, 0.25, 0.5, 0.75, 0.98],
        #         quantile_lines=True,
        #     )
        #     + theme(axis_line_y="blank")
        #     # + ggtitle(v)
        #     + ylab("pmv")
        # )
        #
        # # Display plot in 'SciView'.
        # p.show()


if __name__ == "__old_code__":
    # accuracies calculation
    for limit in [3, 2, 1]:
        data = df[df["thermal_sensation_round"].abs() <= limit]
        data_iso = data[df["pmv_round"].abs() <= limit]
        data_ash = data[df["pmv_ce_round"].abs() <= limit]

        acc_iso = (
            data_iso[
                data_iso["thermal_sensation_round"] == data_iso["pmv_round"]
            ].shape[0]
            / data_iso.shape[0]
        )
        save_var_latex(f"Overall PMV ISO accuracy - limit {limit}", int(acc_iso * 100))
        acc_ash = (
            data_ash[
                data_ash["thermal_sensation_round"] == data_ash["pmv_ce_round"]
            ].shape[0]
            / data_ash.shape[0]
        )
        save_var_latex(
            f"Overall PMV ASHRAE accuracy - limit {limit}", int(acc_ash * 100)
        )

    # logistic regression models
    plt.figure()
    sns.boxenplot(df.pmv_gagge_hb)
    print(df.groupby("thermal_preference")["ta"].count())

    # gagge heat balance vs thermal sensation
    clf = LogisticRegression(random_state=0).fit(
        df["pmv_gagge_hb"].values.reshape(-1, 1), df["thermal_sensation_round"]
    )
    set_range = np.arange(-60, 60, 0.5)
    prob = clf.predict_proba(set_range.reshape(-1, 1))
    df_prob = pd.DataFrame(prob, columns=sorted(df["thermal_sensation_round"].unique()))
    plt.figure()
    for col in df_prob.columns:
        plt.plot(set_range, df_prob[col], label=col)
    plt.tight_layout()
    plt.legend()

    # gagge heat balance vs thermal preference
    df_dropna = df[["pmv_set_hb", "thermal_preference"]].dropna().sample(frac=1)
    df_log = pd.DataFrame()
    for preference in df_dropna.thermal_preference.unique():
        _df = df_dropna.query("thermal_preference == @preference").head(10000)
        df_log = pd.concat([df_log, _df])
    print(df_log.groupby("thermal_preference")["pmv_set_hb"].count())
    clf = LogisticRegression(
        random_state=0,
    ).fit(df_log["pmv_set_hb"].values.reshape(-1, 1), df_log["thermal_preference"])
    set_range = np.arange(-60, 60, 0.5)
    prob = clf.predict_proba(set_range.reshape(-1, 1))
    df["tp_fede"] = clf.predict(df["pmv_set_hb"].values.reshape(-1, 1))
    print(
        df.groupby(["thermal_preference", "tp_fede"])["ta"]
        .count()
        .unstack("thermal_preference")
    )
    df_prob = pd.DataFrame(prob, columns=sorted(df_log["thermal_preference"].unique()))
    plt.figure()
    for col in df_prob.columns:
        plt.plot(set_range, df_prob[col], label=col)
    plt.tight_layout()
    plt.legend()

    # gagge heat balance vs thermal preference
    df_dropna = (
        df[["pmv_set_hb", "met", "clo", "t_mot_isd", "thermal_preference"]]
        .dropna()
        .sample(frac=1)
    )
    df_log = pd.DataFrame()
    for preference in df_dropna.thermal_preference.unique():
        _df = df_dropna.query("thermal_preference == @preference").head(3000)
        df_log = pd.concat([df_log, _df])
    print(df_log.groupby("thermal_preference")["pmv_set_hb"].count())
    clf = LogisticRegression(
        random_state=0,
    ).fit(
        df_log[["pmv_set_hb", "met", "clo", "t_mot_isd"]].values,
        df_log["thermal_preference"],
    )
    set_range = np.arange(-60, 60, 0.5)
    prob = clf.predict_proba([[x[0], 1.2, 0.6, 15] for x in set_range.reshape(-1, 1)])
    df_dropna["tp_fede"] = clf.predict(
        df_dropna[["pmv_set_hb", "met", "clo", "t_mot_isd"]].values
    )
    print(
        df_dropna.groupby(["thermal_preference", "tp_fede"])["tp_fede"]
        .count()
        .unstack("thermal_preference")
    )
    df_prob = pd.DataFrame(prob, columns=sorted(df_log["thermal_preference"].unique()))
    plt.figure()
    for col in df_prob.columns:
        plt.plot(set_range, df_prob[col], label=col)
    plt.tight_layout()
    plt.legend()

    plt.figure()
    plt.plot(set_range, df_prob["no change"], label="no change")
    plt.plot(set_range, df_prob[["cooler", "warmer"]].sum(axis=1), label="change")
    plt.tight_layout()
    plt.legend()

    clf = LogisticRegression(random_state=0).fit(
        df["set"].values.reshape(-1, 1), df["thermal_sensation_round"]
    )
    set_range = np.arange(5, 40, 0.5)
    prob = clf.predict_proba(set_range.reshape(-1, 1))
    df_prob = pd.DataFrame(prob, columns=sorted(df["thermal_sensation_round"].unique()))
    plt.figure()
    for col in df_prob.columns:
        plt.plot(set_range, df_prob[col], label=col)
    plt.tight_layout()
    plt.legend()

    plt.figure()
    plt.plot(set_range, df_prob[[0]].sum(axis=1), label="neutral")
    plt.plot(set_range, df_prob[[-3, -2, -1, 1, 3, 2]].sum(axis=1), label="hot or cold")
    plt.tight_layout()
    plt.legend()

    df_log = df[["set", "thermal_preference"]].dropna()
    clf = LogisticRegression(random_state=0).fit(
        df_log["set"].values.reshape(-1, 1), df_log["thermal_preference"]
    )
    set_range = np.arange(5, 40, 0.5)
    prob = clf.predict_proba(set_range.reshape(-1, 1))
    df_prob = pd.DataFrame(prob, columns=sorted(df_log["thermal_preference"].unique()))
    plt.figure()
    for col in df_prob.columns:
        plt.plot(set_range, df_prob[col], label=col)
    plt.tight_layout()
    plt.legend()

    plt.figure()
    plt.plot(set_range, df_prob["no change"], label="no change")
    plt.plot(set_range, df_prob[["cooler", "warmer"]].sum(axis=1), label="change")
    plt.tight_layout()
    plt.legend()

    df_log = df[["ta", "thermal_preference"]].dropna()
    clf = LogisticRegression(random_state=0).fit(
        df_log["ta"].values.reshape(-1, 1), df_log["thermal_preference"]
    )
    set_range = np.arange(5, 40, 0.5)
    prob = clf.predict_proba(set_range.reshape(-1, 1))
    df_prob = pd.DataFrame(prob, columns=sorted(df_log["thermal_preference"].unique()))
    plt.figure()
    for col in df_prob.columns:
        plt.plot(set_range, df_prob[col], label=col)
    plt.tight_layout()
    plt.legend()

    plt.figure()
    plt.plot(set_range, df_prob["no change"], label="no change")
    plt.plot(set_range, df_prob[["cooler", "warmer"]].sum(axis=1), label="change")
    plt.tight_layout()
    plt.legend()

    # # Old Figure 2
    # bar_chart(data=df, ind="tsv", show_per=False, figletter="a")
    # bar_chart(
    #     data=df,
    #     ind="tsv",
    #     show_per=False,
    #     figletter="a",
    #     variables=["pmv_gagge_round", "pmv_set_round"],
    # )
    # bar_chart(
    #     data=df,
    #     ind="tsv",
    #     show_per=False,
    #     figletter="a",
    #     variables=["pmv_round", "athb_round"],
    # )
    # legend_pmv()

    # Figure 3
    plot_error_prediction(data=df[df.vel_r > 0.1])
    plot_error_prediction(data=df)


if __name__ == "__testing_pmv_set_agreement__":

    number_combinations = 10000
    # todo change the values below programmatically rather than a static input
    combinations = {
        "tdb": np.random.uniform(size=number_combinations, low=18.5, high=29),
        "tr": np.random.uniform(size=number_combinations, low=18.4, high=29.5),
        "rh": np.random.uniform(size=number_combinations, low=21, high=72),
        "v": np.random.uniform(size=number_combinations, low=0.1, high=1),
        "met": np.random.uniform(size=number_combinations, low=1, high=1.9),
        "clo": np.random.uniform(size=number_combinations, low=0.3, high=1.31),
    }
    df_comb = pd.DataFrame(combinations)
    print(pd.DataFrame(combinations).describe().T[["min", "max"]])

    for standard in ["iso", "ashrae"]:
        df_comb[f"pmv_{standard}"] = pmv(
            tdb=df_comb["tdb"],
            tr=df_comb["tr"],
            vr=df_comb["v"],
            rh=df_comb["rh"],
            met=df_comb["met"],
            clo=df_comb["clo"],
            limit_inputs=False,
            standard=standard,
        )
        df_comb[f"pmv_{standard}_hl"] = pmv_ppd(
            tdb=df_comb["tdb"],
            tr=df_comb["tr"],
            vr=df_comb["v"],
            rh=df_comb["rh"],
            met=df_comb["met"],
            clo=df_comb["clo"],
            limit_inputs=False,
            standard=standard,
        )["heat loss"]
    results = two_nodes(
        tdb=df_comb["tdb"],
        tr=df_comb["tr"],
        v=df_comb["v"],
        rh=df_comb["rh"],
        met=df_comb["met"],
        clo=df_comb["clo"],
        limit_inputs=False,
    )
    df_comb["pmv_set"] = results["pmv_set"]
    df_comb["pmv_gagge"] = results["pmv_gagge"]
    df_comb["two_node_hl"] = results["heat loss"]
    df_comb["pmv_delta"] = df_comb["pmv_iso"] - df_comb["pmv_ashrae"]
    df_comb["pmv_ashrae_gagge"] = df_comb["pmv_ashrae"] - df_comb["pmv_gagge"]
    df_comb["pmv_iso_gagge"] = df_comb["pmv_iso"] - df_comb["pmv_gagge"]
    df_comb["pmv_ashrae_set"] = df_comb["pmv_ashrae"] - df_comb["pmv_set"]
    df_comb["pmv_iso_set"] = df_comb["pmv_iso"] - df_comb["pmv_set"]
    df_comb["hl_set_iso"] = df_comb["two_node_hl"] - df_comb["pmv_iso_hl"]
    df_comb["hl_set_ashrae"] = df_comb["two_node_hl"] - df_comb["pmv_ashrae_hl"]
    plt.close("all")

    sns.set_palette("viridis", 2)
    f, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.axis("equal")
    sns.regplot(
        df_comb,
        x="pmv_gagge",
        y="pmv_ashrae",
        label="ashrae",
        scatter_kws={"s": 1},
        lowess=True,
    )
    sns.regplot(
        df_comb,
        x="pmv_gagge",
        y="pmv_iso",
        label="iso",
        scatter_kws={"s": 1},
        lowess=True,
    )
    plt.plot([-2, 2], [-2, 2], c="k")
    limits = (-2, 2)
    ax.set(ylim=limits, xlim=limits, xlabel=r"PMV$_{two-node}$", ylabel=r"PMV")
    plt.annotate(
        r"R$^2_{ASHRAE}$"
        + f"={r2_score(df_comb['pmv_gagge'], df_comb['pmv_ashrae']):.2f}"
        + r" - R$^2_{ISO}$"
        + f"={r2_score(df_comb['pmv_gagge'], df_comb['pmv_iso']):.2f}",
        (0, 2.0),
        ha="center",
        va="bottom",
    )
    plt.legend()
    sns.despine(left=True, bottom=True)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"./Manuscript/src/figures/pmv_two_node_comparison.pdf")

    f, (ax1) = plt.subplots(1, 1, sharex=True, sharey=True, constrained_layout=True)
    sns.kdeplot(df_comb, x="hl_set_iso", color="red", ax=ax1, label="iso")
    sns.kdeplot(df_comb, x="hl_set_ashrae", color="blue", ax=ax1, label="ashrae")
    ax1.set(title="comparison with pmv set")
    ax1.legend()
    sns.despine()

    f, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, sharey=True, constrained_layout=True
    )
    sns.kdeplot(df_comb, x="pmv_iso_set", color="red", ax=ax1, label="iso")
    sns.kdeplot(df_comb, x="pmv_ashrae_set", color="blue", ax=ax1, label="ashrae")
    ax1.set(title="comparison with pmv set")
    ax1.legend()
    sns.kdeplot(df_comb, x="pmv_iso_gagge", color="red", ax=ax2, label="iso")
    sns.kdeplot(df_comb, x="pmv_ashrae_gagge", color="blue", ax=ax2, label="ashrae")
    ax2.set(title="comparison with pmv gagge")
    ax2.legend()
    sns.despine()

    f, axs = plt.subplots(
        len(["tdb", "tr", "rh", "v", "met", "clo"]),
        1,
        sharex=True,
        constrained_layout=True,
        figsize=(7, 7),
    )
    for ix, var in enumerate(["tdb", "tr", "rh", "v", "met", "clo"]):
        sns.kdeplot(
            df_comb,
            x="pmv_iso_gagge",
            y=var,
            color="red",
            ax=axs[ix],
            label="iso",
        )
        sns.kdeplot(
            df_comb,
            x="pmv_ashrae_gagge",
            y=var,
            color="blue",
            ax=axs[ix],
            label="ashrae",
        )
        axs[ix].axvline(0)
    axs[0].legend()
    axs[0].set(xlim=(-3, 3))
    sns.despine()

    f, axs = plt.subplots(
        len(["tdb", "tr", "rh", "v", "met", "clo"]),
        1,
        sharex=True,
        constrained_layout=True,
        figsize=(7, 7),
    )
    for ix, var in enumerate(["tdb", "tr", "rh", "v", "met", "clo"]):
        sns.kdeplot(
            df_comb, x="pmv_iso_set", y=var, color="red", ax=axs[ix], label="iso"
        )
        sns.kdeplot(
            df_comb, x="pmv_ashrae_set", y=var, color="blue", ax=axs[ix], label="ashrae"
        )
        axs[ix].axvline(0)
    axs[0].legend()
    axs[0].set(xlim=(-3, 3))
    sns.despine()

    for var in ["tdb", "tr", "rh", "v", "met", "clo"]:
        plt.figure()
        df_comb["cut"] = pd.cut(df_comb[var], 4)
        sns.kdeplot(df_comb, x="pmv_iso_gagge", hue="cut", palette="winter")
        sns.kdeplot(df_comb, x="pmv_ashrae_gagge", hue="cut", palette="autumn")
    # for var in ["tdb", "tr", "rh", "v", "met", "clo"]:
    #     plt.figure()
    #     sns.kdeplot(df_comb, x="pmv_delta", hue=var)
