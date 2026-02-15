"""
Reward utility — standalone reference for reward calculation.

The reward is computed directly inside HydronicHeatingEnv.step().
This file is kept as documentation and for standalone analysis.
"""


def compute_reward(T_in, T_set, energy_kWh, delta_shift, valve_osc,
                   alpha=1.0, beta=0.1, gamma_r=0.5, delta_w=0.1,
                   asymmetry=1.5, hard_band=2.0, hard_penalty=5.0):
    """
    R = - alpha * comfort_penalty
        - beta  * energy_used
        - gamma * |ΔT_shift|
        - delta * valve_oscillation

    Parameters
    ----------
    T_in         : float – indoor air temperature (°C)
    T_set        : float – room setpoint (°C)
    energy_kWh   : float – energy consumed this step (kWh)
    delta_shift  : float – absolute curve-shift change this step (K)
    valve_osc    : float – total valve movement this step (% cumulative)
    alpha … hard_penalty : reward weight parameters

    Returns
    -------
    total, r_comfort, r_energy, r_smooth, r_valve
    """
    temp_error = T_in - T_set

    if temp_error < 0:
        r_comfort = -alpha * asymmetry * temp_error ** 2
    else:
        r_comfort = -alpha * temp_error ** 2

    if abs(temp_error) > hard_band:
        r_comfort -= hard_penalty * (abs(temp_error) - hard_band)

    r_energy = -beta * energy_kWh
    r_smooth = -gamma_r * delta_shift
    r_valve = -delta_w * (valve_osc / 100.0)

    total = r_comfort + r_energy + r_smooth + r_valve
    return total, r_comfort, r_energy, r_smooth, r_valve
