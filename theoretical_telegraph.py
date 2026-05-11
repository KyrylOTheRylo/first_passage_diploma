import numpy as np


def _as_array(x):
    return np.asarray(x, dtype=np.float64)


def local_coordinate(x0, left_side):
    return _as_array(x0) - float(left_side)


def domain_length(left_side, right_side):
    return float(right_side) - float(left_side)


def _common(x0, left_side, right_side, speed, lambda_rate):
    v = float(speed)
    a = float(lambda_rate)
    L = domain_length(left_side, right_side)
    y = local_coordinate(x0, left_side)
    return y, L, v, a


def theory_p_right(x0, left_side, right_side, speed, lambda_rate):
    y, L, v, a = _common(x0, left_side, right_side, speed, lambda_rate)
    return (v + 2.0 * a * y) / (2.0 * (v + a * L))


def theory_p_left(x0, left_side, right_side, speed, lambda_rate):
    return 1.0 - theory_p_right(x0, left_side, right_side, speed, lambda_rate)


def theory_mean_either(x0, left_side, right_side, speed, lambda_rate):
    y, L, v, a = _common(x0, left_side, right_side, speed, lambda_rate)
    return L / (2.0 * v) + (a / (v * v)) * y * (L - y)


def theory_mean_plus(x0, left_side, right_side, speed, lambda_rate):
    y, L, v, a = _common(x0, left_side, right_side, speed, lambda_rate)
    return (L - y) / v + (a / (v * v)) * y * (L - y)


def theory_mean_minus(x0, left_side, right_side, speed, lambda_rate):
    y, L, v, a = _common(x0, left_side, right_side, speed, lambda_rate)
    return y / v + (a / (v * v)) * y * (L - y)


def _w_r_plus(y, L, v, a):
    a_plus = (
        (L * L) * (a ** 3) * y
        + (L * L) * (a ** 2) * v
        + L * (a ** 3) * (y ** 2)
        + 4.0 * L * (a ** 2) * v * y
        + 3.0 * L * a * (v ** 2)
        + (a ** 2) * v * (y ** 2)
        + 3.0 * a * (v ** 2) * y
        + 3.0 * (v ** 3)
    )
    denom = 3.0 * (v ** 2) * ((v + a * L) ** 2)
    return (L - y) * a_plus / denom


def _w_r_minus(y, L, v, a):
    a_minus = (
        (L ** 3) * (a ** 2)
        + 3.0 * (L ** 2) * a * v
        - L * (a ** 2) * (y ** 2)
        + 3.0 * L * (v ** 2)
        - a * v * (y ** 2)
    )
    denom = 3.0 * (v ** 2) * ((v + a * L) ** 2)
    return a * y * a_minus / denom


def theory_joint_right_time(x0, left_side, right_side, speed, lambda_rate):
    y, L, v, a = _common(x0, left_side, right_side, speed, lambda_rate)
    return 0.5 * (_w_r_plus(y, L, v, a) + _w_r_minus(y, L, v, a))


def theory_mean_right_conditional(x0, left_side, right_side, speed, lambda_rate):
    p_r = theory_p_right(x0, left_side, right_side, speed, lambda_rate)
    w_r = theory_joint_right_time(x0, left_side, right_side, speed, lambda_rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = w_r / p_r
    return out


def theory_mean_left_conditional(x0, left_side, right_side, speed, lambda_rate):
    p_l = theory_p_left(x0, left_side, right_side, speed, lambda_rate)
    t = theory_mean_either(x0, left_side, right_side, speed, lambda_rate)
    w_r = theory_joint_right_time(x0, left_side, right_side, speed, lambda_rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (t - w_r) / p_l
    return out
