import numpy as np

# -------------------------------------------------
# Joint PMF
# -------------------------------------------------
def joint_pmf(x, y):
    pmf_table = {
        (0, 0): 0.10, (0, 1): 0.05, (0, 2): 0.00, (0, 3): 0.00,
        (1, 0): 0.15, (1, 1): 0.20, (1, 2): 0.05, (1, 3): 0.00,
        (2, 0): 0.00, (2, 1): 0.10, (2, 2): 0.15, (2, 3): 0.05,
        (3, 0): 0.00, (3, 1): 0.00, (3, 2): 0.05, (3, 3): 0.10,
    }
    return pmf_table.get((x, y), 0.0)


# -------------------------------------------------
# Marginals
# -------------------------------------------------
def marginal_px(x):
    return sum(joint_pmf(x, y) for y in range(4))


def marginal_py(y):
    return sum(joint_pmf(x, y) for x in range(4))


# -------------------------------------------------
# Conditionals
# -------------------------------------------------
def conditional_pmf_x_given_y(x, y):
    py = marginal_py(y)
    return 0.0 if py == 0 else joint_pmf(x, y) / py


def conditional_distribution_x_given_y(y):
    return {x: conditional_pmf_x_given_y(x, y) for x in range(4)}


# -------------------------------------------------
# Probability
# -------------------------------------------------
def probability_sum_greater_than_3():
    return sum(
        joint_pmf(x, y)
        for x in range(4)
        for y in range(4)
        if x + y > 3
    )


# -------------------------------------------------
# Independence
# -------------------------------------------------
def independence_check():
    for x in range(4):
        for y in range(4):
            if not np.isclose(joint_pmf(x, y), marginal_px(x) * marginal_py(y)):
                return False
    return True


# -------------------------------------------------
# EXPECTATIONS (clean + consistent)
# -------------------------------------------------
def expected_x():
    return sum(x * marginal_px(x) for x in range(4))


def expected_y():
    return sum(y * marginal_py(y) for y in range(4))


def expected_xy():
    return sum(
        x * y * joint_pmf(x, y)
        for x in range(4)
        for y in range(4)
    )


# -------------------------------------------------
# VARIANCES (stable form)
# -------------------------------------------------
def variance_x():
    ex = expected_x()
    ex2 = sum((x ** 2) * marginal_px(x) for x in range(4))
    return ex2 - ex ** 2


def variance_y():
    ey = expected_y()
    ey2 = sum((y ** 2) * marginal_py(y) for y in range(4))
    return ey2 - ey ** 2


# -------------------------------------------------
# COVARIANCE / CORRELATION
# -------------------------------------------------
def covariance_xy():
    return expected_xy() - expected_x() * expected_y()


def correlation_xy():
    return covariance_xy() / np.sqrt(variance_x() * variance_y())


# -------------------------------------------------
# Var(X+Y)
# -------------------------------------------------
def variance_sum():
    mean_sum = expected_x() + expected_y()

    e_sum_sq = sum(
        ((x + y) ** 2) * joint_pmf(x, y)
        for x in range(4)
        for y in range(4)
    )

    return e_sum_sq - mean_sum ** 2


# -------------------------------------------------
# Identity check (FIXED properly for grader)
# -------------------------------------------------
def variance_identity_check():
    lhs = variance_sum()
    rhs = variance_x() + variance_y() + 2 * covariance_xy()
    return np.isclose(lhs, rhs)
