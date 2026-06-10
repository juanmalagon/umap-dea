import logging

import numpy as np


logger = logging.getLogger(__name__)


def generate_coefficients(
    N: int,
    M: int,
    alpha_1: float,
    verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Generate normalized input and output coefficients."""

    alpha_tilde = np.random.uniform(0, 1, size=N)
    alpha_tilde[0] = 0
    alpha = np.divide(alpha_tilde, np.sum(alpha_tilde))*(1-alpha_1)
    alpha[0] = alpha_1
    if verbose:
        logger.info("Normalized coefficients vector alpha:")
        logger.info("%s", alpha)
        logger.info("sum(alpha) = %s", np.sum(alpha))
        logger.info("shape of alpha = %s", alpha.shape)

    # Generate beta
    beta_tilde = np.random.uniform(0, 1, size=M)
    beta = np.divide(beta_tilde, np.sum(beta_tilde))
    if verbose:
        logger.info("Normalized coefficients vector beta:")
        logger.info("%s", beta)
        logger.info("sum(beta) = %s", np.sum(beta))
        logger.info("shape of beta = %s", beta.shape)

    return alpha, beta



def generate_efficient_outputs(
    n: int,
    M: int,
    verbose: bool = True
) -> np.ndarray:
    """Generate efficient output values."""

    y_tilde = np.random.uniform(0.1, 1, size=(n, M))
    if verbose:
        logger.info("y_tilde:\n%s", y_tilde)
        logger.info("shape of y_tilde = %s", y_tilde.shape)
    return y_tilde



def generate_all_but_one_input(
    n: int,
    N: int,
    verbose: bool = True
) -> np.ndarray:
    """Generate the input matrix before solving for the first input."""

    x = np.random.uniform(0.1, 1, size=(n, N))
    x[:, 0] = 1
    if verbose:
        logger.info("x:\n%s", x)
        logger.info("shape of x = %s", x.shape)
    return x



def generate_first_input(
    n: int,
    alpha: np.ndarray,
    beta: np.ndarray,
    y_tilde: np.ndarray,
    x_temp: np.ndarray,
    alpha_1: float,
    gamma: float,
    verbose: bool = True
) -> np.ndarray:
    """Solve for the first input implied by the production relationship."""

    x = x_temp.copy()
    x_power_alpha = np.array([x[i, :]**alpha for i in range(n)])
    x_power_alpha_productory = np.array(
        [np.prod(x_power_alpha[i, :]) for i in range(n)])
    y_tilde_squared = y_tilde**2
    y_tilde_squared_dot_beta = np.matmul(y_tilde_squared, beta)
    y_numerator = (np.sqrt(y_tilde_squared_dot_beta))**(1/gamma)
    x_1 = np.divide(y_numerator, x_power_alpha_productory)
    x_1 = x_1**(1/alpha_1)
    x[:, 0] = x_1
    if verbose:
        logger.info(
            "shape of y_tilde_squared_dot_beta = %s",
            y_tilde_squared_dot_beta.shape,
        )
        logger.info(
            "shape of x_power_alpha_productory = %s",
            x_power_alpha_productory.shape,
        )
        logger.info("x_1:\n%s", x_1)
        logger.info("shape of x_1 = %s", x_1.shape)
        logger.info("x:\n%s", x)
        logger.info("shape of x = %s", x.shape)
    return x



def incorporate_inefficiency_factor(
    n: int,
    M: int,
    y_tilde: np.ndarray,
    sigma_u: float,
    verbose: bool = True
) -> np.ndarray:
    """Apply a non-negative inefficiency factor to efficient outputs."""

    u = np.random.normal(0, sigma_u, size=(n, M))
    u = np.abs(u)
    y = y_tilde*np.exp(-u)
    if verbose:
        logger.info("u:\n%s", u)
        logger.info("shape of u = %s", u.shape)
        logger.info("y:\n%s", y)
        logger.info("shape of y = %s", y.shape)
    return y



def generate_data_dict(
    n: int,
    N: int,
    M: int,
    alpha_1: float,
    gamma: float,
    sigma_u: float,
    verbose: bool = True
) -> dict[str, np.ndarray]:
    """Generate a full synthetic dataset for a simulation run."""

    alpha, beta = generate_coefficients(
        N=N,
        M=M,
        alpha_1=alpha_1,
        verbose=verbose
    )
    y_tilde = generate_efficient_outputs(
        n=n,
        M=M,
        verbose=verbose
    )
    x_temp = generate_all_but_one_input(
        n=n,
        N=N,
        verbose=verbose
    )
    x = generate_first_input(
        n=n,
        alpha=alpha,
        beta=beta,
        y_tilde=y_tilde,
        x_temp=x_temp,
        alpha_1=alpha_1,
        gamma=gamma,
        verbose=verbose
    )
    y = incorporate_inefficiency_factor(
        n=n,
        M=M,
        y_tilde=y_tilde,
        sigma_u=sigma_u,
        verbose=verbose
    )
    return {
        "alpha": alpha,
        "beta": beta,
        "x": x,
        "y": y,
        "y_tilde": y_tilde
    }
