import numpy as np


def generate_coefficients(
    N: int,
    M: int,
    alpha_1: float,
    verbose: bool = True
):
    # Generate alpha
    alpha_tilde = np.random.uniform(0, 1, size=N)
    alpha_tilde[0] = 0
    alpha = np.divide(alpha_tilde, np.sum(alpha_tilde))*(1-alpha_1)
    alpha[0] = alpha_1
    if verbose:
        print("\n Normalized coefficients vector alpha:\n")
        print(alpha)
        print(f"\n sum(alpha) = {np.sum(alpha)}")
        print(f"\n shape of alpha = {alpha.shape}")

    # Generate beta
    beta_tilde = np.random.uniform(0, 1, size=M)
    beta = np.divide(beta_tilde, np.sum(beta_tilde))
    if verbose:
        print("\n Normalized coefficients vector beta:\n")
        print(beta)
        print(f"\n sum(beta) = {np.sum(beta)}")
        print(f"\n shape of beta = {beta.shape}")

    return alpha, beta


def generate_efficient_outputs(
    n: int,
    M: int,
    verbose: bool = True
):
    y_tilde = np.random.uniform(0.1, 1, size=(n, M))
    if verbose:
        print("\n y_tilde:\n\n", y_tilde)
        print(f"\n shape of y_tilde = {y_tilde.shape}")
    return y_tilde


def generate_all_but_one_input(
    n: int,
    N: int,
    verbose: bool = True
):
    x = np.random.uniform(0.1, 1, size=(n, N))
    x[:, 0] = 1
    if verbose:
        print("\n x:\n\n", x)
        print(f"\n shape of x = {x.shape}")
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
):
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
        print("\n shape of y_tilde_squared_dot_beta "
              + f"= {y_tilde_squared_dot_beta.shape}")
        print("\n shape of x_power_alpha_productory "
              + f"= {x_power_alpha_productory.shape}")
        print("\n x_1:\n\n", x_1)
        print("\n shape of x_1 = ", x_1.shape)
        print("\n x:\n\n", x)
        print(f"\n shape of x = {x.shape}")
    return x


def incorporate_inefficiency_factor(
    n: int,
    M: int,
    y_tilde: np.ndarray,
    sigma_u: float,
    verbose: bool = True
):
    u = np.random.normal(0, sigma_u, size=(n, M))
    u = np.abs(u)
    y = y_tilde*np.exp(-u)
    if verbose:
        print("\n u:\n\n", u)
        print(f"\n shape of u = {u.shape}")
        print("\n y:\n\n", y)
        print(f"\n shape of y = {y.shape}")
    return y


def generate_data(
    n: int,
    N: int,
    M: int,
    alpha_1: float,
    gamma: float,
    sigma_u: float,
    verbose: bool = True
):
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
    return x, y
