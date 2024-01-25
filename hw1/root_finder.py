from typing import Callable

from tqdm import tqdm


def bisection(
    fn: Callable[[float], float],
    x_low: float,
    x_high: float,
    *,
    max_iteration: int = 500,
    tolerance: float = 1e-8,
):
    """Find a root of a function using bisection method.

    Args:
        fn: The function to find the root of.
        x_low: The lower bound of the search interval.
        x_high: The upper bound of the search interval.
        tolerance: The tolerance to which the root should be found.

    Returns:
        The root of the function.
    """
    x_mid = None

    for iter_idx in tqdm(range(max_iteration)):
        x_mid_old = x_high
        x_mid = 0.5 * (x_low + x_high)

        # Check if we found the root
        if abs(x_mid_old - x_mid) <= tolerance:
            print(f"Found root at {x_mid} after {iter_idx} iterations.")
            return x_mid

        # Rebracket depending on if the root is in the lower or upper half of the interval
        elif fn(x_mid) * fn(x_low) < 0:
            x_high = x_mid
        else:
            x_low = x_mid

    print(
        f"Failed to find root after {max_iteration} iterations. Returning last value."
    )
    return x_mid


def newton_raphson(
    fn: Callable[[float], float],
    fn_prime: Callable[[float], float],
    x_guess: float,
    *,
    max_iteration: int = 500,
    tolerance: float = 1e-8,
):
    """Find a root of a function using Newton-Raphson method.

    Args:
        fn: The function to find the root of.
        fn_prime: The derivative of the function.
        x: The initial guess for the root.
        tolerance: The tolerance to which the root should be found.

    Returns:
        The root of the function.
    """
    for iter_idx in tqdm(range(max_iteration)):
        x_old = x_guess
        x_guess = x_guess - fn(x_guess) / fn_prime(x_guess)

        # Check if we found the root
        if abs(x_old - x_guess) <= tolerance:
            print(f"Found root at {x_guess} after {iter_idx} iterations.")
            return x_guess

    print(
        f"Failed to find root after {max_iteration} iterations. Returning last value."
    )
    return x_guess


def newton_bisection(
    fn: Callable[[float], float],
    fn_prime: Callable[[float], float],
    x_low: float,
    x_high: float,
    *,
    max_iteration: int = 500,
    tolerance: float = 1e-8,
):
    """Find a root of a function using a hybrid Newton-Raphson/Bisection method.

    The Newton-Raphson is used when the step

    Args:
        fn: The function to find the root of.
        fn_prime: The derivative of the function.
        x: The initial guess for the root.
        tolerance: The tolerance to which the root should be found.

    Returns:
        The root of the function.
    """

    x_guess = None

    # Initialize the old guess value for later use in the loop
    x_old = x_high

    for iter_idx in tqdm(range(max_iteration)):
        # Perfom bisection and a Newton-Raphson step
        x_mid = 0.5 * (x_low + x_high)
        x_newton = x_mid - fn(x_mid) / fn_prime(x_mid)

        # Check if the Newton-Raphson step is in the interval
        if x_low < x_newton and x_newton < x_high:
            x_guess = x_newton
        else:
            x_guess = x_mid

        # Check if we found the root
        if abs(x_old - x_guess) <= tolerance:
            print(f"Found root at {x_guess} after {iter_idx} iterations.")
            return x_guess

        # Rebracket depending on if the root is in the lower or upper half of the interval
        elif fn(x_guess) * fn(x_low) < 0:
            x_high = x_guess
        else:
            x_low = x_guess

        # Update the old guess value
        x_old = x_guess

    print(
        f"Failed to find root after {max_iteration} iterations. Returning last value."
    )
    return x_guess
