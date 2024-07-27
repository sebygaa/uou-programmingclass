# %%
# Bisection Method

# %%
def bisection_method(f, a, b, tol=1e-7, max_iter=1000):
    """
    Bisection method to find a root of the function f in the interval [a, b].
    
    Parameters:
    - f: The function for which we are finding a root.
    - a: The start of the interval.
    - b: The end of the interval.
    - tol: The tolerance for the solution.
    - max_iter: The maximum number of iterations.
    
    Returns:
    - The approximate root of the function.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("The function must have different signs at a and b.")
    
    iteration = 0
    c = (a + b) / 2.0
    
    while abs(f(c)) > tol and iteration < max_iter:
        iteration += 1
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        c = (a + b) / 2.0
    
    return c

# Example usage:
if __name__ == "__main__":
    # Define the function for which we want to find the root.
    def func(x):
        return x**3 - x - 2
    
    root = bisection_method(func, 1, 2)
    print(f"The root is approximately: {root}")

# %%
