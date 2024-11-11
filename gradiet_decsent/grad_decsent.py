import torch


def gradient_descent_auto(func, start, learning_rate=0.01, max_iter=100000,
                          tolerance=1e-6):
    """
    Finds the minimum of a multivariable function using gradient descent with autodiff,
    handling cases where the function might not have a minimum.

    Parameters:
    - func: the function to minimize (can be multivariable)
    - start: initial point (vector) to start the descent
    - learning_rate: step size for each iteration
    - max_iter: maximum number of iterations
    - tolerance: threshold for stopping (when improvement is very small)
    - min_value_threshold: lower bound for function values to stop in case of no minimum

    Returns:
    - min_x: the point (vector) at which the function reaches its minimum, if any
    - min_value: the minimum value of the function, if converged; otherwise, None
    """
    # Convert start to a torch tensor with requires_grad=True to track gradients
    x = torch.tensor(start, dtype=torch.float32, requires_grad=True)

    flg = False
    for i in range(max_iter):
        # Calculate the function value
        y = func(x)

        # Perform backpropagation to compute gradients
        y.backward()

        # Perform a gradient descent step
        with torch.no_grad():
            new_x = x - learning_rate * x.grad

            # Check if the change is within the tolerance
            if torch.norm(new_x - x) < tolerance:
                flg = True
                break

            # Update x and clear gradients
            x.copy_(new_x)
            x.grad.zero_()

    if flg:
        min_x = x.detach().numpy()
        min_value = func(x).item()
        return min_x, min_value
    else:
        print('Function appears to decrease indefinitely')
        return None, None


# Example usage:
# Function: f(x) = x (no minimum)
func = lambda vars: 1 / vars[0] ** 2 + vars[1] ** 2

start = [2, 2]  # Initial guess
min_x, min_value = gradient_descent_auto(func, start)
if min_x is not None:
    print(f"The minimum is at {min_x}, with value = {min_value}")
else:
    print("The function does not appear to have a minimum.")
