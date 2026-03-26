import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# "Derive in Symbols. Compile to C. Execute in Vectors."

def find_inverse(func_expr, var, y_target, initial_guess):
    error_expr = func_expr - y_target
    derivative_expr = sp.diff(error_expr, var)
    
    calculate_error = sp.lambdify(var, error_expr, 'math')
    calculate_slope = sp.lambdify(var, derivative_expr, 'math')

    x_current = float(initial_guess)
    
    for step in range(100): 
        error_val = calculate_error(x_current)
        slope_val = calculate_slope(x_current)

        x_next = x_current - (error_val / slope_val)
        
        if abs(x_next - x_current) < 0.0000001:
            return x_next
            
        x_current = x_next
        
    return None

def compile_newton_solver(func_expr, var):
    derivative_expr = sp.diff(func_expr, var)
    
    f_numpy = sp.lambdify(var, func_expr, 'numpy')
    df_numpy = sp.lambdify(var, derivative_expr, 'numpy')
    
    def solver(y_target, initial_guess, tolerance=1e-7, max_iter=100):
        x_n = float(initial_guess)
        for _ in range(max_iter):
            f_val = f_numpy(x_n)
            df_val = df_numpy(x_n)
            
            if df_val == 0:
                raise ValueError("Derivative hit zero. Newton method fails.")
                
            x_n1 = x_n - (f_val - y_target) / df_val
            
            if abs(x_n1 - x_n) < tolerance:
                return x_n1
                
            x_n = x_n1
        raise ValueError("Maximum iterations reached.")
        
    return solver,  f_numpy

if __name__ == "__main__":
    x = sp.Symbol('x')
    func = x**3 + x  
    
    find_inverse_tree, numb_tree = compile_newton_solver(func, x)

    target_y = 10.0
    guess = 2.0

    calculated_x = find_inverse_tree(y_target=target_y, initial_guess=guess)
    print(f"To get y={target_y}, the input x must be: {calculated_x:.4f}")
    

    # Plotting the function and the target point
    x_vals = np.linspace(-3, 3, 400)
    y_vals = numb_tree(x_vals)
    plt.plot(x_vals, y_vals, label='f(x)')          
    plt.plot(y_vals, x_vals, label='f⁻¹(x)')    
    plt.show()



    