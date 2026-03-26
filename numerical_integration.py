import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def midpoint(y_mids, dx):
    return dx * np.sum(y_mids)

def trapezoidal(y, dx):
    return (dx / 2.0) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])

def simpson_13(y, dx, n):
    if n % 2 != 0: return np.nan
    return (dx / 3.0) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

def simpson_38(y, dx, n):
    if n % 3 != 0: return np.nan
    weights = np.full(n + 1, 3.0)
    weights[0] = weights[-1] = 1.0
    weights[3:-1:3] = 2.0 
    return (3.0 * dx / 8.0) * np.sum(weights * y)


def taylor_bound(dx, M, method_name):
    '''Estimate the error using Taylor series expansion.'''
    if method_name == 'midpoint':    return (dx**2) * M / 24.0
    if method_name == 'trapezoidal': return (dx**2) * M / 12.0
    if method_name == 'simpson_1/3': return (dx**4) * M / 180.0
    if method_name == 'simpson_3/8': return (dx**4) * M / 80.0
        


if __name__ == "__main__":
    a, b = 0.0, 3.0
    x_sym = sp.Symbol('x')
    
    test_func_sym = sp.exp(-x_sym) * sp.sin(5*x_sym)
    f_num = sp.lambdify(x_sym, test_func_sym, 'numpy')
    
    exact_val = float(sp.integrate(test_func_sym, (x_sym, a, b)).evalf())
    
    grid = np.linspace(a, b, 10000)
    M2 = np.max(np.abs(sp.lambdify(x_sym, sp.diff(test_func_sym, x_sym, 2), 'numpy')(grid))) * (b - a)
    M4 = np.max(np.abs(sp.lambdify(x_sym, sp.diff(test_func_sym, x_sym, 4), 'numpy')(grid))) * (b - a)
    
    N_vals = np.array([6, 18, 30, 60, 100, 300, 600, 1200, 3000, 6000, 12000, 24000, 42000, 60000, 120000, 300000, 600000, 1200000, 3000000, 6000000, 12000000])
    
    # Clean dictionary structure for state management
    results = {
        'mid': {'val': [], 'err': [], 'bound': []},
        'trap': {'val': [], 'err': [], 'bound': []},
        's13': {'val': [], 'err': [], 'bound': []},
        's38': {'val': [], 'err': [], 'bound': []}
    }


    for n in N_vals:
        dx = (b - a) / n

        x = np.linspace(a, b, n + 1)
        x_mids = (x[:-1] + x[1:]) / 2
        
        y = f_num(x)
        y_mids = f_num(x_mids)
        
        v_mid = midpoint(y_mids, dx)
        v_trap = trapezoidal(y, dx)
        v_s13 = simpson_13(y, dx, n)
        v_s38 = simpson_38(y, dx, n)
        
        results['mid']['val'].append(v_mid)
        results['trap']['val'].append(v_trap)
        results['s13']['val'].append(v_s13)
        results['s38']['val'].append(v_s38)
        
        results['mid']['err'].append(abs(exact_val - v_mid))
        results['trap']['err'].append(abs(exact_val - v_trap))
        results['s13']['err'].append(abs(exact_val - v_s13) if not np.isnan(v_s13) else np.nan)
        results['s38']['err'].append(abs(exact_val - v_s38) if not np.isnan(v_s38) else np.nan)
        
        results['mid']['bound'].append(taylor_bound(dx, M2, 'midpoint'))
        results['trap']['bound'].append(taylor_bound(dx, M2, 'trapezoidal'))
        results['s13']['bound'].append(taylor_bound(dx, M4, 'simpson_1/3') if not np.isnan(v_s13) else np.nan)
        results['s38']['bound'].append(taylor_bound(dx, M4, 'simpson_3/8') if not np.isnan(v_s38) else np.nan)



    plt.figure(figsize=(12, 6))
    plt.plot(N_vals, results['mid']['val'], label='Midpoint', marker='o', alpha=0.7)
    plt.plot(N_vals, results['trap']['val'], label='Trapezoidal', marker='s', alpha=0.7)
    plt.plot(N_vals, results['s13']['val'], label='Simpson 1/3', marker='^', alpha=0.7)
    plt.plot(N_vals, results['s38']['val'], label='Simpson 3/8', marker='d', alpha=0.7)
    plt.axhline(exact_val, color='k', linestyle='--', linewidth=2, label='Exact Analytical Area')
    
    plt.xscale('log')
    plt.xlabel('Number of Intervals (N)', fontsize=12)
    plt.ylabel('Calculated Area', fontsize=12)
    plt.title('Algorithm Convergence to Exact Area', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 7))
    
    plt.plot(N_vals, results['mid']['err'], label='Actual: Midpoint', color='#1f77b4', marker='o', markersize=4)
    plt.plot(N_vals, results['trap']['err'], label='Actual: Trapezoidal', color='#ff7f0e', marker='s', markersize=4)
    plt.plot(N_vals, results['s13']['err'], label='Actual: Simpson 1/3', color='#2ca02c', marker='^', markersize=4)
    plt.plot(N_vals, results['s38']['err'], label='Actual: Simpson 3/8', color='#d62728', marker='d', markersize=4)

    plt.plot(N_vals, results['mid']['bound'], label='Bound: Midpoint', color='#1f77b4', linestyle=':', alpha=0.7)
    plt.plot(N_vals, results['trap']['bound'], label='Bound: Trapezoidal', color='#ff7f0e', linestyle='--', alpha=0.7)
    plt.plot(N_vals, results['s13']['bound'], label='Bound: Simpson 1/3', color='#2ca02c', linestyle='-.', alpha=0.7)
    plt.plot(N_vals, results['s38']['bound'], label='Bound: Simpson 3/8', color='#d62728', linestyle=':', alpha=0.7)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Intervals (N)', fontsize=12)
    plt.ylabel('Absolute Error', fontsize=12)
    plt.title('Log-Log Error Analysis: Truncation vs. Round-off Drift', fontsize=14)
    
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    target_N = 120000
    idx = np.where(N_vals == target_N)[0][0]
    
    labels = ['Midpoint', 'Trapezoidal', 'Simpson 1/3', 'Simpson 3/8']
    errors_at_target = [
        results['mid']['err'][idx], 
        results['trap']['err'][idx], 
        results['s13']['err'][idx], 
        results['s38']['err'][idx]
    ]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, errors_at_target, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    plt.yscale('log')
    plt.ylabel('Absolute Error (Log Scale)', fontsize=12)
    plt.title(f'Algorithm Precision Comparison at N = {target_N:,}', fontsize=14)
    plt.grid(True, axis='y', ls="--", alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        # Saftey check in case a NaN value sneaks in
        if not np.isnan(yval): 
            plt.text(bar.get_x() + bar.get_width()/2, yval * 1.5, f'{yval:.1e}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()