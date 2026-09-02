"""
KKT (Karush-Kuhn-Tucker) Conditions Analysis Module

This module provides comprehensive KKT conditions analysis for constrained optimization problems.
It computes Lagrange multipliers, checks KKT conditions, and provides visualization.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import re
from typing import List, Tuple, Optional, Dict, Any

def parse_single_constraint(constraint_str):
    """Parse a single constraint string in the format 'expression => value' or 'expression >= value'."""
    clean = constraint_str.replace(' ', '')

    if '=>' in clean:
        parts = clean.split('=>')
        operator = '=>'
    elif '>=' in clean:
        parts = clean.split('>=')
        operator = '>='

    else:
        return constraint_str, constraint_str
    if len(parts) != 2:
        return constraint_str, constraint_str

    expr = parts[0]
    value = parts[1]

    return f"({expr}) - ({value})", f"{expr} {operator} {value}"

def parse_constraint(constraint_str):
    """Parse a constraint string with support for multiple constraints using 'and'."""
    if ' and ' in constraint_str.lower():
        parts = re.split(r'\s+and\s+', constraint_str, flags=re.IGNORECASE)
        constraints = []
        displays = []
        for part in parts:
            parsed, display = parse_single_constraint(part)
            constraints.append(parsed)
            displays.append(display)
        combined = f"min({', '.join(constraints)})"
        combined_display = ' and '.join(displays)
        return combined, combined_display
    else:
        return parse_single_constraint(constraint_str)

def compute_lagrange_multipliers(x: List[float], func_str: str, constraint_str: str,
                                  namespace: Dict[str, Any], tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Compute Lagrange multipliers for KKT conditions.

    For inequality constraints g_i(x) >= 0, the KKT conditions are:
    1. Stationarity: ∇f(x) + Σ λ_i * ∇g_i(x) = 0
    2. Primal feasibility: g_i(x) >= 0 for all i
    3. Dual feasibility: λ_i >= 0 for all i
    4. Complementary slackness: λ_i * g_i(x) = 0 for all i

    Returns dictionary with multipliers, active constraints, and violation metrics.
    """
    num_vars = len(x)

    # Parse individual constraints
    if ' and ' in constraint_str.lower():
        parts = re.split(r'\s+and\s+', constraint_str, flags=re.IGNORECASE)
        individual_constraints = [parse_single_constraint(part)[0] for part in parts]
    else:
        individual_constraints = [parse_single_constraint(constraint_str)[0]]

    # Evaluate constraints and compute gradients
    g_values = []
    for constr in individual_constraints:
        try:
            for i, val in enumerate(x):
                namespace[f'x{i+1}'] = val
            g_values.append(eval(constr, {"__builtins__": {}}, namespace))
        except:
            g_values.append(float('inf'))

    # Compute gradients using finite differences
    h = 1e-7
    grad_f = []
    grad_g_list = []

    for i in range(num_vars):
        x_plus = x.copy()
        x_plus[i] += h
        try:
            for j, val in enumerate(x_plus):
                namespace[f'x{j+1}'] = val
            f_plus = eval(func_str, {"__builtins__": {}}, namespace)

            for j, val in enumerate(x):
                namespace[f'x{j+1}'] = val
            f_current = eval(func_str, {"__builtins__": {}}, namespace)

            partial = (f_plus - f_current) / h
            grad_f.append(partial if not np.isinf(partial) and not np.isnan(partial) else 0.0)
        except:
            grad_f.append(0.0)

    for constr in individual_constraints:
        grad_g = []
        for i in range(num_vars):
            x_plus = x.copy()
            x_plus[i] += h
            try:
                for j, val in enumerate(x_plus):
                    namespace[f'x{j+1}'] = val
                g_plus = eval(constr, {"__builtins__": {}}, namespace)

                for j, val in enumerate(x):
                    namespace[f'x{j+1}'] = val
                g_current = eval(constr, {"__builtins__": {}}, namespace)

                partial = (g_plus - g_current) / h
                grad_g.append(partial if not np.isinf(partial) and not np.isnan(partial) else 0.0)
            except:
                grad_g.append(0.0)
        grad_g_list.append(grad_g)

    # Identify active constraints
    active_constraints = [i for i, g in enumerate(g_values) if g <= tolerance]

    # Solve for multipliers using least squares
    if active_constraints:
        A_matrix = np.array([grad_g_list[i] for i in active_constraints]).T
        b_vector = np.array([-g for g in grad_f])
        try:
            multipliers_active, _, _, _ = np.linalg.lstsq(A_matrix, b_vector, rcond=None)
            multipliers = [0.0] * len(individual_constraints)
            for idx, mult in zip(active_constraints, multipliers_active):
                multipliers[idx] = max(0.0, mult)  # Ensure dual feasibility
        except:
            multipliers = [0.0] * len(individual_constraints)
    else:
        multipliers = [0.0] * len(individual_constraints)

    # Calculate violations
    stationarity_violation = sum(abs(grad_f[i] + sum(multipliers[j] * grad_g_list[j][i] for j in range(len(multipliers)))) for i in range(num_vars))
    comp_slack_violation = sum(abs(g * mult) for g, mult in zip(g_values, multipliers))
    primal_violation = sum(max(0, -g) for g in g_values)
    dual_violation = sum(max(0, -mult) for mult in multipliers)

    return {
        'multipliers': multipliers,
        'active_constraints': active_constraints,
        'kkt_violation': stationarity_violation + comp_slack_violation + primal_violation + dual_violation,
        'stationarity_violation': stationarity_violation,
        'complementary_slackness_violation': comp_slack_violation,
        'primal_violation': primal_violation,
        'dual_violation': dual_violation,
        'g_values': g_values,
        'grad_f': grad_f,
        'grad_g_list': grad_g_list
    }

def check_kkt_conditions(x: List[float], func_str: str, constraint_str: str,
                         namespace: Dict[str, Any], tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Check if KKT conditions are satisfied at a given point.
    Returns dictionary with satisfaction status and violation details.
    """
    result = compute_lagrange_multipliers(x, func_str, constraint_str, namespace, tolerance)

    is_satisfied = (
        result['stationarity_violation'] < tolerance and
        result['complementary_slackness_violation'] < tolerance and
        result['primal_violation'] < tolerance and
        result['dual_violation'] < tolerance
    )

    return {
        'is_kkt_satisfied': is_satisfied,
        'kkt_violation': result['kkt_violation'],
        'details': result,
        'tolerance': tolerance
    }

def kkt_based_optimization(func_str: str, constraint_str: str, start_values: List[float],
                           max_iterations: int = 200, learning_rate: float = 0.001,
                           alpha: float = 1.0, beta1: float = 0.9, beta2: float = 0.999,
                           epsilon: float = 1e-8, kkt_tolerance: float = 1e-6,
                           verbose: bool = True, quiet: bool = False):
    """
    Run safe gradient flow optimization with KKT analysis.
    """
    from safegradientflow import safe_gradient_flow_adam

    namespace = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'exp': math.exp, 'log': math.log, 'log10': math.log10,
        'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e,
        'abs': abs, 'min': min, 'max': max
    }
    for i, val in enumerate(start_values):
        namespace[f'x{i+1}'] = val

    # Run optimization
    history, f_history, g_history, final_x, final_f, final_g, constraint_display, violation_history = safe_gradient_flow_adam(
        func_str=func_str, constraint_str=constraint_str, start_values=start_values,
        learning_rate=learning_rate, alpha=alpha, max_iterations=max_iterations,
        beta1=beta1, beta2=beta2, epsilon=epsilon, verbose=verbose, quiet=quiet
    )

    if history is None:
        return None, None, None, None

    # Check KKT conditions
    kkt_check = check_kkt_conditions(final_x, func_str, constraint_str, namespace, kkt_tolerance)

    if verbose and not quiet:
        print("\n" + "=" * 70)
        print("KKT CONDITIONS ANALYSIS")
        print("=" * 70)
        print(f"KKT conditions satisfied: {'YES' if kkt_check['is_kkt_satisfied'] else 'NO'}")
        print(f"Total KKT violation: {kkt_check['kkt_violation']:.6e}")
        # ... (additional output)

    kkt_info = {
        'is_kkt_satisfied': kkt_check['is_kkt_satisfied'],
        'kkt_violation': kkt_check['kkt_violation'],
        'multipliers': kkt_check['details']['multipliers'],
        'active_constraints': kkt_check['details']['active_constraints'],
        'g_values': kkt_check['details']['g_values'],
        'stationarity_violation': kkt_check['details']['stationarity_violation'],
        'complementary_slackness_violation': kkt_check['details']['complementary_slackness_violation'],
        'primal_violation': kkt_check['details']['primal_violation'],
        'dual_violation': kkt_check['details']['dual_violation'],
        'tolerance': kkt_tolerance,
        'final_f': final_f,
        'final_g': final_g
    }

    return history, final_f, final_g, kkt_info

def kkt_visualization(kkt_info: Dict[str, Any], var_names: List[str],
                     func_str: str, constraint_str: str):
    """Create visualization of KKT conditions and Lagrange multipliers."""
    if not kkt_info:
        return None, None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Lagrange multipliers
    multipliers = kkt_info['multipliers']
    g_values = kkt_info['g_values']
    active_constraints = kkt_info['active_constraints']

    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(multipliers)), multipliers, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    for i, bar in enumerate(bars):
        if i in active_constraints:
            bar.set_color('#FF6B6B')
            bar.set_edgecolor('black')
            bar.set_linewidth(2)
    ax1.set_title('Lagrange Multipliers')
    ax1.set_xlabel('Constraint')
    ax1.set_ylabel('Multiplier (λ)')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Constraint values
    ax2 = axes[0, 1]
    ax2.bar(range(len(g_values)), g_values, alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--', label='g=0')
    ax2.set_title('Constraint Values')
    ax2.set_xlabel('Constraint')
    ax2.set_ylabel('g(x)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Violation breakdown
    ax3 = axes[1, 0]
    violations = {
        'Stationarity': kkt_info['stationarity_violation'],
        'Complementary Slackness': kkt_info['complementary_slackness_violation'],
        'Primal Feasibility': kkt_info['primal_violation'],
        'Dual Feasibility': kkt_info['dual_violation']
    }
    display_values = [v for v in violations.values() if v > 1e-10]
    display_labels = [k for k, v in violations.items() if v > 1e-10]
    if display_values:
        ax3.pie(display_values, labels=display_labels, autopct='%1.1f%%')
    else:
        ax3.text(0.5, 0.5, 'All KKT conditions\nsatisfied!', ha='center', va='center', fontsize=14, color='green')
    ax3.set_title('KKT Violation Breakdown')

    # Plot 4: Status
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.6, f'Total Violation: {kkt_info["kkt_violation"]:.2e}', ha='center', va='center', fontsize=14)
    ax4.text(0.5, 0.4, f'KKT Satisfied: {"YES" if kkt_info["is_kkt_satisfied"] else "NO"}',
            ha='center', va='center', fontsize=14, color='green' if kkt_info["is_kkt_satisfied"] else 'red')
    ax4.text(0.5, 0.2, f'Tolerance: {kkt_info["tolerance"]:.2e}', ha='center', va='center', fontsize=12)
    ax4.set_title('KKT Status')
    ax4.axis('off')

    fig.suptitle(f'KKT Analysis: {func_str}\n{constraint_str}')
    plt.tight_layout()
    return fig, axes

def export_kkt_data(kkt_info: Dict[str, Any], var_names: List[str], func_str: str,
                   constraint_str: str, final_x: List[float], final_f: float,
                   prefix: str = "kkt_analysis"):
    """Export KKT analysis to CSV and text files."""
    import csv

    # Export to CSV
    with open(f"{prefix}_kkt.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['KKT Satisfied', 'YES' if kkt_info['is_kkt_satisfied'] else 'NO'])
        writer.writerow(['Total KKT Violation', kkt_info['kkt_violation']])
        writer.writerow(['Stationarity Violation', kkt_info['stationarity_violation']])
        writer.writerow(['Complementary Slackness Violation', kkt_info['complementary_slackness_violation']])
        writer.writerow(['Primal Feasibility Violation', kkt_info['primal_violation']])
        writer.writerow(['Dual Feasibility Violation', kkt_info['dual_violation']])
        for i, mult in enumerate(kkt_info['multipliers']):
            writer.writerow([f'Lagrange Multiplier λ_{i+1}', mult])
        for i, g in enumerate(kkt_info['g_values']):
            writer.writerow([f'Constraint g_{i+1} Value', g])

    # Export solution summary
    with open(f"{prefix}_solution.txt", 'w') as f:
        f.write(f"KKT Analysis Results\n{'='*50}\n")
        f.write(f"Objective: {func_str}\nConstraints: {constraint_str}\n\nSolution:\n")
        for name, val in zip(var_names, final_x):
            f.write(f"  {name} = {val:.10f}\n")
        if final_f is not None:
            f.write(f"  f(x) = {final_f:.10f}\n")
        f.write(f"\nKKT Conditions:\n  Satisfied: {'YES' if kkt_info['is_kkt_satisfied'] else 'NO'}\n")
        f.write(f"  Total violation: {kkt_info['kkt_violation']:.6e}\n\n")
        f.write(f"Lagrange Multipliers:\n")
        for i, mult in enumerate(kkt_info['multipliers']):
            f.write(f"  λ_{i+1} = {mult:.10f}\n")
        f.write(f"\nConstraint Values:\n")
        for i, g in enumerate(kkt_info['g_values']):
            f.write(f"  g_{i+1}(x) = {g:.10f}\n")

    print(f"Exported KKT data to {prefix}_kkt.csv and {prefix}_solution.txt")
