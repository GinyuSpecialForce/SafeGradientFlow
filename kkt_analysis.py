"""
KKT (Karush-Kuhn-Tucker) Conditions Analysis Module

This module provides comprehensive KKT conditions analysis for constrained optimization problems.
It computes Lagrange multipliers, checks KKT conditions, and prints results.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import re
from typing import List, Tuple, Optional, Dict, Any
import csv


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
    1. Stationarity: ∇f(x) - Σ λ_i * ∇g_i(x) = 0  (CORRECTED: minus sign)
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
    
    # Evaluate constraints
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
    
    # Identify active constraints (g_i(x) <= tolerance)
    active_constraints = [i for i, g in enumerate(g_values) if g <= tolerance]
    
    # Initialize multipliers
    multipliers = [0.0] * len(individual_constraints)
    
    # ============= FIXED MULTIPLIER COMPUTATION WITH CORRECT SIGN =============
    # For minimization with inequality constraints g(x) >= 0:
    # Stationarity condition: ∇f(x) - Σ λ_i * ∇g_i(x) = 0  (λ_i >= 0)
    # This means: ∇f(x) = Σ λ_i * ∇g_i(x)
    # For a single active constraint: λ = (∇f · ∇g) / ||∇g||²
    
    if active_constraints:
        for idx in active_constraints:
            grad_g = np.array(grad_g_list[idx])
            grad_f_np = np.array(grad_f)
            
            # Compute the projection of ∇f onto ∇g
            grad_g_norm_sq = np.dot(grad_g, grad_g)
            
            if grad_g_norm_sq > 1e-10:
                # CORRECT FORMULA: λ = (∇f · ∇g) / ||∇g||² (positive sign)
                lambda_val = np.dot(grad_f_np, grad_g) / grad_g_norm_sq
                multipliers[idx] = max(0.0, lambda_val)  # Dual feasibility: λ >= 0
            else:
                multipliers[idx] = 0.0
    
    # Calculate violations using the CORRECT stationarity condition
    # Stationarity: ∇f - Σ λ_i * ∇g_i = 0
    stationarity_violation = 0.0
    for i in range(num_vars):
        gradient_sum = grad_f[i]
        for j in range(len(multipliers)):
            gradient_sum -= multipliers[j] * grad_g_list[j][i]
        stationarity_violation += abs(gradient_sum)
    
    comp_slack_violation = sum(abs(g * mult) for g, mult in zip(g_values, multipliers))
    primal_violation = sum(max(0, -g) for g in g_values)
    dual_violation = sum(max(0, -mult) for mult in multipliers)
    
    # If stationarity is still violated, try an alternative approach
    if stationarity_violation > tolerance and active_constraints:
        for idx in active_constraints:
            grad_g = np.array(grad_g_list[idx])
            grad_f_np = np.array(grad_f)
            
            if np.linalg.norm(grad_g) > 1e-10:
                # λ = (∇f · ∇g) / ||∇g||²
                lambda_val = np.dot(grad_f_np, grad_g) / np.dot(grad_g, grad_g)
                multipliers[idx] = max(0.0, lambda_val)
                
                # Recompute stationarity violation with this multiplier
                new_violation = 0.0
                for i in range(num_vars):
                    gradient_sum = grad_f[i] - multipliers[idx] * grad_g[i]
                    new_violation += abs(gradient_sum)
                
                if new_violation < stationarity_violation:
                    stationarity_violation = new_violation
    
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
        'grad_g_list': grad_g_list,
        'grad_f_norm': np.linalg.norm(grad_f),
        'grad_g_norm': np.linalg.norm(np.array(grad_g_list[0])) if grad_g_list else 0
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


def print_kkt_results(kkt_info: Dict[str, Any], var_names: List[str], 
                      func_str: str, constraint_str: str, final_x: List[float]):
    """
    Print KKT results in a clean, formatted way.
    """
    if not kkt_info:
        print("No KKT information available.")
        return
    
    # Extract data
    if 'details' in kkt_info:
        details = kkt_info['details']
        is_satisfied = kkt_info.get('is_kkt_satisfied', False)
        kkt_violation = kkt_info.get('kkt_violation', 0)
        tolerance = kkt_info.get('tolerance', 1e-6)
        multipliers = details.get('multipliers', [])
        g_values = details.get('g_values', [])
        active_constraints = details.get('active_constraints', [])
        stationarity = details.get('stationarity_violation', 0)
        comp_slack = details.get('complementary_slackness_violation', 0)
        primal = details.get('primal_violation', 0)
        dual = details.get('dual_violation', 0)
        grad_f = details.get('grad_f', [])
        grad_g_list = details.get('grad_g_list', [])
    else:
        is_satisfied = kkt_info.get('is_kkt_satisfied', False)
        kkt_violation = kkt_info.get('kkt_violation', 0)
        tolerance = kkt_info.get('tolerance', 1e-6)
        multipliers = kkt_info.get('multipliers', [])
        g_values = kkt_info.get('g_values', [])
        active_constraints = kkt_info.get('active_constraints', [])
        stationarity = kkt_info.get('stationarity_violation', 0)
        comp_slack = kkt_info.get('complementary_slackness_violation', 0)
        primal = kkt_info.get('primal_violation', 0)
        dual = kkt_info.get('dual_violation', 0)
        grad_f = kkt_info.get('grad_f', [])
        grad_g_list = kkt_info.get('grad_g_list', [])
    
    print("\n" + "=" * 70)
    print("KKT CONDITIONS ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Objective Function: {func_str}")
    print(f"Constraint: {constraint_str}")
    print("-" * 70)
    
    # Print solution
    print("\nOPTIMAL SOLUTION:")
    for i, name in enumerate(var_names):
        print(f"  {name} = {final_x[i]:.10f}")
    
    # Print gradients
    print("\nGRADIENTS AT OPTIMUM:")
    print(f"  ∇f = [{', '.join([f'{g:.6f}' for g in grad_f])}]")
    for i, grad_g in enumerate(grad_g_list):
        print(f"  ∇g_{i+1} = [{', '.join([f'{g:.6f}' for g in grad_g])}]")
    
    # Print Lagrange multipliers
    print("\nLAGRANGE MULTIPLIERS:")
    for i, lam in enumerate(multipliers):
        active_status = " (ACTIVE)" if i in active_constraints else " (INACTIVE)"
        print(f"  λ_{i+1} = {lam:.10f}{active_status}")
    
    # Print constraint values
    print("\nCONSTRAINT VALUES:")
    for i, g in enumerate(g_values):
        status = "✓ Feasible" if g >= 0 else "✗ VIOLATED"
        print(f"  g_{i+1}(x) = {g:.10f}  {status}")
    
    # Print active constraints
    print("\nACTIVE CONSTRAINTS:")
    if active_constraints:
        for i in active_constraints:
            print(f"  Constraint {i+1} is active (g_{i+1}(x) ≈ 0)")
    else:
        print("  No active constraints")
    
    # Print KKT condition checks
    print("\nKKT CONDITIONS CHECK:")
    print(f"  Stationarity:        {'✓' if stationarity < tolerance else '✗'}  (violation: {stationarity:.6e})")
    print(f"  Complementary Slack: {'✓' if comp_slack < tolerance else '✗'}  (violation: {comp_slack:.6e})")
    print(f"  Primal Feasibility:  {'✓' if primal < tolerance else '✗'}  (violation: {primal:.6e})")
    print(f"  Dual Feasibility:    {'✓' if dual < tolerance else '✗'}  (violation: {dual:.6e})")
    
    # Final verdict
    print("-" * 70)
    print(f"OVERALL STATUS: {'✅ KKT CONDITIONS SATISFIED' if is_satisfied else '❌ KKT CONDITIONS NOT SATISFIED'}")
    print(f"Total KKT Violation: {kkt_violation:.6e}")
    print(f"Tolerance: {tolerance:.2e}")
    print("=" * 70)


def export_kkt_data(kkt_info: Dict[str, Any], var_names: List[str], func_str: str,
                   constraint_str: str, final_x: List[float], final_f: float,
                   prefix: str = "kkt_analysis"):
    """Export KKT analysis to CSV and text files."""
    
    # Extract data
    if 'details' in kkt_info:
        details = kkt_info['details']
        is_satisfied = kkt_info.get('is_kkt_satisfied', False)
        kkt_violation = kkt_info.get('kkt_violation', 0)
        tolerance = kkt_info.get('tolerance', 1e-6)
        multipliers = details.get('multipliers', [])
        g_values = details.get('g_values', [])
        stationarity = details.get('stationarity_violation', 0)
        comp_slack = details.get('complementary_slackness_violation', 0)
        primal = details.get('primal_violation', 0)
        dual = details.get('dual_violation', 0)
        active = details.get('active_constraints', [])
        grad_f = details.get('grad_f', [])
        grad_g_list = details.get('grad_g_list', [])
    else:
        is_satisfied = kkt_info.get('is_kkt_satisfied', False)
        kkt_violation = kkt_info.get('kkt_violation', 0)
        tolerance = kkt_info.get('tolerance', 1e-6)
        multipliers = kkt_info.get('multipliers', [])
        g_values = kkt_info.get('g_values', [])
        stationarity = kkt_info.get('stationarity_violation', 0)
        comp_slack = kkt_info.get('complementary_slackness_violation', 0)
        primal = kkt_info.get('primal_violation', 0)
        dual = kkt_info.get('dual_violation', 0)
        active = kkt_info.get('active_constraints', [])
        grad_f = kkt_info.get('grad_f', [])
        grad_g_list = kkt_info.get('grad_g_list', [])
    
    # Export to CSV
    with open(f"{prefix}_kkt.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['KKT Satisfied', 'YES' if is_satisfied else 'NO'])
        writer.writerow(['Total KKT Violation', kkt_violation])
        writer.writerow(['Stationarity Violation', stationarity])
        writer.writerow(['Complementary Slackness Violation', comp_slack])
        writer.writerow(['Primal Feasibility Violation', primal])
        writer.writerow(['Dual Feasibility Violation', dual])
        writer.writerow(['Tolerance', tolerance])
        for i, mult in enumerate(multipliers):
            writer.writerow([f'Lagrange Multiplier λ_{i+1}', mult])
            writer.writerow([f'Active λ_{i+1}', 'YES' if i in active else 'NO'])
        for i, g in enumerate(g_values):
            writer.writerow([f'Constraint g_{i+1} Value', g])
        for i, g in enumerate(grad_f):
            writer.writerow([f'∇f_{i+1}', g])
        for j, grad_g in enumerate(grad_g_list):
            for i, g in enumerate(grad_g):
                writer.writerow([f'∇g_{j+1}_{i+1}', g])
    
    # Export solution summary
    with open(f"{prefix}_solution.txt", 'w') as f:
        f.write(f"KKT Analysis Results\n{'='*50}\n")
        f.write(f"Objective: {func_str}\nConstraints: {constraint_str}\n\nSolution:\n")
        for name, val in zip(var_names, final_x):
            f.write(f"  {name} = {val:.10f}\n")
        if final_f is not None:
            f.write(f"  f(x) = {final_f:.10f}\n")
        f.write(f"\nKKT Conditions:\n  Satisfied: {'YES' if is_satisfied else 'NO'}\n")
        f.write(f"  Total violation: {kkt_violation:.6e}\n")
        f.write(f"  Tolerance: {tolerance:.2e}\n\n")
        f.write(f"Lagrange Multipliers:\n")
        for i, mult in enumerate(multipliers):
            active_str = ' (active)' if i in active else ''
            f.write(f"  λ_{i+1} = {mult:.10f}{active_str}\n")
        f.write(f"\nConstraint Values:\n")
        for i, g in enumerate(g_values):
            f.write(f"  g_{i+1}(x) = {g:.10f}\n")
        f.write(f"\nActive constraints: {[i+1 for i in active]}\n")
        f.write(f"\nKKT Condition Violations:\n")
        f.write(f"  Stationarity: {stationarity:.6e}\n")
        f.write(f"  Complementary Slackness: {comp_slack:.6e}\n")
        f.write(f"  Primal Feasibility: {primal:.6e}\n")
        f.write(f"  Dual Feasibility: {dual:.6e}\n")
    
    print(f"Exported KKT data to {prefix}_kkt.csv and {prefix}_solution.txt")


def kkt_analysis_main(func_str: str, constraint_str: str, start_values: List[float],
                      max_iterations: int = 200, learning_rate: float = None,
                      alpha: float = None, beta1: float = 0.9, beta2: float = 0.999,
                      epsilon: float = 1e-8, kkt_tolerance: float = 1e-6,
                      save_prefix: str = None, no_plots: bool = False,
                      quiet: bool = False):
    """
    Run safe gradient flow optimization with KKT analysis.
    """
    # Import the safe gradient flow function
    from safegradientflow import safe_gradient_flow_adam
    
    namespace = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'exp': math.exp, 'log': math.log, 'log10': math.log10,
        'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e,
        'abs': abs, 'min': min, 'max': max
    }
    for i, val in enumerate(start_values):
        namespace[f'x{i+1}'] = val
    
    # Set default learning rate and alpha if not provided
    if learning_rate is None:
        learning_rate = 0.001
    if alpha is None:
        alpha = 1.0
    
    # Run optimization
    print("\n" + "=" * 70)
    print("RUNNING SAFE GRADIENT FLOW OPTIMIZATION")
    print("=" * 70)
    
    history, f_history, g_history, final_x, final_f, final_g, constraint_display, violation_history = safe_gradient_flow_adam(
        func_str=func_str, constraint_str=constraint_str, start_values=start_values,
        learning_rate=learning_rate, alpha=alpha, max_iterations=max_iterations,
        beta1=beta1, beta2=beta2, epsilon=epsilon, verbose=True, quiet=quiet
    )
    
    if history is None:
        print("Optimization failed.")
        return None, None, None, None
    
    # Check KKT conditions
    kkt_check = check_kkt_conditions(final_x, func_str, constraint_str, namespace, kkt_tolerance)
    
    var_names = [f'x{i+1}' for i in range(len(start_values))]
    kkt_info = {
        'is_kkt_satisfied': kkt_check['is_kkt_satisfied'],
        'kkt_violation': kkt_check['kkt_violation'],
        'details': kkt_check['details'],
        'tolerance': kkt_tolerance,
        'final_f': final_f,
        'final_g': final_g,
        'final_x': final_x
    }
    
    # Print KKT results
    print_kkt_results(kkt_info, var_names, func_str, constraint_str, final_x)
    
    # Export KKT data if requested
    if save_prefix:
        export_kkt_data(kkt_info, var_names, func_str, constraint_str, final_x, final_f, save_prefix)
    
    return history, final_f, final_g, kkt_info


# For backward compatibility
def kkt_visualization(kkt_info: Dict[str, Any], var_names: List[str],
                     func_str: str, constraint_str: str):
    """Legacy function - now prints results instead of showing graphs."""
    print_kkt_results(kkt_info, var_names, func_str, constraint_str, 
                     kkt_info.get('final_x', [0, 0]))
    return None, None


def kkt_based_optimization(func_str: str, constraint_str: str, start_values: List[float],
                           max_iterations: int = 200, learning_rate: float = 0.001,
                           alpha: float = 1.0, beta1: float = 0.9, beta2: float = 0.999,
                           epsilon: float = 1e-8, kkt_tolerance: float = 1e-6,
                           verbose: bool = True, quiet: bool = False):
    """
    Legacy function - use kkt_analysis_main instead.
    """
    return kkt_analysis_main(
        func_str=func_str,
        constraint_str=constraint_str,
        start_values=start_values,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        alpha=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        kkt_tolerance=kkt_tolerance,
        save_prefix=None,
        no_plots=False,
        quiet=quiet
    )
