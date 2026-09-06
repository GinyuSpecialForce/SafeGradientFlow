import math
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re


def parse_constraint(constraint_str):
    """
    Parse a constraint string in the format 'expression => value' or 'expression >= value'.
    Returns the constraint function as g(x) >= 0.
    
    Examples:
    - "x1 + x2 => 1" -> g(x) = x1 + x2 - 1
    - "x1**2 + x2**2 => 1" -> g(x) = x1**2 + x2**2 - 1
    - "x1 + x2 => 1 and x1 - x2 => 0.5" -> combined constraints
    """
    # Check if there are multiple constraints separated by 'and'
    if ' and ' in constraint_str.lower():
        # Split by 'and' (case insensitive)
        parts = re.split(r'\s+and\s+', constraint_str, flags=re.IGNORECASE)
        constraints = []
        displays = []
        for part in parts:
            parsed, display = parse_single_constraint(part)
            constraints.append(parsed)
            displays.append(display)
        # Combine constraints using min (g(x) = min(g1(x), g2(x), ...))
        combined = f"min({', '.join(constraints)})"
        combined_display = ' and '.join(displays)
        return combined, combined_display
    else:
        return parse_single_constraint(constraint_str)


def parse_single_constraint(constraint_str):
    """
    Parse a single constraint string in the format 'expression => value' or 'expression >= value'.
    Returns the constraint function as g(x) >= 0.
    """
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


def suggest_parameters(func_str, constraint_str, start_values):
    """
    Suggest appropriate learning rate and alpha values based on the function structure.
    These are conservative values that prioritize stability over speed.
    """
    clean_func = func_str.replace(' ', '')
    clean_constraint = constraint_str.replace(' ', '')
    
    # Start with conservative defaults
    suggested_lr = 0.001
    suggested_alpha = 1.0
    reason = ""
    
    # Check if multiple constraints
    num_constraints = len(re.findall(r'and', constraint_str.lower())) + 1 if 'and' in constraint_str.lower() else 1
    if num_constraints > 1:
        reason = f"Multiple constraints ({num_constraints}). "
        suggested_alpha = 1.5
        suggested_lr = 0.0005
    
    # Detect objective type
    if '**4' in clean_func or '**5' in clean_func or '**6' in clean_func:
        suggested_lr = 0.0001
        reason += "Objective has degree 4+. Using very small LR (0.0001)."
    elif '**3' in clean_func:
        suggested_lr = 0.0005
        reason += "Objective has cubic term. Using small LR (0.0005)."
    elif '**2' in clean_func or '^2' in clean_func:
        if 'x1**2 + x2**2' in clean_func or 'x1^2 + x2^2' in clean_func:
            suggested_lr = 0.001
            reason += "Simple quadratic. Using LR=0.001."
        elif '100*' in clean_func or '50*' in clean_func:
            suggested_lr = 0.0005
            reason += "Quadratic with large coefficients. Using LR=0.0005."
        else:
            suggested_lr = 0.001
            reason += "Quadratic objective. Using LR=0.001."
    elif 'exp(' in clean_func or 'e**' in clean_func:
        suggested_lr = 0.0001
        reason += "Exponential objective. Using very small LR (0.0001)."
    elif 'sin' in clean_func or 'cos' in clean_func or 'tan' in clean_func:
        suggested_lr = 0.001
        reason += "Trigonometric objective. Using LR=0.001."
    else:
        suggested_lr = 0.001
        reason += "Default conservative LR=0.001."
    
    # Adjust based on constraint complexity
    if '**4' in clean_constraint or '**5' in clean_constraint or '**6' in clean_constraint:
        suggested_alpha = max(suggested_alpha, 1.5)
        suggested_lr = suggested_lr * 0.5
        reason += " High-degree constraint."
    elif 'sin' in clean_constraint or 'cos' in clean_constraint:
        suggested_alpha = max(suggested_alpha, 1.5)
        suggested_lr = suggested_lr * 0.5
        reason += " Trigonometric constraint."
    elif 'exp' in clean_constraint:
        suggested_alpha = max(suggested_alpha, 2.0)
        suggested_lr = suggested_lr * 0.3
        reason += " Exponential constraint."
    
    # Special case: Circle constraint with quadratic objective
    if ('x1**2 + x2**2' in clean_func or 'x1^2 + x2^2' in clean_func) and \
       ('x1**2 + x2**2' in clean_constraint or 'x1^2 + x2^2' in clean_constraint):
        suggested_lr = 0.001
        suggested_alpha = 1.0
        reason = "Circle constraint with quadratic objective. Using LR=0.001, Alpha=1.0."
    
    # Adjust for starting values
    max_start = max(abs(v) for v in start_values)
    if max_start > 10:
        suggested_lr = suggested_lr / (max_start / 5)
        reason += f" Starting values far from origin ({max_start:.1f}). LR reduced."
    
    # Ensure reasonable bounds
    suggested_lr = min(max(suggested_lr, 0.00001), 0.01)
    suggested_alpha = min(max(suggested_alpha, 0.1), 2.5)
    
    reason += " If oscillation occurs, try: -lr 0.001 -a 1.0"
    
    return suggested_lr, suggested_alpha, reason


class LossSurface:
    """A loss surface for 2D functions with safe gradient flow visualization."""
    
    def __init__(self, func_str, constraint_str, x1_range=(-3, 3), x2_range=(-3, 3), num_points=200, alpha=0.5):
        self.func_str = func_str
        self.constraint_raw = constraint_str
        self.constraint_parsed, self.constraint_display = parse_constraint(constraint_str)
        self.constraint_str = self.constraint_parsed
        self.x1_min, self.x1_max = x1_range
        self.x2_min, self.x2_max = x2_range
        self.alpha = alpha
        
        x1_list = np.linspace(x1_range[0], x1_range[1], num_points)
        x2_list = np.linspace(x2_range[0], x2_range[1], num_points)
        self.X1, self.X2 = np.meshgrid(x1_list, x2_list)
        
        self.namespace = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'exp': math.exp, 'log': math.log, 'log10': math.log10,
            'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e,
            'abs': abs, 'min': min, 'max': max,
            'x1': 0, 'x2': 0
        }
        
        self.Z = np.zeros_like(self.X1)
        for i in range(num_points):
            for j in range(num_points):
                self.namespace['x1'] = self.X1[i, j]
                self.namespace['x2'] = self.X2[i, j]
                try:
                    val = eval(func_str, {"__builtins__": {}}, self.namespace)
                    if np.isinf(val) or np.isnan(val) or abs(val) > 1e100:
                        self.Z[i, j] = np.nan
                    else:
                        self.Z[i, j] = val
                except:
                    self.Z[i, j] = np.nan
        
        self.G = np.zeros_like(self.X1)
        for i in range(num_points):
            for j in range(num_points):
                self.namespace['x1'] = self.X1[i, j]
                self.namespace['x2'] = self.X2[i, j]
                try:
                    val = eval(self.constraint_str, {"__builtins__": {}}, self.namespace)
                    if np.isinf(val) or np.isnan(val) or abs(val) > 1e100:
                        self.G[i, j] = np.nan
                    else:
                        self.G[i, j] = val
                except:
                    self.G[i, j] = np.nan
    
    def plot_3d(self, trajectories=None, best_trajectory=None, title=None, alpha_val=None):
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        Z_masked = np.ma.masked_invalid(self.Z)
        surf = ax.plot_surface(self.X1, self.X2, Z_masked, cmap='viridis', 
                               alpha=0.7, linewidth=0, antialiased=True)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='f(x)')
        
        ax.contour(self.X1, self.X2, self.G, levels=[0], colors=['red'], 
                  linewidths=2, alpha=0.8)
        
        if trajectories:
            colors = ['cyan', 'magenta', 'yellow', 'orange', 'purple', 'pink', 'lime', 'white']
            for idx, trajectory in enumerate(trajectories):
                traj_x1 = [p[0] for p in trajectory]
                traj_x2 = [p[1] for p in trajectory]
                traj_z = [self.evaluate_point(p[0], p[1]) for p in trajectory]
                
                valid_indices = [i for i, z in enumerate(traj_z) if np.isfinite(z) and abs(z) < 1e100]
                if valid_indices:
                    traj_x1 = [traj_x1[i] for i in valid_indices]
                    traj_x2 = [traj_x2[i] for i in valid_indices]
                    traj_z = [traj_z[i] for i in valid_indices]
                    
                    color = colors[idx % len(colors)]
                    alpha = 0.3 if idx > 0 else 0.5
                    linewidth = 1.0 if idx > 0 else 2.0
                    
                    ax.plot(traj_x1, traj_x2, traj_z, '-', color=color, 
                           linewidth=linewidth, alpha=alpha, label=f'Run {idx+1}')
                    
                    ax.scatter(traj_x1[0], traj_x2[0], traj_z[0], 
                              color=color, s=30, marker='o', alpha=0.5)
        
        if best_trajectory:
            traj_x1 = [p[0] for p in best_trajectory]
            traj_x2 = [p[1] for p in best_trajectory]
            traj_z = [self.evaluate_point(p[0], p[1]) for p in best_trajectory]
            
            valid_indices = [i for i, z in enumerate(traj_z) if np.isfinite(z) and abs(z) < 1e100]
            if valid_indices:
                traj_x1 = [traj_x1[i] for i in valid_indices]
                traj_x2 = [traj_x2[i] for i in valid_indices]
                traj_z = [traj_z[i] for i in valid_indices]
                
                ax.plot(traj_x1, traj_x2, traj_z, 'gold', linewidth=4, label='BEST Path')
                ax.scatter(traj_x1[0], traj_x2[0], traj_z[0], 
                          color='green', s=120, label='Start', edgecolor='black', linewidth=1)
                ax.scatter(traj_x1[-1], traj_x2[-1], traj_z[-1], 
                          color='blue', s=120, label='End', edgecolor='black', linewidth=1)
        
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('x2', fontsize=12)
        ax.set_zlabel('f(x1, x2)', fontsize=12)
        
        Z_finite = self.Z[np.isfinite(self.Z)]
        if len(Z_finite) > 0:
            z_min, z_max = np.percentile(Z_finite, [1, 99])
            if z_min == z_max:
                z_min, z_max = -10, 10
            ax.set_zlim(z_min, z_max)
        
        if title:
            display_title = title
        else:
            display_title = f'Safe Gradient Flow\nf(x1, x2) = {self.func_str}\n{self.constraint_display}'
        
        if alpha_val is not None:
            display_title += f'\nAlpha = {alpha_val:.4f}'
        elif self.alpha is not None:
            display_title += f'\nAlpha = {self.alpha:.4f}'
        
        ax.set_title(display_title, fontsize=14)
        
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='Constraint boundary g(x)=0')
        ]
        
        if trajectories and len(trajectories) > 1:
            colors = ['cyan', 'magenta', 'yellow', 'orange', 'purple', 'pink', 'lime', 'white']
            for idx in range(min(3, len(trajectories))):
                color = colors[idx % len(colors)]
                legend_elements.append(Line2D([0], [0], color=color, linewidth=2, 
                                             label=f'Run {idx+1}'))
        
        if best_trajectory:
            legend_elements.append(Line2D([0], [0], color='gold', linewidth=4, label='BEST Path'))
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                                          markersize=10, label='Start'))
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                                          markersize=10, label='End'))
        
        ax.legend(handles=legend_elements, loc='upper right')
        ax.view_init(elev=25, azim=-60)
        plt.tight_layout()
        return fig, ax
    
    def plot_contour(self, trajectories=None, best_trajectory=None, title=None, alpha_val=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        Z_masked = np.ma.masked_invalid(self.Z)
        cp = ax.contour(self.X1, self.X2, Z_masked, 30, cmap='viridis', alpha=0.6)
        fig.colorbar(cp, ax=ax, label='f(x)')
        
        ax.contour(self.X1, self.X2, self.G, levels=[0], colors=['red'], 
                  linewidths=2, alpha=0.8)
        ax.contourf(self.X1, self.X2, self.G, levels=[0, 1e10], 
                   colors=['lightgreen'], alpha=0.3)
        
        if trajectories:
            colors = ['cyan', 'magenta', 'yellow', 'orange', 'purple', 'pink', 'lime', 'brown']
            for idx, trajectory in enumerate(trajectories):
                traj_x1 = [p[0] for p in trajectory]
                traj_x2 = [p[1] for p in trajectory]
                
                color = colors[idx % len(colors)]
                alpha = 0.3 if idx > 0 else 0.5
                linewidth = 1.0 if idx > 0 else 2.0
                
                ax.plot(traj_x1, traj_x2, '-', color=color, linewidth=linewidth, 
                       alpha=alpha, label=f'Run {idx+1}')
                ax.scatter(traj_x1[0], traj_x2[0], color=color, s=20, marker='o', alpha=0.5)
        
        if best_trajectory:
            traj_x1 = [p[0] for p in best_trajectory]
            traj_x2 = [p[1] for p in best_trajectory]
            
            ax.plot(traj_x1, traj_x2, 'gold', linewidth=3, label='BEST Path')
            ax.scatter(traj_x1[0], traj_x2[0], color='green', s=100, label='Start', zorder=5)
            ax.scatter(traj_x1[-1], traj_x2[-1], color='blue', s=100, label='End', zorder=5)
        
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('x2', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if title:
            display_title = title
        else:
            display_title = f'Safe Gradient Flow - 2D View\nf(x1, x2) = {self.func_str}\n{self.constraint_display}'
        
        if alpha_val is not None:
            display_title += f'\nAlpha = {alpha_val:.4f}'
        elif self.alpha is not None:
            display_title += f'\nAlpha = {self.alpha:.4f}'
        
        ax.set_title(display_title, fontsize=14)
        
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Patch(facecolor='lightgreen', alpha=0.5, label='Safe region g(x) > 0'),
            Line2D([0], [0], color='red', linewidth=2, label='Constraint boundary g(x)=0')
        ]
        
        if trajectories and len(trajectories) > 1:
            colors = ['cyan', 'magenta', 'yellow', 'orange', 'purple', 'pink', 'lime', 'brown']
            for idx in range(min(3, len(trajectories))):
                color = colors[idx % len(colors)]
                legend_elements.append(Line2D([0], [0], color=color, linewidth=2, 
                                             label=f'Run {idx+1}'))
        
        if best_trajectory:
            legend_elements.append(Line2D([0], [0], color='gold', linewidth=3, label='BEST Path'))
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                                          markersize=10, label='Start'))
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                                          markersize=10, label='End'))
        
        ax.legend(handles=legend_elements, loc='upper right')
        plt.tight_layout()
        return fig, ax
    
    def evaluate_point(self, x1, x2):
        try:
            self.namespace['x1'] = x1
            self.namespace['x2'] = x2
            val = eval(self.func_str, {"__builtins__": {}}, self.namespace)
            if np.isinf(val) or np.isnan(val) or abs(val) > 1e100:
                return float('inf')
            return val
        except:
            return float('inf')
    
    def evaluate_constraint(self, x1, x2):
        try:
            self.namespace['x1'] = x1
            self.namespace['x2'] = x2
            val = eval(self.constraint_str, {"__builtins__": {}}, self.namespace)
            if np.isinf(val) or np.isnan(val) or abs(val) > 1e100:
                return float('inf')
            return val
        except:
            return float('inf')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Safe Gradient Flow with Adam Adaptive Learning Rate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single constraint
  python safe_gradient.py -f "x1**2 + x2**2" -g "x1 + x2 => 1" -s "0.5,0.5" -i 100
  
  # Multiple constraints (use 'and' to separate)
  python safe_gradient.py -f "x1**2 + x2**2" -g "x1 + x2 => 1 and x1 - x2 => 0.5" -s "0.8,0.8" -i 200
  
  # Circle constraint with quadratic objective
  python safe_gradient.py -f "x1**2 + x2**2" -g "x1**2 + x2**2 => 0.25" -s "0.1,0.1" -i 200
  
  # Multiple complex constraints
  python safe_gradient.py -f "20 + x1**2 + x2**2 - 10*(cos(2*pi*x1) + cos(2*pi*x2))" -g "sin(x1) + cos(x2) => 0.5 and x1**2 + x2**2 => 0.25" -s "0.5,0.5" -i 300
        """
    )
    parser.add_argument(
        '-f', '--function',
        type=str,
        required=True,
        help='Objective function to minimize (e.g., "x1**2 + x2**2")'
    )
    parser.add_argument(
        '-g', '--constraint',
        type=str,
        required=True,
        help='Constraint(s) in format "expression => value" (e.g., "x1 + x2 => 1"). Use "and" for multiple constraints'
    )
    parser.add_argument(
        '-s', '--start',
        type=str,
        required=True,
        help='Starting values, comma-separated (e.g., "0.5, 0.5")'
    )
    parser.add_argument(
        '-lr', '--learning_rate',
        type=float,
        default=None,
        help='Initial learning rate (auto-selected if not provided)'
    )
    parser.add_argument(
        '-a', '--alpha',
        type=float,
        default=None,
        help='Safety parameter alpha for restoring force (auto-selected if not provided)'
    )
    parser.add_argument(
        '-i', '--iterations',
        type=int,
        default=200,
        help='Maximum number of iterations per run (default: 200)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plotting (run in headless mode)'
    )
    parser.add_argument(
        '--save',
        type=str,
        metavar='PREFIX',
        help='Save plots to files with this prefix'
    )
    parser.add_argument(
        '--multi',
        type=int,
        metavar='N',
        help='Multi-start: run safe gradient flow N times from different starting points'
    )
    parser.add_argument(
        '--range',
        type=str,
        default='-1,1',
        help='Range for random starting points in multi-start mode'
    )
    parser.add_argument(
        '--noise',
        type=float,
        metavar='AMOUNT',
        help='Add random noise to learning rate at intervals (e.g., 0.5)'
    )
    parser.add_argument(
        '--noise_freq',
        type=int,
        default=10,
        help='Frequency of noise injection (every N iterations, default: 10)'
    )
    parser.add_argument(
        '--adam_beta1',
        type=float,
        default=0.9,
        help='Adam beta1 parameter (default: 0.9)'
    )
    parser.add_argument(
        '--adam_beta2',
        type=float,
        default=0.999,
        help='Adam beta2 parameter (default: 0.999)'
    )
    parser.add_argument(
        '--adam_epsilon',
        type=float,
        default=1e-8,
        help='Adam epsilon parameter (default: 1e-8)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress iteration-by-iteration output (only show final results)'
    )
    parser.add_argument(
        '--kkt',
        action='store_true',
        help='Run KKT analysis after optimization'
    )
    parser.add_argument(
        '--kkt-tolerance',
        type=float,
        default=1e-6,
        help='Tolerance for KKT conditions (default: 1e-6)'
    )
    parser.add_argument(
        '--kkt-export',
        action='store_true',
        help='Export KKT analysis to CSV files'
    )
    return parser.parse_args()


def safe_get_input(prompt, input_type=str, validation=None, error_msg="Invalid input. Try again."):
    while True:
        try:
            user_input = input(prompt)
            if input_type != str:
                user_input = input_type(user_input)
            if validation and not validation(user_input):
                print(error_msg)
                continue
            return user_input
        except ValueError:
            print(f"  Invalid input. Expected {input_type.__name__}. Try again.")


def validate_functions(func_str, constraint_str, namespace, num_vars=2):
    test_vals = [0.5] * num_vars
    try:
        for i, val in enumerate(test_vals):
            namespace[f'x{i+1}'] = val
        result = eval(func_str, {"__builtins__": {}}, namespace)
        if not isinstance(result, (int, float)):
            print(f"   Function returned {type(result)} instead of number")
            return False
        parsed_constraint, _ = parse_constraint(constraint_str)
        result = eval(parsed_constraint, {"__builtins__": {}}, namespace)
        if not isinstance(result, (int, float)):
            print(f"   Constraint returned {type(result)} instead of number")
            return False
        return True
    except Exception as e:
        print(f"   Function validation failed: {e}")
        return False


def safe_gradient_flow_adam(func_str, constraint_str, start_values, learning_rate, alpha, 
                             max_iterations, noise_amount=0, noise_freq=10,
                             beta1=0.9, beta2=0.999, epsilon=1e-8, verbose=True, quiet=False):
    """
    Run safe gradient flow with Adam adaptive learning rate.
    
    The safe gradient flow is:
    dx/dt = -∇f + (∇g/||∇g||²) * max(0, -α*g + ∇g·∇f)
    
    When constraint is violated (g < 0), a strong correction pushes toward the feasible region.
    """
    
    num_vars = len(start_values)
    var_names = [f'x{i+1}' for i in range(num_vars)]
    
    parsed_constraint, constraint_display = parse_constraint(constraint_str)
    
    namespace = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'exp': math.exp, 'log': math.log, 'log10': math.log10,
        'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e,
        'abs': abs, 'min': min, 'max': max
    }
    
    for i, val in enumerate(start_values):
        namespace[f'x{i+1}'] = val
    
    if not validate_functions(func_str, constraint_str, namespace, num_vars):
        print("Function validation failed. Please check your syntax.")
        return None, None, None, None, None, None, None, None
    
    def func(vals):
        try:
            for i, val in enumerate(vals):
                namespace[f'x{i+1}'] = val
            val = eval(func_str, {"__builtins__": {}}, namespace)
            if np.isinf(val) or np.isnan(val) or abs(val) > 1e100:
                return float('inf')
            return val
        except:
            return float('inf')
    
    # Parse the constraint string to get individual constraints for gradient computation
    def get_individual_constraints(constraint_str):
        """Parse constraint string and return list of individual constraint functions."""
        if ' and ' in constraint_str.lower():
            parts = re.split(r'\s+and\s+', constraint_str, flags=re.IGNORECASE)
            constraints = []
            for part in parts:
                parsed, _ = parse_single_constraint(part)
                constraints.append(parsed)
            return constraints
        else:
            parsed, _ = parse_single_constraint(constraint_str)
            return [parsed]
    
    # Get individual constraints
    individual_constraints = get_individual_constraints(constraint_str)
    
    def constraint(vals):
        """Combined constraint function."""
        try:
            for i, val in enumerate(vals):
                namespace[f'x{i+1}'] = val
            val = eval(parsed_constraint, {"__builtins__": {}}, namespace)
            if np.isinf(val) or np.isnan(val) or abs(val) > 1e100:
                return -float('inf')
            return val
        except:
            return -float('inf')
    
    def compute_gradients(vals, h=1e-7):
        f_current = func(vals)
        g_current = constraint(vals)
        
        grad_f = []
        grad_g = []
        
        # Compute gradient of f
        for i in range(len(vals)):
            vals_plus = vals.copy()
            vals_plus[i] += h
            f_plus = func(vals_plus)
            partial_f = (f_plus - f_current) / h
            if np.isinf(partial_f) or np.isnan(partial_f):
                partial_f = 1e6 if f_plus > f_current else -1e6
            if abs(partial_f) > 1e6:
                partial_f = math.copysign(1e6, partial_f)
            grad_f.append(partial_f)
        
        # Compute gradient of active constraint (the one with smallest g value)
        if len(individual_constraints) > 1:
            min_g = float('inf')
            active_idx = 0
            for idx, constr in enumerate(individual_constraints):
                try:
                    for i, val in enumerate(vals):
                        namespace[f'x{i+1}'] = val
                    g_val = eval(constr, {"__builtins__": {}}, namespace)
                    if g_val < min_g:
                        min_g = g_val
                        active_idx = idx
                except:
                    pass
            active_constr = individual_constraints[active_idx]
        else:
            active_constr = individual_constraints[0]
        
        # Compute gradient of active constraint
        for i in range(len(vals)):
            vals_plus = vals.copy()
            vals_plus[i] += h
            try:
                for j, val in enumerate(vals_plus):
                    namespace[f'x{j+1}'] = val
                g_plus = eval(active_constr, {"__builtins__": {}}, namespace)
                
                for j, val in enumerate(vals):
                    namespace[f'x{j+1}'] = val
                g_current_active = eval(active_constr, {"__builtins__": {}}, namespace)
                
                partial_g = (g_plus - g_current_active) / h
            except:
                partial_g = 0.0
            
            if np.isinf(partial_g) or np.isnan(partial_g):
                partial_g = 1e6 if g_plus > g_current_active else -1e6
            if abs(partial_g) > 1e6:
                partial_g = math.copysign(1e6, partial_g)
            grad_g.append(partial_g)
        
        return grad_f, grad_g, f_current, g_current
    
    m = [0.0] * num_vars
    v = [0.0] * num_vars
    t = 0
    
    x = start_values.copy()
    history = [x.copy()]
    f_history = [func(x)]
    g_history = [constraint(x)]
    
    # Track constraint violations
    violation_count = 0
    max_violation = 0.0
    violation_history = [0]
    
    if g_history[0] < 0:
        violation_count += 1
        max_violation = abs(g_history[0])
        if not quiet:
            print(f"Warning: Start point violates constraint! g(x) = {g_history[0]:.6f}")
            print("The algorithm will prioritize constraint satisfaction.")
    
    if verbose and not quiet:
        print(f"\nSafe Gradient Flow with Adam")
        print(f"f(x) = {func_str}")
        print(f"Constraint: {constraint_display}")
        print(f"Start: {', '.join([f'{v:.6f}' for v in start_values])}")
        print(f"Initial Learning Rate: {learning_rate:.6f}")
        print(f"Alpha: {alpha:.6f}")
        print(f"Adam beta1: {beta1:.4f}, beta2: {beta2:.4f}")
        if noise_amount > 0:
            print(f"Noise: {noise_amount} (every {noise_freq} iterations)")
        print("-" * 70)
        print(f"{'Iter':<6} | {'x1':<12} | {'x2':<12} | {'f(x)':<14} | {'g(x)':<14} | {'LR':<10}")
        print("-" * 70)
    
    current_lr = learning_rate
    lr_reductions = 0
    max_lr_reductions = 10
    
    prev_f_vals = []
    prev_g_vals = []
    
    for i in range(max_iterations):
        t += 1
        
        grad_f, grad_g, f_val, g_val = compute_gradients(x)
        grad_g_mag = math.sqrt(sum(g**2 for g in grad_g))
        
        # ============= SAFE GRADIENT FLOW UPDATE =============
        # The safe gradient flow is:
        # dx/dt = -∇f + (∇g/||∇g||²) * max(0, -α*g + ∇g·∇f)
        #
        # This naturally keeps the solution on the constraint boundary
        # by adding a correction term when needed
        
        # Start with descent direction
        descent = [-grad_f[j] for j in range(num_vars)]
        
        if grad_g_mag > 1e-10:
            # Compute the dot product between ∇f and ∇g
            dot_product = sum(grad_f[j] * grad_g[j] for j in range(num_vars))
            
            # The correction term: max(0, -α*g + ∇g·∇f) / ||∇g||²
            # This pushes toward the feasible region when constraint is active
            scalar = -alpha * g_val + dot_product
            
            # If scalar > 0, we need to add a correction
            if scalar > 0:
                # Normalize the correction
                correction = [(grad_g[j] / (grad_g_mag ** 2)) * scalar for j in range(num_vars)]
                update = [descent[j] + correction[j] for j in range(num_vars)]
            else:
                update = descent
        else:
            # If gradient of constraint is zero, just use descent
            update = descent
        
        # ============= CONSTRAINED CORRECTION (Additional Safety) =============
        # If constraint is violated, apply a stronger correction
        # This ensures we never stay in the infeasible region
        if g_val < -0.001:
            violation_count += 1
            if abs(g_val) > max_violation:
                max_violation = abs(g_val)
            
            if grad_g_mag > 1e-10:
                # Move directly toward the feasible region
                norm_grad_g = [grad_g[j] / grad_g_mag for j in range(num_vars)]
                step_size = min(0.5, -g_val * 0.5)
                
                # Add the correction to the update
                correction = [step_size * norm_grad_g[j] for j in range(num_vars)]
                
                # Blend correction with descent
                blend = min(1.0, abs(g_val) * 2.0)  # More correction when more violated
                for j in range(num_vars):
                    update[j] = (1 - blend) * update[j] + blend * correction[j]
                
                if verbose and not quiet and i % 10 == 0:
                    print(f"        Constraint correction: g={g_val:.4f}, step={step_size:.4f}")
        
        # ============= CONSTRAINT PROJECTION (Hard Enforcement) =============
        # After the update, if constraint is still violated, project back
        # This is a hard enforcement mechanism
        
        # Apply the update
        for j in range(num_vars):
            x[j] += current_lr * update[j]
        
        # Project back to feasible region if violated
        g_new = constraint(x)
        if g_new < -0.001:
            # Use Newton-like projection
            _, grad_g_proj, _, _ = compute_gradients(x)
            grad_g_mag_proj = math.sqrt(sum(g**2 for g in grad_g_proj))
            if grad_g_mag_proj > 1e-10:
                norm_grad_g = [grad_g_proj[j] / grad_g_mag_proj for j in range(num_vars)]
                # Step size to bring g back to 0
                step_size = max(0.01, -g_new * 0.8)
                for j in range(num_vars):
                    x[j] += step_size * norm_grad_g[j]
        
        # ============= ADAM UPDATE (Only if not in violation correction mode) =============
        # If we're not in the violation correction mode, use Adam
        # Note: We already applied the update above, so we need to use the correct update
        
        # Recompute for Adam
        grad_f, grad_g, _, g_val = compute_gradients(x)
        grad_g_mag = math.sqrt(sum(g**2 for g in grad_g))
        
        # Safe gradient flow update for Adam
        descent = [-grad_f[j] for j in range(num_vars)]
        if grad_g_mag > 1e-10:
            dot_product = sum(grad_f[j] * grad_g[j] for j in range(num_vars))
            scalar = -alpha * g_val + dot_product
            if scalar > 0:
                correction = [(grad_g[j] / (grad_g_mag ** 2)) * scalar for j in range(num_vars)]
                update_adam = [descent[j] + correction[j] for j in range(num_vars)]
            else:
                update_adam = descent
        else:
            update_adam = descent
        
        # Add noise if enabled
        if noise_amount > 0 and i > 0 and i % noise_freq == 0:
            noise = random.uniform(-noise_amount, noise_amount)
            current_lr = max(0.0000001, learning_rate + noise * learning_rate)
            if verbose and not quiet:
                print(f"        Noise injected: LR {learning_rate:.6f} -> {current_lr:.6f}")
        
        # Adam update
        for j in range(num_vars):
            m[j] = beta1 * m[j] + (1 - beta1) * update_adam[j]
            v[j] = beta2 * v[j] + (1 - beta2) * (update_adam[j] ** 2)
            
            m_hat = m[j] / (1 - beta1 ** t)
            v_hat = v[j] / (1 - beta2 ** t)
            
            step = current_lr * m_hat / (math.sqrt(v_hat) + epsilon)
            
            # Clamp step to prevent instability
            max_step = 0.1
            if abs(step) > max_step:
                step = math.copysign(max_step, step)
            
            x[j] = x[j] + step
        
        # Project back to feasible region after Adam update
        g_new = constraint(x)
        if g_new < -0.001:
            _, grad_g_proj, _, _ = compute_gradients(x)
            grad_g_mag_proj = math.sqrt(sum(g**2 for g in grad_g_proj))
            if grad_g_mag_proj > 1e-10:
                norm_grad_g = [grad_g_proj[j] / grad_g_mag_proj for j in range(num_vars)]
                step_size = max(0.01, -g_new * 0.8)
                for j in range(num_vars):
                    x[j] += step_size * norm_grad_g[j]
        
        f_new = func(x)
        g_new = constraint(x)
        
        # Track violation in history
        violation_history.append(1 if g_new < -0.001 else 0)
        
        # Oscillation detection
        prev_f_vals.append(f_new)
        if len(prev_f_vals) > 10:
            prev_f_vals.pop(0)
        
        prev_g_vals.append(g_new)
        if len(prev_g_vals) > 10:
            prev_g_vals.pop(0)
        
        oscillating = False
        if len(prev_f_vals) >= 10:
            diffs = [prev_f_vals[i+1] - prev_f_vals[i] for i in range(len(prev_f_vals)-1)]
            sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
            if sign_changes >= 3:
                oscillating = True
        
        if len(prev_g_vals) >= 10:
            crosses_zero = sum(1 for i in range(len(prev_g_vals)-1) if prev_g_vals[i] * prev_g_vals[i+1] < 0)
            if crosses_zero >= 4:
                oscillating = True
        
        if oscillating and lr_reductions < max_lr_reductions:
            current_lr = current_lr * 0.5
            lr_reductions += 1
            if verbose and not quiet:
                print(f"        Oscillation detected! Reducing LR to {current_lr:.8f}")
            prev_f_vals = []
            prev_g_vals = []
        
        if verbose and not quiet:
            if abs(f_new) < 0.001 or abs(f_new) > 1000:
                f_str = f"{f_new:<14.6e}"
            else:
                f_str = f"{f_new:<14.8f}"
            if abs(g_new) < 0.001 or abs(g_new) > 1000:
                g_str = f"{g_new:<14.6e}"
            else:
                g_str = f"{g_new:<14.8f}"
            row = f"{i:<6} | {x[0]:<12.6f} | {x[1]:<12.6f} | {f_str} | {g_str} | {current_lr:<10.8f}"
            print(row)
        
        grad_magnitude = math.sqrt(sum(update_adam[j] ** 2 for j in range(num_vars)))
        if grad_magnitude < 1e-8 and abs(g_new) < 1e-6:
            if verbose and not quiet:
                print("-" * 70)
                print(f"    Converged after {i+1} iterations! (gradient magnitude = {grad_magnitude:.2e})")
            break
        
        history.append(x.copy())
        f_history.append(f_new)
        g_history.append(g_new)
        
        if any(abs(val) > 1e10 for val in x):
            if verbose and not quiet:
                print("-" * 70)
                print("      WARNING: Values are exploding! Try a smaller learning rate.")
            break
    
    final_f = func(x)
    final_g = constraint(x)
    
    if verbose and not quiet:
        print("-" * 70)
        print(f"FINAL RESULT:")
        for j, name in enumerate(var_names):
            print(f"  {name} = {x[j]:.10f}")
        print(f"  f(x) = {final_f:.16f}")
        print(f"  g(x) = {final_g:.16f} {'(SAFE)' if final_g >= 0 else '(VIOLATED)'}")
        if violation_count > 0:
            print(f"  Constraint violations: {violation_count}")
            print(f"  Maximum violation: {max_violation:.6f}")
            violation_percent = (violation_count / len(history)) * 100
            print(f"  Violation percentage: {violation_percent:.1f}%")
        else:
            print(f"  Constraint violations: 0 (Perfectly safe!)")
        if lr_reductions > 0:
            print(f"  Learning rate reductions: {lr_reductions}")
        print("-" * 70)
    
    return history, f_history, g_history, x, final_f, final_g, constraint_display, violation_history


def plot_convergence(history, f_history, g_history, var_names, title=None, alpha_val=None, violation_history=None):
    iterations = list(range(len(history)))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for i, name in enumerate(var_names):
        values = [point[i] for point in history]
        color = colors[i % len(colors)]
        axes[0].plot(iterations, values, 'o-', color=color, linewidth=2, 
                    markersize=3, label=name)
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Variable Value', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title('Variable Convergence', fontsize=14)
    
    axes[1].plot(iterations, f_history, 'b-', linewidth=2)
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('f(x)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Objective Function Value', fontsize=14)
    if f_history and max(f_history) > 0:
        axes[1].set_yscale('log')
    
    axes[2].plot(iterations, g_history, 'g-', linewidth=2)
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Safety boundary g(x)=0')
    if g_history and max(g_history) > 0:
        axes[2].fill_between(iterations, 0, max(g_history), 
                             color='green', alpha=0.1, label='Safe region')
    axes[2].set_xlabel('Iteration', fontsize=12)
    axes[2].set_ylabel('g(x)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    # Plot violation markers on the constraint plot
    if violation_history:
        violation_iters = [i for i, v in enumerate(violation_history) if v > 0]
        if violation_iters:
            # Get g values at violation points
            violation_vals = [g_history[i] if i < len(g_history) else 0 for i in violation_iters]
            axes[2].scatter(violation_iters, violation_vals, color='red', s=40, 
                          marker='x', zorder=5, label='Constraint violation')
    
    axes[2].legend()
    axes[2].set_title('Constraint Function (g(x) >= 0 for safety)', fontsize=14)
    
    if title:
        display_title = title
    else:
        display_title = 'Safe Gradient Flow Convergence'
    
    if alpha_val is not None:
        display_title += f' (Alpha = {alpha_val:.4f})'
    
    fig.suptitle(display_title, fontsize=16)
    plt.tight_layout()
    return fig, axes


def export_trajectory_data(history, f_history, g_history, var_names, constraint_display, prefix="trajectory"):
    """
    Export trajectory data to CSV files for external analysis.
    
    Parameters:
    - history: List of variable value lists from gradient descent
    - f_history: History of objective values
    - g_history: History of constraint values
    - var_names: List of variable names
    - constraint_display: Display string for the constraint
    - prefix: Prefix for output filenames
    """
    import csv
    
    # Export variables history
    var_filename = f"{prefix}_variables.csv"
    with open(var_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ['Iteration'] + var_names
        writer.writerow(header)
        # Data
        for i, vals in enumerate(history):
            row = [i] + vals
            writer.writerow(row)
    print(f"    Exported variable history to: {var_filename}")
    
    # Export objective and constraint history
    func_filename = f"{prefix}_functions.csv"
    with open(func_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['Iteration', 'f(x)', 'g(x)'])
        # Data
        for i in range(len(f_history)):
            writer.writerow([i, f_history[i], g_history[i] if i < len(g_history) else ''])
    print(f"    Exported function history to: {func_filename}")
    
    # Export metadata
    meta_filename = f"{prefix}_metadata.txt"
    with open(meta_filename, 'w') as f:
        f.write(f"Objective function: {', '.join(var_names)}\n")
        f.write(f"Constraint: {constraint_display}\n")
        f.write(f"Number of iterations: {len(history)}\n")
        f.write(f"Final f(x): {f_history[-1] if f_history else 'N/A'}\n")
        f.write(f"Final g(x): {g_history[-1] if g_history else 'N/A'}\n")
    print(f"    Exported metadata to: {meta_filename}")


def kkt_analysis_main(args):
    """Run KKT analysis on optimization results (prints results instead of showing graphs)."""
    try:
        from kkt_analysis import kkt_visualization, export_kkt_data, check_kkt_conditions, print_kkt_results
        KKT_AVAILABLE = True
    except ImportError:
        KKT_AVAILABLE = False
        print("Warning: kkt_analysis module not found. KKT analysis disabled.")
        return
    
    if args.start:
        start_values = [float(x.strip()) for x in args.start.split(',')]
    else:
        print("Error: Starting values required. Use -s")
        return
    
    # Get constraint display
    _, constraint_display = parse_constraint(args.constraint)
    
    # Set default learning rate and alpha if not provided
    if args.learning_rate is None or args.alpha is None:
        suggested_lr, suggested_alpha, reason = suggest_parameters(
            args.function, args.constraint, start_values
        )
        if args.learning_rate is None:
            learning_rate = suggested_lr
        else:
            learning_rate = args.learning_rate
        if args.alpha is None:
            alpha = suggested_alpha
        else:
            alpha = args.alpha
    else:
        learning_rate = args.learning_rate
        alpha = args.alpha
    
    # Run optimization
    print("\n" + "=" * 70)
    print("RUNNING SAFE GRADIENT FLOW OPTIMIZATION")
    print("=" * 70)
    
    history, f_history, g_history, final_x, final_f, final_g, constraint_display, violation_history = safe_gradient_flow_adam(
        func_str=args.function,
        constraint_str=args.constraint,
        start_values=start_values,
        learning_rate=learning_rate,
        alpha=alpha,
        max_iterations=args.iterations,
        noise_amount=args.noise or 0,
        noise_freq=args.noise_freq,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        epsilon=args.adam_epsilon,
        verbose=True,
        quiet=args.quiet if hasattr(args, 'quiet') else False
    )
    
    if history is None:
        print("Optimization failed.")
        return
    
    # Check KKT conditions
    namespace = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'exp': math.exp, 'log': math.log, 'log10': math.log10,
        'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e,
        'abs': abs, 'min': min, 'max': max
    }
    for i, val in enumerate(start_values):
        namespace[f'x{i+1}'] = val
    
    kkt_check = check_kkt_conditions(final_x, args.function, args.constraint, namespace, args.kkt_tolerance)
    
    var_names = [f'x{i+1}' for i in range(len(start_values))]
    kkt_info = {
        'is_kkt_satisfied': kkt_check['is_kkt_satisfied'],
        'kkt_violation': kkt_check['kkt_violation'],
        'details': kkt_check['details'],
        'tolerance': args.kkt_tolerance,
        'final_f': final_f,
        'final_g': final_g,
        'final_x': final_x
    }
    
    # Print KKT results
    print_kkt_results(kkt_info, var_names, args.function, args.constraint, final_x)
    
    # Export KKT data
    if args.kkt_export and args.save:
        export_kkt_data(kkt_info, var_names, args.function, args.constraint, final_x, final_f, args.save)
    
    # Generate standard visualizations (3D, contour, convergence) if not in no_plots mode
    if not args.no_plots:
        # Generate 3D visualization
        all_x1 = [p[0] for p in history]
        all_x2 = [p[1] for p in history]
        x1_range = (min(all_x1) - 0.5, max(all_x1) + 0.5)
        x2_range = (min(all_x2) - 0.5, max(all_x2) + 0.5)
        x1_range = (min(x1_range[0], -1), max(x1_range[1], 1))
        x2_range = (min(x2_range[0], -1), max(x2_range[1], 1))
        
        print("\n[1/3] Generating 3D visualization...")
        ls = LossSurface(args.function, args.constraint, x1_range, x2_range, alpha=alpha)
        
        fig_3d, ax_3d = ls.plot_3d(
            trajectories=[history],
            best_trajectory=history,
            title=f'Safe Gradient Flow\nf: {args.function}\n{constraint_display}',
            alpha_val=alpha
        )
        
        if args.save:
            fig_3d.savefig(f'{args.save}_3d.png', dpi=300, bbox_inches='tight')
            print(f"    Saved: {args.save}_3d.png")
        else:
            plt.figure(fig_3d.number)
            plt.show()
            plt.pause(0.1)
        
        print("\n[2/3] Generating 2D contour visualization...")
        fig_contour, ax_contour = ls.plot_contour(
            trajectories=[history],
            best_trajectory=history,
            title=f'Safe Gradient Flow - 2D View\nf: {args.function}\n{constraint_display}',
            alpha_val=alpha
        )
        
        if args.save:
            fig_contour.savefig(f'{args.save}_contour.png', dpi=300, bbox_inches='tight')
            print(f"    Saved: {args.save}_contour.png")
        else:
            plt.figure(fig_contour.number)
            plt.show()
            plt.pause(0.1)
        
        print("\n[3/3] Generating convergence plots...")
        fig_conv, axes_conv = plot_convergence(
            history=history,
            f_history=f_history,
            g_history=g_history,
            var_names=var_names,
            title=f'Safe Gradient Flow Convergence\nf: {args.function}',
            alpha_val=alpha,
            violation_history=violation_history
        )
        
        if args.save:
            fig_conv.savefig(f'{args.save}_convergence.png', dpi=300, bbox_inches='tight')
            print(f"    Saved: {args.save}_convergence.png")
        else:
            plt.figure(fig_conv.number)
            plt.show()
            plt.pause(0.1)
        
        # Keep all plots open at the end
        print("\nAll visualizations complete. Close the plot windows to exit.")
        plt.show(block=True)


def main():
    args = parse_arguments()
    
    # Check for KKT analysis mode
    if hasattr(args, 'kkt') and args.kkt:
        kkt_analysis_main(args)
        return
    
    # ... rest of your existing main() code ...
    if args.start:
        start_values = [float(x.strip()) for x in args.start.split(',')]
    else:
        print("Error: Starting values required. Use -s")
        return
    
    num_vars = len(start_values)
    
    if num_vars != 2:
        print("Warning: This visualization is optimized for 2D functions.")
        print("The algorithm will still run, but 3D plots will not be generated.")
    
    namespace = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'exp': math.exp, 'log': math.log, 'log10': math.log10,
        'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e,
        'abs': abs, 'min': min, 'max': max
    }
    
    if not validate_functions(args.function, args.constraint, namespace, num_vars):
        print("Function validation failed. Exiting.")
        return
    
    if args.learning_rate is None or args.alpha is None:
        suggested_lr, suggested_alpha, reason = suggest_parameters(
            args.function, args.constraint, start_values
        )
        
        if args.learning_rate is None:
            learning_rate = suggested_lr
            if not args.quiet:
                print(f"Auto-selected learning rate: {learning_rate:.6f}")
        else:
            learning_rate = args.learning_rate
        
        if args.alpha is None:
            alpha = suggested_alpha
            if not args.quiet:
                print(f"Auto-selected alpha: {alpha:.4f}")
        else:
            alpha = args.alpha
        
        if not args.quiet:
            print(f"Reason: {reason}")
            print("-" * 70)
    else:
        learning_rate = args.learning_rate
        alpha = args.alpha
    
    if learning_rate > 0.01 and not args.quiet:
        print(f"Warning: Learning rate ({learning_rate}) is high. Consider using -lr 0.001 for stability.")
    
    if alpha > 2.0 and not args.quiet:
        print(f"Warning: Alpha ({alpha}) is high. Consider using -a 1.0 for stability.")
    
    multi_start = args.multi is not None and args.multi > 1
    num_starts = args.multi if multi_start else 1
    
    range_parts = args.range.split(',')
    min_val = float(range_parts[0].strip())
    max_val = float(range_parts[1].strip())
    
    all_trajectories = []
    all_f_histories = []
    all_g_histories = []
    all_violation_histories = []  # Initialize this variable
    best_trajectory = None
    best_f = float('inf')
    best_x = None
    best_g = None
    best_f_history = None
    best_g_history = None
    best_violation_history = None
    best_run_idx = 0
    constraint_display = args.constraint
    
    if multi_start:
        if not args.quiet:
            print(f"\nMulti-start mode: Running {num_starts} starts...")
            print(f"Random start range: [{min_val}, {max_val}]")
            print(f"Alpha: {alpha:.6f}")
            print(f"Constraint: {args.constraint}")
            print("-" * 70)
        
        for run in range(num_starts):
            current_start = [random.uniform(min_val, max_val) for _ in range(num_vars)]
            
            if not args.quiet:
                print(f"\n--- Run {run + 1}/{num_starts} ---")
                print(f"Start: {current_start}")
            
            history, f_history, g_history, final_x, final_f, final_g, constraint_display, violation_history = safe_gradient_flow_adam(
                func_str=args.function,
                constraint_str=args.constraint,
                start_values=current_start,
                learning_rate=learning_rate,
                alpha=alpha,
                max_iterations=args.iterations,
                noise_amount=args.noise or 0,
                noise_freq=args.noise_freq,
                beta1=args.adam_beta1,
                beta2=args.adam_beta2,
                epsilon=args.adam_epsilon,
                verbose=False,
                quiet=args.quiet
            )
            
            if history is not None:
                all_trajectories.append(history)
                all_f_histories.append(f_history)
                all_g_histories.append(g_history)
                all_violation_histories.append(violation_history)
                
                # Print violation info for this run
                violation_count = sum(1 for v in violation_history if v > 0)
                max_viol = max(abs(v) for v in g_history if v < 0) if any(v < 0 for v in g_history) else 0
                if not args.quiet:
                    print(f"  Final: f(x) = {final_f:.6f}, g(x) = {final_g:.6f}")
                    print(f"  Violations: {violation_count} (max: {max_viol:.6f})")
                
                if final_f < best_f and final_g >= 0:
                    best_f = final_f
                    best_trajectory = history
                    best_x = final_x
                    best_g = final_g
                    best_f_history = f_history
                    best_g_history = g_history
                    best_violation_history = violation_history
                    best_run_idx = run
                    if not args.quiet:
                        print(f"    NEW BEST: f(x) = {best_f:.6f}")
        
        if not args.quiet:
            print("\n" + "=" * 70)
            print("Multi-start Summary")
            print("=" * 70)
            print(f"Best f(x): {best_f:.10f}")
            print(f"Best x: {best_x}")
            print(f"Best run: {best_run_idx + 1}/{num_starts}")
            if best_violation_history:
                best_violations = sum(1 for v in best_violation_history if v > 0)
                print(f"Best run violations: {best_violations}")
            print(f"Runs completed: {num_starts}")
            print(f"Alpha: {alpha:.6f}")
            print(f"Constraint: {args.constraint}")
            print("=" * 70)
        
        if best_trajectory is None and all_trajectories:
            if not args.quiet:
                print("Warning: No run satisfied the constraint. Using best available.")
            best_trajectory = all_trajectories[0]
            best_x = all_trajectories[0][-1]
            best_f_history = all_f_histories[0]
            best_g_history = all_g_histories[0]
            best_violation_history = all_violation_histories[0]
    
    else:
        history, f_history, g_history, final_x, final_f, final_g, constraint_display, violation_history = safe_gradient_flow_adam(
            func_str=args.function,
            constraint_str=args.constraint,
            start_values=start_values,
            learning_rate=learning_rate,
            alpha=alpha,
            max_iterations=args.iterations,
            noise_amount=args.noise or 0,
            noise_freq=args.noise_freq,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.adam_epsilon,
            verbose=True,
            quiet=args.quiet
        )
        
        if history is None:
            return
        
        all_trajectories = [history]
        all_violation_histories = [violation_history]
        best_trajectory = history
        best_f = final_f
        best_x = final_x
        best_g = final_g
        best_f_history = f_history
        best_g_history = g_history
        best_violation_history = violation_history
    
    # Define var_names here for use in plots and export
    var_names = [f'x{i+1}' for i in range(num_vars)]
    
    if not args.no_plots:
        if num_vars == 2:
            all_x1 = []
            all_x2 = []
            for traj in all_trajectories:
                for point in traj:
                    all_x1.append(point[0])
                    all_x2.append(point[1])
            
            if all_x1 and all_x2:
                x1_range = (min(all_x1) - 0.5, max(all_x1) + 0.5)
                x2_range = (min(all_x2) - 0.5, max(all_x2) + 0.5)
                x1_range = (min(x1_range[0], -1), max(x1_range[1], 1))
                x2_range = (min(x2_range[0], -1), max(x2_range[1], 1))
            else:
                x1_range = (-3, 3)
                x2_range = (-3, 3)
            
            if not args.quiet:
                print("\n[1/3] Generating 3D visualization...")
            ls = LossSurface(args.function, args.constraint, x1_range, x2_range, alpha=alpha)
            
            fig_3d, ax_3d = ls.plot_3d(
                trajectories=all_trajectories if multi_start else None,
                best_trajectory=best_trajectory,
                title=f'Safe Gradient Flow\nf: {args.function}\n{constraint_display}',
                alpha_val=alpha
            )
            
            if args.save:
                fig_3d.savefig(f'{args.save}_3d.png', dpi=300, bbox_inches='tight')
                if not args.quiet:
                    print(f"    Saved: {args.save}_3d.png")
            else:
                plt.show()
            
            if not args.quiet:
                print("\n[2/3] Generating 2D contour visualization...")
            fig_contour, ax_contour = ls.plot_contour(
                trajectories=all_trajectories if multi_start else None,
                best_trajectory=best_trajectory,
                title=f'Safe Gradient Flow - 2D View\nf: {args.function}\n{constraint_display}',
                alpha_val=alpha
            )
            
            if args.save:
                fig_contour.savefig(f'{args.save}_contour.png', dpi=300, bbox_inches='tight')
                if not args.quiet:
                    print(f"    Saved: {args.save}_contour.png")
            else:
                plt.show()
        
        if not args.quiet:
            print("\n[3/3] Generating convergence plots...")
        fig_conv, axes_conv = plot_convergence(
            history=best_trajectory if best_trajectory else history,
            f_history=best_f_history if best_f_history else f_history,
            g_history=best_g_history if best_g_history else g_history,
            var_names=var_names,
            title=f'Safe Gradient Flow Convergence\nf: {args.function}',
            alpha_val=alpha,
            violation_history=best_violation_history
        )
        
        if args.save:
            fig_conv.savefig(f'{args.save}_convergence.png', dpi=300, bbox_inches='tight')
            if not args.quiet:
                print(f"    Saved: {args.save}_convergence.png")
        else:
            plt.show()
    
    # Export trajectory data
    if not args.quiet:
        export_choice = input("\nWould you like to export trajectory data to CSV? (y/n, default = n): ").lower()
        if export_choice == 'y':
            prefix = args.save if args.save else "trajectory"
            export_trajectory_data(
                history=best_trajectory if best_trajectory else history,
                f_history=best_f_history if best_f_history else f_history,
                g_history=best_g_history if best_g_history else g_history,
                var_names=var_names,
                constraint_display=constraint_display,
                prefix=prefix
            )
    else:
        # In quiet mode, automatically export if --save is provided
        if args.save:
            prefix = args.save
            export_trajectory_data(
                history=best_trajectory if best_trajectory else history,
                f_history=best_f_history if best_f_history else f_history,
                g_history=best_g_history if best_g_history else g_history,
                var_names=var_names,
                constraint_display=constraint_display,
                prefix=prefix
            )
    
    return best_x, best_f, best_g


if __name__ == "__main__":
    try:
        import numpy
        import matplotlib
        import argparse
    except ImportError:
        print("\n      Required libraries not installed.")
        print("   Please install them with:")
        print("   pip install numpy matplotlib")
        print("-" * 70)
    
    main()
