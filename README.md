# Safe Gradient Flow with Adam Adaptive Learning Rate

<div style="display: flex; overflow-x: auto; gap: 10px; padding: 10px 0;">
  <img width="1920" height="1200" alt="ezgif com-gif-maker" src="https://github.com/user-attachments/assets/60deab62-73b0-4685-ae3c-ef49112951e6" />
</div>

A Python program that implements safe gradient flow with hard constraint enforcement using Adam adaptive learning rate, 3D visualization, multi-start optimization, and CSV data export.

## What is Safe Gradient Flow?

Safe gradient flow is an optimization algorithm that finds the minimum of a function while **guaranteeing** that constraints are never violated. Unlike standard gradient descent that may leave the feasible region, safe gradient flow actively enforces constraints by modifying the descent direction.

The update rule is:

```
dx/dt = -∇f(x) + (∇g(x)/||∇g(x)||²) * max(0, -α*g(x) + ∇g(x)·∇f(x))
```

Where:
- `f(x)` is the objective function to minimize
- `g(x)` is the constraint function (requires `g(x) >= 0`)
- `α` is the safety parameter (higher = stronger push to safety)
- `∇f` and `∇g` are the gradients

## What This Program Does

This tool helps you understand how safe gradient flow works by:

- Running safe gradient flow on any function with constraints
- Automatically suggesting learning rate and alpha values based on function structure
- Showing each iteration step-by-step with variable values (with optional quiet mode)
- Visualizing the safe flow path in 3D and 2D
- Plotting convergence of variables, objective, and constraint
- Supporting multi-start optimization with noise injection
- Handling multiple constraints with `and` operator
- **Exporting trajectory data to CSV files** for external analysis
- **Running in headless mode** without displaying plots
- **Quiet mode** to suppress iteration-by-iteration output

## Features

### Hard Constraint Enforcement

The algorithm enforces constraints **strictly**:
- When `g(x) < 0` (constraint violated), the algorithm moves directly toward the feasible region
- When `g(x) >= 0` (safe), the algorithm descends the objective while staying on the boundary
- Multiple constraints are handled using the `min()` function
- **Constraint violation tracking**: The algorithm tracks and reports constraint violations including count and maximum violation magnitude

### Adam Adaptive Learning Rate

The algorithm uses the Adam optimizer with adaptive learning rates:
- Automatically adjusts learning rates per parameter
- Handles sparse gradients and noisy objectives
- **Oscillation detection**: Automatically reduces learning rate when oscillations are detected
- Learning rate clipping prevents exploding gradients

### Local Minima Escape Strategies

**1. Multi-start**: Runs safe gradient flow from multiple random starting points and keeps the best result.

**2. Noise Injection**: Adds random "jumps" to the learning rate at specified intervals to escape local minima.

**3. Combined**: Use both strategies together for maximum effectiveness.

### Data Export

**CSV Export**: Export trajectory data for external analysis:
- `*_variables.csv`: Iteration history of all variables
- `*_functions.csv`: f(x) and g(x) values at each iteration
- `*_metadata.txt`: Metadata including final results and parameters

### Plot Management

- **3D Surface Plot**: Shows objective surface, constraint boundary (red), and safe flow path
- **2D Contour Plot**: Shows objective contours, safe region (light green), and path
- **Convergence Plots**: Shows variables, objective value, and constraint over iterations
- **Constraint Violation Markers**: Red 'X' markers on convergence plots showing where constraints were violated
- **Automatic Plot Saving**: Save all plots as PNG files using `--save` flag
- **Headless Mode**: Run without displaying plots using `--no-plots` flag

### Output Control

- **Quiet Mode**: Suppress iteration-by-iteration output with `--quiet` flag
- **Verbose Mode**: Full iteration details (default)
- **Final Results**: Always displayed with constraint satisfaction status
- **Violation Statistics**: Count and percentage of constraint violations

### Parameter Auto-Selection

The program intelligently suggests parameters based on:
- Function complexity (quadratic, cubic, exponential, trigonometric)
- Constraint complexity (multiple constraints, high-degree, trigonometric)
- Starting values (adjusts for values far from origin)
- Special cases (circle constraint with quadratic objective)

## Requirements

- Python 3.6 or higher
- numpy
- matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

## How to Run

### Interactive Mode

1. Save the code as `safe_gradient.py`

2. Run it:

```bash
python3 safe_gradient.py
```

3. Follow the prompts:

   - Enter your function using `x1`, `x2`, etc.
   - Enter your constraint using `=>` or `>=`
   - Enter starting values (comma-separated)
   - Choose initial learning rate (or accept the suggestion)
   - Choose alpha value (or accept the suggestion)
   - Set max iterations
   - Optionally export data to CSV

### Command-Line Mode

Run the program with arguments for non-interactive use:

```bash
# Single constraint
python3 safe_gradient.py -f "x1**2 + x2**2" -g "x1 + x2 => 1" -s "0.5,0.5" -i 100

# Multiple constraints (use 'and' to separate)
python3 safe_gradient.py -f "x1**2 + x2**2" -g "x1 + x2 => 1 and x1 - x2 => 0.5" -s "0.8,0.8" -i 200

# Circle constraint with quadratic objective
python3 safe_gradient.py -f "x1**2 + x2**2" -g "x1**2 + x2**2 => 0.25" -s "0.1,0.1" -i 200

# Multi-start with noise and quiet mode
python3 safe_gradient.py -f "sin(x1)*cos(x2) + 0.1*(x1**2 + x2**2)" -g "sin(x1) + cos(x2) => 0.5" --multi 10 --range -2,2 --noise 0.3 --noise_freq 8 --quiet --save my_run

# Complex constraints with automatic export
python3 safe_gradient.py -f "20 + x1**2 + x2**2 - 10*(cos(2*pi*x1) + cos(2*pi*x2))" -g "sin(x1) + cos(x2) => 0.5 and x1**2 + x2**2 => 0.25" -s "0.5,0.5" -i 300 --save complex_run
```

### Command-Line Options

| Flag | Description | Example |
|------|-------------|---------|
| `-f, --function` | Objective function to minimize | `"x1**2 + x2**2"` |
| `-g, --constraint` | Constraint(s) in format `expression => value` | `"x1 + x2 => 1"` |
| `-s, --start` | Starting values (comma-separated) | `"0.5, 0.5"` |
| `-lr, --learning_rate` | Initial learning rate (auto-selected) | `0.01` |
| `-a, --alpha` | Safety parameter (auto-selected) | `1.0` |
| `-i, --iterations` | Max iterations | `200` |
| `--multi` | Number of random starts | `10` |
| `--range` | Range for random starts | `-2,2` |
| `--noise` | Random noise amount | `0.3` |
| `--noise_freq` | Noise injection frequency | `8` |
| `--adam_beta1` | Adam beta1 parameter | `0.9` |
| `--adam_beta2` | Adam beta2 parameter | `0.999` |
| `--adam_epsilon` | Adam epsilon parameter | `1e-8` |
| `--no-plots` | Disable plotting | (flag) |
| `--save` | Save plots and export CSV with prefix | `"my_run"` |
| `--quiet` | Suppress iteration-by-iteration output | (flag) |

## Examples

### Interactive Example

```
Enter your function: x1**2 + x2**2
Enter your constraint: x1**2 + x2**2 => 0.25
Enter starting values: 0.5, 0.5
Use suggested learning rate? y
Use suggested alpha? y
Max iterations: 200

Would you like to export trajectory data to CSV? (y/n): y
```

### Multi-Start with Quiet Mode and Export

```bash
# Find the global minimum of a complex function with 10 starts, quiet mode, and auto-export
python3 safe_gradient.py -f "20 + x1**2 + x2**2 - 10*(cos(2*pi*x1) + cos(2*pi*x2))" -g "x1**2 + x2**2 => 1" --multi 10 --range -3,3 --noise 0.3 --noise_freq 8 --quiet --save my_run
```

### Headless Mode with Custom Adam Parameters

```bash
# Run without displaying plots, with custom Adam parameters
python3 safe_gradient.py -f "x1**2 + 4*x2**2" -g "0.5*x1**2 + x2**2 => 1" -s "1.0,1.0" --no-plots --adam_beta1 0.85 --adam_beta2 0.99 --save headless_run
```

## Constraint Syntax

| Math | Type This |
|------|-----------|
| x₁ + x₂ ≥ 1 | `x1 + x2 => 1` |
| x₁² + x₂² ≥ 1 | `x1**2 + x2**2 => 1` |
| x₁ ≥ 0.5 and x₂ ≥ 0.5 | `x1 => 0.5 and x2 => 0.5` |
| sin(x₁) + cos(x₂) ≥ 0.5 | `sin(x1) + cos(x2) => 0.5` |

## Visualizations

- **3D Plot**: Shows the objective surface, constraint boundary (red), and the safe flow path
- **2D Contour Plot**: Shows the objective contours, safe region (light green), and the path
- **Convergence Plots**: Shows variables, objective value, and constraint over iterations with violation markers

All plots can be saved as PNG files using the `--save` flag.

### Convergence Plot Details

The convergence plot includes three subplots:
1. **Variable Convergence**: Shows how each variable evolves over iterations
2. **Objective Function Value**: Shows f(x) decreasing over time (log scale)
3. **Constraint Function**: Shows g(x) with:
   - Red dashed line at g(x)=0 (safety boundary)
   - Green shaded region for g(x)>0 (safe region)
   - Red 'X' markers for constraint violations

## Adam Parameters

The algorithm uses Adam optimizer with the following parameters:
- **β₁ (beta1)**: Controls momentum (default: 0.9)
- **β₂ (beta2)**: Controls RMS propagation (default: 0.999)
- **ε (epsilon)**: Small constant for numerical stability (default: 1e-8)

## Function Syntax

| Math | Type This |
|------|-----------|
| x₁² | `x1**2` |
| (x₁ - 2)² | `(x1 - 2)**2` |
| sin(x₁) + cos(x₂) | `sin(x1) + cos(x2)` |
| x₁² + x₂² + x₃² | `x1**2 + x2**2 + x3**2` |
| exp(x₁) | `exp(x1)` |
| sqrt(x₁) | `sqrt(x1)` |

## CSV Export Format

When you choose to export data (or use `--save` in quiet mode), three files are created:

### variables.csv
| Iteration | x1 | x2 | ... |
|-----------|----|----|-----|
| 0 | 0.500000 | 0.500000 | ... |
| 1 | 0.499000 | 0.501000 | ... |

### functions.csv
| Iteration | f(x) | g(x) |
|-----------|------|------|
| 0 | 0.500000 | 0.500000 |
| 1 | 0.499000 | 0.501000 |

### metadata.txt
```
Objective function: x1, x2
Constraint: x1**2 + x2**2 => 0.25
Number of iterations: 200
Final f(x): 0.2500000000
Final g(x): 0.0000000000
```

## Tips for Finding Global Minima

1. **Use Multi-start**: For functions with many local minima, run with `--multi 20` or more starts
2. **Add Noise**: Combine multi-start with noise for even better exploration
3. **Wide Range**: Use a larger range like `--range=-10,10` to explore more of the function
4. **Check Plots**: The 3D visualization shows the path, helping you understand the algorithm's behavior
5. **Adjust Alpha**: If the algorithm oscillates, try increasing alpha; if it's too slow, try decreasing it
6. **Adjust Learning Rate**: If the algorithm explodes, reduce the learning rate
7. **Use Quiet Mode**: When running many experiments, use `--quiet` to reduce output clutter
8. **Export Data**: Use CSV export for further analysis in Excel, MATLAB, or other tools
9. **Custom Adam Parameters**: Experiment with different β₁, β₂ values for specific problem types

## Test Functions

| # | Function | Constraint | Difficulty |
|---|----------|------------|------------|
| 1 | x₁² + x₂² | x₁² + x₂² ≥ 0.25 | ★☆☆☆☆ |
| 2 | x₁² + x₂² | x₁ + x₂ ≥ 1 | ★☆☆☆☆ |
| 3 | (x₁-1)² + (x₂-1)² | x₁² + x₂² ≥ 1 | ★★☆☆☆ |
| 4 | x₁² + x₂² | x₁ ≥ 0.5 and x₂ ≥ 0.5 | ★★☆☆☆ |
| 5 | x₁² + 4x₂² | 0.5x₁² + x₂² ≥ 1 | ★★★☆☆ |
| 6 | Rastrigin | x₁² + x₂² ≥ 1 | ★★★☆☆ |
| 7 | Griewank | x₁ + x₂ ≥ 1 and x₁ - x₂ ≥ 0.5 | ★★★☆☆ |
| 8 | Rosenbrock | x₁² + x₂² ≥ 1 | ★★★★☆ |
| 9 | Styblinski-Tang | sin(x₁)cos(x₂) ≥ 0.5 | ★★★★☆ |
| 10 | Ackley | sin(x₁)cos(x₂) ≥ 0.3 | ★★★★★ |

## Notes

- Works with any number of dimensions (2D gets 3D visualization)
- The algorithm automatically detects the number of variables from your input
- Multiple constraints are combined using the `min()` function
- The algorithm enforces constraints strictly using safe gradient flow
- Alpha controls the strength of the safety push (higher = stronger)
- **Constraint violations are tracked** and reported as percentages
- **Oscillation detection** automatically reduces learning rate when needed
- **Adam optimizer** provides adaptive learning rates per parameter
- **All plots can be saved** as PNG files using the `--save` flag
- **CSV export** provides data for external analysis
- **Quiet mode** suppresses iteration output while keeping final results
- **Headless mode** runs without displaying any plots
- **Auto-parameter selection** suggests appropriate values based on problem structure
