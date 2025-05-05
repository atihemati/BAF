"""
Piecewise Linear Fitting Functions

Monotonically decreasing piecewise linear fitting functions, one continuous and one discontinuous.

Created on 02.05.2025
@author: claude.ai 
Prompted by Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling). 
Conversation link: https://claude.ai/share/e7fc33bd-c77e-4cbb-b782-25db93f864bb
"""
#%% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class ConstrainedPiecewiseLinFit:
    def __init__(self, x, y):
        """
        Initialize with x and y data points
        
        Parameters
        ----------
        x : array_like
            The x data
        y : array_like
            The y data
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.n_data = len(x)
        self.break_points = None
        self.slopes = None
        self.intercepts = None
        self.n_segments = None
        self.fit_breaks = None
        self.parameters = None
        
    def fit(self, n_segments, allow_discontinuities=True):
        """
        Fit a monotonically decreasing piecewise linear function
        
        Parameters
        ----------
        n_segments : int
            The number of line segments to fit
        allow_discontinuities : bool
            If True, allow discontinuities at breakpoints but
            still enforce that the function never increases
            
        Returns
        -------
        parameters : array_like
            The fitted parameters
        """
        self.n_segments = n_segments
        n_breakpoints = n_segments - 1
        
        # Initial guess for breakpoints: evenly spaced
        x_min, x_max = min(self.x), max(self.x)
        initial_breakpoints = np.linspace(x_min, x_max, n_breakpoints + 2)[1:-1]
        
        # Define the objective function to minimize (MSE)
        def objective(breakpoints):
            return self._calculate_mse(breakpoints, allow_discontinuities)
        
        # Constraints for breakpoints to be within data range and in order
        bounds = [(x_min, x_max) for _ in range(n_breakpoints)]
        
        # If there are breakpoints, enforce ordering
        constraints = []
        if n_breakpoints > 1:
            for i in range(n_breakpoints - 1):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda bp, i=i: bp[i+1] - bp[i] - 1e-6  # Ensure increasing order
                })
        
        # Try several initial points to improve chances of finding global minimum
        best_result = None
        best_mse = float('inf')
        
        # Try the evenly spaced initialization
        result = minimize(
            objective, 
            initial_breakpoints, 
            method='SLSQP', 
            bounds=bounds,
            constraints=constraints if constraints else None
        )
        
        if result.success and result.fun < best_mse:
            best_result = result
            best_mse = result.fun
        
        # Try random initializations
        for _ in range(3):
            initial_breakpoints = np.sort(np.random.uniform(x_min, x_max, n_breakpoints))
            result = minimize(
                objective, 
                initial_breakpoints, 
                method='SLSQP', 
                bounds=bounds,
                constraints=constraints if constraints else None
            )
            
            if result.success and result.fun < best_mse:
                best_result = result
                best_mse = result.fun
        
        if best_result is None:
            raise ValueError("Optimization failed to find a valid solution")
        
        # Store the results
        self.fit_breaks = np.concatenate(([x_min], best_result.x, [x_max]))
        self._calculate_parameters(self.fit_breaks)
        self.parameters = best_result.x
        
        # If continuity is required, make sure the function is continuous
        if not allow_discontinuities:
            self.enforce_continuity()
        
        return self.parameters
    
    def _calculate_parameters(self, breakpoints):
        """Calculate slopes and intercepts for each segment"""
        sorted_breaks = np.sort(breakpoints)
        self.break_points = sorted_breaks
        
        # Get slopes and intercepts for each segment
        slopes = []
        intercepts = []
        
        for i in range(len(sorted_breaks) - 1):
            # Get points in this segment
            segment_mask = (self.x >= sorted_breaks[i]) & (self.x <= sorted_breaks[i+1])
            if np.sum(segment_mask) > 1:
                # If we have multiple points, perform linear regression
                x_segment = self.x[segment_mask]
                y_segment = self.y[segment_mask]
                
                # Calculate slope and intercept
                slope, intercept = np.polyfit(x_segment, y_segment, 1)
            else:
                # If we don't have enough points, use adjacent breakpoints
                # Find closest points to the breakpoints
                idx_before = np.argmin(np.abs(self.x - sorted_breaks[i]))
                idx_after = np.argmin(np.abs(self.x - sorted_breaks[i+1]))
                
                if idx_before == idx_after:
                    # If we got the same point, try to get another
                    x_values = np.sort(np.unique(self.x))
                    idx = np.where(x_values == self.x[idx_before])[0][0]
                    if idx > 0:
                        idx_before = np.where(self.x == x_values[idx-1])[0][0]
                    elif idx < len(x_values) - 1:
                        idx_after = np.where(self.x == x_values[idx+1])[0][0]
                
                if idx_before != idx_after:
                    slope = (self.y[idx_after] - self.y[idx_before]) / (self.x[idx_after] - self.x[idx_before])
                    intercept = self.y[idx_before] - slope * self.x[idx_before]
                else:
                    # Default to flat line if we can't do better
                    slope = 0
                    intercept = np.mean(self.y)
            
            slopes.append(slope)
            intercepts.append(intercept)
        
        self.slopes = np.array(slopes)
        self.intercepts = np.array(intercepts)
    
    def _calculate_mse(self, breakpoints, allow_discontinuities=True):
        """
        Calculate MSE with constraints for monotonically decreasing function
        """
        # Add the min and max x values as breakpoints
        full_breakpoints = np.concatenate(([min(self.x)], breakpoints, [max(self.x)]))
        full_breakpoints.sort()
        
        # Calculate parameters for the segments
        self._calculate_parameters(full_breakpoints)
        
        # Check if all slopes are negative (function is decreasing)
        valid = True
        for slope in self.slopes:
            if slope > 0:  # Positive slope means increasing function
                valid = False
                break
        
        # If we allow discontinuities, check that each segment starts lower than
        # the previous segment ended (function never increases)
        if valid and allow_discontinuities:
            for i in range(1, len(full_breakpoints) - 1):
                # Calculate y value at breakpoint from both adjacent segments
                bp = full_breakpoints[i]
                y_left = self.slopes[i-1] * bp + self.intercepts[i-1]
                y_right = self.slopes[i] * bp + self.intercepts[i]
                
                # Check if right segment starts higher than left segment ends
                if y_right > y_left:
                    valid = False
                    break
        
        # Calculate predictions
        y_pred = self.predict(self.x)
        
        # Calculate MSE
        mse = np.mean((self.y - y_pred) ** 2)
        
        # If continuity is required, add a penalty for discontinuities
        if not allow_discontinuities:
            continuity_penalty = 0
            for i in range(1, len(full_breakpoints) - 1):
                bp = full_breakpoints[i]
                y_left = self.slopes[i-1] * bp + self.intercepts[i-1]
                y_right = self.slopes[i] * bp + self.intercepts[i]
                continuity_penalty += (y_left - y_right)**2
            
            # Add the continuity penalty to the MSE
            mse += 1000 * continuity_penalty
        
        # Penalize if monotonicity constraints are violated
        if not valid:
            return mse * 1000  # Large penalty for constraint violation
        
        return mse
    
    def predict(self, x):
        """
        Predict y values for given x values
        
        Parameters
        ----------
        x : array_like
            The x values to predict
            
        Returns
        -------
        y : array_like
            The predicted y values
        """
        if self.break_points is None:
            raise ValueError("Model has not been fitted yet")
        
        x = np.array(x)
        y = np.zeros_like(x, dtype=float)
        
        # For each segment, calculate y values
        for i in range(len(self.break_points) - 1):
            segment_mask = (x >= self.break_points[i]) & (x <= self.break_points[i+1])
            y[segment_mask] = self.slopes[i] * x[segment_mask] + self.intercepts[i]
        
        return y
        
    def enforce_continuity(self):
        """
        Adjust the intercepts to ensure the fitted function is continuous
        at all breakpoints while maintaining the fitted slopes.
        
        This should be called after fit() if continuity is required
        but wasn't achieved during optimization.
        """
        if self.break_points is None:
            raise ValueError("Model has not been fitted yet")
            
        # Start with the first segment unchanged
        for i in range(1, len(self.break_points) - 1):
            # Get the breakpoint
            bp = self.break_points[i]
            
            # Calculate y value from left segment
            y_left = self.slopes[i-1] * bp + self.intercepts[i-1]
            
            # Adjust intercept of right segment to match
            self.intercepts[i] = y_left - self.slopes[i] * bp
    
    def r_squared(self):
        """
        Calculate R^2 value for the fit
        
        Returns
        -------
        r_squared : float
            The R^2 value
        """
        if self.break_points is None:
            raise ValueError("Model has not been fitted yet")
        
        y_pred = self.predict(self.x)
        ss_total = np.sum((self.y - np.mean(self.y)) ** 2)
        ss_residual = np.sum((self.y - y_pred) ** 2)
        
        return 1 - (ss_residual / ss_total)


# Example usage:
if __name__ == "__main__":
    # Generate example data
    np.random.seed(0)
    x = np.sort(np.random.rand(100) * 10)
    
    # Create monotonically decreasing function with discontinuities
    y = np.zeros_like(x)
    breakpoints = [0, 3, 6, 10]
    slopes = [-0.5, -1.0, -0.7]
    
    # Values at breakpoints (discontinuous)
    values_at_breaks = [8, 6, 3, 0]
    
    # Calculate intercepts to achieve the values at breakpoints
    intercepts = []
    for i in range(len(breakpoints) - 1):
        # Value at right end of segment
        val_right = values_at_breaks[i+1]
        # Calculate intercept to achieve this value
        intercept = val_right - slopes[i] * breakpoints[i+1]
        intercepts.append(intercept)
    
    # Generate data
    for i in range(len(breakpoints) - 1):
        mask = (x >= breakpoints[i]) & (x < breakpoints[i+1])
        y[mask] = slopes[i] * x[mask] + intercepts[i]
    
    # Add noise
    y += np.random.normal(0, 0.3, size=len(x))
    
    # Fit with our custom model - with discontinuities
    model_disc = ConstrainedPiecewiseLinFit(x, y)
    model_disc.fit(3, allow_discontinuities=True)  # Try to find 3 segments with discontinuities allowed
    
    # Fit with our custom model - continuous
    model_cont = ConstrainedPiecewiseLinFit(x, y)
    model_cont.fit(3, allow_discontinuities=False)  # Try to find 3 segments with continuity
    
    # Print results - discontinuous model
    print("DISCONTINUOUS MODEL:")
    print("Fitted breakpoints:", model_disc.fit_breaks)
    print("Fitted slopes:", model_disc.slopes)
    print("Fitted intercepts:", model_disc.intercepts)
    print("R-squared:", model_disc.r_squared())
    
    # Print results - continuous model
    print("\nCONTINUOUS MODEL:")
    print("Fitted breakpoints:", model_cont.fit_breaks)
    print("Fitted slopes:", model_cont.slopes)
    print("Fitted intercepts:", model_cont.intercepts)
    print("R-squared:", model_cont.r_squared())
    
    # Plot
    fig = plt.figure(figsize=(12, 8))
    
    # Subplot for discontinuous fit
    plt.subplot(2, 1, 1)
    plt.scatter(x, y, alpha=0.5, label='Data')
    
    # Plot the true function
    x_true = np.linspace(min(x), max(x), 1000)
    y_true = np.zeros_like(x_true)
    for i in range(len(breakpoints) - 1):
        mask = (x_true >= breakpoints[i]) & (x_true < breakpoints[i+1])
        y_true[mask] = slopes[i] * x_true[mask] + intercepts[i]
    plt.plot(x_true, y_true, 'b--', linewidth=1.5, label='True Function')
    
    # Plot the fitted function - discontinuous
    x_fit = np.linspace(min(x), max(x), 1000)
    y_fit = model_disc.predict(x_fit)
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='Discontinuous Fit')
    
    # Mark breakpoints
    for bp in model_disc.fit_breaks:
        plt.axvline(bp, color='green', linestyle='--', alpha=0.5)
    
    plt.legend()
    plt.title('Monotonically Decreasing Piecewise Linear Fit with Discontinuities')
    
    # Subplot for continuous fit
    plt.subplot(2, 1, 2)
    plt.scatter(x, y, alpha=0.5, label='Data')
    
    # Plot the true function
    plt.plot(x_true, y_true, 'b--', linewidth=1.5, label='True Function')
    
    # Plot the fitted function - continuous
    y_fit_cont = model_cont.predict(x_fit)
    plt.plot(x_fit, y_fit_cont, 'r-', linewidth=2, label='Continuous Fit')
    
    # Mark breakpoints
    for bp in model_cont.fit_breaks:
        plt.axvline(bp, color='green', linestyle='--', alpha=0.5)
    
    plt.legend()
    plt.title('Monotonically Decreasing Continuous Piecewise Linear Fit')
    
    plt.tight_layout()
    fig.savefig('test.png')