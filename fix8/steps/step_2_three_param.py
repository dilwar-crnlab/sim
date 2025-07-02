#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: Three-Parameter Fitting Implementation
Fits the power evolution profile from Step 1 to the approximate model:
P(f,z) ≈ P(f,0) × exp(-2α₀(f)z + (2α₁(f)/σ(f))(exp(-σ(f)z) - 1))

Based on equations (24), (30.1), and (30.2) from the research paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
import warnings

class Step2_ParameterFitting:
    """
    Step 2: Fit power evolution to three-parameter model
    """
    
    def __init__(self, step1_results: Dict):
        """
        Initialize with results from Step 1
        
        Args:
            step1_results: Dictionary containing Step 1 PEP calculation results
        """
        self.step1_results = step1_results
        
        # Extract key parameters from Step 1
        self.frequencies_hz = np.array(step1_results['system_parameters']['frequencies_hz'])
        self.wavelengths_nm = np.array(step1_results['system_parameters']['wavelengths_nm'])
        self.num_channels = step1_results['system_parameters']['num_channels']
        self.alpha_db_km = np.array(step1_results['system_parameters']['alpha_db_km'])
        
        # Power evolution data from Step 1
        self.distances_m = np.array(step1_results['distances_m'])
        self.power_evolution_w = np.array(step1_results['power_evolution_w'])
        self.span_length_km = step1_results['span_length_km']
        
        # Convert to per-meter units
        self.alpha_linear_per_m = self.alpha_db_km * np.log(10) / (10 * 1000)  # 1/m
        
        print(f"Step 2 Parameter Fitting Initialized:")
        print(f"  Processing {self.num_channels} channels")
        print(f"  Span length: {self.span_length_km} km")
        print(f"  Distance points: {len(self.distances_m)}")
        
    def approximate_model(self, z: np.ndarray, P0: float, alpha0: float, alpha1: float, sigma: float) -> np.ndarray:
        """
        Approximate power evolution model from equation (13)
        
        Args:
            z: Distance array (m)
            P0: Initial power (W)
            alpha0: α₀ coefficient (1/m)
            alpha1: α₁ coefficient (1/m)  
            sigma: σ coefficient (1/m)
            
        Returns:
            Power evolution according to approximate model
        """
        if sigma <= 0:
            # Avoid division by zero
            return P0 * np.exp(-2 * alpha0 * z)
        
        exponent = -2 * alpha0 * z + (2 * alpha1 / sigma) * (np.exp(-sigma * z) - 1)
        return P0 * np.exp(exponent)
    
    def cost_function(self, alpha0: float, alpha1: float, sigma: float, 
                     P_num: np.ndarray, P0: float, z: np.ndarray, m_c: float = 2.0) -> float:
        """
        Cost function from equation (24)
        
        Args:
            alpha0, alpha1, sigma: Model parameters
            P_num: Numerical power evolution from Step 1
            P0: Initial power
            z: Distance array
            m_c: Weighting exponent (default 2.0)
            
        Returns:
            Cost function value
        """
        try:
            # Calculate approximate model
            P_approx = self.approximate_model(z, P0, alpha0, alpha1, sigma)
            
            # Avoid division by very small numbers
            P_num_safe = np.maximum(P_num, 1e-15)
            P_approx_safe = np.maximum(P_approx, 1e-15)
            
            # Calculate relative error
            relative_error = np.log(P_num_safe / P0) + 2 * alpha0 * z + 2 * alpha1 * (1 - np.exp(-sigma * z)) / sigma
            
            # Weight by power^m_c and integrate
            weight = np.power(P_num_safe, m_c)
            integrand = weight * relative_error**2
            
            # Numerical integration (trapezoidal rule)
            cost = np.trapz(integrand, z)
            
            return cost
            
        except (OverflowError, ZeroDivisionError, RuntimeWarning):
            return 1e10  # Return large cost for invalid parameters
    
    def solve_alpha0_alpha1(self, sigma: float, P_num: np.ndarray, P0: float, 
                           z: np.ndarray, m_c: float = 2.0) -> Tuple[float, float]:
        """
        Solve for optimal α₀ and α₁ given σ using equations (30.1) and (30.2)
        
        Args:
            sigma: Fixed σ value
            P_num: Numerical power evolution
            P0: Initial power
            z: Distance array
            m_c: Weighting exponent
            
        Returns:
            Tuple of (α₀, α₁)
        """
        try:
            # Avoid numerical issues
            P_num_safe = np.maximum(P_num, 1e-15)
            
            # Calculate terms for matrix equation
            exp_neg_sigma_z = np.exp(-sigma * z)
            one_minus_exp = (1 - exp_neg_sigma_z) / sigma
            
            # Weight function
            weight = np.power(P_num_safe, m_c)
            
            # Matrix elements (using trapezoidal integration)
            A11 = np.trapz(weight * z**2, z)
            A12 = np.trapz(weight * z * one_minus_exp, z)
            A22 = np.trapz(weight * one_minus_exp**2, z)
            
            # Right-hand side
            log_ratio = np.log(P_num_safe / P0)
            b1 = -0.5 * np.trapz(weight * z * log_ratio, z)
            b2 = -0.5 * np.trapz(weight * one_minus_exp * log_ratio, z)
            
            # Solve 2x2 system
            det = A11 * A22 - A12**2
            
            if abs(det) < 1e-12:
                # Matrix is nearly singular
                alpha0 = self.alpha_linear_per_m[0]  # Use intrinsic loss as fallback
                alpha1 = 0.0
            else:
                alpha0 = (A22 * b1 - A12 * b2) / det
                alpha1 = (A11 * b2 - A12 * b1) / det
            
            return alpha0, alpha1
            
        except Exception as e:
            # Fallback values
            alpha0 = self.alpha_linear_per_m[0]
            alpha1 = 0.0
            return alpha0, alpha1
    
    def fit_single_channel(self, channel_idx: int, m_c: float = 2.0, 
                          max_iterations: int = 50) -> Dict:
        """
        Fit three parameters for a single channel
        
        Args:
            channel_idx: Channel index to fit
            m_c: Weighting exponent for cost function
            max_iterations: Maximum iterations for σ search
            
        Returns:
            Dictionary with fitted parameters and metrics
        """
        print(f"Fitting channel {channel_idx + 1}/{self.num_channels} "
              f"(f = {self.frequencies_hz[channel_idx]/1e12:.2f} THz)")
        
        # Extract power evolution for this channel
        P_num = self.power_evolution_w[:, channel_idx]
        P0 = P_num[0]  # Initial power
        z = self.distances_m
        
        # Initial intrinsic loss for this frequency
        alpha_intrinsic = self.alpha_linear_per_m[channel_idx]
        
        # Search bounds for σ: [α_intrinsic, 4×α_intrinsic]
        sigma_min = alpha_intrinsic
        sigma_max = 4 * alpha_intrinsic
        
        def objective_function(sigma):
            """Objective function for σ optimization"""
            alpha0, alpha1 = self.solve_alpha0_alpha1(sigma, P_num, P0, z, m_c)
            cost = self.cost_function(alpha0, alpha1, sigma, P_num, P0, z, m_c)
            return cost
        
        # Golden section search for optimal σ
        try:
            result = minimize_scalar(objective_function, bounds=(sigma_min, sigma_max), 
                                   method='bounded', options={'maxiter': max_iterations})
            
            optimal_sigma = result.x
            optimal_cost = result.fun
            
        except Exception as e:
            print(f"  Warning: Optimization failed for channel {channel_idx}, using fallback")
            optimal_sigma = 2 * alpha_intrinsic
            optimal_cost = 1e10
        
        # Calculate final α₀ and α₁
        alpha0, alpha1 = self.solve_alpha0_alpha1(optimal_sigma, P_num, P0, z, m_c)
        
        # Calculate final cost and relative error
        final_cost = self.cost_function(alpha0, alpha1, optimal_sigma, P_num, P0, z, m_c)
        
        # Calculate R² goodness of fit
        P_fitted = self.approximate_model(z, P0, alpha0, alpha1, optimal_sigma)
        ss_res = np.sum((P_num - P_fitted)**2)
        ss_tot = np.sum((P_num - np.mean(P_num))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate maximum relative error
        rel_error = np.abs((P_num - P_fitted) / np.maximum(P_num, 1e-15))
        max_rel_error = np.max(rel_error)
        rms_rel_error = np.sqrt(np.mean(rel_error**2))
        
        return {
            'channel_idx': channel_idx,
            'frequency_hz': self.frequencies_hz[channel_idx],
            'frequency_thz': self.frequencies_hz[channel_idx] / 1e12,
            'wavelength_nm': self.wavelengths_nm[channel_idx],
            'alpha0_per_m': alpha0,
            'alpha1_per_m': alpha1,
            'sigma_per_m': optimal_sigma,
            'alpha_intrinsic_per_m': alpha_intrinsic,
            'cost_function_value': final_cost,
            'r_squared': r_squared,
            'max_relative_error': max_rel_error,
            'rms_relative_error': rms_rel_error,
            'P_numerical': P_num.tolist(),
            'P_fitted': P_fitted.tolist(),
            'initial_power_w': P0
        }
    
    def step2_fit_all_channels(self, m_c: float = 2.0, max_iterations: int = 50) -> Dict:
        """
        Fit three parameters for all channels
        
        Args:
            m_c: Weighting exponent for cost function
            max_iterations: Maximum iterations for σ search
            
        Returns:
            Complete Step 2 results dictionary
        """
        print(f"\n{'='*80}")
        print(f"STEP 2: THREE-PARAMETER FITTING")
        print(f"{'='*80}")
        print(f"Fitting power evolution to approximate model (Equation 13)")
        print(f"Cost function weighting exponent: {m_c}")
        print(f"Maximum optimization iterations: {max_iterations}")
        
        start_time = time.time()
        
        # Fit each channel
        channel_results = []
        for ch_idx in range(self.num_channels):
            try:
                result = self.fit_single_channel(ch_idx, m_c, max_iterations)
                channel_results.append(result)
            except Exception as e:
                print(f"  Error fitting channel {ch_idx}: {e}")
                # Add fallback result
                channel_results.append({
                    'channel_idx': ch_idx,
                    'frequency_hz': self.frequencies_hz[ch_idx],
                    'alpha0_per_m': self.alpha_linear_per_m[ch_idx],
                    'alpha1_per_m': 0.0,
                    'sigma_per_m': 2 * self.alpha_linear_per_m[ch_idx],
                    'cost_function_value': 1e10,
                    'r_squared': 0.0,
                    'max_relative_error': 1.0,
                    'rms_relative_error': 1.0
                })
        
        computation_time = time.time() - start_time
        
        # Calculate statistics
        r_squared_values = [r['r_squared'] for r in channel_results]
        max_errors = [r['max_relative_error'] for r in channel_results]
        rms_errors = [r['rms_relative_error'] for r in channel_results]
        
        # Compile results
        results = {
            'step2_parameter_fitting': True,
            'computation_time_s': computation_time,
            'weighting_exponent_mc': m_c,
            'max_optimization_iterations': max_iterations,
            'num_channels_fitted': len(channel_results),
            'channel_results': channel_results,
            'distances_m': self.distances_m.tolist(),
            'distances_km': (self.distances_m / 1000).tolist(),
            'statistics': {
                'mean_r_squared': np.mean(r_squared_values),
                'min_r_squared': np.min(r_squared_values),
                'max_r_squared': np.max(r_squared_values),
                'mean_max_relative_error': np.mean(max_errors),
                'max_max_relative_error': np.max(max_errors),
                'mean_rms_relative_error': np.mean(rms_errors),
                'max_rms_relative_error': np.max(rms_errors)
            },
            'frequencies_hz': self.frequencies_hz.tolist(),
            'frequencies_thz': (self.frequencies_hz / 1e12).tolist(),
            'wavelengths_nm': self.wavelengths_nm.tolist(),
            'alpha0_per_m': [r['alpha0_per_m'] for r in channel_results],
            'alpha1_per_m': [r['alpha1_per_m'] for r in channel_results], 
            'sigma_per_m': [r['sigma_per_m'] for r in channel_results],
            'alpha_intrinsic_per_m': [r['alpha_intrinsic_per_m'] for r in channel_results]
        }
        
        print(f"\n✓ Step 2 parameter fitting completed in {computation_time:.2f}s")
        print(f"✓ Mean R²: {np.mean(r_squared_values):.4f}")
        print(f"✓ Mean RMS relative error: {np.mean(rms_errors):.2e}")
        print(f"✓ Max relative error: {np.max(max_errors):.2e}")
        
        return results
    
    def plot_step2_results(self, results: Dict, save_path: str = None, 
                          selected_channels: List[int] = None):
        """
        Plot Step 2 fitting results
        
        Args:
            results: Step 2 results dictionary
            save_path: Path to save plot
            selected_channels: Channels to show in detail (default: evenly spaced selection)
        """
        if selected_channels is None:
            # Select evenly spaced channels for detailed plots
            selected_channels = np.linspace(0, self.num_channels-1, 6, dtype=int).tolist()
        
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.3)
        
        frequencies_thz = np.array(results['frequencies_thz'])
        
        # 1. Parameter profiles vs frequency
        ax1 = fig.add_subplot(gs[0, 0])
        alpha0_values = np.array(results['alpha0_per_m'])
        alpha_intrinsic = np.array(results['alpha_intrinsic_per_m'])
        ax1.plot(frequencies_thz, alpha0_values * 1000, 'b-', linewidth=2, label='α₀ (fitted)')
        ax1.plot(frequencies_thz, alpha_intrinsic * 1000, 'r--', linewidth=2, label='α (intrinsic)')
        ax1.set_xlabel('Frequency (THz)', fontweight='bold')
        ax1.set_ylabel('Loss Coefficient (1/km)', fontweight='bold')
        ax1.set_title('α₀ vs Intrinsic Loss', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2 = fig.add_subplot(gs[0, 1])
        alpha1_values = np.array(results['alpha1_per_m'])
        ax2.plot(frequencies_thz, alpha1_values * 1000, 'g-', linewidth=2)
        ax2.set_xlabel('Frequency (THz)', fontweight='bold')
        ax2.set_ylabel('α₁ (1/km)', fontweight='bold')
        ax2.set_title('ISRS Parameter α₁', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[0, 2])
        sigma_values = np.array(results['sigma_per_m'])
        ax3.plot(frequencies_thz, sigma_values * 1000, 'm-', linewidth=2)
        ax3.set_xlabel('Frequency (THz)', fontweight='bold')
        ax3.set_ylabel('σ (1/km)', fontweight='bold')
        ax3.set_title('ISRS Parameter σ', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 2. Fitting quality metrics
        ax4 = fig.add_subplot(gs[1, 0])
        r_squared = [r['r_squared'] for r in results['channel_results']]
        bars = ax4.bar(range(len(r_squared)), r_squared, alpha=0.7, color='blue')
        ax4.set_xlabel('Channel Index', fontweight='bold')
        ax4.set_ylabel('R² (Goodness of Fit)', fontweight='bold')
        ax4.set_title('Fitting Quality per Channel', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1])
        
        ax5 = fig.add_subplot(gs[1, 1])
        max_errors = [r['max_relative_error'] for r in results['channel_results']]
        ax5.semilogy(frequencies_thz, max_errors, 'ro-', markersize=4, linewidth=1)
        ax5.set_xlabel('Frequency (THz)', fontweight='bold')
        ax5.set_ylabel('Max Relative Error', fontweight='bold')
        ax5.set_title('Maximum Fitting Error', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[1, 2])
        rms_errors = [r['rms_relative_error'] for r in results['channel_results']]
        ax6.semilogy(frequencies_thz, rms_errors, 'go-', markersize=4, linewidth=1)
        ax6.set_xlabel('Frequency (THz)', fontweight='bold')
        ax6.set_ylabel('RMS Relative Error', fontweight='bold')
        ax6.set_title('RMS Fitting Error', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 3. Power evolution comparison for selected channels
        distances_km = np.array(results['distances_km'])
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_channels)))
        
        ax7 = fig.add_subplot(gs[2, :])
        for i, ch_idx in enumerate(selected_channels):
            ch_result = results['channel_results'][ch_idx]
            P_num = np.array(ch_result['P_numerical'])
            P_fitted = np.array(ch_result['P_fitted'])
            
            # Convert to dBm
            P_num_dbm = 10 * np.log10(P_num * 1000 + 1e-15)
            P_fitted_dbm = 10 * np.log10(P_fitted * 1000 + 1e-15)
            
            label_base = f'Ch {ch_idx}: {ch_result["frequency_thz"]:.1f} THz'
            ax7.plot(distances_km, P_num_dbm, '-', color=colors[i], linewidth=2, 
                    label=f'{label_base} (Step 1)')
            ax7.plot(distances_km, P_fitted_dbm, '--', color=colors[i], linewidth=2,
                    label=f'{label_base} (Fitted)')
        
        ax7.set_xlabel('Distance (km)', fontweight='bold')
        ax7.set_ylabel('Power (dBm)', fontweight='bold')
        ax7.set_title('Power Evolution: Step 1 vs Fitted Model', fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Relative error for selected channels
        ax8 = fig.add_subplot(gs[3, :])
        for i, ch_idx in enumerate(selected_channels):
            ch_result = results['channel_results'][ch_idx]
            P_num = np.array(ch_result['P_numerical'])
            P_fitted = np.array(ch_result['P_fitted'])
            
            rel_error = np.abs((P_num - P_fitted) / np.maximum(P_num, 1e-15))
            
            ax8.semilogy(distances_km, rel_error, '-', color=colors[i], linewidth=2,
                        label=f'Ch {ch_idx}: {ch_result["frequency_thz"]:.1f} THz')
        
        ax8.set_xlabel('Distance (km)', fontweight='bold')
        ax8.set_ylabel('Relative Error', fontweight='bold')
        ax8.set_title('Fitting Relative Error vs Distance', fontweight='bold')
        ax8.grid(True, alpha=0.3)
        ax8.legend()
        
        plt.suptitle('Step 2: Three-Parameter Fitting Results', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Step 2 results plot saved to: {save_path}")
        
        plt.show()
        return fig

def main():
    """Demonstrate Step 2 parameter fitting"""
    
    print("="*80)
    print("STEP 2: THREE-PARAMETER FITTING DEMONSTRATION")
    print("="*80)
    
    # You would typically load Step 1 results from a file
    # For demo purposes, we'll create a minimal example
    
    print("Note: This demo requires Step 1 results.")
    print("Please run Step 1 first and save the results, then load them here.")
    print("\nExample usage:")
    print("# Load Step 1 results")
    print("with open('step1_pep_ground_truth_YYYYMMDD_HHMMSS.json', 'r') as f:")
    print("    step1_results = json.load(f)")
    print("")
    print("# Extract the split-step results")
    print("step1_data = step1_results['split_step_results']")
    print("")
    print("# Initialize Step 2")
    print("step2_fitter = Step2_ParameterFitting(step1_data)")
    print("")
    print("# Perform fitting")
    print("step2_results = step2_fitter.step2_fit_all_channels(m_c=2.0)")
    print("")
    print("# Plot results")
    print("step2_fitter.plot_step2_results(step2_results, 'step2_fitting_results.png')")
    print("")
    print("# Save results")
    print("with open('step2_parameter_fitting_results.json', 'w') as f:")
    print("    json.dump(step2_results, f, indent=2)")

if __name__ == "__main__":
    main()