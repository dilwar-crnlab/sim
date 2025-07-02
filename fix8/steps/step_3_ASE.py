#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3: Amplifier Gains Calculation
Calculates required amplifier gains to compensate for fiber loss and ISRS effects
Based on Equation (5): G^(l,s,i) = P / P^(l,s,i)(z = L^(l,s)_s)
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class AmplifierConfiguration:
    """Configuration parameters for amplifiers"""
    # Target power levels
    target_power_dbm: float = 0.0  # Target power per channel at amplifier output
    
    # Amplifier constraints
    max_gain_db: float = 30.0      # Maximum amplifier gain
    min_gain_db: float = 10.0      # Minimum amplifier gain
    max_output_power_dbm: float = 20.0  # Maximum total output power
    
    # Amplifier types and noise figures
    edfa_noise_figure_db: float = 4.5    # EDFA noise figure (C-band)
    edfa_l_noise_figure_db: float = 5.0  # EDFA noise figure (L-band) 
    tdfa_noise_figure_db: float = 6.0    # TDFA noise figure (S-band)
    
    # Operational parameters
    gain_flatness_tolerance_db: float = 1.0  # Maximum gain variation across band
    power_control_mode: str = "constant_gain"  # "constant_gain" or "constant_power"

class Step3_AmplifierGains:
    """
    Step 3: Calculate required amplifier gains for each span and channel
    
    Implements equation (5) from the paper:
    G^(l,s,i) = P / P^(l,s,i)(z = L^(l,s)_s)
    
    Where:
    - G^(l,s,i): Gain for link l, span s, channel i
    - P: Target power level
    - P^(l,s,i)(z = L^(l,s)_s): Power at end of span before amplification
    """
    
    def __init__(self, step1_results: Dict, step2_results: Dict, 
                 amplifier_config: Optional[AmplifierConfiguration] = None):
        """
        Initialize Step 3 with results from previous steps
        
        Args:
            step1_results: Power evolution profile from Step 1
            step2_results: Fitted parameters from Step 2  
            amplifier_config: Amplifier configuration parameters
        """
        self.step1_results = step1_results
        self.step2_results = step2_results
        self.config = amplifier_config or AmplifierConfiguration()
        
        # Extract key parameters
        self.frequencies_hz = np.array(step1_results['system_parameters']['frequencies_hz'])
        self.wavelengths_nm = np.array(step1_results['system_parameters']['wavelengths_nm'])
        self.num_channels = step1_results['system_parameters']['num_channels']
        self.span_length_km = step1_results['span_length_km']
        
        # Power evolution data
        self.power_evolution_w = np.array(step1_results['power_evolution_w'])
        self.initial_powers_w = np.array(step1_results['initial_powers_w'])
        self.final_powers_w = np.array(step1_results['final_powers_w'])
        
        # Reference wavelength for band classification
        self.ref_wavelength_nm = 1550.0
        
        print(f"Step 3 Amplifier Gains Calculator Initialized:")
        print(f"  Processing {self.num_channels} channels")
        print(f"  Span length: {self.span_length_km} km")
        print(f"  Target power: {self.config.target_power_dbm} dBm")
        print(f"  Power control mode: {self.config.power_control_mode}")
    
    def classify_channel_band(self, wavelength_nm: float) -> str:
        """
        Classify channel into optical band (S, C, L)
        
        Args:
            wavelength_nm: Channel wavelength in nm
            
        Returns:
            Band classification: 'S', 'C', or 'L'
        """
        if wavelength_nm < 1460:
            return 'S'  # S-band: ~1460-1530 nm
        elif wavelength_nm < 1565:
            return 'C'  # C-band: ~1530-1565 nm
        else:
            return 'L'  # L-band: ~1565-1625 nm
    
    def get_noise_figure_for_band(self, band: str) -> float:
        """
        Get appropriate noise figure for optical band
        
        Args:
            band: Optical band ('S', 'C', 'L')
            
        Returns:
            Noise figure in dB
        """
        noise_figures = {
            'S': self.config.tdfa_noise_figure_db,  # Thulium-doped fiber amplifier
            'C': self.config.edfa_noise_figure_db,  # Erbium-doped fiber amplifier
            'L': self.config.edfa_l_noise_figure_db # L-band EDFA
        }
        return noise_figures.get(band, self.config.edfa_noise_figure_db)
    
    def calculate_target_power(self, channel_idx: int) -> float:
        """
        Calculate target power for a specific channel
        
        Args:
            channel_idx: Channel index
            
        Returns:
            Target power in watts
        """
        if self.config.power_control_mode == "constant_power":
            # Maintain constant power per channel
            target_power_w = 10**(self.config.target_power_dbm / 10) * 1e-3
        else:
            # Constant gain mode - restore input power
            target_power_w = self.initial_powers_w[channel_idx]
        
        return target_power_w
    
    def calculate_amplifier_gain_single_channel(self, channel_idx: int) -> Dict:
        """
        Calculate amplifier gain for a single channel
        
        Args:
            channel_idx: Channel index
            
        Returns:
            Dictionary with gain calculation results
        """
        # Get powers
        target_power_w = self.calculate_target_power(channel_idx)
        received_power_w = self.final_powers_w[channel_idx]
        
        # Avoid division by zero
        if received_power_w <= 0:
            received_power_w = 1e-15
        
        # Calculate required gain
        gain_linear = target_power_w / received_power_w
        gain_db = 10 * np.log10(gain_linear)
        
        # Get channel characteristics
        wavelength_nm = self.wavelengths_nm[channel_idx]
        frequency_hz = self.frequencies_hz[channel_idx]
        band = self.classify_channel_band(wavelength_nm)
        noise_figure_db = self.get_noise_figure_for_band(band)
        
        # Check gain constraints
        gain_valid = (self.config.min_gain_db <= gain_db <= self.config.max_gain_db)
        
        if not gain_valid:
            if gain_db > self.config.max_gain_db:
                gain_db_clamped = self.config.max_gain_db
                actual_output_power_w = received_power_w * 10**(gain_db_clamped / 10)
            else:
                gain_db_clamped = self.config.min_gain_db
                actual_output_power_w = received_power_w * 10**(gain_db_clamped / 10)
        else:
            gain_db_clamped = gain_db
            actual_output_power_w = target_power_w
        
        # Calculate span loss for reference
        if self.initial_powers_w[channel_idx] > 0:
            span_loss_db = 10 * np.log10(self.initial_powers_w[channel_idx] / received_power_w)
        else:
            span_loss_db = 0.0
        
        return {
            'channel_idx': channel_idx,
            'frequency_hz': frequency_hz,
            'frequency_thz': frequency_hz / 1e12,
            'wavelength_nm': wavelength_nm,
            'band': band,
            'noise_figure_db': noise_figure_db,
            'initial_power_w': self.initial_powers_w[channel_idx],
            'received_power_w': received_power_w,
            'target_power_w': target_power_w,
            'actual_output_power_w': actual_output_power_w,
            'required_gain_db': gain_db,
            'applied_gain_db': gain_db_clamped,
            'gain_linear': 10**(gain_db_clamped / 10),
            'span_loss_db': span_loss_db,
            'gain_valid': gain_valid,
            'initial_power_dbm': 10 * np.log10(self.initial_powers_w[channel_idx] * 1000 + 1e-15),
            'received_power_dbm': 10 * np.log10(received_power_w * 1000 + 1e-15),
            'target_power_dbm': 10 * np.log10(target_power_w * 1000 + 1e-15),
            'actual_output_power_dbm': 10 * np.log10(actual_output_power_w * 1000 + 1e-15)
        }
    
    def step3_calculate_amplifier_gains(self) -> Dict:
        """
        Calculate amplifier gains for all channels
        
        Returns:
            Complete Step 3 results dictionary
        """
        print(f"\n{'='*80}")
        print(f"STEP 3: AMPLIFIER GAINS CALCULATION")
        print(f"{'='*80}")
        print(f"Calculating required amplifier gains using Equation (5)")
        print(f"Power control mode: {self.config.power_control_mode}")
        print(f"Target power level: {self.config.target_power_dbm} dBm")
        
        start_time = time.time()
        
        # Calculate gains for all channels
        channel_results = []
        for ch_idx in range(self.num_channels):
            try:
                result = self.calculate_amplifier_gain_single_channel(ch_idx)
                channel_results.append(result)
            except Exception as e:
                print(f"  Error calculating gain for channel {ch_idx}: {e}")
                # Add fallback result
                channel_results.append({
                    'channel_idx': ch_idx,
                    'frequency_hz': self.frequencies_hz[ch_idx],
                    'required_gain_db': 20.0,  # Default gain
                    'applied_gain_db': 20.0,
                    'gain_linear': 100.0,
                    'gain_valid': False,
                    'error': str(e)
                })
        
        computation_time = time.time() - start_time
        
        # Calculate statistics
        gain_values_db = [r['applied_gain_db'] for r in channel_results]
        span_losses_db = [r.get('span_loss_db', 0) for r in channel_results]
        gain_valid_flags = [r['gain_valid'] for r in channel_results]
        
        # Total power calculations
        total_input_power_w = np.sum(self.initial_powers_w)
        total_output_power_w = np.sum([r['actual_output_power_w'] for r in channel_results])
        
        # Band-wise statistics
        band_stats = {}
        for band in ['S', 'C', 'L']:
            band_channels = [r for r in channel_results if r.get('band') == band]
            if band_channels:
                band_gains = [r['applied_gain_db'] for r in band_channels]
                band_stats[band] = {
                    'num_channels': len(band_channels),
                    'mean_gain_db': np.mean(band_gains),
                    'std_gain_db': np.std(band_gains),
                    'min_gain_db': np.min(band_gains),
                    'max_gain_db': np.max(band_gains),
                    'noise_figure_db': band_channels[0]['noise_figure_db']
                }
        
        # Check gain flatness
        gain_flatness_db = np.max(gain_values_db) - np.min(gain_values_db)
        gain_flatness_acceptable = gain_flatness_db <= self.config.gain_flatness_tolerance_db
        
        # Compile results
        results = {
            'step3_amplifier_gains': True,
            'computation_time_s': computation_time,
            'amplifier_configuration': {
                'target_power_dbm': self.config.target_power_dbm,
                'power_control_mode': self.config.power_control_mode,
                'max_gain_db': self.config.max_gain_db,
                'min_gain_db': self.config.min_gain_db,
                'gain_flatness_tolerance_db': self.config.gain_flatness_tolerance_db
            },
            'channel_results': channel_results,
            'span_length_km': self.span_length_km,
            'num_channels': self.num_channels,
            'frequencies_hz': self.frequencies_hz.tolist(),
            'frequencies_thz': (self.frequencies_hz / 1e12).tolist(),
            'wavelengths_nm': self.wavelengths_nm.tolist(),
            'gain_values_db': gain_values_db,
            'gain_values_linear': [10**(g/10) for g in gain_values_db],
            'span_loss_values_db': span_losses_db,
            'statistics': {
                'mean_gain_db': np.mean(gain_values_db),
                'std_gain_db': np.std(gain_values_db),
                'min_gain_db': np.min(gain_values_db),
                'max_gain_db': np.max(gain_values_db),
                'gain_flatness_db': gain_flatness_db,
                'gain_flatness_acceptable': gain_flatness_acceptable,
                'num_valid_gains': np.sum(gain_valid_flags),
                'num_invalid_gains': np.sum(~np.array(gain_valid_flags)),
                'mean_span_loss_db': np.mean(span_losses_db),
                'total_input_power_dbm': 10 * np.log10(total_input_power_w * 1000 + 1e-15),
                'total_output_power_dbm': 10 * np.log10(total_output_power_w * 1000 + 1e-15)
            },
            'band_statistics': band_stats,
            'power_budget': {
                'total_input_power_w': total_input_power_w,
                'total_output_power_w': total_output_power_w,
                'power_restoration_ratio': total_output_power_w / total_input_power_w if total_input_power_w > 0 else 0,
                'average_power_per_channel_input_dbm': 10 * np.log10(total_input_power_w / self.num_channels * 1000 + 1e-15),
                'average_power_per_channel_output_dbm': 10 * np.log10(total_output_power_w / self.num_channels * 1000 + 1e-15)
            }
        }
        
        print(f"\n✓ Step 3 amplifier gains calculation completed in {computation_time:.3f}s")
        print(f"✓ Mean gain: {np.mean(gain_values_db):.2f} dB")
        print(f"✓ Gain flatness: {gain_flatness_db:.2f} dB ({'✓' if gain_flatness_acceptable else '✗'})")
        print(f"✓ Valid gains: {np.sum(gain_valid_flags)}/{self.num_channels}")
        print(f"✓ Band coverage: {list(band_stats.keys())}")
        
        return results
    
    def validate_amplifier_gains(self, results: Dict) -> Dict:
        """
        Validate amplifier gain results against physical constraints
        
        Args:
            results: Step 3 results dictionary
            
        Returns:
            Validation report
        """
        validation = {
            'overall_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check gain flatness
        if not results['statistics']['gain_flatness_acceptable']:
            validation['warnings'].append(
                f"Gain flatness {results['statistics']['gain_flatness_db']:.2f} dB exceeds "
                f"tolerance {self.config.gain_flatness_tolerance_db} dB"
            )
        
        # Check for invalid gains
        if results['statistics']['num_invalid_gains'] > 0:
            validation['errors'].append(
                f"{results['statistics']['num_invalid_gains']} channels have invalid gain requirements"
            )
            validation['overall_valid'] = False
        
        # Check total output power
        total_output_dbm = results['statistics']['total_output_power_dbm']
        if total_output_dbm > self.config.max_output_power_dbm:
            validation['errors'].append(
                f"Total output power {total_output_dbm:.1f} dBm exceeds limit "
                f"{self.config.max_output_power_dbm} dBm"
            )
            validation['overall_valid'] = False
        
        # Check power restoration
        power_ratio = results['power_budget']['power_restoration_ratio']
        if power_ratio < 0.9:
            validation['warnings'].append(
                f"Power restoration ratio {power_ratio:.3f} is low (< 0.9)"
            )
        
        return validation

def main():
    """Demonstrate Step 3 amplifier gains calculation"""
    
    print("="*80)
    print("STEP 3: AMPLIFIER GAINS CALCULATION DEMONSTRATION")
    print("="*80)
    
    print("This step calculates the required amplifier gains using:")
    print("• Equation (5): G^(l,s,i) = P / P^(l,s,i)(z = L^(l,s)_s)")
    print("• Power evolution results from Step 1")
    print("• Fitted parameters from Step 2 (for validation)")
    print("")
    print("Usage example:")
    print("# Load previous results")
    print("with open('step1_results.json', 'r') as f:")
    print("    step1_results = json.load(f)")
    print("with open('step2_results.json', 'r') as f:")
    print("    step2_results = json.load(f)")
    print("")
    print("# Configure amplifiers")
    print("amp_config = AmplifierConfiguration(")
    print("    target_power_dbm=0.0,")
    print("    power_control_mode='constant_gain',")
    print("    max_gain_db=25.0")
    print(")")
    print("")
    print("# Calculate gains")
    print("step3_calc = Step3_AmplifierGains(step1_results, step2_results, amp_config)")
    print("step3_results = step3_calc.step3_calculate_amplifier_gains()")
    print("")
    print("# Validate results")
    print("validation = step3_calc.validate_amplifier_gains(step3_results)")
    print("")
    print("# Save results")
    print("with open('step3_amplifier_gains.json', 'w') as f:")
    print("    json.dump(step3_results, f, indent=2)")

if __name__ == "__main__":
    main()