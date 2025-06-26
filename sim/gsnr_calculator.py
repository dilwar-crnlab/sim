#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GSNR Calculator - Integration with Steps 1-7
Wrapper class that integrates the existing step implementations for GSNR calculation
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import time

# Add steps directory to path
steps_dir = os.path.join(os.path.dirname(__file__), '..', 'steps')
if steps_dir not in sys.path:
    sys.path.insert(0, steps_dir)

# Import existing step implementations
try:
    from step1 import SplitStepGroundTruthGenerator
    from step_2_three_param import Step2_ParameterFitting
    from step_3_ASE import Step3_AmplifierGains, AmplifierConfiguration
    from step_4567 import Step4567_GSNRComputation
except ImportError as e:
    print(f"Warning: Could not import step implementations: {e}")
    print("Please ensure the steps directory contains the required files")

class GSNRCalculationResult:
    """Results from GSNR calculation"""
    
    def __init__(self, channel_index: int, core_index: int, path_links: List,
                 gsnr_db: float, osnr_db: float, supported_modulation: str,
                 max_bitrate_gbps: float, ase_power: float, nli_power: float, 
                 icxt_power: float):
        self.channel_index = channel_index
        self.core_index = core_index
        self.path_links = path_links
        self.gsnr_db = gsnr_db
        self.osnr_db = osnr_db
        self.supported_modulation = supported_modulation
        self.max_bitrate_gbps = max_bitrate_gbps
        self.ase_power = ase_power
        self.nli_power = nli_power
        self.icxt_power = icxt_power
        
        # Calculate SNR components
        signal_power = 1e-3  # 0 dBm reference
        self.snr_ase_db = 10 * np.log10(signal_power / (ase_power + 1e-15))
        self.snr_nli_db = 10 * np.log10(signal_power / (nli_power + 1e-15))
        self.snr_icxt_db = 10 * np.log10(signal_power / (icxt_power + 1e-15))

class GSNRCalculator:
    """
    GSNR Calculator integrating Steps 1-7
    Provides GSNR calculation for MCF EON resource allocation
    """
    
    def __init__(self, mcf_config, band_config):
        """
        Initialize GSNR calculator with MCF and band configurations
        
        Args:
            mcf_config: MCF configuration from MCF4CoreCLBandConfig
            band_config: Band configuration dictionary
        """
        self.mcf_config = mcf_config
        self.band_config = band_config
        
        # Extract channel information
        self.channels = mcf_config.channels
        self.frequencies_hz = [ch['frequency_hz'] for ch in self.channels]
        self.wavelengths_nm = [ch['wavelength_nm'] for ch in self.channels]
        self.num_channels = len(self.channels)
        
        # Physical parameters
        self.frequency_params = mcf_config.frequency_dependent_params
        
        # Modulation format thresholds and bitrates
        self.modulation_thresholds_db = {
            'PM-BPSK': 3.45,
            'PM-QPSK': 6.5,
            'PM-8QAM': 8.4,
            'PM-16QAM': 12.4,
            'PM-32QAM': 16.5,
            'PM-64QAM': 19.3
        }
        
        self.modulation_bitrates_gbps = {
            'PM-BPSK': 100,
            'PM-QPSK': 200,
            'PM-8QAM': 300,
            'PM-16QAM': 400,
            'PM-32QAM': 500,
            'PM-64QAM': 600
        }
        
        # Cache for calculated results
        self.gsnr_cache = {}
        
        print(f"GSNR Calculator initialized:")
        print(f"  MCF: {mcf_config.mcf_params.num_cores} cores, {mcf_config.mcf_params.core_pitch_um} μm pitch")
        print(f"  Channels: {self.num_channels} ({len([ch for ch in self.channels if ch['band']=='C'])} C-band, {len([ch for ch in self.channels if ch['band']=='L'])} L-band)")
    
    def create_span_configuration_for_step1(self, path_links: List, total_launch_power_dbm: float = 0.0) -> Dict:
        """
        Create span configuration for Step 1 based on path links
        
        Args:
            path_links: List of link objects in the path
            total_launch_power_dbm: Total launch power per channel (dBm)
            
        Returns:
            Configuration dictionary for Step 1
        """
        # Calculate total path length
        total_length_km = sum(link.length_km for link in path_links)
        
        # For simplification, treat entire path as single span
        # In practice, each link would be processed separately
        
        # Create input power configuration
        # Uniform loading across all channels
        launch_power_per_channel_w = 10**(total_launch_power_dbm / 10) * 1e-3  # Convert dBm to W
        channel_powers_w = np.full(self.num_channels, launch_power_per_channel_w)
        
        # Create scenario for Step 1
        scenario = {
            'name': f'path_length_{total_length_km:.1f}km',
            'description': f'Path with {len(path_links)} links, total {total_length_km:.1f} km',
            'channel_powers_w': channel_powers_w,
            'transponder_channels': [0, 1],  # First two channels as transponders
            'ase_channels': list(range(2, self.num_channels)),
            'channel_types': ['transponder', 'transponder'] + ['ase'] * (self.num_channels - 2)
        }
        
        # System parameters based on MCF configuration
        system_parameters = {
            'frequencies_hz': self.frequencies_hz,
            'wavelengths_nm': self.wavelengths_nm,
            'num_channels': self.num_channels,
            'alpha_db_km': [self.frequency_params['loss_coefficient_db_km'][freq] 
                           for freq in self.frequencies_hz],
            'span_length_km': total_length_km,
            'mcf_parameters': self.mcf_config.mcf_params.__dict__
        }
        
        return {
            'scenario': scenario,
            'system_parameters': system_parameters,
            'span_length_km': total_length_km
        }
    
    def run_step1_power_evolution(self, path_links: List, launch_power_dbm: float = 0.0) -> Dict:
        """
        Run Step 1 power evolution calculation
        
        Args:
            path_links: List of link objects
            launch_power_dbm: Launch power per channel (dBm)
            
        Returns:
            Step 1 results
        """
        try:
            # Create split-step ground truth generator
            generator = SplitStepGroundTruthGenerator()
            
            # Create configuration
            config = self.create_span_configuration_for_step1(path_links, launch_power_dbm)
            
            # Run simulation for the scenario
            results = generator.simulate_multi_span_system(config['scenario'])
            
            # Add system parameters
            results['system_parameters'] = config['system_parameters']
            results['span_length_km'] = config['span_length_km']
            
            # Extract final power evolution data
            power_evolution_w = np.array(results['cumulative_evolution']['power_evolution_w'])
            distances_m = np.array(results['cumulative_evolution']['distances_km']) * 1000  # Convert to meters
            
            results['power_evolution_w'] = power_evolution_w
            results['distances_m'] = distances_m
            results['initial_powers_w'] = power_evolution_w[0, :] if power_evolution_w.size > 0 else np.zeros(self.num_channels)
            results['final_powers_w'] = power_evolution_w[-1, :] if power_evolution_w.size > 0 else np.zeros(self.num_channels)
            
            return results
            
        except Exception as e:
            print(f"Error in Step 1: {e}")
            # Return fallback results
            return self._create_fallback_step1_results(path_links, launch_power_dbm)
    
    def _create_fallback_step1_results(self, path_links: List, launch_power_dbm: float) -> Dict:
        """Create fallback Step 1 results if import fails"""
        total_length_km = sum(link.length_km for link in path_links)
        launch_power_w = 10**(launch_power_dbm / 10) * 1e-3
        
        # Simple exponential decay model
        final_powers_w = np.zeros(self.num_channels)
        for i, freq_hz in enumerate(self.frequencies_hz):
            loss_db_km = self.frequency_params['loss_coefficient_db_km'][freq_hz]
            loss_linear = 10**(-loss_db_km * total_length_km / 10)
            final_powers_w[i] = launch_power_w * loss_linear
        
        return {
            'system_parameters': {
                'frequencies_hz': self.frequencies_hz,
                'wavelengths_nm': self.wavelengths_nm,
                'num_channels': self.num_channels,
                'alpha_db_km': [self.frequency_params['loss_coefficient_db_km'][freq] 
                               for freq in self.frequencies_hz]
            },
            'span_length_km': total_length_km,
            'power_evolution_w': np.array([np.full(self.num_channels, launch_power_w), final_powers_w]),
            'distances_m': np.array([0, total_length_km * 1000]),
            'initial_powers_w': np.full(self.num_channels, launch_power_w),
            'final_powers_w': final_powers_w,
            'fallback_mode': True
        }
    
    def run_step2_parameter_fitting(self, step1_results: Dict) -> Dict:
        """Run Step 2 parameter fitting"""
        try:
            fitter = Step2_ParameterFitting(step1_results)
            results = fitter.step2_fit_all_channels(m_c=2.0)
            return results
        except Exception as e:
            print(f"Error in Step 2: {e}")
            return self._create_fallback_step2_results(step1_results)
    
    def _create_fallback_step2_results(self, step1_results: Dict) -> Dict:
        """Create fallback Step 2 results"""
        alpha_intrinsic = step1_results['system_parameters']['alpha_db_km']
        alpha_linear = [a * np.log(10) / (10 * 1000) for a in alpha_intrinsic]  # Convert to 1/m
        
        return {
            'step2_parameter_fitting': True,
            'alpha0_per_m': alpha_linear,
            'alpha1_per_m': [0.1 * a for a in alpha_linear],  # 10% ISRS effect
            'sigma_per_m': [2 * a for a in alpha_linear],
            'frequencies_hz': step1_results['system_parameters']['frequencies_hz'],
            'fallback_mode': True
        }
    
    def run_step3_amplifier_gains(self, step1_results: Dict, step2_results: Dict) -> Dict:
        """Run Step 3 amplifier gains calculation"""
        try:
            amp_config = AmplifierConfiguration(
                target_power_dbm=0.0,
                power_control_mode="constant_gain",
                max_gain_db=25.0
            )
            calculator = Step3_AmplifierGains(step1_results, step2_results, amp_config)
            results = calculator.step3_calculate_amplifier_gains()
            return results
        except Exception as e:
            print(f"Error in Step 3: {e}")
            return self._create_fallback_step3_results(step1_results)
    
    def _create_fallback_step3_results(self, step1_results: Dict) -> Dict:
        """Create fallback Step 3 results"""
        initial_powers = step1_results['initial_powers_w']
        final_powers = step1_results['final_powers_w']
        
        # Calculate required gains
        gains_linear = []
        gains_db = []
        for i in range(len(initial_powers)):
            if final_powers[i] > 0:
                gain_linear = initial_powers[i] / final_powers[i]
                gain_db = 10 * np.log10(gain_linear)
            else:
                gain_linear = 100.0  # 20 dB default
                gain_db = 20.0
            
            gains_linear.append(gain_linear)
            gains_db.append(gain_db)
        
        return {
            'step3_amplifier_gains': True,
            'gain_values_linear': gains_linear,
            'gain_values_db': gains_db,
            'fallback_mode': True
        }
    
    def run_steps_4567_gsnr_computation(self, step1_results: Dict, step2_results: Dict, 
                                       step3_results: Dict, channel_index: int, 
                                       core_index: int) -> Dict:
        """Run Steps 4-7 GSNR computation"""
        try:
            # MCF configuration for ICXT calculation
            mcf_config_dict = {
                'num_cores': self.mcf_config.mcf_params.num_cores,
                'core_pitch_um': self.mcf_config.mcf_params.core_pitch_um,
                'cladding_diameter_um': self.mcf_config.mcf_params.cladding_diameter_um,
                'core_radius_um': self.mcf_config.mcf_params.core_radius_um,
                'trench_width_ratio': self.mcf_config.mcf_params.trench_width_ratio,
                'bending_radius_mm': self.mcf_config.mcf_params.bending_radius_mm,
                'ncore': self.mcf_config.mcf_params.core_refractive_index
            }
            
            calculator = Step4567_GSNRComputation(step1_results, step2_results, step3_results)
            results = calculator.run_complete_gsnr_computation(mcf_config_dict)
            
            # Extract results for specific channel
            gsnr_db = results['step7_gsnr_results']['GSNR_db'][channel_index]
            
            # Extract noise components for the channel
            ase_power = results['step4_ase_results']['P_ASE_per_channel'][channel_index]
            nli_power = results['step5_nli_results']['P_NLI_per_channel'][channel_index]
            
            if results['step6_icxt_results']:
                icxt_power = results['step6_icxt_results']['P_ICXT_total_per_channel'][channel_index]
            else:
                icxt_power = 0.0
            
            return {
                'gsnr_db': gsnr_db,
                'ase_power': ase_power,
                'nli_power': nli_power,
                'icxt_power': icxt_power,
                'full_results': results
            }
            
        except Exception as e:
            print(f"Error in Steps 4-7: {e}")
            return self._create_fallback_gsnr_results(step1_results, channel_index, core_index)
    
    def _create_fallback_gsnr_results(self, step1_results: Dict, channel_index: int, 
                                    core_index: int) -> Dict:
        """Create fallback GSNR results"""
        # Simple GSNR estimation based on distance and frequency
        span_length_km = step1_results['span_length_km']
        frequency_hz = self.frequencies_hz[channel_index]
        
        # Basic penalties
        distance_penalty_db = span_length_km * 0.01  # 0.01 dB/km
        frequency_penalty_db = abs(frequency_hz - 193.4e12) / 1e12 * 0.5  # Frequency-dependent
        
        # Base GSNR
        base_gsnr_db = 25.0 - distance_penalty_db - frequency_penalty_db
        
        # ICXT penalty for MCF
        num_adjacent = self.mcf_config.get_adjacent_cores_count(core_index)
        icxt_penalty_db = num_adjacent * 0.5  # 0.5 dB per adjacent core
        
        gsnr_db = base_gsnr_db - icxt_penalty_db
        
        # Estimate noise components (rough approximation)
        signal_power_w = 1e-3  # 0 dBm
        noise_power_w = signal_power_w / (10**(gsnr_db / 10))
        
        return {
            'gsnr_db': gsnr_db,
            'ase_power': noise_power_w * 0.5,  # ASE dominates
            'nli_power': noise_power_w * 0.3,  # NLI contribution
            'icxt_power': noise_power_w * 0.2,  # ICXT contribution
            'fallback_mode': True
        }
    
    def calculate_gsnr(self, path_links: List, channel_index: int, core_index: int,
                      launch_power_dbm: float = 0.0, use_cache: bool = True) -> GSNRCalculationResult:
        """
        Calculate GSNR for specific path, channel, and core
        
        Args:
            path_links: List of link objects in the path
            channel_index: Channel index
            core_index: Core index
            launch_power_dbm: Launch power per channel (dBm)
            use_cache: Whether to use cached results
            
        Returns:
            GSNRCalculationResult object
        """
        # Create cache key
        path_key = tuple(link.link_id for link in path_links)
        cache_key = (path_key, channel_index, core_index, launch_power_dbm)
        
        if use_cache and cache_key in self.gsnr_cache:
            return self.gsnr_cache[cache_key]
        
        print(f"Calculating GSNR for channel {channel_index}, core {core_index}, path length {sum(link.length_km for link in path_links):.1f} km")
        
        # Step 1: Power evolution
        step1_results = self.run_step1_power_evolution(path_links, launch_power_dbm)
        
        # Step 2: Parameter fitting
        step2_results = self.run_step2_parameter_fitting(step1_results)
        
        # Step 3: Amplifier gains
        step3_results = self.run_step3_amplifier_gains(step1_results, step2_results)
        
        # Steps 4-7: GSNR computation
        gsnr_results = self.run_steps_4567_gsnr_computation(
            step1_results, step2_results, step3_results, channel_index, core_index
        )
        
        # Determine supported modulation format
        gsnr_db = gsnr_results['gsnr_db']
        supported_modulation = 'None'
        max_bitrate_gbps = 0
        
        for mod_format in ['PM-64QAM', 'PM-32QAM', 'PM-16QAM', 'PM-8QAM', 'PM-QPSK', 'PM-BPSK']:
            if gsnr_db >= self.modulation_thresholds_db[mod_format]:
                supported_modulation = mod_format
                max_bitrate_gbps = self.modulation_bitrates_gbps[mod_format]
                break
        
        # Calculate OSNR (simplified conversion)
        osnr_db = gsnr_db + 3.0  # Approximate OSNR from GSNR
        
        # Create result object
        result = GSNRCalculationResult(
            channel_index=channel_index,
            core_index=core_index,
            path_links=path_links,
            gsnr_db=gsnr_db,
            osnr_db=osnr_db,
            supported_modulation=supported_modulation,
            max_bitrate_gbps=max_bitrate_gbps,
            ase_power=gsnr_results['ase_power'],
            nli_power=gsnr_results['nli_power'],
            icxt_power=gsnr_results['icxt_power']
        )
        
        # Cache result
        if use_cache:
            self.gsnr_cache[cache_key] = result
        
        return result
    
    def batch_calculate_gsnr(self, path_links: List, channel_indices: List[int],
                           core_indices: List[int], launch_power_dbm: float = 0.0) -> List[GSNRCalculationResult]:
        """
        Calculate GSNR for multiple channel-core combinations
        
        Args:
            path_links: List of link objects
            channel_indices: List of channel indices
            core_indices: List of core indices
            launch_power_dbm: Launch power per channel (dBm)
            
        Returns:
            List of GSNRCalculationResult objects
        """
        results = []
        
        for channel_idx in channel_indices:
            for core_idx in core_indices:
                try:
                    result = self.calculate_gsnr(path_links, channel_idx, core_idx, launch_power_dbm)
                    results.append(result)
                except Exception as e:
                    print(f"Error calculating GSNR for channel {channel_idx}, core {core_idx}: {e}")
                    continue
        
        return results
    
    def clear_cache(self):
        """Clear GSNR calculation cache"""
        self.gsnr_cache.clear()
    
    def get_cache_statistics(self) -> Dict:
        """Get cache statistics"""
        return {
            'cache_size': len(self.gsnr_cache),
            'cache_keys': list(self.gsnr_cache.keys())[:10] if self.gsnr_cache else []  # First 10 keys
        }

# Example usage
if __name__ == "__main__":
    # This would normally be imported from the MCF configuration
    print("GSNR Calculator - requires MCF configuration and network links for testing")
    print("Use within the main MCF EON simulator framework")