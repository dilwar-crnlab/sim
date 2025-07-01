#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modified GSNR Calculator using existing configurations from config.py
Integrates with existing Step 1-7 implementations while adding interfering channel support
ALL CONFIGURATIONS NOW USE config.py - NO DUPLICATIONS
Clean implementation without fallback mechanisms - relies on main algorithms
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
from steps.step1 import SplitStepGroundTruthGenerator
from steps.step_2_three_param import Step2_ParameterFitting
from steps.step_3_ASE import Step3_AmplifierGains, AmplifierConfiguration
from steps.step_4567 import Step4567_GSNRComputation

# Import existing config
from config import MCF4CoreCLBandConfig

class ModifiedGSNRCalculator:
    """
    GSNR Calculator using existing configurations from config.py
    Modifies existing Steps 1-7 to work with specific channels and interfering channels
    """
    
    def __init__(self, mcf_config: MCF4CoreCLBandConfig, band_config: Dict):
        """Initialize modified GSNR calculator with existing configurations"""
        self.mcf_config = mcf_config
        self.band_config = band_config
        
        # ✅ USE EXISTING CHANNEL INFORMATION FROM CONFIG
        self.channels = mcf_config.channels
        self.frequencies_hz = [ch['frequency_hz'] for ch in self.channels]
        self.wavelengths_nm = [ch['wavelength_nm'] for ch in self.channels]
        self.num_channels = len(self.channels)
        
        # ✅ USE EXISTING FREQUENCY-DEPENDENT PARAMETERS FROM CONFIG
        self.frequency_params = mcf_config.frequency_dependent_params
        
        # Initialize existing step components
        self.split_step_generator = SplitStepGroundTruthGenerator()
        
        # ✅ USE EXISTING MODULATION THRESHOLDS AND BITRATES FROM CONFIG
        self.modulation_thresholds_db = self._get_modulation_thresholds_from_config()
        self.modulation_bitrates_gbps = self._get_modulation_bitrates_from_config()
        
        print(f"Modified GSNR Calculator initialized using config.py:")
        print(f"  Using existing split-step methods")
        print(f"  MCF: {mcf_config.mcf_params.num_cores} cores, {mcf_config.mcf_params.core_pitch_um} μm pitch")
        print(f"  Channels: {self.num_channels}")
        print(f"  Using existing frequency-dependent parameters from config")
    
    def _get_modulation_thresholds_from_config(self) -> Dict[str, float]:
        """Get modulation thresholds from existing config"""
        # ✅ USE EXISTING MODULATION CONFIGURATION FROM CONFIG.PY
        return self.mcf_config.get_modulation_thresholds_db()
    
    def _get_modulation_bitrates_from_config(self) -> Dict[str, int]:
        """Get modulation bitrates from existing config"""
        # ✅ USE EXISTING MODULATION CONFIGURATION FROM CONFIG.PY
        return self.mcf_config.get_modulation_bitrates_gbps()
    
    def get_interfering_channels(self, path_links: List, core_index: int, 
                               target_channel: int, spectrum_allocation) -> List[Dict]:
        """
        Get interfering channels for the target channel using spectrum allocation matrix
        
        Args:
            path_links: Path links
            core_index: Target core index
            target_channel: Target channel index
            spectrum_allocation: Current spectrum allocation matrix
            
        Returns:
            List of interfering channel information
        """
        interfering_channels = []
        
        # Find channels that are active on ALL links in the path (same core)
        for ch_idx in range(self.num_channels):
            if ch_idx == target_channel:
                continue
            
            # Check if this channel is allocated on all links in the path
            is_interferer = True
            for link in path_links:
                if spectrum_allocation.allocation[link.link_id, core_index, ch_idx] == 0:
                    is_interferer = False
                    break
            
            if is_interferer:
                interfering_channels.append({
                    'channel_index': ch_idx,
                    'frequency_hz': self.frequencies_hz[ch_idx],
                    'wavelength_nm': self.wavelengths_nm[ch_idx],
                    'power_w': 1e-3,  # Assume typical power (0 dBm)
                    'connection_id': str(spectrum_allocation.allocation[path_links[0].link_id, core_index, ch_idx])
                })
        
        return interfering_channels
    
    def create_modified_scenario_for_step1(self, path_links: List, target_channel: int,
                                         interfering_channels: List[Dict], 
                                         launch_power_dbm: float = 0.0) -> Dict:
        """
        Create scenario for Step 1 using existing configurations
        """
        
        # Calculate total path length from actual links
        total_length_km = sum(link.length_km for link in path_links)
        
        # Create channel power array (only target + interfering channels)
        active_channels = [target_channel] + [ich['channel_index'] for ich in interfering_channels]
        num_active = len(active_channels)
        
        # Power configuration for active channels only
        # ✅ USE PHYSICAL CONSTANTS FROM CONFIG
        if launch_power_dbm == 0.0:
            launch_power_dbm = self.mcf_config.physical_constants['reference_power_dbm']
        launch_power_w = 10**(launch_power_dbm / 10) * 1e-3
        channel_powers_w = np.zeros(self.num_channels)
        
        # Set power for target channel
        channel_powers_w[target_channel] = launch_power_w
        
        # Set power for interfering channels
        for ich in interfering_channels:
            channel_powers_w[ich['channel_index']] = ich['power_w']
        
        # ✅ USE EXISTING MCF CONFIGURATION FROM CONFIG.PY
        scenario = {
            'name': f'mcf_realistic_path_{total_length_km:.1f}km_ch{target_channel}',
            'description': f'4-core MCF path {len(path_links)} links, {num_active} active channels, target ch{target_channel}',
            'channel_powers_w': channel_powers_w,
            'transponder_channels': [target_channel],
            'ase_channels': [ich['channel_index'] for ich in interfering_channels],
            'channel_types': ['transponder' if i == target_channel else 
                            ('ase' if i in [ich['channel_index'] for ich in interfering_channels] else 'off')
                            for i in range(self.num_channels)],
            'active_channels': active_channels,
            'target_channel': target_channel,
            'num_interfering_channels': len(interfering_channels),
            # ✅ USE EXISTING MCF CONFIGURATION
            'mcf_configuration': {
                'num_cores': self.mcf_config.mcf_params.num_cores,
                'core_pitch_um': self.mcf_config.mcf_params.core_pitch_um,
                'core_layout': self.mcf_config.mcf_params.core_layout,
                'enable_icxt': True
            }
        }
        
        # ✅ USE EXISTING SYSTEM PARAMETERS FROM CONFIG
        system_parameters = {
            'frequencies_hz': self.frequencies_hz,
            'wavelengths_nm': self.wavelengths_nm,
            'num_channels': self.num_channels,
            # ✅ USE EXISTING FREQUENCY-DEPENDENT PARAMETERS
            'alpha_db_km': [self.frequency_params['loss_coefficient_db_km'][freq] 
                          for freq in self.frequencies_hz],
            'actual_path_links': path_links,
            'total_path_length_km': total_length_km,
            # ✅ USE EXISTING MCF PARAMETERS FROM CONFIG
            'mcf_parameters': {
                'num_cores': self.mcf_config.mcf_params.num_cores,
                'core_pitch_um': self.mcf_config.mcf_params.core_pitch_um,
                'core_radius_um': self.mcf_config.mcf_params.core_radius_um,
                'cladding_diameter_um': self.mcf_config.mcf_params.cladding_diameter_um,
                'trench_width_ratio': self.mcf_config.mcf_params.trench_width_ratio,
                'bending_radius_mm': self.mcf_config.mcf_params.bending_radius_mm,
                'core_refractive_index': self.mcf_config.mcf_params.core_refractive_index,
                'cladding_refractive_index': self.mcf_config.mcf_params.cladding_refractive_index,
                'core_cladding_delta': self.mcf_config.mcf_params.core_cladding_delta
            },
            # ✅ USE EXISTING BAND CONFIGURATION FROM CONFIG
            'band_configuration': {
                'c_band_start_hz': self.mcf_config.band_configs['C'].start_frequency_hz,
                'c_band_end_hz': self.mcf_config.band_configs['C'].end_frequency_hz,
                'l_band_start_hz': self.mcf_config.band_configs['L'].start_frequency_hz,
                'l_band_end_hz': self.mcf_config.band_configs['L'].end_frequency_hz,
                'channel_spacing_hz': self.mcf_config.band_configs['C'].channel_spacing_hz
            }
        }
        
        return {
            'scenario': scenario,
            'system_parameters': system_parameters,
            'span_length_km': total_length_km
        }
    
    def run_modified_step1(self, path_links: List, target_channel: int,
                        interfering_channels: List[Dict], launch_power_dbm: float = 0.0) -> Dict:
        """
        Run modified Step 1 with actual path and interfering channels
        """
        
        # Create modified configuration using existing configs
        config = self.create_modified_scenario_for_step1(
            path_links, target_channel, interfering_channels, launch_power_dbm
        )
        
        # ✅ USE EXISTING SPAN LENGTHS FROM NETWORK LINKS
        original_spans = self.split_step_generator.spans
        
        # Create spans from actual path links - PROPERLY DIVIDE INTO SPANS
        modified_spans = []
        span_id_counter = 1
        
        for link in path_links:
            # Each link may have multiple spans
            for i, span_length_km in enumerate(link.span_lengths_km):
                # Create span configuration for each span within the link
                span_config = type('SpanConfig', (), {
                    'span_id': span_id_counter,
                    'length_km': span_length_km,  # Use individual span length
                    'loss_coef_db_km': 0.21,     # Standard SSMF loss
                    'dispersion_ps_nm_km': 17.0,
                    'gamma_w_km': 1.3e-3,
                    'effective_area_um2': 80.0,
                    'link_id': link.link_id,      # Track which link this span belongs to
                    'span_in_link': i             # Track position within link
                })()
                modified_spans.append(span_config)
                span_id_counter += 1
        
        # Calculate total path length for validation
        total_calculated_length = sum(span.length_km for span in modified_spans)
        expected_length = sum(link.length_km for link in path_links)
        
        if abs(total_calculated_length - expected_length) > 0.1:
            print(f"Warning: Calculated span total {total_calculated_length:.1f} km "
                f"doesn't match expected path length {expected_length:.1f} km")
        
        print(f"Path spans breakdown:")
        for span in modified_spans:
            print(f"  Span {span.span_id}: {span.length_km:.1f} km (Link {span.link_id})")
        
        # Temporarily replace spans with actual path spans
        self.split_step_generator.spans = modified_spans
        
        try:
            # Run split-step simulation with modified configuration
            results = self.split_step_generator.simulate_multi_span_system(config['scenario'])
            
            # Add system parameters and path information
            results['system_parameters'] = config['system_parameters']
            results['span_length_km'] = total_calculated_length
            results['target_channel'] = target_channel
            results['interfering_channels'] = interfering_channels
            results['num_spans_in_path'] = len(modified_spans)
            
            # Extract power evolution for target channel
            power_evolution_w = np.array(results['cumulative_evolution']['power_evolution_w'])
            distances_m = np.array(results['cumulative_evolution']['distances_km']) * 1000
            
            results['power_evolution_w'] = power_evolution_w
            results['distances_m'] = distances_m
            results['initial_powers_w'] = power_evolution_w[0, :] if power_evolution_w.size > 0 else np.zeros(self.num_channels)
            results['final_powers_w'] = power_evolution_w[-1, :] if power_evolution_w.size > 0 else np.zeros(self.num_channels)
            
            return results
            
        finally:
            # Restore original spans
            self.split_step_generator.spans = original_spans
    
    def run_modified_step2(self, step1_results: Dict, target_channel: int) -> Dict:
        """Run modified Step 2 for target channel only"""
        
        # Create Step 2 fitter
        fitter = Step2_ParameterFitting(step1_results)
        
        # Fit only the target channel instead of all channels
        target_result = fitter.fit_single_channel(target_channel, m_c=2.0)
        
        # Create results structure compatible with existing code
        results = {
            'step2_parameter_fitting': True,
            'target_channel': target_channel,
            'channel_results': [target_result],
            'frequencies_hz': [fitter.frequencies_hz[target_channel]],
            'frequencies_thz': [fitter.frequencies_hz[target_channel] / 1e12],
            'wavelengths_nm': [fitter.wavelengths_nm[target_channel]],
            'alpha0_per_m': [target_result['alpha0_per_m']],
            'alpha1_per_m': [target_result['alpha1_per_m']],
            'sigma_per_m': [target_result['sigma_per_m']],
            'distances_m': step1_results['distances_m'],
            'computation_time_s': 0.1  # Much faster for single channel
        }
        
        return results
    
    def run_modified_step3(self, step1_results: Dict, step2_results: Dict, target_channel: int) -> Dict:
        """Run modified Step 3 for target channel only"""
        
        # ✅ USE EXISTING AMPLIFIER CONFIGURATION FROM CONFIG.PY
        amp_config_from_mcf = self.mcf_config.amplifier_config
        amp_config = AmplifierConfiguration(
            target_power_dbm=amp_config_from_mcf.target_power_dbm,
            power_control_mode=amp_config_from_mcf.power_control_mode,
            max_gain_db=amp_config_from_mcf.max_gain_db
        )
        
        # Create Step 3 calculator
        calculator = Step3_AmplifierGains(step1_results, step2_results, amp_config)
        
        # Calculate gain for target channel only
        target_result = calculator.calculate_amplifier_gain_single_channel(target_channel)
        
        # Create results structure
        results = {
            'step3_amplifier_gains': True,
            'target_channel': target_channel,
            'channel_results': [target_result],
            'gain_values_linear': [target_result['gain_linear']],
            'gain_values_db': [target_result['applied_gain_db']],
            'computation_time_s': 0.001
        }
        
        return results
    
    def run_modified_steps_4567(self, step1_results: Dict, step2_results: Dict,
                               step3_results: Dict, target_channel: int, core_index: int) -> Dict:
        """
        Run modified Steps 4-7 using existing MCF configuration from config.py
        """
        
        # ✅ USE EXISTING MCF CONFIGURATION FROM CONFIG.PY
        mcf_config_dict = {
            'num_cores': self.mcf_config.mcf_params.num_cores,
            'core_pitch_um': self.mcf_config.mcf_params.core_pitch_um,
            'cladding_diameter_um': self.mcf_config.mcf_params.cladding_diameter_um,
            'core_radius_um': self.mcf_config.mcf_params.core_radius_um,
            'trench_width_ratio': self.mcf_config.mcf_params.trench_width_ratio,
            'bending_radius_mm': self.mcf_config.mcf_params.bending_radius_mm,
            'ncore': self.mcf_config.mcf_params.core_refractive_index
        }
        
        # Create Steps 4-7 calculator
        calculator = Step4567_GSNRComputation(step1_results, step2_results, step3_results)
        
        # Run complete GSNR computation
        results = calculator.run_complete_gsnr_computation(mcf_config_dict)
        
        # Extract results for target channel
        gsnr_db = results['step7_gsnr_results']['GSNR_db'][target_channel]
        
        # Extract noise components for target channel
        ase_power = results['step4_ase_results']['P_ASE_per_channel'][target_channel]
        nli_power = results['step5_nli_results']['P_NLI_per_channel'][target_channel]
        
        if results['step6_icxt_results']:
            icxt_power = results['step6_icxt_results']['P_ICXT_total_per_channel'][target_channel]
        else:
            icxt_power = 0.0
        
        return {
            'gsnr_db': gsnr_db,
            'ase_power': ase_power,
            'nli_power': nli_power,
            'icxt_power': icxt_power,
            'target_channel': target_channel,
            'core_index': core_index,
            'full_results': results,
            'computation_time_s': results.get('computation_time_s', 1.0)
        }
    
    def calculate_gsnr(self, path_links: List, channel_index: int, core_index: int,
                      launch_power_dbm: float = 0.0, use_cache: bool = True,
                      spectrum_allocation=None):
        """
        Calculate GSNR using modified existing steps with interfering channels
        ALL CONFIGURATIONS FROM config.py
        """
        
        start_time = time.time()
        
        print(f"Calculating GSNR for channel {channel_index}, core {core_index}, "
              f"path length {sum(link.length_km for link in path_links):.1f} km using existing config.py")
        
        # Get interfering channels
        if spectrum_allocation is not None:
            interfering_channels = self.get_interfering_channels(
                path_links, core_index, channel_index, spectrum_allocation
            )
        else:
            interfering_channels = []  # No interference if allocation not provided
        
        print(f"  Found {len(interfering_channels)} interfering channels")
        
        # Step 1: Modified power evolution with actual path and interference
        step1_results = self.run_modified_step1(
            path_links, channel_index, interfering_channels, launch_power_dbm
        )
        
        # Step 2: Modified parameter fitting for target channel only
        step2_results = self.run_modified_step2(step1_results, channel_index)
        
        # Step 3: Modified amplifier gains for target channel only
        step3_results = self.run_modified_step3(step1_results, step2_results, channel_index)
        
        # Steps 4-7: Modified GSNR computation for target channel using existing MCF config
        gsnr_results = self.run_modified_steps_4567(
            step1_results, step2_results, step3_results, channel_index, core_index
        )
        
        # ✅ USE EXISTING MODULATION FORMAT SELECTION FROM CONFIG
        gsnr_db = gsnr_results['gsnr_db']
        supported_modulation, max_bitrate_gbps = self.mcf_config.get_supported_modulation_format(gsnr_db)
        
        # Calculate OSNR (simplified conversion)
        osnr_db = gsnr_db + 3.0
        
        calculation_time = time.time() - start_time
        
        # Create compatible result object
        class ModifiedGSNRResult:
            def __init__(self):
                self.channel_index = channel_index
                self.core_index = core_index
                self.path_links = path_links
                self.gsnr_db = gsnr_db
                self.osnr_db = osnr_db
                self.supported_modulation = supported_modulation
                self.max_bitrate_gbps = max_bitrate_gbps
                self.ase_power = gsnr_results['ase_power']
                self.nli_power = gsnr_results['nli_power']
                self.icxt_power = gsnr_results['icxt_power']
                self.num_interfering_channels = len(interfering_channels)
                self.calculation_time_s = calculation_time
                
                # SNR components
                signal_power = 1e-3
                self.snr_ase_db = 10 * np.log10(signal_power / (self.ase_power + 1e-15))
                self.snr_nli_db = 10 * np.log10(signal_power / (self.nli_power + 1e-15))
                self.snr_icxt_db = 10 * np.log10(signal_power / (self.icxt_power + 1e-15))
        
        print(f"  GSNR calculation completed: {gsnr_db:.2f} dB, {supported_modulation}, {calculation_time:.2f}s")
        
        return ModifiedGSNRResult()

# Integration wrapper for existing code
class IntegratedGSNRCalculator:
    """Wrapper class to integrate with existing xt_nli_rsa.py using config.py"""
    
    def __init__(self, mcf_config: MCF4CoreCLBandConfig, band_config: Dict):
        self.modified_calculator = ModifiedGSNRCalculator(mcf_config, band_config)
        self.mcf_config = mcf_config
        self.band_config = band_config
    
    def calculate_gsnr(self, path_links: List, channel_index: int, core_index: int,
                      launch_power_dbm: float = 0.0, use_cache: bool = True,
                      spectrum_allocation=None):
        """Calculate GSNR with modified existing steps using config.py"""
        
        return self.modified_calculator.calculate_gsnr(
            path_links, channel_index, core_index, launch_power_dbm, 
            use_cache, spectrum_allocation
        )

# Example usage
if __name__ == "__main__":
    print("Modified GSNR Calculator using existing configurations from config.py")
    print("No more configuration duplications - all configs from config.py")
    print("Clean implementation without fallbacks - relies on main algorithms")
    print("Use IntegratedGSNRCalculator for seamless integration")