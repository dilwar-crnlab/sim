#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCF Parameters - 4-core C+L Band Configuration
Physical parameters for 4-core Multi-Core Fiber based on research papers
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import math

@dataclass
class BandConfiguration:
    """Optical band configuration for C+L band systems"""
    name: str
    start_frequency_hz: float
    end_frequency_hz: float
    channel_spacing_hz: float
    num_channels: int
    amplifier_type: str
    noise_figure_db: float

@dataclass
class MCFPhysicalParameters:
    """Physical parameters of Multi-Core Fiber"""
    # Core configuration
    num_cores: int
    core_layout: str  # "square", "hexagonal"
    core_pitch_um: float  # Distance between core centers (μm)
    core_radius_um: float  # Core radius (μm)
    
    # Cladding parameters
    cladding_diameter_um: float
    cladding_thickness_um: float
    
    # Trench parameters (for trench-assisted MCF)
    trench_width_ratio: float  # wtr/r1
    trench_depth: float  # Relative refractive index difference

    
    
    # Optical parameters
    core_refractive_index: float
    cladding_refractive_index: float
    core_cladding_delta: float  # Relative refractive index difference
    
    # Mechanical parameters
    bending_radius_mm: float
    coating_diameter_um: float

    trench_width_um: float = 6.75
    numerical_aperture: float = 0.12

    def get_trench_parameters_for_icxt(self) -> Dict[str, float]:
        return {
            'w_tr_m': self.trench_width_um * 1e-6,
            'r_1_m': self.core_radius_um * 1e-6,
            'Lambda_m': self.core_pitch_um * 1e-6,
            'r_b_m': self.bending_radius_mm * 1e-3,
            'n_core': self.core_refractive_index,
            'Delta_1': self.core_cladding_delta
        }


# Add these to config.py to eliminate ALL duplications

@dataclass
class ModulationConfiguration:
    """Modulation format configuration with thresholds and bitrates"""
    format_name: str
    cardinality: int
    threshold_db: float  # GSNR threshold in dB
    bitrate_gbps: int    # Bitrate at 64 GBaud

@dataclass
class AmplifierConfiguration:
    """Amplifier configuration parameters"""
    target_power_dbm: float = 0.0
    max_gain_db: float = 25.0
    min_gain_db: float = 10.0
    c_band_noise_figure_db: float = 4.5
    l_band_noise_figure_db: float = 5.0
    power_control_mode: str = "constant_gain"

    
class MCF4CoreCLBandConfig:
    """Configuration for 4-core C+L band Multi-Core Fiber EON"""
    
    def __init__(self):
        # Physical constants
        self.c_light = 299792458  # Speed of light (m/s)
        self.h_planck = 6.626e-34  # Planck constant (J⋅s)
        
        # Setup 4-core MCF parameters
        self.mcf_params = self._setup_4core_mcf_parameters()
        
        # Setup C+L band configuration
        self.band_configs = self._setup_cl_band_configuration()
        
        # Generate channel grid
        self.channels = self._generate_channel_grid()
        
        # Calculate frequency-dependent parameters
        self.frequency_dependent_params = self._calculate_frequency_dependent_parameters()

        # ✅ ADD MODULATION CONFIGURATIONS
        self.modulation_configs = self._setup_modulation_configurations()
        
        # ✅ ADD AMPLIFIER CONFIGURATION
        self.amplifier_config = self._setup_amplifier_configuration()
        
        # ✅ ADD PHYSICAL CONSTANTS
        self.physical_constants = self._setup_physical_constants()

    def _setup_modulation_configurations(self) -> Dict[str, ModulationConfiguration]:
        """Setup modulation format configurations"""
        return {
            'PM-BPSK': ModulationConfiguration('PM-BPSK', 1, 3.45, 100),
            'PM-QPSK': ModulationConfiguration('PM-QPSK', 2, 6.5, 200),
            'PM-8QAM': ModulationConfiguration('PM-8QAM', 3, 8.4, 300),
            'PM-16QAM': ModulationConfiguration('PM-16QAM', 4, 12.4, 400),
            'PM-32QAM': ModulationConfiguration('PM-32QAM', 5, 16.5, 500),
            'PM-64QAM': ModulationConfiguration('PM-64QAM', 6, 19.3, 600)
        }
    
    def _setup_amplifier_configuration(self) -> AmplifierConfiguration:
        """Setup amplifier configuration"""
        return AmplifierConfiguration(
            target_power_dbm=0.0,
            max_gain_db=25.0,
            min_gain_db=10.0,
            c_band_noise_figure_db=4.5,
            l_band_noise_figure_db=5.0,
            power_control_mode="constant_gain"
        )
    
    def _setup_physical_constants(self) -> Dict[str, float]:
        """Setup physical constants"""
        return {
            'c_light': 299792458,      # Speed of light (m/s)
            'h_planck': 6.626e-34,     # Planck constant (J⋅s)
            'symbol_rate_hz': 64e9,    # Symbol rate (Hz)
            'reference_power_dbm': 0.0 # Reference launch power
        }
    
    def get_modulation_thresholds_db(self) -> Dict[str, float]:
        """Get modulation format GSNR thresholds"""
        return {name: config.threshold_db for name, config in self.modulation_configs.items()}
    
    def get_modulation_bitrates_gbps(self) -> Dict[str, int]:
        """Get modulation format bitrates"""
        return {name: config.bitrate_gbps for name, config in self.modulation_configs.items()}
    
    def get_supported_modulation_format(self, gsnr_db: float) -> Tuple[str, int]:
        """
        Get highest supported modulation format for given GSNR
        
        Args:
            gsnr_db: GSNR in dB
            
        Returns:
            Tuple of (modulation_format_name, bitrate_gbps)
        """
        for mod_format in ['PM-64QAM', 'PM-32QAM', 'PM-16QAM', 'PM-8QAM', 'PM-QPSK', 'PM-BPSK']:
            if gsnr_db >= self.modulation_configs[mod_format].threshold_db:
                return mod_format, self.modulation_configs[mod_format].bitrate_gbps
        
        return 'None', 0

        
    def _setup_4core_mcf_parameters(self) -> MCFPhysicalParameters:
        """Setup 4-core MCF parameters based on research papers"""
        return MCFPhysicalParameters(
            # Core configuration - 4 cores in square layout
            num_cores=4,
            core_layout="square",
            core_pitch_um=43.0,  # From paper: 43 μm core pitch for ultra-low ICXT
            core_radius_um=4.5,  # Standard core radius
            
            # Cladding parameters - standard diameter
            cladding_diameter_um=125.0,  # Standard SMF cladding diameter
            cladding_thickness_um=40.0,  # Sufficient for low bending loss
            
            # Trench parameters - trench-assisted design
            trench_width_ratio=1.5,  # wtr/r1 = 1.5 from paper analysis
            trench_depth=-0.002,  # -0.2% refractive index difference
            trench_width_um=6.75,  # ADD this line
            numerical_aperture=0.12,  # ADD this line
            
            # Optical parameters
            core_refractive_index=1.4504,  # At 1550nm
            cladding_refractive_index=1.4444,  # Standard SMF values
            core_cladding_delta=0.0042,  # ~0.42% index difference
            
            # Mechanical parameters
            bending_radius_mm=144.0,  # 144mm bending radius from papers
            coating_diameter_um=245.0  # Standard coating
        )
    
    def _setup_cl_band_configuration(self) -> Dict[str, BandConfiguration]:
        """Setup C+L band configuration"""
        # C-band: 191.4 - 196.1 THz (1565 - 1530 nm)
        c_band = BandConfiguration(
            name="C",
            start_frequency_hz=191.4e12,
            end_frequency_hz=196.1e12,
            channel_spacing_hz=100e9,  # 100 GHz spacing
            num_channels=47,  # 4.7 THz / 100 GHz
            amplifier_type="EDFA",
            noise_figure_db=4.5
        )
        
        # L-band: 186.1 - 190.8 THz (1625 - 1565 nm)
        l_band = BandConfiguration(
            name="L",
            start_frequency_hz=186.1e12,
            end_frequency_hz=190.8e12,
            channel_spacing_hz=100e9,  # 100 GHz spacing
            num_channels=47,  # 4.7 THz / 100 GHz
            amplifier_type="EDFA_L",
            noise_figure_db=5.0
        )
        
        return {"C": c_band, "L": l_band}
    
    def _generate_channel_grid(self) -> List[Dict]:
        """Generate complete channel grid for C+L bands"""
        channels = []
        channel_index = 0
        
        # Generate L-band channels first (lower frequencies)
        l_band = self.band_configs["L"]
        for i in range(l_band.num_channels):
            frequency_hz = l_band.start_frequency_hz + i * l_band.channel_spacing_hz
            wavelength_nm = self.c_light / frequency_hz * 1e9
            
            channels.append({
                'index': channel_index,
                'frequency_hz': frequency_hz,
                'frequency_thz': frequency_hz / 1e12,
                'wavelength_nm': wavelength_nm,
                'band': 'L',
                'channel_in_band': i,
                'spacing_hz': l_band.channel_spacing_hz
            })
            channel_index += 1
        
        # Generate C-band channels
        c_band = self.band_configs["C"]
        for i in range(c_band.num_channels):
            frequency_hz = c_band.start_frequency_hz + i * c_band.channel_spacing_hz
            wavelength_nm = self.c_light / frequency_hz * 1e9
            
            channels.append({
                'index': channel_index,
                'frequency_hz': frequency_hz,
                'frequency_thz': frequency_hz / 1e12,
                'wavelength_nm': wavelength_nm,
                'band': 'C',
                'channel_in_band': i,
                'spacing_hz': c_band.channel_spacing_hz
            })
            channel_index += 1
        
        return channels
    
    def _calculate_frequency_dependent_parameters(self) -> Dict:
        """Calculate frequency-dependent fiber parameters"""
        params = {
            'effective_area_um2': {},
            'loss_coefficient_db_km': {},
            'dispersion_ps_nm_km': {},
            'nonlinear_coefficient_w_km': {},
            'mode_coupling_coefficient': {},
            'power_coupling_coefficient': {}
        }
        
        for channel in self.channels:
            freq_hz = channel['frequency_hz']
            wavelength_nm = channel['wavelength_nm']
            
            # Effective area (frequency dependent)
            params['effective_area_um2'][freq_hz] = self._calculate_effective_area(wavelength_nm)
            
            # Loss coefficient (frequency dependent)
            params['loss_coefficient_db_km'][freq_hz] = self._calculate_loss_coefficient(wavelength_nm)
            
            # Dispersion parameter
            params['dispersion_ps_nm_km'][freq_hz] = self._calculate_dispersion(wavelength_nm)
            
            # Nonlinear coefficient
            params['nonlinear_coefficient_w_km'][freq_hz] = self._calculate_nonlinear_coefficient(
                wavelength_nm, params['effective_area_um2'][freq_hz]
            )
            
            # ICXT parameters for 4-core MCF
            params['mode_coupling_coefficient'][freq_hz] = self._calculate_mode_coupling_coefficient(freq_hz)
            params['power_coupling_coefficient'][freq_hz] = self._calculate_power_coupling_coefficient(freq_hz)
        
        return params
    
    def _calculate_effective_area(self, wavelength_nm: float) -> float:
        """Calculate effective area (μm²) vs wavelength"""
        # Based on standard SMF with slight wavelength dependence
        # Aeff(λ) ≈ 80 + 0.05 * (λ - 1550)  [μm²]
        aeff = 80.0 + 0.05 * (wavelength_nm - 1550.0)
        return max(70.0, min(90.0, aeff))  # Reasonable bounds
    
    def _calculate_loss_coefficient(self, wavelength_nm: float) -> float:
        """Calculate loss coefficient (dB/km) vs wavelength"""
        # Standard SMF loss profile
        if wavelength_nm < 1530:  # L-band region
            # Higher loss in L-band due to Rayleigh scattering
            loss = 0.22 + 0.001 * (1530 - wavelength_nm)
        elif wavelength_nm < 1565:  # C-band region
            # Minimum loss region
            loss = 0.19 + 0.0001 * abs(wavelength_nm - 1550)
        else:  # Extended C-band
            loss = 0.20 + 0.0002 * (wavelength_nm - 1565)
        
        return max(0.18, min(0.25, loss))
    
    def _calculate_dispersion(self, wavelength_nm: float) -> float:
        """Calculate chromatic dispersion (ps/nm/km)"""
        # Standard SMF dispersion formula
        # D(λ) = (λ/4) * d²n/dλ² where zero dispersion at 1308 nm
        lambda_0 = 1308.0  # Zero dispersion wavelength (nm)
        S0 = 0.092  # Dispersion slope (ps/nm²/km)
        
        dispersion = (wavelength_nm / 4) * S0 * (1 - (lambda_0 / wavelength_nm)**4)
        return dispersion
    
    def _calculate_nonlinear_coefficient(self, wavelength_nm: float, aeff_um2: float) -> float:
        """Calculate nonlinear coefficient γ (1/W/km)"""
        n2 = 2.6e-20  # Nonlinear refractive index (m²/W)
        aeff_m2 = aeff_um2 * 1e-12  # Convert μm² to m²
        
        # γ = 2πn₂/(λAeff)
        gamma = (2 * math.pi * n2) / (wavelength_nm * 1e-9 * aeff_m2) / 1000  # 1/W/km
        return gamma
    
    def _calculate_mode_coupling_coefficient(self, frequency_hz: float) -> float:
        """Calculate mode coupling coefficient κ(f) for ICXT (Equation 3)"""
        # Parameters for 4-core trench-assisted MCF
        r1 = self.mcf_params.core_pitch_um * 1e-6  # Core pitch (m)
        wtr = self.mcf_params.trench_width_ratio * self.mcf_params.core_radius_um * 1e-6  # Trench width (m)
        ncore = self.mcf_params.core_refractive_index
        
        # Frequency-dependent V-parameter
        wavelength_m = self.c_light / frequency_hz
        delta_1 = self.mcf_params.core_cladding_delta
        
        V1 = 2 * math.pi * frequency_hz * r1 * ncore * math.sqrt(2 * delta_1) / self.c_light
        W1 = 1.143 * V1 - 0.22
        
        if W1 <= 0:
            W1 = 0.1  # Avoid numerical issues
        
        # Simplified mode coupling coefficient calculation
        # Based on Equation (3) from the paper
        U1_squared = (2 * math.pi * frequency_hz * r1 * ncore / self.c_light)**2 * (delta_1**2 - 1)
        K1_W1 = math.sqrt(math.pi / (2 * W1)) * math.exp(-W1)
        
        # Trench factor
        Gamma = W1 / (W1 + 1.2 * (1 + V1) * wtr / r1)
        
        # Mode coupling coefficient
        kappa = (math.sqrt(0.11 / r1) * U1_squared * V1**3 * K1_W1**2) / (math.sqrt(math.pi * r1) * W1**3)
        kappa *= math.exp(-(W1**3 + 1.2 * (1 + V1) * wtr / r1))
        
        return max(kappa, 1e-15)  # Avoid zero values
    
    def _calculate_power_coupling_coefficient(self, frequency_hz: float) -> float:
        """Calculate power coupling coefficient Ω(f) for ICXT (Equation 2)"""
        kappa = self._calculate_mode_coupling_coefficient(frequency_hz)
        rb = self.mcf_params.bending_radius_mm * 1e-3  # Convert to meters
        ncore = self.mcf_params.core_refractive_index
        Lambda = self.mcf_params.core_pitch_um * 1e-6  # Convert to meters
        
        # Equation (2): Ω(f) = cκ²rb*ncore / (πf*Λ)
        omega = (self.c_light * kappa**2 * rb * ncore) / (math.pi * frequency_hz * Lambda)
        
        return omega
    
    def get_icxt_threshold(self, modulation_format: str, qot_penalty_db: float = 1.0) -> float:
        """Calculate ICXT threshold for modulation format (Equation 9)"""
        # Modulation format parameters χ_m
        chi_values = {
            'PM-BPSK': 0.5,    # m=1
            'PM-QPSK': 1.0,    # m=2
            'PM-8QAM': 3.41,   # m=3
            'PM-16QAM': 5.0,   # m=4
            'PM-32QAM': 10.0,  # m=5
            'PM-64QAM': 21.0   # m=6
        }
        
        # GSNR thresholds (dB)
        gsnr_thresholds = {
            'PM-BPSK': 3.45,
            'PM-QPSK': 6.5,
            'PM-8QAM': 8.4,
            'PM-16QAM': 12.4,
            'PM-32QAM': 16.5,
            'PM-64QAM': 19.3
        }
        
        if modulation_format not in chi_values:
            raise ValueError(f"Unknown modulation format: {modulation_format}")
        
        chi_m = chi_values[modulation_format]
        Gth_db = gsnr_thresholds[modulation_format]
        Gth_linear = 10**(Gth_db / 10)
        Gamma_linear = 10**(qot_penalty_db / 10)
        
        # Equation (9): μ_ICXT_th = 10*log10((1 - 10^(-Γ/10)) / (χ_m * 10^(Gth/10)))
        numerator = 1 - (1 / Gamma_linear)
        denominator = chi_m * Gth_linear
        
        mu_icxt_th_linear = numerator / denominator
        mu_icxt_th_db = 10 * math.log10(max(mu_icxt_th_linear, 1e-10))
        
        return mu_icxt_th_db
    
    def get_adjacent_cores_count(self, core_index: int) -> int:
        """Get number of adjacent cores for 4-core square layout"""
        # 4-core square layout: each core has 2 adjacent cores
        # Core layout:
        #  0 -- 1
        #  |    |
        #  3 -- 2
        adjacency_map = {
            0: [1, 3],  # Core 0 adjacent to cores 1 and 3
            1: [0, 2],  # Core 1 adjacent to cores 0 and 2
            2: [1, 3],  # Core 2 adjacent to cores 1 and 3
            3: [0, 2]   # Core 3 adjacent to cores 0 and 2
        }
        
        if core_index in adjacency_map:
            return len(adjacency_map[core_index])
        return 0
    
    def get_configuration_summary(self) -> Dict:
        """Get complete configuration summary"""
        return {
            'mcf_type': '4-core C+L band',
            'mcf_parameters': self.mcf_params.__dict__,
            'band_configurations': {k: v.__dict__ for k, v in self.band_configs.items()},
            'total_channels': len(self.channels),
            'c_band_channels': len([ch for ch in self.channels if ch['band'] == 'C']),
            'l_band_channels': len([ch for ch in self.channels if ch['band'] == 'L']),
            'frequency_range_thz': [
                min(ch['frequency_thz'] for ch in self.channels),
                max(ch['frequency_thz'] for ch in self.channels)
            ],
            'wavelength_range_nm': [
                min(ch['wavelength_nm'] for ch in self.channels),
                max(ch['wavelength_nm'] for ch in self.channels)
            ],
            'total_bandwidth_thz': sum(self.band_configs[band].channel_spacing_hz for band in self.band_configs) * len(self.channels) / 1e12
        }

# Example usage and testing
if __name__ == "__main__":
    # Create 4-core C+L band configuration
    config = MCF4CoreCLBandConfig()
    
    # Print configuration summary
    summary = config.get_configuration_summary()
    print("4-Core C+L Band MCF Configuration:")
    print("=" * 50)
    print(f"MCF Type: {summary['mcf_type']}")
    print(f"Number of cores: {config.mcf_params.num_cores}")
    print(f"Core pitch: {config.mcf_params.core_pitch_um} μm")
    print(f"Total channels: {summary['total_channels']}")
    print(f"C-band channels: {summary['c_band_channels']}")
    print(f"L-band channels: {summary['l_band_channels']}")
    print(f"Frequency range: {summary['frequency_range_thz'][0]:.1f} - {summary['frequency_range_thz'][1]:.1f} THz")
    print(f"Wavelength range: {summary['wavelength_range_nm'][1]:.1f} - {summary['wavelength_range_nm'][0]:.1f} nm")
    
    # Test ICXT thresholds
    print("\nICXT Thresholds:")
    for mod_format in ['PM-BPSK', 'PM-QPSK', 'PM-16QAM', 'PM-64QAM']:
        threshold = config.get_icxt_threshold(mod_format)
        print(f"  {mod_format}: {threshold:.2f} dB")
    
    # Test frequency-dependent parameters for a few channels
    print("\nSample Frequency-Dependent Parameters:")
    test_channels = [0, len(config.channels)//4, len(config.channels)//2, 3*len(config.channels)//4, -1]
    for ch_idx in test_channels:
        ch = config.channels[ch_idx]
        freq = ch['frequency_hz']
        print(f"  Channel {ch['index']} ({ch['band']}-band, {ch['frequency_thz']:.2f} THz):")
        print(f"    Effective area: {config.frequency_dependent_params['effective_area_um2'][freq]:.1f} μm²")
        print(f"    Loss: {config.frequency_dependent_params['loss_coefficient_db_km'][freq]:.3f} dB/km")
        print(f"    Dispersion: {config.frequency_dependent_params['dispersion_ps_nm_km'][freq]:.2f} ps/nm/km")