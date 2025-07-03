#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean GSNR Integration Interface
Simple wrapper to integrate the clean GSNR calculator with existing codebase
Replaces the modified_gsnr_steps.py functionality with clean implementation
"""

from typing import List, Dict, Optional
import numpy as np
from clean_gsnr_calculator import CleanGSNRCalculator, GSNRResult

class CleanGSNRIntegration:
    """
    Integration wrapper for clean GSNR calculator
    Compatible interface with existing XT_NLIA_RSA.py code
    """
    
    def __init__(self, mcf_config, band_config: Dict = None):
        """
        Initialize clean GSNR integration
        
        Args:
            mcf_config: MCF4CoreCLBandConfig instance from config.py
            band_config: Optional band config (not used - kept for compatibility)
        """
        self.calculator = CleanGSNRCalculator(mcf_config)
        self.mcf_config = mcf_config
        
        print(f"Clean GSNR Integration initialized - using exact paper methodology")
    
    def calculate_gsnr(self, path_links: List, channel_index: int, core_index: int,
                      launch_power_dbm: float = 0.0, use_cache: bool = False,
                      link_wise_interference: Dict[int, List[int]] = None,
                      spectrum_allocation=None) -> 'CompatibleGSNRResult':
        """
        Calculate GSNR with compatible interface
        
        Args:
            path_links: List of path link objects
            channel_index: Target channel index
            core_index: Target core index
            launch_power_dbm: Launch power in dBm
            use_cache: Cache flag (ignored - kept for compatibility)
            link_wise_interference: Dict mapping link_id to list of interfering channels
            spectrum_allocation: Spectrum allocation matrix (optional)
            
        Returns:
            Compatible GSNR result object
        """
        
        # Extract interfering channels
        interfering_channels = []
        
        if link_wise_interference:
            # Use the most restrictive interference (worst case)
            all_interferers = set()
            for link_id, interferers in link_wise_interference.items():
                all_interferers.update(interferers)
            interfering_channels = list(all_interferers)
        
        elif spectrum_allocation is not None:
            # Extract from spectrum allocation matrix
            interfering_channels = self._extract_interfering_channels(
                path_links, core_index, channel_index, spectrum_allocation
            )
        
        # Calculate GSNR using clean implementation
        result = self.calculator.calculate_gsnr(
            path_links=path_links,
            channel_index=channel_index,
            core_index=core_index,
            launch_power_dbm=launch_power_dbm,
            interfering_channels=interfering_channels
        )
        
        # Return compatible result object
        return CompatibleGSNRResult(result)
    
    def _extract_interfering_channels(self, path_links: List, core_index: int, 
                                    target_channel: int, spectrum_allocation) -> List[int]:
        """Extract interfering channels from spectrum allocation matrix"""
        
        interfering_channels = []
        
        # Find channels that are active on ALL links in the path (same core)
        for ch_idx in range(len(self.mcf_config.channels)):
            if ch_idx == target_channel:
                continue
            
            # Check if this channel is allocated on all links in the path
            is_interferer = True
            for link in path_links:
                if spectrum_allocation.allocation[link.link_id, core_index, ch_idx] == 0:
                    is_interferer = False
                    break
            
            if is_interferer:
                interfering_channels.append(ch_idx)
        
        return interfering_channels
    
    def clear_cache(self):
        """Clear cache (no-op for compatibility)"""
        pass

class CompatibleGSNRResult:
    """
    Compatible GSNR result wrapper
    Provides same interface as existing code expects
    """
    
    def __init__(self, clean_result: GSNRResult):
        """Initialize from clean GSNR result"""
        
        # Map clean result to compatible interface
        self.channel_index = clean_result.channel_index
        self.core_index = clean_result.core_index
        self.gsnr_db = clean_result.gsnr_db
        self.osnr_db = clean_result.osnr_db
        self.supported_modulation = clean_result.supported_modulation
        self.max_bitrate_gbps = clean_result.max_bitrate_gbps
        
        # Noise components
        self.ase_power = clean_result.ase_power_w
        self.nli_power = clean_result.nli_power_w
        self.icxt_power = clean_result.icxt_power_w
        
        # SNR components
        self.snr_ase_db = clean_result.snr_ase_db
        self.snr_nli_db = clean_result.snr_nli_db
        self.snr_icxt_db = clean_result.snr_icxt_db
        
        # Additional metrics
        self.path_length_km = clean_result.path_length_km
        self.num_interfering_channels = clean_result.num_interfering_channels
        self.calculation_time_s = clean_result.calculation_time_s
        
        # Legacy compatibility attributes
        self.path_links = None  # Set externally if needed
    
    def __repr__(self):
        return (f"GSNRResult(ch={self.channel_index}, core={self.core_index}, "
                f"GSNR={self.gsnr_db:.2f}dB, {self.supported_modulation}, "
                f"{self.max_bitrate_gbps}Gbps)")

# Simple factory function for backward compatibility
def create_clean_gsnr_calculator(mcf_config, band_config: Dict = None):
    """
    Factory function to create clean GSNR calculator
    Compatible with existing integration patterns
    """
    return CleanGSNRIntegration(mcf_config, band_config)

# Direct replacement for IntegratedGSNRCalculator
class IntegratedGSNRCalculator(CleanGSNRIntegration):
    """
    Direct replacement for the existing IntegratedGSNRCalculator
    Uses clean implementation instead of legacy step-based approach
    """
    pass

# Usage example and testing
if __name__ == "__main__":
    print("Clean GSNR Integration Interface")
    print("=" * 50)
    print("This module provides a clean replacement for modified_gsnr_steps.py")
    print("Key improvements:")
    print("  ✓ Exact implementation of paper equations")
    print("  ✓ No configuration duplication")
    print("  ✓ Proper error handling")
    print("  ✓ Compatible interface with existing code")
    print("  ✓ Clean, maintainable codebase")
    print("\nIntegration:")
    print("  Replace: from modified_gsnr_steps import IntegratedGSNRCalculator")
    print("  With:    from clean_gsnr_integration import IntegratedGSNRCalculator")
    print("  All existing code will work unchanged!")