"""Spatial evaluator for ice cover spatial patterns.

Evaluates spatial pattern similarity between ice cover datasets using:
- SSIM (Structural Similarity Index)
- Spearman and Kendall Tau-B correlations  
- Phenology timing anomaly maps (FUS/BUS)

Uses custom SSIM implementation to avoid scikit-image dependency.
Implements 3-source data range computation for fair comparison across models.
"""

from typing import Dict, Any
import warnings
import sys
import io
import numpy as np
import pandas as pd
from esda.moran import Moran_Local
from libpysal.weights import lat2W
from libpysal.weights import W
from src.utils.constants import EVAL_START, EVAL_END, BREAKUP_SEASON, FREEZEUP_SEASON
from src.eval.utils import spatial_break_up, spatial_freeze_up
from src.eval.evaluators import BaseEvaluator

class SpatialEvaluator(BaseEvaluator):
    """
    Evaluator for spatial pattern metrics on phenology timing anomalies.
    
    Implements 3-source data range computation to ensure fair comparison
    between different models (LIF-DL vs FLake) against observations (IMS).
    
    Note: Not compatible with CIS data (non-spatial FIC timeseries).
    """
    
    name = "spatial"
    requires_spatial = True  # CIS data not supported
    
    @staticmethod
    def calculate_anomaly_map(data, mask):
        """
        Process ice cover data into average timing anomaly maps.
        
        Args:
            data: xarray Dataset with 'time' coordinate and ice cover variable
            mask: 2D numpy array (height, width) - lake mask (1=lake, 0=land)
            
        Returns:
            Dictionary containing:
            - 'FUS': 2D anomaly map for Freeze-Up Start (averaged across years)
            - 'BUS': 2D anomaly map for Break-Up Start (averaged across years)
        """
        start_year = pd.to_datetime(EVAL_START).year
        end_year = pd.to_datetime(EVAL_END).year
        
        # Storage for yearly phenology maps
        fus_yearly = []
        bus_yearly = []
        
        for year in range(start_year, end_year + 1):
            # === Freeze-Up Start (FUS) ===
            freezeup_start = pd.Timestamp(f"{year}-{FREEZEUP_SEASON['start_month']:02d}-{FREEZEUP_SEASON['start_day']:02d}")
            freezeup_end = pd.Timestamp(f"{year + 1}-{FREEZEUP_SEASON['end_month']:02d}-{FREEZEUP_SEASON['end_day']:02d}")
            offset_fz = freezeup_start.day_of_year
            
            # Get freeze-up season data
            freezeup_period = data.sel(time=slice(freezeup_start, freezeup_end))
            
            if len(freezeup_period.time) == 0:
                continue
            
            # Calculate Freeze Onset spatial map
            freeze_onset, _ = spatial_freeze_up(
                freezeup_period.values, mask, offset=offset_fz
            )
            fus_yearly.append(freeze_onset)
            
            # === Break-Up Start (BUS) ===
            breakup_start = pd.Timestamp(f"{year}-{BREAKUP_SEASON['start_month']:02d}-{BREAKUP_SEASON['start_day']:02d}")
            breakup_end = pd.Timestamp(f"{year}-{BREAKUP_SEASON['end_month']:02d}-{BREAKUP_SEASON['end_day']:02d}")
            offset_bu = breakup_start.day_of_year
            
            # Get break-up season data
            breakup_period = data.sel(time=slice(breakup_start, breakup_end))
            
            if len(breakup_period.time) == 0:
                continue
            
            # Calculate First Open Water spatial map
            first_open_water, _ = spatial_break_up(
                breakup_period.values, mask, offset=offset_bu
            )
            bus_yearly.append(first_open_water)
        
        # Average across years (suppress warnings for empty slices - expected for some lakes/years)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Mean of empty slice.*')
            fus_avg = np.nanmean(np.array(fus_yearly), axis=0)
            bus_avg = np.nanmean(np.array(bus_yearly), axis=0)
        
        # Calculate anomalies (deviation from mean timing)
        fus_anomaly = fus_avg - np.nanmean(fus_avg)
        bus_anomaly = bus_avg - np.nanmean(bus_avg)
        
        return {
            'FUS': fus_anomaly,
            'BUS': bus_anomaly
        }
     
    def preprocess(
        self,
        context,  # EvaluationContext
        obs_name: str,
        pred_name: str
    ) -> Dict[str, Any]:
        """
        Preprocess spatial data with 3-source data range computation.
        
        Process:
        1. Access ALL spatial sources (IMS, LIF-DL, FLake) from context
        2. Compute phenology anomalies for each source
        3. Calculate unified data ranges across all 3 sources
        4. Return preprocessed anomalies and ranges for the requested comparison
        
        Args:
            context: EvaluationContext with all sources
            obs_name: Observation source ('ims')
            pred_name: Prediction source ('lif_dl' or 'flake')
        
        Returns:
            Dictionary with:
            - 'obs_anomalies': Dict with 'FUS' and 'BUS' anomaly maps for obs
            - 'pred_anomalies': Dict with 'FUS' and 'BUS' anomaly maps for pred
            - 'data_ranges': Dict with 'FUS' and 'BUS' unified data ranges
            - 'mask': Lake mask
        """
        # Validate we have spatial sources
        if not context.has_source('ims'):
            raise ValueError("Spatial evaluator requires 'ims' source")
        if not context.has_source('lif_dl'):
            raise ValueError("Spatial evaluator requires 'lif_dl' source")
        if not context.has_source('flake'):
            raise ValueError("Spatial evaluator requires 'flake' source")
        
        mask = context.mask
        
        # Compute phenology anomalies for ALL 3 sources (for unified range)
        ims_anomalies = SpatialEvaluator.calculate_anomaly_map(
            context.get_source('ims'), mask
        )
        lifdl_anomalies = SpatialEvaluator.calculate_anomaly_map(
            context.get_source('lif_dl'), mask
        )
        flake_anomalies = SpatialEvaluator.calculate_anomaly_map(
            context.get_source('flake'), mask
        )
        
        # Compute unified data ranges across ALL 3 sources (99th percentile)
        data_ranges = {}
        for event in ['FUS', 'BUS']:
            combined_abs = np.abs(np.stack([
                ims_anomalies[event],
                lifdl_anomalies[event],
                flake_anomalies[event]
            ]))
            data_ranges[event] = float(np.nanpercentile(combined_abs, 99))
        
        # Select the specific sources for this comparison
        all_anomalies = {
            'ims': ims_anomalies,
            'lif_dl': lifdl_anomalies,
            'flake': flake_anomalies
        }
        
        return {
            'obs_anomalies': all_anomalies[obs_name],
            'pred_anomalies': all_anomalies[pred_name],
            'data_ranges': data_ranges,
            'mask': mask
        }
    
    @staticmethod
    def _compute_local_ssim(window_a, window_b, window_mask, data_range=1.0, K1=0.01, K2=0.03):
        """
        Compute SSIM over a masked window (1D array of valid pixels).
        
        Based on Wang et al. 2004: "Image Quality Assessment: From Error Visibility 
        to Structural Similarity"
        
        Args:
            window_a: First window (flattened valid pixels)
            window_b: Second window (flattened valid pixels)
            window_mask: Boolean mask of valid pixels
            data_range: Range of data values (default 1.0 for binary ice)
            K1, K2: Stability constants
            
        Returns:
            SSIM value for this window
        """
        valid = window_mask
        if np.sum(valid) < 4:  # Need at least 4 valid pixels
            return np.nan

        a = window_a[valid]
        b = window_b[valid]

        if a.size == 0 or b.size == 0:
            return np.nan

        mean_a = np.mean(a)
        mean_b = np.mean(b)
        var_a = np.var(a, ddof=1)
        var_b = np.var(b, ddof=1)
        
        # Handle edge case where variance calculation might fail
        if len(a) < 2:
            return np.nan
            
        cov_ab = np.cov(a, b, ddof=1)[0, 1]

        C1 = (K1 * data_range) ** 2
        C2 = (K2 * data_range) ** 2

        numerator = (2 * mean_a * mean_b + C1) * (2 * cov_ab + C2)
        denominator = (mean_a**2 + mean_b**2 + C1) * (var_a + var_b + C2)

        if denominator == 0:
            return np.nan

        return numerator / denominator
    
    @staticmethod
    def _masked_ssim(a, b, mask, window_size=7, min_valid=3, data_range=1.0):
        """
        Compute mean SSIM between two 2D images with lake mask.
        
        Args:
            a: First image (2D array)
            b: Second image (2D array)
            mask: Lake mask (1=lake, 0=land)
            window_size: Size of sliding window (default 7x7)
            min_valid: Minimum valid pixels per window
            data_range: Range of data values
            
        Returns:
            Mean SSIM across all valid windows
        """
        if a.shape != b.shape:
            raise ValueError("Shapes of a, b must match.")

        H, W = a.shape
        half_w = window_size // 2

        # Pad arrays to handle edges
        pad_width = half_w
        a_pad = np.pad(a, pad_width, mode='constant', constant_values=np.nan)
        b_pad = np.pad(b, pad_width, mode='constant', constant_values=np.nan)
        mask_pad = np.pad(mask, pad_width, mode='constant', constant_values=False)

        count_valid_windows = 0
        ssim_sum = 0

        # Slide window
        for i in range(H):
            for j in range(W):
                # Only compute SSIM if center pixel is valid (in lake)
                if not mask[i, j]:
                    continue
                
                win_a = a_pad[i:i+window_size, j:j+window_size]
                win_b = b_pad[i:i+window_size, j:j+window_size]
                win_mask = mask_pad[i:i+window_size, j:j+window_size]

                # Only consider pixels in this window that are valid
                win_valid = win_mask & ~np.isnan(win_a) & ~np.isnan(win_b)
                if np.sum(win_valid) < min_valid:
                    continue

                local_ssim = SpatialEvaluator._compute_local_ssim(
                    win_a, win_b, win_valid, data_range
                )

                if not np.isnan(local_ssim):
                    ssim_sum += local_ssim
                    count_valid_windows += 1

        if count_valid_windows == 0:
            return np.nan
        else:
            return ssim_sum / count_valid_windows

    @staticmethod
    def _kendall_tau_b(a, b, mask):
        """
        Compute the Kendall Tau-b correlation coefficient for two masked 2D arrays.
        
        Parameters:
            a (np.ndarray): First 2D array.
            b (np.ndarray): Second 2D array.
            mask (np.ndarray): Boolean mask indicating valid pixels.
        """
        valid = mask & ~np.isnan(a) & ~np.isnan(b)
        if np.sum(valid) < 2:
            return np.nan

        a_valid = a[valid]
        b_valid = b[valid]

        if a_valid.size == 0 or b_valid.size == 0:
            return np.nan

        # Use pandas to compute Kendall Tau-b
        return pd.Series(a_valid).corr(pd.Series(b_valid), method='kendall')

    @staticmethod
    def _local_moran_i(data, mask):
        """
        Compute Local Moran's I for spatial autocorrelation analysis.
        
        Args:
            data: 2D array of data values
            mask: 2D boolean array indicating valid pixels
        
        Returns:
            2D array of Local Moran's I cluster labels (1=HH, 2=LL, 3=HL, 4=LH) with non-significant values masked out (set to NaN).
        """
        
        if np.sum(mask) < 10:
            return np.nan

        residuals_valid = data[mask]
        
        # Create spatial weights matrix (4-neighbourhood)
        rows, cols = data.shape
        w = lat2W(rows, cols, rook=True)
        
        # Filter weights to only include valid pixels
        valid_indices = np.where(mask.flatten())[0]
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
        neighbors = {}
        weights_dict = {}
        
        # Build filtered weights based on valid pixels - land excluded
        for new_idx, old_idx in enumerate(valid_indices):
            original_neighbors = w.neighbors.get(old_idx, [])
            filtered_neighbors = [index_map[n] for n in original_neighbors if n in index_map]
            neighbors[new_idx] = filtered_neighbors
            weights_dict[new_idx] = [1.0] * len(filtered_neighbors)
        
        # Suppress warnings about disconnected components and islands
        # These are expected when working with irregular lake shapes
        # Also suppress stdout messages from libpysal
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            # Redirect stdout to suppress print statements from libpysal
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                w_filtered = W(neighbors, weights_dict)
                
                # Calculate Local Moran's I
                moran_local = Moran_Local(residuals_valid, w_filtered, permutations=999)
            finally:
                # Restore stdout
                sys.stdout = old_stdout
        
        # Cast back to 2D array for storage
        # local_moran_map = np.full(data.shape, np.nan)
        # local_moran_map[mask] = moran_local.Is
        local_moran_p = np.full(data.shape, np.nan)
        local_moran_p[mask] = moran_local.p_sim
        local_moran_q = np.full(data.shape, np.nan)
        local_moran_q[mask] = moran_local.q
        
        # Mask out non-significant values (p >= 0.05) by setting to 0, but only over lake pixels
        ns_mask = (local_moran_p >= 0.05) & (mask)
        local_moran_q[ns_mask] = 0
        
        return local_moran_q

    def _lmi_iou(self, a_q, b_q):
        """
        Calculate the weighted Intersection over Union (IoU) between two Local Moran's I maps.
        IoU is calculated for each cluster type (High-High, Low-Low, High-Low, Low-High), and then
        combined into a weighted average IoU.
        
        Args:
            a_q: First Local Moran's I cluster labels (2D array)
            b_q: Second Local Moran's I cluster labels (2D array)
            
        Returns:
            Weighted IoU value across all cluster types.
        """
        
        # Calculate IoU for each cluster type and accumulate weighted sum
        weighted_sum = 0
        total_weight = 0
        for cluster_type in [1, 2, 3, 4]:  # HH=1, LL=2, HL=3, LH=4
            a_mask = (a_q == cluster_type)
            b_mask = (b_q == cluster_type)
            
            intersection = np.logical_and(a_mask, b_mask).sum()
            union = np.logical_or(a_mask, b_mask).sum()
            
            if union == 0:
                iou = np.nan
            else:
                iou = intersection / union
                weighted_sum += iou * union
                total_weight += union

        if total_weight == 0:
            return np.nan
        
        weighted_iou = weighted_sum / total_weight
        return weighted_iou

    def evaluate(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute spatial pattern metrics on preprocessed phenology anomalies.
        
        Uses unified data ranges computed from all 3 sources to ensure
        fair comparison between models.
        
        Args:
            preprocessed: Dictionary from preprocess() with:
                - 'obs_anomalies': Observation anomaly maps (FUS, BUS)
                - 'pred_anomalies': Prediction anomaly maps (FUS, BUS)
                - 'data_ranges': Unified data ranges (FUS, BUS)
                - 'mask': Lake mask
            
        Returns:
            Dictionary with spatial metrics for FUS and BUS events:
            - FUS_SSIM: SSIM for freeze-up start anomaly map
            - FUS_Kendall_Tau_B: Kendall Tau-B for FUS
            - BUS_SSIM: SSIM for break-up start anomaly map
            - BUS_Kendall_Tau_B: Kendall Tau-B for BUS
        """
        obs_anomalies = preprocessed['obs_anomalies']
        pred_anomalies = preprocessed['pred_anomalies']
        data_ranges = preprocessed['data_ranges']
        mask = preprocessed['mask']
        
        results = {}
        
        # Process each event (FUS and BUS)
        for event in ['FUS', 'BUS']:
            obs_anom = obs_anomalies[event]
            pred_anom = pred_anomalies[event]
            data_range = data_ranges[event]
            
            # Handle edge case where data_range is 0
            if data_range == 0 or np.isnan(data_range):
                results[f"{event}_SSIM"] = np.nan
                results[f"{event}_Kendall_Tau_B"] = np.nan
                results[f"{event}_LMI_IoU"] = np.nan
                continue
            
            # Normalize anomalies by unified data range
            obs_scaled = obs_anom / data_range
            pred_scaled = pred_anom / data_range
            
            # Create valid pixel mask (lake pixels with non-NaN values in both)
            valid_mask = (mask == 1) & ~np.isnan(obs_scaled) & ~np.isnan(pred_scaled)
            
            # Compute SSIM on scaled anomalies
            ssim_value = self._masked_ssim(
                obs_scaled, pred_scaled, valid_mask,
                window_size=7, min_valid=3, data_range=1.0
            )
            
            # Compute Kendall Tau-B
            kendall = self._kendall_tau_b(
                obs_scaled, pred_scaled, valid_mask
            )
            
            # Compute local Moran's I
            obs_lmi = self._local_moran_i(
                obs_anom, valid_mask
            )
            pred_lmi = self._local_moran_i(
                pred_anom, valid_mask
            )
            lmi_iou = self._lmi_iou(
                obs_lmi, pred_lmi
            )
            
            # Store results for this event
            results[f"{event}_SSIM"] = float(ssim_value) if not np.isnan(ssim_value) else np.nan
            results[f"{event}_Kendall_Tau_B"] = float(kendall) if not np.isnan(kendall) else np.nan
            results[f"{event}_LMI_IoU"] = float(lmi_iou) if not np.isnan(lmi_iou) else np.nan
        
        return results
    
    def extract_intermediate_data(self, context, source_name: str) -> Dict[str, Any]:
        """
        Extract timing maps and Local Moran's I for a single source.
        
        This method extracts spatial timing anomaly maps and LMI cluster
        classifications for both seasons, which can be saved for downstream
        figure generation.
        
        Args:
            context: EvaluationContext with all sources
            source_name: Name of source (e.g., 'ims', 'lif_dl', 'flake')
            
        Returns:
            Dictionary with:
            - bus_timing: Break-up season timing map (2D array)
            - fus_timing: Freeze-up season timing map (2D array)
            - bus_lmi_clusters: Break-up season cluster labels (2D array)
            - fus_lmi_clusters: Freeze-up season cluster labels (2D array)
            - mask: Lake mask (2D array)
            - site: Site name
            - source: Source name
        """
        source = context.get_source(source_name)
        mask = context.mask
        
        # Calculate anomaly maps for both seasons
        anomalies = self.calculate_anomaly_map(source, mask)
        
        bus_timing = anomalies['BUS']
        fus_timing = anomalies['FUS']
        
        # Calculate Local Moran's I for both seasons
        # BUS
        bus_valid_mask = (mask == 1) & ~np.isnan(bus_timing)
        bus_lmi = self._local_moran_i(bus_timing, bus_valid_mask)
        
        # FUS
        fus_valid_mask = (mask == 1) & ~np.isnan(fus_timing)
        fus_lmi = self._local_moran_i(fus_timing, fus_valid_mask)
        
        return {
            'bus_timing': bus_timing,
            'fus_timing': fus_timing,
            'bus_lmi_clusters': bus_lmi,
            'fus_lmi_clusters': fus_lmi,
            'mask': mask,
            'site': context.site,
            'source': source_name
        }

