"""
Basic operations and utilities for wavelength calibration pipeline.

This module provides the CalibrationParameters class that orchestrates the entire
wavelength calibration pipeline:
  1. Loads SUMER spectral data
  2. Performs multi-gaussian fits on spectral lines
  3. Extracts line centroids
  4. Fits a calibration line to convert pixels to wavelength
  5. Returns calibration parameters
"""

import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits
import sys
import os
import warnings
warnings.filterwarnings('ignore')


def _mask_all_defective_pixels_DetA(array_to_mask):
    """
    Mask all defective pixels detected in SUMER detector A.
    Use when the spectrum array has been flipped in y-axis.
    """
    array_to_mask_copy = np.copy(array_to_mask)
    
    # Defective pixels in the detector frame (list of [x,y] positions)
    defects_xy_px_0 = [[210, 255], [213, 254], [213, 253], [213, 252], [212, 255], [212, 254], [212, 253], [212, 252], [212, 251], [212, 250], [211, 255], [211, 254], [211, 253], [211, 252], [211, 251], [211, 250], [210, 256], [210, 255], [210, 254], [210, 253], [210, 252], [210, 251], [210, 250], [209, 256], [209, 255], [209, 254], [209, 253], [209, 252], [209, 251], [209, 250], [208, 256], [208, 255], [208, 254], [208, 253], [208, 252], [208, 251], [208, 250], [207, 255], [207, 254], [207, 253], [207, 252], [207, 251], [207, 250], [206, 254], [206, 253], [206, 252], [206, 251], [205, 253], [205, 252]]
    defects_xy_px_1 = [[442, 8], [443, 8], [444, 8], [445, 8], [446, 8], [442, 9], [443, 9], [444, 9], [445, 9], [446, 9], [442, 10], [443, 10], [444, 10], [445, 10], [446, 10], [442, 11], [443, 11], [444, 11], [445, 11], [446, 11], [442, 12], [443, 12], [444, 12], [445, 12], [446, 12]]
    defects_xy_px_2 = [[470, 19], [471, 19], [472, 19], [473, 19], [474, 19], [470, 20], [471, 20], [472, 20], [473, 20], [474, 20], [470, 21], [471, 21], [472, 21], [473, 21], [474, 21], [470, 22], [471, 22], [472, 22], [473, 22], [474, 22], [470, 23], [471, 23], [472, 23], [473, 23], [474, 23]]
    defects_xy_px_3 = [[471, 61], [472, 61], [473, 61], [474, 61], [471, 62], [472, 62], [473, 62], [474, 62], [471, 63], [472, 63], [473, 63], [474, 63], [471, 64], [472, 64], [473, 64], [474, 64]]
    defects_xy_px_4 = [[458, 92], [459, 92], [460, 92], [461, 92], [462, 92], [463, 92], [458, 93], [459, 93], [460, 93], [461, 93], [462, 93], [463, 93], [458, 94], [459, 94], [460, 94], [461, 94], [462, 94], [463, 94], [458, 95], [459, 95], [460, 95], [461, 95], [462, 95], [463, 95], [458, 96], [459, 96], [460, 96], [461, 96], [462, 96], [463, 96], [458, 97], [459, 97], [460, 97], [461, 97], [462, 97], [463, 97]]
    defects_xy_px_5 = [[459, 88], [460, 88], [461, 88], [462, 88], [463, 88], [459, 89], [460, 89], [461, 89], [462, 89], [463, 89], [459, 90], [460, 90], [461, 90], [462, 90], [463, 90], [459, 91], [460, 91], [461, 91], [462, 91], [463, 91], [459, 92], [460, 92], [461, 92], [462, 92], [463, 92]]
    defects_xy_px_6 = [[488, 110], [489, 110], [490, 110], [491, 110], [492, 110], [493, 110], [494, 110], [488, 111], [489, 111], [490, 111], [491, 111], [492, 111], [493, 111], [494, 111], [488, 112], [489, 112], [490, 112], [491, 112], [492, 112], [493, 112], [494, 112], [488, 113], [489, 113], [490, 113], [491, 113], [492, 113], [493, 113], [494, 113], [488, 114], [489, 114], [490, 114], [491, 114], [492, 114], [493, 114], [494, 114], [488, 115], [489, 115], [490, 115], [491, 115], [492, 115], [493, 115], [494, 115], [488, 116], [489, 116], [490, 116], [491, 116], [492, 116], [493, 116], [494, 116]]
    defects_xy_px = np.concatenate([defects_xy_px_0, defects_xy_px_1, defects_xy_px_2, defects_xy_px_3, defects_xy_px_4, defects_xy_px_5, defects_xy_px_6])
    
    # Assign a rare negative value to the defective pixels
    for col_defective, row_defective in defects_xy_px:
        array_to_mask_copy[row_defective, col_defective] = -999.9
    
    # Mask the negative values
    array_masked = np.ma.masked_less(array_to_mask_copy, 0)
    return array_masked


class CalibrationParameters:
    """
    Compute wavelength calibration parameters from SUMER spectral data.
    
    This class performs the complete wavelength calibration pipeline for rows
    of SUMER spectral data. For each row:
      1. Loads SUMER spectral FITS data from files
      2. Averages all spectral images
      3. Performs multi-gaussian fits on selected spectral intervals
      4. Extracts line centroids from individual gaussians
      5. Fits a straight line: wavelength = slope * pixel + intercept
      6. Returns calibration parameters (slope, intercept, and uncertainties)
    
    The row-specific fitting parameters (initial guesses, interval ranges) are
    loaded from a configuration module to keep the code clean and parameters safe.
    
    This class is self-contained and does not depend on external auxiliary functions.
    All necessary data loading and processing is integrated internally.
    
    Parameters
    ----------
    row_start : int, default=6
        First row to process (SUMER detector rows typically 6-323)
    row_end : int, default=323
        Last row to process
    show_figures : str, default='no'
        Whether to display diagnostic plots ('yes' or 'no')
    exposure_time : float, default=150.0
        Exposure time in seconds
    factor_fullspectrum : float, default=1.0
        Scaling factor for full spectrum
    rest_wavelengths : list, default=None
        Rest wavelengths (in Angstroms) of calibration lines [1537.94, 1542.18, 1543.72, 1543.96]
    rough_pixel_estimates : list, default=None
        Rough pixel positions corresponding to rest wavelengths
    
    Attributes
    ----------
    pixelscale_list : list
        Slopes of calibration lines for each row
    pixelscale_unc_list : list
        Uncertainties of slopes
    pixelscale_intercept_list : list
        Intercepts of calibration lines for each row
    pixelscale_intercept_unc_list : list
        Uncertainties of intercepts
    
    Examples
    --------
    >>> from utils.basic_operations import CalibrationParameters
    >>> 
    >>> # Create instance and compute calibration for rows 6-50
    >>> calibrator = CalibrationParameters(row_start=6, row_end=50, show_figures='no')
    >>> calibrator.compute_calibration(
    ...     data_path='../data/soho/sumer/',
    ...     sumer_filename_list=['file1.fits', 'file2.fits', ...],
    ... )
    >>> 
    >>> # Access results
    >>> print(calibrator.pixelscale_list)
    >>> print(calibrator.pixelscale_intercept_list)
    >>> 
    >>> # Get results as arrays
    >>> slopes, slopes_unc = calibrator.get_slopes()
    >>> intercepts, intercepts_unc = calibrator.get_intercepts()
    """
    
    def __init__(
        self,
        row_start: int = 6,
        row_end: int = 323,
        show_figures: str = 'no',
        exposure_time: float = 150.0,
        factor_fullspectrum: float = 1.0,
        rest_wavelengths: list = None,
        rough_pixel_estimates: list = None,
    ):
        """Initialize calibration parameters object."""
        self.row_start = row_start
        self.row_end = row_end
        self.show_figures = show_figures
        self.exposure_time = exposure_time
        self.factor_fullspectrum = factor_fullspectrum
        
        # Default calibration line info
        self.rest_wavelengths = rest_wavelengths or [153.7935, 154.2177, 154.3724, 154.3960]
        self.rough_pixel_estimates = rough_pixel_estimates or [178., 279., 316., 321.]
        
        # Output lists
        self.pixelscale_list = []
        self.pixelscale_unc_list = []
        self.pixelscale_intercept_list = []
        self.pixelscale_intercept_unc_list = []
        
        # Internal state
        self._raster_average = None
        self._raster_average_unc = None
        self._color_list = ['blue', 'red', 'green', 'orange', 'magenta', 'olive', 'brown', 'lime']
    
    def compute_calibration(
        self,
        data_path: str,
        sumer_filename_list: list,
    ):
        """
        Compute wavelength calibration for all specified rows.
        
        Parameters
        ----------
        data_path : str
            Path to SUMER data directory
        sumer_filename_list : list
            List of SUMER FITS filenames to process
        """
        from modules.calibration_params_loader import get_parameters_for_row
        
        # Load and average SUMER data
        self._load_data(data_path, sumer_filename_list)
        
        # Process each row
        for row in np.arange(self.row_start, self.row_end + 1):
            print(f'Row: {row}')
            
            try:
                params = get_parameters_for_row(row)
            except ValueError:
                print(f"  Parameters not defined for row {row}, skipping...")
                continue
            
            idx_interval_dic = params['idx_interval']
            init_parameters_dic = params['init_parameters']
            
            # Process this row
            slope_fit, slope_unc_fit, intercept_fit, intercept_unc_fit = self._process_row(
                row, idx_interval_dic, init_parameters_dic
            )
            
            # Store results
            self.pixelscale_list.append(float(slope_fit))
            self.pixelscale_unc_list.append(float(slope_unc_fit))
            self.pixelscale_intercept_list.append(float(intercept_fit))
            self.pixelscale_intercept_unc_list.append(float(intercept_unc_fit))
    
    def _load_data(self, data_path: str, sumer_filename_list: list):
        """Load and average SUMER spectral data from FITS files."""
        print("Loading SUMER data...")
        
        # Load all data files
        sumer_data_list = []
        sumer_data_unc_list = []
        files_loaded = 0
        
        for filename in sumer_filename_list:
            # Skip Level 1 files (they have different structure)
            if '_l1.fits' in filename.lower():
                continue
            
            filepath = os.path.join(data_path, filename)
            try:
                # Use fits.getdata() which handles different HDUs automatically
                data = fits.getdata(filepath)
                
                # Check if data is None
                if data is None:
                    print(f"Warning: No data in {filename}")
                    continue
                
                # Convert to float and reverse row order (as in original code)
                data = data.astype(float)[::-1, :]
                
                # Mask defective pixels
                data = _mask_all_defective_pixels_DetA(data)
                
                # Calculate uncertainties (assuming Poisson noise)
                # Data uncertainty = sqrt(data * factor_fullspectrum) / t_exp
                data_unc = np.sqrt(np.abs(data) * self.factor_fullspectrum) / self.exposure_time
                
                sumer_data_list.append(data)
                sumer_data_unc_list.append(data_unc)
                files_loaded += 1
                
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
        
        # Average all spectral images
        if not sumer_data_list:
            raise ValueError(f"No data files loaded from {data_path}")
        
        # Use np.mean() on masked arrays directly - automatically ignores masked values
        # This matches the original code behavior
        self._raster_average = np.mean(sumer_data_list, axis=0)
        
        # Average uncertainty: sqrt(sum(unc^2)) / N
        data_unc_sumsquare = np.zeros(self._raster_average.shape)
        for data_unc_i in sumer_data_unc_list:
            data_unc_sumsquare = data_unc_sumsquare + data_unc_i**2
        self._raster_average_unc = (1/files_loaded) * np.sqrt(data_unc_sumsquare)
        
        print(f"Loaded {files_loaded} FITS files from {data_path}")
    
    def _process_row(self, row: int, idx_interval_dic: dict, init_parameters_dic: dict):
        """
        Process a single row: fit gaussians and compute calibration line.
        
        Returns
        -------
        tuple
            (slope, slope_unc, intercept, intercept_unc)
        """
        x_pixels = np.arange(0, 512)
        
        # Extract row data and handle masked arrays
        y_radiance_row = self._raster_average[row, :]
        y_unc_radiance_row = self._raster_average_unc[row, :]
        
        # Convert masked arrays to regular arrays if necessary
        if np.ma.is_masked(y_radiance_row):
            y_radiance_row = np.ma.filled(y_radiance_row, np.mean(y_radiance_row.compressed()))
        if np.ma.is_masked(y_unc_radiance_row):
            y_unc_radiance_row = np.ma.filled(y_unc_radiance_row, np.mean(y_unc_radiance_row.compressed()))
        
        y_radiance = 10 * y_radiance_row
        y_unc_radiance = 10 * y_unc_radiance_row
        
        # Perform multi-gaussian fits and extract means
        means_fit, means_unc_fit = self._fit_spectral_intervals(
            x_pixels, y_radiance, y_unc_radiance, idx_interval_dic, init_parameters_dic
        )
        
        # Match fitted means to calibration lines based on rough estimates
        means_px, means_unc_px = self._match_lines_to_calibration(means_fit, means_unc_fit)
        
        # Fit calibration line
        slope_fit, slope_unc_fit, intercept_fit, intercept_unc_fit = (
            self._fit_calibration_line(means_px, means_unc_px)
        )
        
        return slope_fit, slope_unc_fit, intercept_fit, intercept_unc_fit
    
    def _fit_spectral_intervals(self, x_pixels, y_radiance, y_unc_radiance, idx_interval_dic, init_parameters_dic):
        """Fit multi-gaussian functions to spectral intervals."""
        means_fit, means_unc_fit = [], []
        
        for interval_str in sorted(idx_interval_dic.keys()):
            init_parameters = init_parameters_dic[interval_str]
            idx_interval = idx_interval_dic[interval_str]
            
            x_data = x_pixels[idx_interval[0]:idx_interval[1]+1]
            y_data = y_radiance[idx_interval[0]:idx_interval[1]+1]
            y_unc_data = y_unc_radiance[idx_interval[0]:idx_interval[1]+1]
            
            # Convert masked arrays to regular arrays for curve_fit
            if np.ma.is_masked(y_data):
                y_data = np.ma.filled(y_data, np.mean(y_data.compressed()))
            if np.ma.is_masked(y_unc_data):
                y_unc_data = np.ma.filled(y_unc_data, np.mean(y_unc_data.compressed()))
            
            # Fit multi-gaussian with robust error handling
            try:
                popt, pcov = curve_fit(
                    self._multigaussian_for_curvefit,
                    x_data, y_data,
                    p0=init_parameters,
                    sigma=y_unc_data,
                    absolute_sigma=True,
                )
            except RuntimeError as e:
                # Retry with larger maxfev, then without sigma as fallback
                try:
                    popt, pcov = curve_fit(
                        self._multigaussian_for_curvefit,
                        x_data, y_data,
                        p0=init_parameters,
                        sigma=y_unc_data,
                        absolute_sigma=True,
                        maxfev=20000,
                    )
                except Exception:
                    try:
                        popt, pcov = curve_fit(
                            self._multigaussian_for_curvefit,
                            x_data, y_data,
                            p0=init_parameters,
                            maxfev=20000,
                        )
                    except Exception as e_final:
                        # Emit diagnostic info and skip this interval
                        print('  Warning: fit failed for interval', interval_str)
                        print('    idx_interval =', idx_interval)
                        print('    init_parameters =', init_parameters)
                        print('    x_data len =', len(x_data), 'y_data min/max =', np.min(y_data), np.max(y_data))
                        print('    y_unc_data min/max =', np.min(y_unc_data), np.max(y_unc_data))
                        print('    curve_fit error:', e_final)
                        # Append NaNs for each gaussian component means to keep indexing
                        n_gaussians = (len(init_parameters) - 1) // 3
                        for _ in range(n_gaussians):
                            means_fit.append(np.nan)
                            means_unc_fit.append(np.nan)
                        continue

            # Compute parameter uncertainties safely
            try:
                perr = np.sqrt(np.diag(pcov))
            except Exception:
                perr = np.full(len(popt), np.nan)
            
            # Extract means from each gaussian component
            n_gaussians = (len(init_parameters) - 1) // 3
            for n_gaussian_to_analyze in range(n_gaussians):
                mean_fit = popt[3*n_gaussian_to_analyze + 2]
                mean_unc_fit = perr[3*n_gaussian_to_analyze + 2]
                means_fit.append(mean_fit)
                means_unc_fit.append(mean_unc_fit)
        
        return means_fit, means_unc_fit
    
    def _match_lines_to_calibration(self, means_fit, means_unc_fit):
        """Match fitted line means to known calibration wavelengths."""
        means_px, means_unc_px = [], []
        
        for rough_px in self.rough_pixel_estimates:
            # Find the fitted mean closest to this rough estimate
            nearest_index = min(range(len(means_fit)), key=lambda i: abs(means_fit[i] - rough_px))
            means_px.append(means_fit[nearest_index])
            means_unc_px.append(means_unc_fit[nearest_index])
        
        return means_px, means_unc_px
    
    def _fit_calibration_line(self, means_px, means_unc_px):
        """Fit wavelength = slope * pixel + intercept."""
        from scipy.odr import Model, RealData, ODR
        
        # Orthogonal Distance Regression (accounts for uncertainties in both x and y)
        def line_model(B, x):
            return B[0] * x + B[1]
        
        model = Model(line_model)
        data = RealData(means_px, self.rest_wavelengths, sy=means_unc_px)
        odr = ODR(data, model, beta0=[0.005, 153.0])
        output = odr.run()
        
        slope_fit = output.beta[0]
        intercept_fit = output.beta[1]
        slope_unc_fit = output.sd_beta[0]
        intercept_unc_fit = output.sd_beta[1]
        
        return slope_fit, slope_unc_fit, intercept_fit, intercept_unc_fit
    
    @staticmethod
    def _multigaussian_for_curvefit(x, *params):
        """
        Multi-gaussian function for curve_fit.
        
        Parameters
        ----------
        x : array
            X values (pixels)
        *params : tuple
            [background, amplitude1, mean1, fwhm1, amplitude2, mean2, fwhm2, ...]
        
        Returns
        -------
        array
            Evaluated multi-gaussian at x
        """
        n = (len(params) - 1) // 3
        bckg_init = params[0]
        result = np.full_like(x, bckg_init, dtype=float)
        
        for i in range(n):
            amplitude = params[3*i + 1]
            mean = params[3*i + 2]
            fwhm = params[3*i + 3]
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            result = result + amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2))
        
        return result
    
    def get_slopes(self):
        """Get slopes and their uncertainties."""
        return np.array(self.pixelscale_list), np.array(self.pixelscale_unc_list)
    
    def get_intercepts(self):
        """Get intercepts and their uncertainties."""
        return np.array(self.pixelscale_intercept_list), np.array(self.pixelscale_intercept_unc_list)
    
    def get_all_results(self):
        """
        Get all calibration results.
        
        Returns
        -------
        dict
            Dictionary with keys: 'slopes', 'slopes_unc', 'intercepts', 'intercepts_unc'
        """
        return {
            'slopes': np.array(self.pixelscale_list),
            'slopes_unc': np.array(self.pixelscale_unc_list),
            'intercepts': np.array(self.pixelscale_intercept_list),
            'intercepts_unc': np.array(self.pixelscale_intercept_unc_list),
        }
    
    def save_results(self, output_path: str):
        """
        Save results to npz file.
        
        Parameters
        ----------
        output_path : str
            Path to save results (e.g., 'calibration_results.npz')
        """
        np.savez(
            output_path,
            pixelscale_list=np.array(self.pixelscale_list),
            pixelscale_unc_list=np.array(self.pixelscale_unc_list),
            pixelscale_intercept_list=np.array(self.pixelscale_intercept_list),
            pixelscale_intercept_unc_list=np.array(self.pixelscale_intercept_unc_list),
        )
        print(f"Results saved to {output_path}")
    
    def load_results(self, output_path: str):
        """
        Load results from npz file.
        
        Parameters
        ----------
        output_path : str
            Path to load results from (e.g., 'calibration_results.npz')
        
        Returns
        -------
        bool
            True if successfully loaded, False otherwise
        """
        if not os.path.exists(output_path):
            return False
        
        try:
            data = np.load(output_path)
            self.pixelscale_list = data['pixelscale_list'].tolist()
            self.pixelscale_unc_list = data['pixelscale_unc_list'].tolist()
            self.pixelscale_intercept_list = data['pixelscale_intercept_list'].tolist()
            self.pixelscale_intercept_unc_list = data['pixelscale_intercept_unc_list'].tolist()
            print(f"Results loaded from {output_path}")
            return True
        except Exception as e:
            print(f"Could not load results from {output_path}: {e}")
            return False
