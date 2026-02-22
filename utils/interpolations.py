"""
Wavelength interpolation of SUMER spectral images using calibration parameters.

This module provides the PixelInterpolation class that performs linear interpolation
of spectral images to a common wavelength scale, accounting for variations in
wavelength calibration across detector rows.

Algorithm:
  1. Load calibration parameters (slopes and intercepts) for each row
  2. Load SUMER spectral data from FITS files
  3. For each spectral image:
     a. Use a reference row to define the target wavelength scale
     b. Interpolate all rows to this common wavelength scale
     c. Propagate uncertainties through the interpolation
  4. Average all interpolated images
  5. Save results to NPZ format for future use
"""

import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
import os
import warnings
warnings.filterwarnings('ignore')


def pixels_to_wavelength(pixel, slope_cal, intercept_cal):
    """
    Convert pixel positions to wavelength using calibration parameters.
    
    Parameters
    ----------
    pixel : array-like
        Pixel positions
    slope_cal : float
        Calibration slope (wavelength/pixel)
    intercept_cal : float
        Calibration intercept (wavelength at pixel 0)
    
    Returns
    -------
    array
        Wavelength values in Angstroms
    """
    wavelength = slope_cal * pixel + intercept_cal
    return wavelength


class PixelInterpolation:
    """
    Interpolate SUMER spectral images to a common wavelength scale.
    
    This class performs wavelength calibration interpolation on SUMER spectral
    images using pre-computed calibration parameters. For each row of the detector,
    the wavelength scale varies due to optical properties. This class maps all rows
    to a common wavelength scale defined by a reference row, accounting for 
    uncertainties through linear interpolation formulas.
    
    Parameters
    ----------
    row_reference : int, default=120
        Row index to use as the reference wavelength scale
    show_progress : bool, default=True
        Whether to show progress bar during interpolation
    
    Attributes
    ----------
    spectral_image_interpolated_list : list
        Interpolated spectral images
    spectral_image_unc_interpolated_list : list
        Uncertainties of interpolated spectral images
    reference_wavelength : array
        Wavelength scale of the reference row
    extent_reference_wavelength : tuple
        (wavelength_min, wavelength_max) for plotting extent
    spectral_image_interpolated_average : array
        Average of all interpolated spectral images
    spectral_image_unc_interpolated_average : array
        Average of interpolated uncertainty images
    
    Examples
    --------
    >>> from utils.calibration import CalibrationParameters
    >>> from utils.interpolations import PixelInterpolation
    >>> 
    >>> # First, compute calibration parameters
    >>> calibrator = CalibrationParameters(row_start=6, row_end=323)
    >>> calibrator.compute_calibration(...)
    >>> slopes, _ = calibrator.get_slopes()
    >>> intercepts, _ = calibrator.get_intercepts()
    >>> 
    >>> # Then interpolate spectral images
    >>> interpolator = PixelInterpolation(row_reference=120)
    >>> interpolator.interpolate_data(
    ...     data_path='../data/soho/sumer/',
    ...     sumer_filename_list=['file1.fits', 'file2.fits', ...],
    ...     slopes=slopes,
    ...     intercepts=intercepts,
    ... )
    >>> 
    >>> # Access results
    >>> print(interpolator.spectral_image_interpolated_average.shape)
    """
    
    def __init__(self, row_reference: int = 120, show_progress: bool = True):
        """Initialize interpolation parameters."""
        self.row_reference = row_reference
        self.show_progress = show_progress
        
        # Output data
        self.spectral_image_interpolated_list = []
        self.spectral_image_unc_interpolated_list = []
        self.reference_wavelength = None
        self.extent_reference_wavelength = None
        self.spectral_image_interpolated_average = None
        self.spectral_image_unc_interpolated_average = None
        
        # Internal state
        self._spectral_image_list = []
        self._spectral_image_unc_list = []
        self._slopes = None
        self._intercepts = None
    
    @staticmethod
    def _unc_linear_interpolation_1point(x_interp, x_data, y_unc_data):
        """
        Calculate uncertainty of interpolated point using linear interpolation.
        
        Based on error propagation formulas from:
        Propagation of Uncertainty and Comparison of Interpolation Schemes 
        (White 2017, doi: 10.1007/s10765-016-2174-6)
        
        Parameters
        ----------
        x_interp : float
            X position where interpolation is performed
        x_data : array
            X positions of data points
        y_unc_data : array
            Y uncertainties of data points
        
        Returns
        -------
        float
            Uncertainty of interpolated Y value, or np.nan if out of bounds
        """
        if x_interp < x_data[0] or x_interp > x_data[-1]:
            return np.nan  # Out of bounds
        
        # Find index where x_interp would fit
        idx = np.searchsorted(x_data, x_interp)
        
        # Handle edge case at first point
        if idx == 0:
            return y_unc_data[0]
        
        # Get surrounding points
        x1 = x_data[idx - 1]
        x2 = x_data[idx]
        y1_unc = y_unc_data[idx - 1]
        y2_unc = y_unc_data[idx]
        
        # Propagate uncertainties through linear interpolation
        A = (x_interp - x2) / (x1 - x2) * y1_unc
        B = (x_interp - x1) / (x2 - x1) * y2_unc
        y_unc_interp = np.sqrt(A**2 + B**2)
        
        return y_unc_interp
    
    @staticmethod
    def _linear_interpolation_setpoints(x_interp_list, x_data, y_data, y_unc_data):
        """
        Interpolate a set of points with uncertainty propagation.
        
        Parameters
        ----------
        x_interp_list : array
            X positions for interpolation
        x_data : array
            X positions of data points
        y_data : array
            Y values of data points (may be masked)
        y_unc_data : array
            Y uncertainties of data points (may be masked)
        
        Returns
        -------
        tuple
            (y_interp, y_unc_interp) - interpolated values and their uncertainties
        """
        # Calculate uncertainties
        y_unc_interp_list = []
        for x_interp_i in x_interp_list:
            y_unc_interp_i = PixelInterpolation._unc_linear_interpolation_1point(
                x_interp=x_interp_i, x_data=x_data, y_unc_data=y_unc_data
            )
            y_unc_interp_list.append(y_unc_interp_i)
        y_unc_interp = np.array(y_unc_interp_list)
        
        # Calculate interpolated values
        # Convert masked arrays to NaNs for interpolation
        y_data_filled = np.ma.filled(y_data, np.nan)
        y_unc_data_filled = np.ma.filled(y_unc_data, np.nan)
        
        # Create interpolation function
        interp_func = interp1d(
            x_data, y_data_filled, kind='linear',
            bounds_error=False, fill_value=np.nan
        )
        y_interp = interp_func(x_interp_list)
        
        return y_interp, y_unc_interp
    
    def _interpolate_spectral_image(self, spectral_image, spectral_image_unc,
                                    slope_list, intercept_list, row_start=6, row_end=323):
        """
        Interpolate all rows of one spectral image to reference wavelength scale.
        
        Parameters
        ----------
        spectral_image : array
            Spectral image (rows x wavelength pixels)
        spectral_image_unc : array
            Uncertainties of spectral image
        slope_list : array
            Calibration slopes for each row with calibration
        intercept_list : array
            Calibration intercepts for each row with calibration
        row_start : int
            Starting row index for which calibration is available
        row_end : int
            Ending row index for which calibration is available
        
        Returns
        -------
        tuple
            (interpolated_image, interpolated_unc, reference_wavelength, extent)
        """
        N_rows, N_cols = spectral_image.shape
        
        # Determine which calibration row index corresponds to our reference row
        # slope_list and intercept_list only contain data for rows row_start to row_end
        # We need to map from detector row indices to calibration parameter indices
        cal_row_reference_idx = self.row_reference - row_start
        
        # Validate reference row is within calibration range
        if cal_row_reference_idx < 0 or cal_row_reference_idx >= len(slope_list):
            raise ValueError(
                f"Reference row {self.row_reference} not in calibration range ({row_start}-{row_end})"
            )
        
        # Create reference wavelength scale from reference row
        pixel_positions = np.arange(0, N_cols)
        reference_wavelength = pixels_to_wavelength(
            pixel=pixel_positions,
            slope_cal=slope_list[cal_row_reference_idx],
            intercept_cal=intercept_list[cal_row_reference_idx]
        )
        
        # Calculate extent for plotting
        px_half = 0.5 * slope_list[cal_row_reference_idx]
        extent_reference_wavelength = [
            reference_wavelength[0] - px_half,
            reference_wavelength[-1] + px_half
        ]
        
        # Interpolate each row to reference wavelength scale
        # Only process rows that have calibration parameters
        intensity_interpolated = []
        intensity_unc_interpolated = []
        
        for i_row in range(N_rows):
            # Check if this row has calibration parameters
            if i_row < row_start or i_row > row_end:
                # Skip rows outside calibration range
                # Use NaN to indicate no data
                intensity_interpolated.append(np.full(N_cols, np.nan))
                intensity_unc_interpolated.append(np.full(N_cols, np.nan))
                continue
            
            # Get calibration index for this row
            cal_row_idx = i_row - row_start
            
            # Get wavelength positions for this row
            wavelength_row = pixels_to_wavelength(
                pixel=pixel_positions,
                slope_cal=slope_list[cal_row_idx],
                intercept_cal=intercept_list[cal_row_idx]
            )
            
            # Interpolate to reference wavelength scale
            y_interp, y_unc_interp = self._linear_interpolation_setpoints(
                x_interp_list=reference_wavelength,
                x_data=wavelength_row,
                y_data=spectral_image[i_row, :],
                y_unc_data=spectral_image_unc[i_row, :]
            )
            
            intensity_interpolated.append(y_interp)
            intensity_unc_interpolated.append(y_unc_interp)
        
        intensity_interpolated = np.array(intensity_interpolated)
        intensity_unc_interpolated = np.array(intensity_unc_interpolated)
        
        return (
            intensity_interpolated, intensity_unc_interpolated,
            reference_wavelength, extent_reference_wavelength
        )
    
    def _load_data(self, data_path: str, sumer_filename_list: list):
        """
        Load SUMER spectral data from FITS files.
        
        Parameters
        ----------
        data_path : str
            Path to SUMER FITS files
        sumer_filename_list : list
            List of FITS filenames to load
        """
        print("Loading SUMER data for interpolation...")
        
        self._spectral_image_list = []
        self._spectral_image_unc_list = []
        files_loaded = 0
        
        for filename in sumer_filename_list:
            # Skip Level 1 files
            if '_l1.fits' in filename.lower():
                continue
            
            filepath = os.path.join(data_path, filename)
            try:
                # Load data
                data = fits.getdata(filepath)
                
                if data is None:
                    print(f"Warning: No data in {filename}")
                    continue
                
                # Reverse row order and convert to float
                data = data.astype(float)[::-1, :]
                
                # Mask defective pixels (assuming DetA)
                # For interpolation, masked pixels will be handled by NaN values
                
                # Calculate uncertainties (Poisson noise)
                # Assume exposure time and scaling factor are known
                data_unc = np.sqrt(np.abs(data)) / 150.0  # t_exp = 150s
                
                self._spectral_image_list.append(data)
                self._spectral_image_unc_list.append(data_unc)
                files_loaded += 1
                
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
        
        if not self._spectral_image_list:
            raise ValueError(f"No data files loaded from {data_path}")
        
        print(f"Loaded {files_loaded} FITS files")
    
    def interpolate_data(self, data_path: str, sumer_filename_list: list,
                        slopes: np.ndarray, intercepts: np.ndarray,
                        row_start: int = 6, row_end: int = 323):
        """
        Perform interpolation on all SUMER spectral images.
        
        Parameters
        ----------
        data_path : str
            Path to SUMER FITS files
        sumer_filename_list : list
            List of FITS filenames to process
        slopes : array
            Calibration slopes for each row (only for rows row_start to row_end)
        intercepts : array
            Calibration intercepts for each row (only for rows row_start to row_end)
        row_start : int, default=6
            Starting row index for calibration data
        row_end : int, default=323
            Ending row index for calibration data
        """
        # Load data
        self._load_data(data_path, sumer_filename_list)
        self._slopes = slopes
        self._intercepts = intercepts
        
        # Interpolate all spectral images
        self.spectral_image_interpolated_list = []
        self.spectral_image_unc_interpolated_list = []
        
        try:
            from tqdm import tqdm
            n_images = len(self._spectral_image_list)
            iterator = tqdm(
                range(n_images),
                desc="Interpolating spectral images",
                unit="image"
            )
        except ImportError:
            iterator = range(len(self._spectral_image_list))
        
        for i_img in iterator:
            spectral_image_interp, spectral_image_unc_interp, \
                reference_wavelength, extent_ref = self._interpolate_spectral_image(
                    spectral_image=self._spectral_image_list[i_img],
                    spectral_image_unc=self._spectral_image_unc_list[i_img],
                    slope_list=self._slopes,
                    intercept_list=self._intercepts,
                    row_start=row_start,
                    row_end=row_end
                )
            
            self.spectral_image_interpolated_list.append(spectral_image_interp)
            self.spectral_image_unc_interpolated_list.append(spectral_image_unc_interp)
            
            # Store reference wavelength (same for all images)
            if self.reference_wavelength is None:
                self.reference_wavelength = reference_wavelength
                self.extent_reference_wavelength = extent_ref
        
        # Average all interpolated images
        self._compute_average()
        
        print(f"Interpolation complete.")
    
    def _compute_average(self):
        """Compute average of interpolated spectral images."""
        if not self.spectral_image_interpolated_list:
            return
        
        # Convert to arrays
        spectral_array = np.array(self.spectral_image_interpolated_list)
        unc_array = np.array(self.spectral_image_unc_interpolated_list)
        
        # Average values
        self.spectral_image_interpolated_average = np.mean(spectral_array, axis=0)
        
        # Average uncertainties: sqrt(sum(unc^2)) / N
        n_images = len(self.spectral_image_interpolated_list)
        unc_sumsquare = np.sum(unc_array**2, axis=0)
        self.spectral_image_unc_interpolated_average = (
            np.sqrt(unc_sumsquare) / n_images
        )
    
    def save_results(self, output_path: str):
        """
        Save interpolated results to NPZ file.
        
        Parameters
        ----------
        output_path : str
            Path to save results (e.g., 'interpolation_results.npz')
        """
        np.savez_compressed(
            output_path,
            spectral_image_interpolated_list=np.array(
                self.spectral_image_interpolated_list, dtype=object
            ),
            spectral_image_unc_interpolated_list=np.array(
                self.spectral_image_unc_interpolated_list, dtype=object
            ),
            reference_wavelength=self.reference_wavelength,
            extent_reference_wavelength=np.array(self.extent_reference_wavelength),
            spectral_image_interpolated_average=self.spectral_image_interpolated_average,
            spectral_image_unc_interpolated_average=self.spectral_image_unc_interpolated_average,
            row_reference=np.int32(self.row_reference),
        )
        print(f"Results saved to {output_path}")
    
    def load_results(self, output_path: str) -> bool:
        """
        Load interpolated results from NPZ file.
        
        Parameters
        ----------
        output_path : str
            Path to load results from
        
        Returns
        -------
        bool
            True if successfully loaded, False otherwise
        """
        if not os.path.exists(output_path):
            return False
        
        try:
            data = np.load(output_path, allow_pickle=True)
            
            self.spectral_image_interpolated_list = [
                arr for arr in data['spectral_image_interpolated_list']
            ]
            self.spectral_image_unc_interpolated_list = [
                arr for arr in data['spectral_image_unc_interpolated_list']
            ]
            self.reference_wavelength = data['reference_wavelength']
            self.extent_reference_wavelength = tuple(
                data['extent_reference_wavelength']
            )
            self.spectral_image_interpolated_average = (
                data['spectral_image_interpolated_average']
            )
            self.spectral_image_unc_interpolated_average = (
                data['spectral_image_unc_interpolated_average']
            )
            self.row_reference = int(data['row_reference'])
            
            print(f"Results loaded from {output_path}")
            return True
        except Exception as e:
            print(f"Could not load results from {output_path}: {e}")
            return False
    
    def get_results(self) -> dict:
        """
        Get all interpolation results as dictionary.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'spectral_image_interpolated_list'
            - 'spectral_image_unc_interpolated_list'
            - 'reference_wavelength'
            - 'extent_reference_wavelength'
            - 'spectral_image_interpolated_average'
            - 'spectral_image_unc_interpolated_average'
        """
        return {
            'spectral_image_interpolated_list': self.spectral_image_interpolated_list,
            'spectral_image_unc_interpolated_list': self.spectral_image_unc_interpolated_list,
            'reference_wavelength': self.reference_wavelength,
            'extent_reference_wavelength': self.extent_reference_wavelength,
            'spectral_image_interpolated_average': self.spectral_image_interpolated_average,
            'spectral_image_unc_interpolated_average': self.spectral_image_unc_interpolated_average,
        }
