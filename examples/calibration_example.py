"""
Example script showing how to use the refactored CalibrationParameters class.

This demonstrates the new organized pipeline for wavelength calibration.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from utils.calibration import CalibrationParameters
import glob


def main():
    """Example usage of CalibrationParameters class."""
    
    # Configuration parameters
    DATA_PATH = '/home/mario/Documents/MPS_PhD/Equatorial_coronal_hole/data/soho/sumer/'  # Path to SUMER FITS files
    OUTPUT_PATH = '../output/calibration_results.npz'  # Where to save/load results
    
    # Get all FITS files from the directory (exclude Level 1 files)
    all_files = sorted(glob.glob(os.path.join(DATA_PATH, '*.fits')))
    SUMER_FILES = [os.path.basename(f) for f in all_files if '_l1.fits' not in f.lower()]
    
    print(f"Found {len(SUMER_FILES)} Level3 SUMER files")
    
    # Create calibrator instance
    calibrator = CalibrationParameters(
        row_start=6,
        row_end=323,
        show_figures='no',
        exposure_time=150.0,        # Exposure time in seconds
        factor_fullspectrum=1.0,    # Scaling factor
        rest_wavelengths=[153.7935, 154.2177, 154.3724, 154.3960],
        rough_pixel_estimates=[178., 279., 316., 321.],
    )
    
    # Check if results already exist and load them
    print(f"\nChecking for existing calibration results at {OUTPUT_PATH}...")
    if os.path.exists(OUTPUT_PATH):
        print("Found existing results. Loading...")
        if calibrator.load_results(OUTPUT_PATH):
            print("✓ Results loaded successfully!")
        else:
            print("⚠ Failed to load results. Computing calibration...")
            calibrator.compute_calibration(
                data_path=DATA_PATH,
                sumer_filename_list=SUMER_FILES,
            )
            calibrator.save_results(OUTPUT_PATH)
    else:
        print("No existing results found. Computing calibration...")
        # Compute calibration for specified rows
        calibrator.compute_calibration(
            data_path=DATA_PATH,
            sumer_filename_list=SUMER_FILES,
        )
        # Save results for future use
        calibrator.save_results(OUTPUT_PATH)
    
    # Access results
    slopes, slopes_unc = calibrator.get_slopes()
    intercepts, intercepts_unc = calibrator.get_intercepts()
    
    print('#' * 50)
    print('pixelscale_list =', calibrator.pixelscale_list)
    print('#' * 50)
    print('pixelscale_unc_list =', calibrator.pixelscale_unc_list)
    print('#' * 50)
    print('pixelscale_intercept_list =', calibrator.pixelscale_intercept_list)
    print('#' * 50)
    print('pixelscale_intercept_unc_list =', calibrator.pixelscale_intercept_unc_list)
    print('#' * 50)
    
    # Example: save results to file
    # calibrator.save_results('calibration_results.npz')
    
    # Example: get all results as dictionary
    # results = calibrator.get_all_results()
    # print(results.keys())


if __name__ == '__main__':
    main()
