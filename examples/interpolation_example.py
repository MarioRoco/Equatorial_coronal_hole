"""
Example script showing how to use the PixelInterpolation class.

This demonstrates the complete interpolation pipeline:
  1. Load calibration parameters (or compute them)
  2. Perform pixel-to-wavelength interpolation
  3. Save results with automatic backup/load logic
"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from utils.calibration import CalibrationParameters
from utils.interpolations import PixelInterpolation
import glob
import numpy as np


def main():
    """Example usage of PixelInterpolation class."""
    
    # Configuration parameters
    DATA_PATH = '/home/mario/Documents/MPS_PhD/Equatorial_coronal_hole/data/soho/sumer/'
    CALIBRATION_OUTPUT = '../output/calibration_results.npz'  # From calibration example
    INTERPOLATION_OUTPUT = '../output/interpolation_results.npz'  # Where to save interpolation results
    
    # Get all FITS files from the directory (exclude Level 1 files)
    all_files = sorted(glob.glob(os.path.join(DATA_PATH, '*.fits')))
    SUMER_FILES = [os.path.basename(f) for f in all_files if '_l1.fits' not in f.lower()]
    
    print(f"Found {len(SUMER_FILES)} Level3 SUMER files")
    
    # ========================================================================
    # STEP 1: Get calibration parameters
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: Loading calibration parameters")
    print("="*70)
    
    # Check if calibration results exist
    if os.path.exists(CALIBRATION_OUTPUT):
        print(f"Loading calibration from {CALIBRATION_OUTPUT}...")
        calibrator = CalibrationParameters(row_start=6, row_end=323, show_figures='no')
        if calibrator.load_results(CALIBRATION_OUTPUT):
            slopes, slopes_unc = calibrator.get_slopes()
            intercepts, intercepts_unc = calibrator.get_intercepts()
            print(f"✓ Calibration loaded: {len(slopes)} rows")
        else:
            raise RuntimeError("Could not load calibration results")
    else:
        print(f"Calibration file not found at {CALIBRATION_OUTPUT}")
        print("Please run calibration_example.py first")
        return
    
    # ========================================================================
    # STEP 2: Perform interpolation (with backup/load logic)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: Performing wavelength interpolation")
    print("="*70)
    
    # Create interpolator instance
    interpolator = PixelInterpolation(
        row_reference=120,  # Use row 120 as reference for wavelength scale
        show_progress=True
    )
    
    # Calibration row range (must match the range used in CalibrationParameters)
    CAL_ROW_START = 6
    CAL_ROW_END = 323
    
    # Check if interpolation results already exist
    print(f"\nChecking for existing interpolation results at {INTERPOLATION_OUTPUT}...")
    if os.path.exists(INTERPOLATION_OUTPUT):
        print("Found existing results. Loading...")
        if interpolator.load_results(INTERPOLATION_OUTPUT):
            print("✓ Interpolation results loaded successfully!")
        else:
            print("⚠ Failed to load results. Computing interpolation...")
            interpolator.interpolate_data(
                data_path=DATA_PATH,
                sumer_filename_list=SUMER_FILES,
                slopes=slopes,
                intercepts=intercepts,
                row_start=CAL_ROW_START,
                row_end=CAL_ROW_END,
            )
            interpolator.save_results(INTERPOLATION_OUTPUT)
    else:
        print("No existing results found. Computing interpolation...")
        # Perform interpolation
        interpolator.interpolate_data(
            data_path=DATA_PATH,
            sumer_filename_list=SUMER_FILES,
            slopes=slopes,
            intercepts=intercepts,
            row_start=CAL_ROW_START,
            row_end=CAL_ROW_END,
        )
        # Save results for future use
        interpolator.save_results(INTERPOLATION_OUTPUT)
    
    # ========================================================================
    # STEP 3: Display results
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: Results Summary")
    print("="*70)
    
    print(f"\nReference row: {interpolator.row_reference}")
    print(f"Number of interpolated spectral images: {len(interpolator.spectral_image_interpolated_list)}")
    
    if len(interpolator.spectral_image_interpolated_list) > 0:
        print(f"Shape of each interpolated image: {interpolator.spectral_image_interpolated_list[0].shape}")
    
    print(f"Shape of average interpolated image: {interpolator.spectral_image_interpolated_average.shape}")
    
    print(f"\nReference wavelength range:")
    print(f"  Min: {interpolator.reference_wavelength[0]:.4f} Å")
    print(f"  Max: {interpolator.reference_wavelength[-1]:.4f} Å")
    print(f"  N points: {len(interpolator.reference_wavelength)}")
    
    print(f"\nAverage spectral image statistics:")
    avg_img = interpolator.spectral_image_interpolated_average
    print(f"  Min: {np.nanmin(avg_img):.6e}")
    print(f"  Max: {np.nanmax(avg_img):.6e}")
    print(f"  Mean: {np.nanmean(avg_img):.6e}")
    
    # ========================================================================
    # STEP 4: Visualization
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: Generating diagnostic plots")
    print("="*70)
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        
        print("\n✓ Generating plots...")
        
        # Only generate plots if we have the original spectral images loaded
        if len(interpolator._spectral_image_list) > 0:
            # Get the first spectral image for comparison
            spectral_image_orig = interpolator._spectral_image_list[0]
            spectral_image_interp = interpolator.spectral_image_interpolated_list[0]
            spectral_image_unc_interp = interpolator.spectral_image_unc_interpolated_list[0]
            
            # Plot 1: Original spectral image
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 7))
            img = ax.imshow(spectral_image_orig, cmap='Greys', aspect='auto', norm=LogNorm())
            cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])
            cbar = fig.colorbar(img, cax=cax)
            cbar.set_label(r'Spectral radiance [W/sr/m$^2$/Å]', fontsize=14)
            ax.set_title('Original spectral image (first file)', fontsize=16)
            ax.set_xlabel('Wavelength direction (pixels)', fontsize=12)
            ax.set_ylabel('Spatial direction (pixels)', fontsize=12)
            ax.grid(False)
            fig.tight_layout()
            plt.savefig('../output/01_original_spectral_image.png', dpi=100, bbox_inches='tight')
            print("  ✓ Saved: output/01_original_spectral_image.png")
            plt.close(fig)
            
            # Plot 2: Interpolated spectral image
            extent = [
                interpolator.extent_reference_wavelength[0],
                interpolator.extent_reference_wavelength[1],
                spectral_image_orig.shape[0] - 0.5,
                -0.5
            ]
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 7))
            img = ax.imshow(spectral_image_interp, cmap='Greys', aspect='auto', norm=LogNorm(), extent=extent)
            cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])
            cbar = fig.colorbar(img, cax=cax)
            cbar.set_label(r'Spectral radiance [W/sr/m$^2$/Å]', fontsize=14)
            ax.set_title('Interpolated spectral image (first file)', fontsize=16)
            ax.set_xlabel('Wavelength (Å)', fontsize=12)
            ax.set_ylabel('Spatial direction (pixels)', fontsize=12)
            ax.grid(False)
            fig.tight_layout()
            plt.savefig('../output/02_interpolated_spectral_image.png', dpi=100, bbox_inches='tight')
            print("  ✓ Saved: output/02_interpolated_spectral_image.png")
            plt.close(fig)
            
            # Plot 3: Average original spectral image
            spectral_image_average_orig = np.mean(
                np.array(interpolator._spectral_image_list), axis=0
            )
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 7))
            img = ax.imshow(spectral_image_average_orig, cmap='Greys', aspect='auto', norm=LogNorm())
            cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])
            cbar = fig.colorbar(img, cax=cax)
            cbar.set_label(r'Av. spectral radiance [W/sr/m$^2$/Å]', fontsize=14)
            ax.set_title(f'Average of {len(interpolator._spectral_image_list)} original spectral images', fontsize=16)
            ax.set_xlabel('Wavelength direction (pixels)', fontsize=12)
            ax.set_ylabel('Spatial direction (pixels)', fontsize=12)
            ax.grid(False)
            fig.tight_layout()
            plt.savefig('../output/03_average_original_spectral_image.png', dpi=100, bbox_inches='tight')
            print("  ✓ Saved: output/03_average_original_spectral_image.png")
            plt.close(fig)
            
            # Plot 5: Zoomed comparison (wavelength range columns 230-270)
            col1, col2 = 230, 270
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # Original zoomed
            img1 = ax1.imshow(
                spectral_image_average_orig[:, col1:col2+1],
                cmap='Greys', aspect='auto', norm=LogNorm()
            )
            cbar1 = plt.colorbar(img1, ax=ax1)
            cbar1.set_label(r'Av. spectral radiance [W/sr/m$^2$/Å]', fontsize=12)
            ax1.set_title('Original (zoomed columns 230-270)', fontsize=14)
            ax1.set_xlabel('Wavelength direction (pixels)', fontsize=11)
            ax1.set_ylabel('Spatial direction (pixels)', fontsize=11)
            
            # Interpolated zoomed
            col_wl_extent = [
                interpolator.reference_wavelength[col1],
                interpolator.reference_wavelength[col2],
                spectral_image_average_orig.shape[0] - 0.5,
                -0.5
            ]
            img2 = ax2.imshow(
                interpolator.spectral_image_interpolated_average[:, col1:col2+1],
                cmap='Greys', aspect='auto', norm=LogNorm(), extent=col_wl_extent
            )
            cbar2 = plt.colorbar(img2, ax=ax2)
            cbar2.set_label(r'Av. spectral radiance [W/sr/m$^2$/Å]', fontsize=12)
            ax2.set_title('Interpolated (zoomed columns 230-270)', fontsize=14)
            ax2.set_xlabel('Wavelength (Å)', fontsize=11)
            ax2.set_ylabel('Spatial direction (pixels)', fontsize=11)
            
            fig.tight_layout()
            plt.savefig('../output/05_zoomed_comparison.png', dpi=100, bbox_inches='tight')
            print("  ✓ Saved: output/05_zoomed_comparison.png")
            plt.close(fig)
        
        # Plot 4: Average interpolated spectral image (always available)
        extent = [
            interpolator.extent_reference_wavelength[0],
            interpolator.extent_reference_wavelength[1],
            interpolator.spectral_image_interpolated_average.shape[0] - 0.5,
            -0.5
        ]
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 7))
        img = ax.imshow(
            interpolator.spectral_image_interpolated_average,
            cmap='Greys', aspect='auto', norm=LogNorm(), extent=extent
        )
        cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])
        cbar = fig.colorbar(img, cax=cax)
        cbar.set_label(r'Av. spectral radiance [W/sr/m$^2$/Å]', fontsize=14)
        ax.set_title(f'Average of {len(interpolator.spectral_image_interpolated_list)} interpolated spectral images', fontsize=16)
        ax.set_xlabel('Wavelength (Å)', fontsize=12)
        ax.set_ylabel('Spatial direction (pixels)', fontsize=12)
        ax.grid(False)
        fig.tight_layout()
        plt.savefig('../output/04_average_interpolated_spectral_image.png', dpi=100, bbox_inches='tight')
        print("  ✓ Saved: output/04_average_interpolated_spectral_image.png")
        plt.close(fig)
        
        # Plot 6: Wavelength calibration profile for different columns
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        col_indices = [10, 100, 450]
        
        for ax, col_idx in zip(axes, col_indices):
            col_wavelength = slopes * col_idx + intercepts
            row_idx = np.arange(len(slopes))
            
            ax.plot(row_idx, col_wavelength, 'b-', linewidth=1.5)
            ax.fill_between(row_idx, col_wavelength - slopes_unc, col_wavelength + slopes_unc,
                           alpha=0.3, color='blue', label='Calibration uncertainty')
            ax.axhline(y=interpolator.reference_wavelength[col_idx], color='red', 
                      linestyle='--', linewidth=2, label=f'Reference row {interpolator.row_reference}')
            ax.set_title(f'Wavelength profile - Column index: {col_idx}', fontsize=13)
            ax.set_xlabel('Row index (spatial direction)', fontsize=11)
            ax.set_ylabel('Wavelength (Å)', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        
        fig.tight_layout()
        plt.savefig('../output/06_wavelength_calibration_profiles.png', dpi=100, bbox_inches='tight')
        print("  ✓ Saved: output/06_wavelength_calibration_profiles.png")
        plt.close(fig)
        
        # Plot 7: Uncertainty map of interpolated average
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 7))
        img = ax.imshow(
            interpolator.spectral_image_unc_interpolated_average,
            cmap='viridis', aspect='auto', extent=extent
        )
        cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])
        cbar = fig.colorbar(img, cax=cax)
        cbar.set_label(r'Uncertainty [W/sr/m$^2$/Å]', fontsize=14)
        ax.set_title('Uncertainty map of average interpolated spectral image', fontsize=16)
        ax.set_xlabel('Wavelength (Å)', fontsize=12)
        ax.set_ylabel('Spatial direction (pixels)', fontsize=12)
        ax.grid(False)
        fig.tight_layout()
        plt.savefig('../output/07_uncertainty_map.png', dpi=100, bbox_inches='tight')
        print("  ✓ Saved: output/07_uncertainty_map.png")
        plt.close(fig)
        
        print("\n✓ All diagnostic plots saved to output/ directory")
        if len(interpolator._spectral_image_list) == 0:
            print("  (Note: Original spectral images not available in loaded results)")
        
    except ImportError:
        print("\n⚠ matplotlib not available, skipping plots")
    except Exception as e:
        print(f"\n⚠ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✓ Interpolation example completed successfully!")
    print("="*70)
    
    # Optional: Get all results as dictionary
    # results = interpolator.get_results()
    # print(f"\nAvailable results keys: {list(results.keys())}")


if __name__ == '__main__':
    main()
