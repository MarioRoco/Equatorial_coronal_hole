
"""
This script computes the wavelength calibration by doing the next for each row of the averaged spectral image (average of the 240 spectral images):
    1) Fits the cold lines with multigaussian function (x axis in pixels, y axis in spectral radiance). Thus it gets the centroid of the lines in pixels.
    2) Fits a stright line to x=[centroid in pixels], y=[rest wavelength (from literature)]
    3) It returns the parameters to convert pixels to wavelength (wavelength = slope * pixel + intercept): 
        - pixelscale_list: slopes for all rows
        - pixelscale_unc_list: uncertainties of pixelscale_list
        - pixelscale_intercept_list: intercepts for all rows
        - pixelscale_intercept_unc_list: uncertainties of pixelscale_intercept_list

The output of this script goes to file calibration_parameters__output.py
"""


import numpy as np

show_figures = 'no'
color_list = ['blue', 'red', 'green', 'orange', 'magenta', 'olive', 'brown', 'lime', 'blue', 'red', 'green', 'orange', 'magenta', 'olive', 'brown', 'lime']


means_label = ['1537.94 Si I', '1542.18 C I', '1543.72 Si I', '1543.96 C I']
means_px_rough = [178., 279., 316., 321.] #[149.5, 
means_Ang_theory = [153.7935, 154.2177, 154.3724, 154.3960] #[153.794, (154.120+154.132)/2, 154.218, 154.372, 154.396]

"""
means_label = ['1537.94 Si I', '1541.20 Si I + 1541.32 Si I', '1542.18 C I', '1543.72 Si I', '1543.96 C I']
means_px_rough = [178., 259., 279., 316., 321.] #[149.5, 
means_Ang_theory = [153.7935, (154.1198*10+154.1322*25)/(10+25), 154.2177, 154.3724, 154.3960] #[153.794, (154.120+154.132)/2, 154.218, 154.372, 154.396]
"""



# import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sys
import os
sys.path.append(os.path.abspath('..'))

import json
from pathlib import Path


def generate_init_parameters(json_path = ".", save=False):
	pixelscale_list_float64, pixelscale_unc_list_float64, pixelscale_intercept_list_float64, pixelscale_intercept_unc_list_float64 = [],[],[],[]

	calibration_parameters_all_rows = {}

	for row in np.arange(6, 323+1):
		print('Row:', row)

		if row == 323:
			idx_interval_dic = {'1':[146,173], '2':[175,204], '3':[241-3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							1.88-bckg_fit, 156., 3.,
							#2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.53-bckg_fit, 178., 3.,
							1.7-bckg_fit, 182., 3.5,
							#1.54-bckg_fit, 184.5, 3.,
							1.68-bckg_fit, 191.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.85-bckg_fit, 200., 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row == 321 or row==322:
			idx_interval_dic = {'1':[146,173], '2':[174,204], '3':[241-3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							#1.88-bckg_fit, 156., 3.,
							2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.53-bckg_fit, 178., 3.,
							1.7-bckg_fit, 182., 3.5,
							#1.54-bckg_fit, 184.5, 3.,
							1.68-bckg_fit, 191.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.85-bckg_fit, 200., 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row == 315 or row ==316 or row ==317 or row ==318 or row ==319 or row ==320:
			idx_interval_dic = {'1':[146,173], '2':[174,204], '3':[241-3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							#1.88-bckg_fit, 156., 3.,
							2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.53-bckg_fit, 178., 3.,
							1.7-bckg_fit, 182., 3.5,
							#1.54-bckg_fit, 184.5, 3.,
							1.68-bckg_fit, 191.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.85-bckg_fit, 200., 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]
						


		if row == 314:
			idx_interval_dic = {'1':[146,173], '2':[174,204], '3':[241-3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							#1.88-bckg_fit, 156., 3.,
							2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.53-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 182., 3.5,
							#1.54-bckg_fit, 184.5, 3.,
							1.68-bckg_fit, 191.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.85-bckg_fit, 200., 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row==313:
			idx_interval_dic = {'1':[146,173], '2':[174,204], '3':[241-3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							#1.88-bckg_fit, 156., 3.,
							2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.53-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 182., 3.5,
							#1.54-bckg_fit, 184.5, 3.,
							1.68-bckg_fit, 191.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.85-bckg_fit, 200., 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row==312:
			idx_interval_dic = {'1':[146,173], '2':[174,204], '3':[241-3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							#1.88-bckg_fit, 156., 3.,
							2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.53-bckg_fit, 178., 3.,
							1.7-bckg_fit, 182., 3.5,
							#1.54-bckg_fit, 184.5, 3.,
							1.68-bckg_fit, 191.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.85-bckg_fit, 200., 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]




		if row==307 or row==308 or row==309 or row==310 or row==311:
			idx_interval_dic = {'1':[146,173], '2':[174,204], '3':[241-3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							#1.88-bckg_fit, 156., 3.,
							2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.53-bckg_fit, 178., 3.,
							1.7-bckg_fit, 182., 3.5,
							#1.54-bckg_fit, 184.5, 3.,
							1.68-bckg_fit, 191.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.85-bckg_fit, 200., 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 261., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]




		if row==304 or row==305 or row==306:
			idx_interval_dic = {'1':[146,173], '2':[174,204], '3':[241-3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							#1.88-bckg_fit, 156., 3.,
							2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.53-bckg_fit, 178., 3.,
							1.7-bckg_fit, 182., 3.5,
							#1.54-bckg_fit, 184.5, 3.,
							1.68-bckg_fit, 191.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.85-bckg_fit, 200., 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 261., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row==302 or row==303:
			idx_interval_dic = {'1':[146,173], '2':[174,204], '3':[241-3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							#1.88-bckg_fit, 156., 3.,
							2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.53-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 182., 3.5,
							1.54-bckg_fit, 184.5, 3.,
							1.68-bckg_fit, 191.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.85-bckg_fit, 200., 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 261., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row==301:
			idx_interval_dic = {'1':[146,173], '2':[174,214-11], '3':[241-3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							#1.88-bckg_fit, 156., 3.,
							2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.44-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 182., 3.5,
							1.54-bckg_fit, 185., 3.,
							1.67-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.97-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 261., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]

		if row==300:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241-3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							#1.88-bckg_fit, 156., 3.,
							2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.42-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 182., 3.5,
							#1.5-bckg_fit, 185., 3.5,
							1.64-bckg_fit, 191.47, 3.,
							1.57-bckg_fit, 195., 2.,
							1.93-bckg_fit, 200.5, 3.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 261., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==299:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241-3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							#1.88-bckg_fit, 156., 3.,
							2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.44-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 182., 3.5,
							1.5-bckg_fit, 184., 5.,
							1.67-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.97-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 261., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row==298:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241-3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							#1.88-bckg_fit, 156., 3.,
							2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.44-bckg_fit, 178., 3.,
							1.7-bckg_fit, 182., 3.5,
							1.5-bckg_fit, 184., 5.,
							1.67-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.97-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 261., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row==288 or row==289 or row==290 or row==291 or row==292 or row==293 or row==294 or row==295 or row==296 or row==297:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241-3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							#1.88-bckg_fit, 156., 3.,
							2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.44-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 180., 3.5,
							1.5-bckg_fit, 184., 5.,
							1.67-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.97-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 261., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==287:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241+1,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							#1.88-bckg_fit, 156., 3.,
							2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.44-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 180., 3.5,
							1.5-bckg_fit, 184., 5.,
							1.67-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.97-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 261., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row==284 or row==285 or row==286:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241+1,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							#1.88-bckg_fit, 156., 3.,
							2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.]
							#2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.44-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 180., 3.5,
							1.5-bckg_fit, 184., 5.,
							1.67-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.97-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 261., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row==282 or row==283:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241+1,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							#1.88-bckg_fit, 156., 3.,
							2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.]
							#2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							1.7-bckg_fit, 180., 3.5,
							#1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 261., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row==281:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241+1,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.1-bckg_fit, 149.5, 2.5,
							1.88-bckg_fit, 156., 3.,
							#1.8-bckg_fit, 158., 3.,
							2.47-bckg_fit, 162.5, 3.,
							2.1-bckg_fit, 168.5, 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							1.7-bckg_fit, 180., 3.5,
							#1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 261., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==279 or row==280:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241-2,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.1-bckg_fit, 149.5, 2.5,
							1.88-bckg_fit, 156., 3.,
							#1.8-bckg_fit, 158., 3.,
							2.47-bckg_fit, 162.5, 3.,
							2.1-bckg_fit, 168.5, 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							1.7-bckg_fit, 180., 3.5,
							#1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 261., 2.,
							2.35-bckg_fit, 264.5, 3.,#]
							2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==277 or row==278:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241+2,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.1-bckg_fit, 149.5, 2.5,
							1.88-bckg_fit, 156., 3.,
							#1.8-bckg_fit, 158., 3.,
							2.47-bckg_fit, 162.5, 3.,
							2.1-bckg_fit, 168.5, 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							1.7-bckg_fit, 180., 2.5,
							#1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 261., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==273 or row== 274 or row==275 or row==276:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241-3,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.1-bckg_fit, 149.5, 2.5,
							1.88-bckg_fit, 156., 3.,
							#1.8-bckg_fit, 158., 3.,
							2.47-bckg_fit, 162.5, 3.,
							2.1-bckg_fit, 168.5, 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							1.7-bckg_fit, 180., 2.5,
							#1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 261., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row==270 or row==271 or row==272:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241-3,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.1-bckg_fit, 149.5, 2.5,
							1.88-bckg_fit, 156., 3.,
							#1.8-bckg_fit, 158., 3.,
							2.47-bckg_fit, 162.5, 3.,
							2.1-bckg_fit, 168.5, 3.]#,
							#2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 180., 2.5,
							#1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 261., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==268 or row==269:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241-3,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.1-bckg_fit, 149.5, 2.5,
							1.88-bckg_fit, 156., 3.,
							#1.8-bckg_fit, 158., 3.,
							2.47-bckg_fit, 162.5, 3.,
							2.1-bckg_fit, 168.5, 3.]#,
							#2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 180., 2.5,
							#1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row==263 or row==264 or row==265 or row==266 or row==267:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241+0,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.1-bckg_fit, 149.5, 2.5,
							1.88-bckg_fit, 156., 3.,
							#1.8-bckg_fit, 158., 3.,
							2.47-bckg_fit, 162.5, 3.,
							2.1-bckg_fit, 168.5, 3.]#,
							#2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 180., 2.5,
							#1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row==262:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241+3,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.1-bckg_fit, 149.5, 2.5,
							1.88-bckg_fit, 156., 3.,
							#1.8-bckg_fit, 158., 3.,
							2.47-bckg_fit, 162.5, 3.,
							2.1-bckg_fit, 168.5, 3.]#,
							#2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 180., 2.5,
							#1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==261:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241-2,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.1-bckg_fit, 149.5, 2.5,
							1.88-bckg_fit, 156., 3.,
							#1.8-bckg_fit, 158., 3.,
							2.47-bckg_fit, 162.5, 3.,
							2.1-bckg_fit, 168.5, 3.]#,
							#2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							1.7-bckg_fit, 180., 2.5,
							#1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if  row==260:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241-2,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.-bckg_fit, 149.5, 3.5,
							1.7-bckg_fit, 154., 3.,
							#1.8-bckg_fit, 158., 3.,
							2.3-bckg_fit, 162., 3.,
							2.14-bckg_fit, 167.5, 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							1.7-bckg_fit, 180., 2.5,
							#1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if  row==259:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241+3,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.-bckg_fit, 149.5, 3.5,
							1.7-bckg_fit, 154., 3.,
							#1.8-bckg_fit, 158., 3.,
							2.3-bckg_fit, 162., 3.,
							2.14-bckg_fit, 167.5, 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							1.7-bckg_fit, 180., 2.5,
							#1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==256 or row==257 or row==258:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241+3,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.-bckg_fit, 149.5, 3.5,
							1.7-bckg_fit, 154., 3.,
							1.8-bckg_fit, 158., 3.,
							2.3-bckg_fit, 162., 3.,
							2.14-bckg_fit, 167.5, 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							1.7-bckg_fit, 180., 2.5,
							#1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==255:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241+3,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.-bckg_fit, 149.5, 3.5,
							1.7-bckg_fit, 154., 3.,
							1.8-bckg_fit, 158., 3.,
							2.3-bckg_fit, 162., 3.,
							2.14-bckg_fit, 167.5, 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 180., 2.5,
							#1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==254:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241+3,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.-bckg_fit, 149.5, 3.5,
							1.7-bckg_fit, 154., 3.,
							1.8-bckg_fit, 158., 3.,
							2.3-bckg_fit, 162., 3.,
							2.14-bckg_fit, 167.5, 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 180., 2.5,
							1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==253:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241-2,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.-bckg_fit, 149.5, 3.5,
							1.7-bckg_fit, 154., 3.,
							1.8-bckg_fit, 158., 3.,
							2.3-bckg_fit, 162., 3.,
							2.14-bckg_fit, 167.5, 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 180., 2.5,
							#1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==252:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241-5,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.-bckg_fit, 149.5, 3.5,
							1.7-bckg_fit, 154., 3.,
							1.8-bckg_fit, 158., 3.,
							2.3-bckg_fit, 162., 3.,
							2.14-bckg_fit, 167.5, 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 180., 2.5,
							#1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]

		if row==249 or row==250 or row==251:
			idx_interval_dic = {'1':[146,173], '2':[173,214-11], '3':[241+3,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.-bckg_fit, 149.5, 3.5,
							1.7-bckg_fit, 154., 3.,
							1.8-bckg_fit, 158., 3.,
							2.3-bckg_fit, 162., 3.,
							2.14-bckg_fit, 167.5, 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.38-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 180., 2.5,
							#1.66-bckg_fit, 184., 3.,
							1.65-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.5]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==248:
			idx_interval_dic = {'1':[146,173], '2':[173,214], '3':[241+3,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.-bckg_fit, 149.5, 3.5,
							1.7-bckg_fit, 154., 3.,
							1.8-bckg_fit, 158., 3.,
							2.3-bckg_fit, 162., 3.,
							2.14-bckg_fit, 167.5, 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.27-bckg_fit, 177.5, 3.5,
							1.7-bckg_fit, 180., 2.5,
							#1.5-bckg_fit, 186.5, 3.,
							1.71-bckg_fit, 192., 3.5,
							1.6-bckg_fit, 197., 3.,
							1.91-bckg_fit, 200., 3.5,
							#1.54-bckg_fit, 204., 2.,
							3.35-bckg_fit, 208.5, 3.,
							2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==247:
			idx_interval_dic = {'1':[146,173], '2':[173,214], '3':[241+3,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.-bckg_fit, 149.5, 3.5,
							1.7-bckg_fit, 154., 3.,
							1.8-bckg_fit, 158., 3.,
							2.3-bckg_fit, 162., 3.,
							2.14-bckg_fit, 167.5, 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.27-bckg_fit, 177.5, 3.5,
							1.7-bckg_fit, 180., 2.5,
							#1.5-bckg_fit, 186.5, 3.,
							1.71-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.91-bckg_fit, 200., 3.5,
							#1.54-bckg_fit, 204., 2.,
							3.35-bckg_fit, 208.5, 3.]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]

		if row==246:
			idx_interval_dic = {'1':[146,173], '2':[173,212], '3':[241+3,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.-bckg_fit, 149.5, 3.5,
							1.7-bckg_fit, 154., 3.,
							1.8-bckg_fit, 158., 3.,
							2.3-bckg_fit, 162., 3.,
							2.14-bckg_fit, 167.5, 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.5-bckg_fit, 178., 3.5,
							1.75-bckg_fit, 181., 2.5,
							1.5-bckg_fit, 186.5, 3.,
							1.71-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.07-bckg_fit, 200., 3.5,
							#1.54-bckg_fit, 204., 2.,
							1.92-bckg_fit, 208.5, 3.]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==245:
			idx_interval_dic = {'1':[146,173], '2':[173,212], '3':[241+3,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.-bckg_fit, 149.5, 3.5,
							1.7-bckg_fit, 154., 3.,
							1.8-bckg_fit, 158., 3.,
							2.3-bckg_fit, 162., 3.,
							2.14-bckg_fit, 167.5, 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.5-bckg_fit, 178., 3.5,
							1.75-bckg_fit, 181., 2.5,
							1.5-bckg_fit, 186.5, 3.,
							1.71-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.07-bckg_fit, 200., 3.5,
							#1.54-bckg_fit, 204., 2.,
							1.92-bckg_fit, 208.5, 3.]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==243 or row==244:
			idx_interval_dic = {'1':[146,173], '2':[173,212], '3':[241+3,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.3-bckg_fit, 149.5, 3.5,
							2.1-bckg_fit, 156., 5.,
							2.7-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.4-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.5-bckg_fit, 178., 3.5,
							1.75-bckg_fit, 181., 2.5,
							#1.8-bckg_fit, 185.5, 3.,
							1.71-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.07-bckg_fit, 200., 3.5,
							#1.54-bckg_fit, 204., 2.,
							1.92-bckg_fit, 208.5, 3.]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==242:
			idx_interval_dic = {'1':[146,173], '2':[173,212], '3':[241+3,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.3-bckg_fit, 149.5, 3.5,
							2.1-bckg_fit, 156., 5.,
							2.7-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.4-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.5-bckg_fit, 178., 3.5,
							1.75-bckg_fit, 181., 2.5,
							#1.8-bckg_fit, 185.5, 3.,
							1.71-bckg_fit, 190., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.07-bckg_fit, 200., 3.5,
							#1.54-bckg_fit, 204., 2.,
							1.92-bckg_fit, 208.5, 3.]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==241:
			idx_interval_dic = {'1':[146,173], '2':[173,212], '3':[241-3,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.3-bckg_fit, 149.5, 3.5,
							2.1-bckg_fit, 156., 5.,
							2.7-bckg_fit, 162.5, 3.,
							#2.4-bckg_fit, 167., 2.,
							1.9-bckg_fit, 168., 5.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.5-bckg_fit, 178., 3.5,
							1.75-bckg_fit, 181., 2.5,
							#1.8-bckg_fit, 185.5, 3.,
							1.71-bckg_fit, 190., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.07-bckg_fit, 200., 3.5,
							#1.54-bckg_fit, 204., 2.,
							1.92-bckg_fit, 208.5, 3.]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==239 or row==240:
			idx_interval_dic = {'1':[146,173], '2':[173,212], '3':[241-3,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.17-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 3.,
							2.7-bckg_fit, 162., 3.5,
							2.4-bckg_fit, 167., 2.,
							1.9-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.5-bckg_fit, 178., 3.5,
							#1.8-bckg_fit, 180.5, 2.5,
							1.8-bckg_fit, 185.5, 3.,
							1.71-bckg_fit, 192.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.,
							#1.54-bckg_fit, 204., 2.,
							1.8-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if  row==236 or row==237 or row==238:
			idx_interval_dic = {'1':[146,173], '2':[173,212], '3':[241-3,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.17-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 3.,
							2.7-bckg_fit, 162., 3.5,
							2.4-bckg_fit, 167., 2.,
							1.9-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.5-bckg_fit, 178., 3.5,
							#1.8-bckg_fit, 180.5, 2.5,
							1.74-bckg_fit, 184.5, 3.,
							1.71-bckg_fit, 192.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.,
							#1.54-bckg_fit, 204., 2.,
							1.8-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==233 or row==234 or row==235:
			idx_interval_dic = {'1':[146,173], '2':[173,213], '3':[241,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.17-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 3.,
							2.7-bckg_fit, 162., 3.5,
							2.4-bckg_fit, 167., 2.,
							1.9-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.5-bckg_fit, 178., 3.5,
							#1.8-bckg_fit, 180.5, 2.5,
							1.74-bckg_fit, 184.5, 3.,
							1.71-bckg_fit, 192.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.,
							#1.54-bckg_fit, 204., 2.,
							1.8-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if  row==229 or row==230 or row==231 or row==232:
			idx_interval_dic = {'1':[146,173], '2':[173,213], '3':[241,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.17-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 3.,
							2.7-bckg_fit, 162., 3.5,
							2.4-bckg_fit, 167., 2.,
							1.9-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.5-bckg_fit, 178., 3.5,
							#1.8-bckg_fit, 180.5, 2.5,
							#1.74-bckg_fit, 184.5, 3.,
							#1.71-bckg_fit, 190.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.,
							#1.54-bckg_fit, 204., 2.,
							1.8-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==225 or row==226 or row==227 or row==228:
			idx_interval_dic = {'1':[146,173], '2':[173,213], '3':[241,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.17-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 3.,
							2.7-bckg_fit, 162., 3.5,
							2.4-bckg_fit, 167., 2.,
							1.9-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.5-bckg_fit, 178., 3.5,
							#1.8-bckg_fit, 180.5, 2.5,
							1.74-bckg_fit, 184.5, 3.,
							#1.71-bckg_fit, 190.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.,
							#1.54-bckg_fit, 204., 2.,
							1.8-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==224:
			idx_interval_dic = {'1':[146,173], '2':[173,213], '3':[241,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.17-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 3.,
							2.7-bckg_fit, 162., 3.5,
							2.4-bckg_fit, 167., 2.,
							1.9-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.5-bckg_fit, 178., 3.5,
							#1.8-bckg_fit, 180.5, 2.5,
							#1.6-bckg_fit, 183., 3.,
							1.7-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.,
							#1.54-bckg_fit, 204., 3.,
							1.8-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==223:
			idx_interval_dic = {'1':[146,173], '2':[173,213], '3':[241,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.17-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 3.,
							2.7-bckg_fit, 162., 3.5,
							2.4-bckg_fit, 167., 2.,
							1.9-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.5-bckg_fit, 178., 3.5,
							#1.8-bckg_fit, 180.5, 2.5,
							1.6-bckg_fit, 183., 3.,
							1.7-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.,
							#1.54-bckg_fit, 204., 3.,
							1.8-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==221 or row==222:
			idx_interval_dic = {'1':[146,173], '2':[173,213], '3':[241,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.17-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 3.,
							2.7-bckg_fit, 162., 3.5,
							2.4-bckg_fit, 167., 2.,
							1.9-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.5-bckg_fit, 178., 3.5,
							#1.8-bckg_fit, 180.5, 2.5,
							1.7-bckg_fit, 185.5, 3.,
							1.7-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.,
							#1.54-bckg_fit, 204., 3.,
							1.8-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==219 or row==220:
			idx_interval_dic = {'1':[146,173], '2':[173,213], '3':[239-3,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.32-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 160., 4.,
							2.7-bckg_fit, 162., 3.5,
							2.4-bckg_fit, 167., 2.,
							2.2-bckg_fit, 169., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.5-bckg_fit, 178., 3.5,
							#1.8-bckg_fit, 180.5, 2.5,
							1.7-bckg_fit, 185.5, 3.,
							1.7-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.,
							#1.54-bckg_fit, 204., 3.,
							1.8-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 260., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==218:
			idx_interval_dic = {'1':[146,173], '2':[173,213], '3':[239+2,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.32-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 160., 4.,
							2.7-bckg_fit, 162., 3.5,
							2.4-bckg_fit, 167., 2.,
							2.2-bckg_fit, 169., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.5-bckg_fit, 178., 3.5,
							#1.8-bckg_fit, 180.5, 2.5,
							1.7-bckg_fit, 185.5, 3.,
							1.7-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.,
							#1.54-bckg_fit, 204., 3.,
							1.8-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]

		if row==216 or row==217:
			idx_interval_dic = {'1':[146,173], '2':[173,213], '3':[239+2,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.32-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 160., 4.,
							2.7-bckg_fit, 162., 3.5,
							2.4-bckg_fit, 167., 2.,
							2.2-bckg_fit, 169., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.5-bckg_fit, 178., 3.5,
							#1.8-bckg_fit, 180.5, 2.5,
							1.7-bckg_fit, 185.5, 3.,
							1.7-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.,
							1.54-bckg_fit, 204., 3.,
							1.8-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==211 or row==212 or row==213 or row==214 or row==215:
			idx_interval_dic = {'1':[146,173], '2':[173,213], '3':[239+2,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.32-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 160., 4.,
							2.7-bckg_fit, 162., 3.5,
							2.4-bckg_fit, 167., 2.,
							2.2-bckg_fit, 169., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.5-bckg_fit, 178., 3.5,
							#1.8-bckg_fit, 180.5, 2.5,
							1.7-bckg_fit, 185.5, 3.,
							1.7-bckg_fit, 192., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.,
							#1.54-bckg_fit, 204., 3.,
							1.8-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==210:
			idx_interval_dic = {'1':[146,173], '2':[173,213], '3':[239+2,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.32-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 160., 4.,
							2.7-bckg_fit, 162., 3.5,
							2.4-bckg_fit, 167., 2.,
							2.2-bckg_fit, 169., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.5-bckg_fit, 178., 3.5,
							1.8-bckg_fit, 180.5, 2.5,
							1.7-bckg_fit, 185.5, 3.,
							#1.7-bckg_fit, 191., 3.5,
							#1.6-bckg_fit, 197., 3.,
							2.-bckg_fit, 200., 2.,
							#1.54-bckg_fit, 204., 3.,
							1.8-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==209:
			idx_interval_dic = {'1':[146,173], '2':[173,213], '3':[239+2,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.5-bckg_fit, 162.5, 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 169., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							#1.6-bckg_fit, 181., 2.5,
							1.46-bckg_fit, 185., 3.,
							1.7-bckg_fit, 191., 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.9-bckg_fit, 199., 2.,
							1.54-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==208:
			idx_interval_dic = {'1':[146,173], '2':[173,213], '3':[239+2,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.5-bckg_fit, 162.5, 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 169., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							#1.6-bckg_fit, 181., 2.5,
							1.46-bckg_fit, 185., 3.,
							1.7-bckg_fit, 191., 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.9-bckg_fit, 199., 2.,
							1.54-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if  row==207:
			idx_interval_dic = {'1':[146,173], '2':[173,213], '3':[239+2,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.5-bckg_fit, 162.5, 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 169., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							1.6-bckg_fit, 181., 2.5,
							1.46-bckg_fit, 185., 3.,
							1.7-bckg_fit, 191., 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.9-bckg_fit, 199., 2.,
							1.54-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 265., 3.,
							2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==206:
			idx_interval_dic = {'1':[146,173], '2':[173,213], '3':[239+2,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 169., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							1.7-bckg_fit, 181., 2.5,
							1.46-bckg_fit, 185., 3.,
							1.7-bckg_fit, 191., 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.9-bckg_fit, 199., 2.,
							1.54-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 268., 2.,
							2.35-bckg_fit, 265., 3.,
							2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]

		if row==205:
			idx_interval_dic = {'1':[146,173], '2':[173,213], '3':[239+5,270], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 169., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							1.7-bckg_fit, 181., 2.5,
							1.46-bckg_fit, 185., 3.,
							#1.7-bckg_fit, 191., 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.9-bckg_fit, 199., 2.,
							1.54-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260.5, 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==204:
			idx_interval_dic = {'1':[146,173], '2':[173,213], '3':[239+5,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 169., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							1.7-bckg_fit, 181., 2.5,
							1.46-bckg_fit, 185., 3.,
							1.7-bckg_fit, 191., 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.9-bckg_fit, 199., 2.,
							1.54-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260.5, 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==203:
			idx_interval_dic = {'1':[146,173], '2':[174,213], '3':[239,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 169., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							1.7-bckg_fit, 181., 2.5,
							1.46-bckg_fit, 185., 3.,
							1.5-bckg_fit, 193., 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.9-bckg_fit, 199., 2.,
							1.54-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260.5, 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==202:
			idx_interval_dic = {'1':[146,173], '2':[174,213], '3':[239+7,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 169., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							1.7-bckg_fit, 181., 2.5,
							1.46-bckg_fit, 185., 3.,
							1.5-bckg_fit, 193., 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.9-bckg_fit, 199., 2.,
							#1.54-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260.5, 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 201:
			idx_interval_dic = {'1':[146,173], '2':[174,213], '3':[239+7,268], '4':[275,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 169., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							#1.7-bckg_fit, 181., 2.5,
							1.46-bckg_fit, 185., 3.,
							1.5-bckg_fit, 193., 3.5,
							1.6-bckg_fit, 197., 3.,
							1.9-bckg_fit, 199., 2.,
							#1.54-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260.5, 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row == 200:
			idx_interval_dic = {'1':[146,173], '2':[174,212], '3':[239+7,271], '4':[274,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 169., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							#1.7-bckg_fit, 181., 2.5,
							1.46-bckg_fit, 185., 3.,
							1.5-bckg_fit, 193., 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.9-bckg_fit, 199., 2.,
							#1.54-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260.5, 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 198 or row ==  199:
			idx_interval_dic = {'1':[146,173], '2':[174,212], '3':[239+7,271], '4':[274,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							#1.7-bckg_fit, 181., 2.5,
							1.46-bckg_fit, 185., 3.,
							1.5-bckg_fit, 193., 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.9-bckg_fit, 199., 2.,
							#1.54-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260.5, 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row == 196 or row == 197:
			idx_interval_dic = {'1':[146,173], '2':[174,213], '3':[239+7,268], '4':[274,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							#1.7-bckg_fit, 181., 2.5,
							1.46-bckg_fit, 185., 3.,
							1.65-bckg_fit, 191.5, 2.5,
							#1.6-bckg_fit, 197., 3.,
							1.9-bckg_fit, 199., 2.,
							#1.54-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 2.5]
							#1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260.5, 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 194 or row == 195:
			idx_interval_dic = {'1':[146,174], '2':[174,213], '3':[239+7,268], '4':[274,289], '5':[312,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							#1.7-bckg_fit, 181., 2.5,
							1.46-bckg_fit, 185., 3.,
							1.65-bckg_fit, 191.5, 2.5,
							1.6-bckg_fit, 197., 3.,
							1.9-bckg_fit, 199., 2.,
							1.54-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 2.5,
							1.6-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260.5, 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row == 193:
			idx_interval_dic = {'1':[145,174], '2':[174,213], '3':[239+7,268], '4':[274,289], '5':[311,332]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							#1.7-bckg_fit, 181., 2.5,
							1.46-bckg_fit, 185., 3.,
							1.65-bckg_fit, 191.5, 2.5,
							1.9-bckg_fit, 199., 3.5,
							1.54-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260.5, 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row == 191 or row == 192:
			idx_interval_dic = {'1':[145,174], '2':[174,213], '3':[239+7,268], '4':[275,289], '5':[313,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							#1.7-bckg_fit, 181., 2.5,
							#1.46-bckg_fit, 185., 3.,
							1.65-bckg_fit, 191.5, 2.5,
							1.9-bckg_fit, 199., 3.5,
							1.54-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260.5, 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]

		if row == 190:
			idx_interval_dic = {'1':[145,174], '2':[174,213], '3':[239+7,268], '4':[275,289], '5':[313,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							1.7-bckg_fit, 181., 2.5,
							1.46-bckg_fit, 185., 3.,
							1.65-bckg_fit, 191.5, 2.5,
							1.9-bckg_fit, 199., 3.5,
							1.54-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							2.16-bckg_fit, 260.5, 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 189:
			idx_interval_dic = {'1':[145,174], '2':[174,213], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							1.7-bckg_fit, 181., 2.5,
							1.46-bckg_fit, 185., 3.,
							1.65-bckg_fit, 191.5, 2.5,
							1.9-bckg_fit, 199., 3.5,
							1.54-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							2.16-bckg_fit, 260.5, 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 187 or row == 188:
			idx_interval_dic = {'1':[145,174], '2':[174,213], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							#1.7-bckg_fit, 182., 2.5,
							#1.65-bckg_fit, 186., 2.5,
							1.65-bckg_fit, 191.5, 2.5,
							1.9-bckg_fit, 199., 3.5,
							#1.66-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 186:
			idx_interval_dic = {'1':[148,174], '2':[174,213], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							1.7-bckg_fit, 182., 2.5,
							#1.65-bckg_fit, 186., 2.5,
							#1.65-bckg_fit, 191.5, 2.5,
							1.9-bckg_fit, 199., 3.5,
							#1.66-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 185:
			idx_interval_dic = {'1':[146,173], '2':[174,213], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							#1.7-bckg_fit, 182., 2.5,
							#1.65-bckg_fit, 186., 2.5,
							1.65-bckg_fit, 191.5, 2.5,
							1.9-bckg_fit, 199., 3.5,
							#1.66-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 184:
			idx_interval_dic = {'1':[146,173], '2':[174,213], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							#1.7-bckg_fit, 182., 2.5,
							#1.65-bckg_fit, 186., 2.5,
							1.65-bckg_fit, 191.5, 2.5,
							1.9-bckg_fit, 199., 3.5,
							#1.66-bckg_fit, 204., 3.,
							1.87-bckg_fit, 209., 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 183:
			idx_interval_dic = {'1':[146,173], '2':[174,213], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							#1.7-bckg_fit, 182., 2.5,
							1.74-bckg_fit, 185., 3.,
							1.6-bckg_fit, 192., 2.5,
							1.87-bckg_fit, 200., 3.,
							1.58-bckg_fit, 207., 3.,
							1.8-bckg_fit, 209., 3.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row == 182:
			idx_interval_dic = {'1':[146,173], '2':[174,213], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							#2.4-bckg_fit, 166.5, 2.,
							2.11-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							#1.7-bckg_fit, 182., 2.5,
							1.74-bckg_fit, 185., 3.,
							1.6-bckg_fit, 192., 2.5,
							1.87-bckg_fit, 200., 3.,
							#1.58-bckg_fit, 207., 3.,
							1.8-bckg_fit, 209., 3.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]

		if row == 181:
			idx_interval_dic = {'1':[146,173], '2':[174,213], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.1-bckg_fit, 166.5, 3.,
							1.95-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							#1.7-bckg_fit, 182., 2.5,
							1.74-bckg_fit, 185., 3.,
							1.6-bckg_fit, 192., 2.5,
							1.87-bckg_fit, 200., 3.,
							#1.58-bckg_fit, 207., 3.,
							1.8-bckg_fit, 209., 3.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 179 or row == 180:
			idx_interval_dic = {'1':[146,173], '2':[174,212], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.1-bckg_fit, 166.5, 3.,
							1.95-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.22-bckg_fit, 178., 3.5,
							#1.7-bckg_fit, 182., 2.5,
							1.74-bckg_fit, 185., 3.,
							1.5-bckg_fit, 191., 4.,
							1.87-bckg_fit, 200., 3.,
							#1.58-bckg_fit, 207., 3.,
							1.8-bckg_fit, 209., 3.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 178:
			idx_interval_dic = {'1':[146,173], '2':[174,212], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.1-bckg_fit, 166.5, 3.,
							1.95-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.28-bckg_fit, 178., 3.5,
							#1.7-bckg_fit, 182., 2.5,
							1.7-bckg_fit, 185., 3.,
							1.5-bckg_fit, 191., 3.,
							1.87-bckg_fit, 200., 3.,
							1.58-bckg_fit, 207., 3.,
							1.8-bckg_fit, 210., 3.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]

		if row == 176 or row == 177:
			idx_interval_dic = {'1':[146,173], '2':[174,212], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.1-bckg_fit, 167., 3.,
							2.1-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.19-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 182., 2.5,
							1.57-bckg_fit, 185., 2.5,
							#1.5-bckg_fit, 190., 3.,
							1.84-bckg_fit, 200., 4.,
							1.58-bckg_fit, 204., 3.,
							1.8-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 175:
			idx_interval_dic = {'1':[146,173], '2':[174,212], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.1-bckg_fit, 167., 3.,
							2.1-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.19-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 182., 2.5,
							1.57-bckg_fit, 185., 2.5,
							1.5-bckg_fit, 190., 3.,
							1.84-bckg_fit, 200., 4.,
							1.58-bckg_fit, 204., 3.,
							1.8-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 174:
			idx_interval_dic = {'1':[146,173], '2':[174,212], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.1-bckg_fit, 167., 3.,
							2.1-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.19-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 182., 2.5,
							1.57-bckg_fit, 185., 2.5,
							1.5-bckg_fit, 190., 3.,
							1.84-bckg_fit, 200., 5.,
							1.58-bckg_fit, 204., 10.,
							1.8-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 173:
			idx_interval_dic = {'1':[146,173], '2':[174,212], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.1-bckg_fit, 167., 3.,
							2.1-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.7-bckg_fit, 182., 2.5,
							1.7-bckg_fit, 185., 1.5,
							#1.6-bckg_fit, 191., 3.,
							1.92-bckg_fit, 200., 5.,
							#1.56-bckg_fit, 204., 3.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row == 170 or row == 171 or row == 172:
			idx_interval_dic = {'1':[146,173], '2':[174,212], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.1-bckg_fit, 167., 3.,
							2.1-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.7-bckg_fit, 182., 2.5,
							1.7-bckg_fit, 185., 1.5,
							#1.6-bckg_fit, 191., 3.,
							1.92-bckg_fit, 200., 5.,
							#1.56-bckg_fit, 204., 3.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 169:
			idx_interval_dic = {'1':[146,173], '2':[174,212], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.1-bckg_fit, 167., 3.,
							2.1-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.7-bckg_fit, 182., 2.5,
							1.7-bckg_fit, 185., 1.5,
							1.6-bckg_fit, 191., 3.,
							1.92-bckg_fit, 200., 5.,
							#1.56-bckg_fit, 204., 3.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 282., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 168:
			idx_interval_dic = {'1':[146,173], '2':[174,212], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.1-bckg_fit, 167., 3.,
							2.1-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.7-bckg_fit, 182., 2.5,
							1.7-bckg_fit, 185., 1.5,
							#1.6-bckg_fit, 192., 3.,
							1.92-bckg_fit, 200., 5.,
							#1.56-bckg_fit, 204., 3.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							#2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 282., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 163 or row == 164 or row == 165 or row == 166 or row == 167:
			idx_interval_dic = {'1':[146,173], '2':[174,212], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.7-bckg_fit, 182., 2.5,
							1.7-bckg_fit, 185., 1.5,
							#1.6-bckg_fit, 192., 3.,
							1.92-bckg_fit, 200., 5.,
							#1.56-bckg_fit, 204., 3.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.5-bckg_fit, 255., 3.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 282., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 161 or row == 162:
			idx_interval_dic = {'1':[146,173], '2':[174,212], '3':[239,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.7-bckg_fit, 182., 2.5,
							1.7-bckg_fit, 185., 1.5,
							#1.6-bckg_fit, 192., 3.,
							1.92-bckg_fit, 200., 5.,
							#1.56-bckg_fit, 204., 3.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 282., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 156 or row == 157 or row == 158 or row == 159 or row == 160:
			idx_interval_dic = {'1':[146,174], '2':[174,212], '3':[239,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.7-bckg_fit, 182., 2.5,
							1.7-bckg_fit, 185., 1.5,
							#1.6-bckg_fit, 192., 3.,
							1.92-bckg_fit, 200., 5.,
							#1.56-bckg_fit, 204., 3.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 282., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 155:
			idx_interval_dic = {'1':[146,174], '2':[174,212], '3':[239,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.7-bckg_fit, 182., 2.5,
							1.7-bckg_fit, 185., 1.5,
							#1.6-bckg_fit, 192., 3.,
							1.92-bckg_fit, 200., 5.,
							#1.56-bckg_fit, 204., 3.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 282., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 150:
			idx_interval_dic = {'1':[146,174], '2':[174,212], '3':[239,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.7-bckg_fit, 182., 2.5,
							1.7-bckg_fit, 185., 1.5,
							1.6-bckg_fit, 192., 3.,
							1.92-bckg_fit, 200., 5.,
							#1.56-bckg_fit, 204., 3.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.2-bckg_fit, 282., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 149 or row == 150 or row == 152 or row == 153 or row == 154:
			idx_interval_dic = {'1':[146,174], '2':[174,212], '3':[239,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.7-bckg_fit, 182., 2.5,
							1.7-bckg_fit, 185., 1.5,
							1.6-bckg_fit, 192., 3.,
							1.92-bckg_fit, 200., 5.,
							#1.56-bckg_fit, 204., 3.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 282., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 143 or row == 144 or row == 145 or row == 146 or row == 147:
			idx_interval_dic = {'1':[146,174], '2':[174,213], '3':[239,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.7-bckg_fit, 182., 2.5,
							1.7-bckg_fit, 185., 1.5,
							1.6-bckg_fit, 192., 3.,
							1.92-bckg_fit, 200., 5.,
							#1.56-bckg_fit, 204., 3.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 282., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 140 or row == 141 or row == 142:
			idx_interval_dic = {'1':[146,174], '2':[174,213], '3':[239,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.7-bckg_fit, 182., 2.5,
							1.7-bckg_fit, 185., 1.5,
							1.6-bckg_fit, 192., 3.,
							1.92-bckg_fit, 200., 5.,
							1.56-bckg_fit, 204., 3.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.43-bckg_fit, 258., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.2-bckg_fit, 282., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 137 or row == 138 or row == 139:
			idx_interval_dic = {'1':[146,174], '2':[174,213], '3':[239+7,268], '4':[275,289], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.7-bckg_fit, 182., 2.5,
							1.7-bckg_fit, 185., 1.5,
							1.6-bckg_fit, 192., 3.,
							1.92-bckg_fit, 200., 5.,
							1.56-bckg_fit, 204., 3.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.43-bckg_fit, 259., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.2-bckg_fit, 282., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 134 or row == 135 or row == 136:
			idx_interval_dic = {'1':[146,173], '2':[174,213], '3':[239+7,268], '4':[275,290], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.7-bckg_fit, 182., 2.5,
							1.7-bckg_fit, 185., 1.5,
							1.6-bckg_fit, 192., 3.,
							1.92-bckg_fit, 200., 5.,
							1.56-bckg_fit, 204., 3.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.43-bckg_fit, 259., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 133:
			idx_interval_dic = {'1':[146,173], '2':[174,213], '3':[239+7,268], '4':[275,290], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							1.7-bckg_fit, 182., 2.5,
							1.7-bckg_fit, 185., 1.5,
							1.6-bckg_fit, 192., 3.,
							1.92-bckg_fit, 200., 5.,
							1.56-bckg_fit, 204., 3.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.43-bckg_fit, 259., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 132:
			idx_interval_dic = {'1':[146,173], '2':[174,213], '3':[239+7,268], '4':[275,290], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							1.7-bckg_fit, 182., 2.5,
							1.7-bckg_fit, 185., 1.5,
							1.6-bckg_fit, 192., 3.,
							#1.6-bckg_fit, 194.5, 3.,
							1.92-bckg_fit, 200., 5.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.43-bckg_fit, 259., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 131:
			idx_interval_dic = {'1':[146,173], '2':[174,212], '3':[239,268], '4':[275,290], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.62-bckg_fit, 182., 2.,
							1.65-bckg_fit, 185., 2.5,
							1.74-bckg_fit, 192., 3.,
							#1.6-bckg_fit, 194.5, 3.,
							1.92-bckg_fit, 200., 5.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.43-bckg_fit, 259., 2.5,
							2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 130:
			idx_interval_dic = {'1':[146,173], '2':[174,212], '3':[239,268], '4':[275,290], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.62-bckg_fit, 182., 2.,
							1.65-bckg_fit, 185., 2.5,
							1.74-bckg_fit, 192., 3.,
							#1.6-bckg_fit, 194.5, 3.,
							1.92-bckg_fit, 200., 5.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.43-bckg_fit, 259., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 129:
			idx_interval_dic = {'1':[146,173], '2':[174,213], '3':[239,268], '4':[275,290], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.78-bckg_fit, 181., 1.5,
							1.74-bckg_fit, 186., 1.5,
							1.6-bckg_fit, 192., 3.,
							#1.6-bckg_fit, 194.5, 3.,
							1.92-bckg_fit, 200., 5.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.43-bckg_fit, 259., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 128:
			idx_interval_dic = {'1':[146,173], '2':[174,213], '3':[239,268], '4':[275,290], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							1.7-bckg_fit, 181., 2.5,
							1.74-bckg_fit, 186., 1.5,
							1.6-bckg_fit, 192., 3.,
							#1.6-bckg_fit, 194.5, 3.,
							1.92-bckg_fit, 200., 5.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.43-bckg_fit, 259., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 127:
			idx_interval_dic = {'1':[146,173], '2':[174,213], '3':[239+7,268], '4':[275,290], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.7-bckg_fit, 181., 2.5,
							1.74-bckg_fit, 186., 1.5,
							1.6-bckg_fit, 192., 3.,
							#1.6-bckg_fit, 194.5, 3.,
							1.92-bckg_fit, 200., 5.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.43-bckg_fit, 259., 3.5,
							#2.16-bckg_fit, 260., 2.,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 126:
			idx_interval_dic = {'1':[148,173], '2':[174,212], '3':[239,268], '4':[275,290], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.47-bckg_fit, 178., 4.,
							#1.62-bckg_fit, 182., 2.,
							#1.47-bckg_fit, 186.5, 1.5,
							#1.5-bckg_fit, 190.5, 2.5,
							#1.6-bckg_fit, 194.5, 3.,
							1.92-bckg_fit, 200., 5.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.74-bckg_fit, 260., 3.5,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 125:
			idx_interval_dic = {'1':[148,173], '2':[174,212], '3':[239,268], '4':[275,290], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.41-bckg_fit, 178., 4.,
							1.62-bckg_fit, 182., 2.,
							1.47-bckg_fit, 186.5, 1.5,
							1.5-bckg_fit, 190.5, 2.5,
							1.6-bckg_fit, 194.5, 3.,
							1.92-bckg_fit, 200., 5.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.74-bckg_fit, 260., 3.5,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 124:
			idx_interval_dic = {'1':[148,173], '2':[174,212], '3':[239,268], '4':[275,290], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.33-bckg_fit, 178., 4.,
							#1.75-bckg_fit, 181., 2.,
							1.61-bckg_fit, 185., 3.5,
							1.7-bckg_fit, 191., 4.,
							1.92-bckg_fit, 200., 5.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.74-bckg_fit, 260., 3.5,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 123:
			idx_interval_dic = {'1':[148,173], '2':[174,212], '3':[239,268], '4':[275,290], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.2-bckg_fit, 167.5, 3.5,
							2.02-bckg_fit, 170., 2.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.33-bckg_fit, 178., 4.,
							#1.75-bckg_fit, 181., 2.,
							1.61-bckg_fit, 185., 3.5,
							1.7-bckg_fit, 191., 4.,
							1.92-bckg_fit, 200., 5.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.74-bckg_fit, 260., 3.5,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row == 122:
			idx_interval_dic = {'1':[148,173], '2':[174,212], '3':[239,268], '4':[275,290], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.17-bckg_fit, 167.5, 4.,
							2.02-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.33-bckg_fit, 178., 4.,
							#1.75-bckg_fit, 181., 2.,
							1.61-bckg_fit, 185., 3.5,
							1.7-bckg_fit, 191., 4.,
							1.92-bckg_fit, 200., 5.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.74-bckg_fit, 260., 3.5,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]

		if row == 121:
			idx_interval_dic = {'1':[142,173], '2':[174,212], '3':[239,268], '4':[275,290], '5':[312,330]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							1.67-bckg_fit, 144., 3.,
							2.13-bckg_fit, 149.5, 3.5,
							2.05-bckg_fit, 156., 4.,
							2.3-bckg_fit, 163., 3.5,
							2.17-bckg_fit, 167.5, 4.,
							2.02-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.33-bckg_fit, 178., 4.,
							1.75-bckg_fit, 181., 2.,
							1.61-bckg_fit, 185., 3.5,
							1.7-bckg_fit, 191., 4.,
							1.92-bckg_fit, 200., 5.,
							1.77-bckg_fit, 208.5, 4.]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.83-bckg_fit, 247., 10.,
							2.74-bckg_fit, 260., 3.5,
							2.35-bckg_fit, 265., 3.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]



		if row==118 or row==119 or row==120:
			idx_interval_dic = {'1':[147,174-1], '2':[174+1,214], '3':[241+2,270-2], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.7-bckg_fit, 175., 2.,
							2.2-bckg_fit, 178.5, 3.5,
							#1.6-bckg_fit, 181., 5.,
							#1.6-bckg_fit, 183., 5.,
							1.59-bckg_fit, 191., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							16.-bckg_fit, 247., 10.,
							#13.5-bckg_fit, 250., 5.,
							#2.75-bckg_fit, 251., 3.,
							2.71-bckg_fit, 259.3, 3.,
							#2.22-bckg_fit, 260., 5.,
							2.67-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.23-bckg_fit, 279., 3.,
							#2.25-bckg_fit, 281., 2.5,
							2.15-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==115 or row==116 or row==117:
			idx_interval_dic = {'1':[147,174-1], '2':[174+1,214], '3':[241-3,270-2], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.7-bckg_fit, 175., 2.,
							2.2-bckg_fit, 178.5, 3.5,
							#1.6-bckg_fit, 181., 5.,
							#1.6-bckg_fit, 183., 5.,
							1.59-bckg_fit, 191., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							16.-bckg_fit, 247., 10.,
							#13.5-bckg_fit, 250., 5.,
							#2.75-bckg_fit, 251., 3.,
							2.71-bckg_fit, 259.3, 3.,
							#2.22-bckg_fit, 260., 5.,
							2.67-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.23-bckg_fit, 279., 3.,
							#2.25-bckg_fit, 281., 2.5,
							2.15-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==112 or row==113 or row==114:
			idx_interval_dic = {'1':[147,174-1], '2':[174+1,214], '3':[241-3,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.7-bckg_fit, 175., 2.,
							2.2-bckg_fit, 178.5, 3.5,
							1.6-bckg_fit, 181., 5.,
							#1.6-bckg_fit, 183., 5.,
							1.59-bckg_fit, 191., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							14.4-bckg_fit, 247., 10.,
							#13.5-bckg_fit, 250., 5.,
							2.75-bckg_fit, 251., 3.,
							2.75-bckg_fit, 259., 3.,
							#2.22-bckg_fit, 260., 5.,
							2.3-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.23-bckg_fit, 279., 3.,
							#2.25-bckg_fit, 281., 2.5,
							2.15-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==109 or row==110 or row==111:
			idx_interval_dic = {'1':[147,174-1], '2':[174+1,214], '3':[241+1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.7-bckg_fit, 175., 2.,
							2.2-bckg_fit, 178.5, 3.5,
							1.6-bckg_fit, 181., 5.,
							#1.6-bckg_fit, 183., 5.,
							1.59-bckg_fit, 191., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							12.8-bckg_fit, 247., 10.,
							#13.5-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.52-bckg_fit, 259., 3.,
							#2.22-bckg_fit, 260., 5.,
							2.28-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.23-bckg_fit, 279., 3.,
							#2.25-bckg_fit, 281., 2.5,
							2.15-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]

		if row==108:
			idx_interval_dic = {'1':[147,174-1], '2':[174+1,214], '3':[241+1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.7-bckg_fit, 175., 2.,
							2.2-bckg_fit, 178.5, 3.5,
							#1.6-bckg_fit, 181., 2.,
							#1.6-bckg_fit, 183., 5.,
							1.59-bckg_fit, 191., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							12.8-bckg_fit, 247., 10.,
							13.5-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.52-bckg_fit, 259., 3.,
							#2.22-bckg_fit, 260., 5.,
							2.28-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.23-bckg_fit, 279., 3.,
							#2.25-bckg_fit, 281., 2.5,
							2.15-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==106 or row==107:
			idx_interval_dic = {'1':[147,174-1], '2':[174+1,214], '3':[241+1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.7-bckg_fit, 175., 2.,
							2.2-bckg_fit, 178.5, 3.5,
							#1.6-bckg_fit, 181., 2.,
							1.6-bckg_fit, 183., 5.,
							#1.74-bckg_fit, 191.5, 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							14.3-bckg_fit, 247., 10.,
							#11.2-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.52-bckg_fit, 259., 3.,
							#2.22-bckg_fit, 260., 5.,
							2.28-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.23-bckg_fit, 279., 3.,
							#2.25-bckg_fit, 281., 2.5,
							2.15-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]

		if row==104 or row==105:
			idx_interval_dic = {'1':[147,174-1], '2':[174+1,214], '3':[241+1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.7-bckg_fit, 175., 2.,
							2.2-bckg_fit, 178.5, 3.5,
							#1.6-bckg_fit, 181., 2.,
							1.47-bckg_fit, 186., 3.,
							1.74-bckg_fit, 191.5, 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							14.3-bckg_fit, 247., 10.,
							#11.2-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.52-bckg_fit, 259., 3.,
							#2.22-bckg_fit, 260., 5.,
							2.28-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.23-bckg_fit, 279., 3.,
							#2.25-bckg_fit, 281., 2.5,
							2.15-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==103:
			idx_interval_dic = {'1':[147,174-1], '2':[174+1,214], '3':[241+1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.7-bckg_fit, 175., 2.,
							2.2-bckg_fit, 178.5, 3.5,
							#1.6-bckg_fit, 181., 2.,
							1.47-bckg_fit, 186., 3.,
							1.74-bckg_fit, 191.5, 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							14.3-bckg_fit, 247., 10.,
							#11.2-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.52-bckg_fit, 259., 3.,
							#2.22-bckg_fit, 260., 5.,
							2.28-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.23-bckg_fit, 279., 3.,
							#2.25-bckg_fit, 281., 2.5,
							2.15-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==101 or row==102:
			idx_interval_dic = {'1':[147,174-1], '2':[174-0,214], '3':[241+1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.7-bckg_fit, 175., 2.,
							2.2-bckg_fit, 178.5, 3.5,
							1.7-bckg_fit, 181., 2.,
							#1.67-bckg_fit, 185.5, 3.,
							1.74-bckg_fit, 191.5, 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							14.3-bckg_fit, 247., 10.,
							#11.2-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.52-bckg_fit, 259., 3.,
							#2.22-bckg_fit, 260., 5.,
							2.28-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.23-bckg_fit, 279., 3.,
							#2.25-bckg_fit, 281., 2.5,
							2.15-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==100:
			idx_interval_dic = {'1':[147,174-1], '2':[174-0,214], '3':[241+1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.7-bckg_fit, 175., 2.,
							2.2-bckg_fit, 178.5, 3.5,
							#1.7-bckg_fit, 181., 2.,
							#1.67-bckg_fit, 185.5, 3.,
							1.74-bckg_fit, 191.5, 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							14.3-bckg_fit, 247., 10.,
							#11.2-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.52-bckg_fit, 259., 3.,
							#2.22-bckg_fit, 260., 5.,
							2.28-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.23-bckg_fit, 279., 3.,
							#2.25-bckg_fit, 281., 2.5,
							2.15-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==99:
			idx_interval_dic = {'1':[147,174-1], '2':[174-0,214], '3':[241-1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.7-bckg_fit, 175., 2.,
							2.2-bckg_fit, 178.5, 3.5,
							#1.7-bckg_fit, 181., 2.,
							1.67-bckg_fit, 185.5, 3.,
							1.74-bckg_fit, 191.5, 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.3-bckg_fit, 247., 10.,
							#11.2-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 258.5, 3.,
							2.26-bckg_fit, 261., 5.,
							2.38-bckg_fit, 264., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.23-bckg_fit, 279., 3.,
							#2.25-bckg_fit, 281., 2.5,
							2.15-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==96 or row==97 or row==98:
			idx_interval_dic = {'1':[147,174-1], '2':[174-0,214], '3':[241-1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.7-bckg_fit, 175., 2.,
							2.2-bckg_fit, 178.5, 3.5,
							#1.7-bckg_fit, 181., 2.,
							1.67-bckg_fit, 185.5, 3.,
							1.74-bckg_fit, 191.5, 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.3-bckg_fit, 247., 10.,
							#11.2-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 258.5, 3.,
							#2.26-bckg_fit, 261., 5.,
							2.38-bckg_fit, 264., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.23-bckg_fit, 279., 3.,
							#2.25-bckg_fit, 281., 2.5,
							2.15-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==95:
			idx_interval_dic = {'1':[147,174-1], '2':[174-0,214], '3':[241-1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.7-bckg_fit, 175., 2.,
							2.2-bckg_fit, 178.5, 3.5,
							#1.7-bckg_fit, 181., 2.,
							1.67-bckg_fit, 185.5, 3.,
							1.74-bckg_fit, 191.5, 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.3-bckg_fit, 247., 10.,
							#11.2-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 258.5, 3.,
							#2.26-bckg_fit, 261., 5.,
							2.38-bckg_fit, 264., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]

		if row==93 or row==94:
			idx_interval_dic = {'1':[147,174-1], '2':[174-0,214], '3':[241-1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.7-bckg_fit, 175., 2.,
							2.2-bckg_fit, 178.5, 3.5,
							#1.7-bckg_fit, 181., 2.,
							1.67-bckg_fit, 185.5, 3.,
							1.74-bckg_fit, 191.5, 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.3-bckg_fit, 247., 10.,
							#11.2-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 258.5, 3.,
							#2.26-bckg_fit, 261., 5.,
							2.38-bckg_fit, 264., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							#2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]

		if row==92:
			idx_interval_dic = {'1':[147,174-1], '2':[174-0,214], '3':[241-1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.7-bckg_fit, 175., 2.,
							2.2-bckg_fit, 178.5, 3.5,
							1.7-bckg_fit, 181., 2.,
							1.67-bckg_fit, 185.5, 3.,
							1.74-bckg_fit, 191.5, 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.3-bckg_fit, 247., 10.,
							11.2-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 258.5, 3.,
							#2.26-bckg_fit, 261., 5.,
							2.38-bckg_fit, 264., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							#2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if  row==89 or row==90 or row==91:
			idx_interval_dic = {'1':[147,174-1], '2':[174-0,214], '3':[241+1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.7-bckg_fit, 175., 2.,
							2.2-bckg_fit, 178.5, 3.5,
							#1.7-bckg_fit, 181., 2.,
							1.58-bckg_fit, 184., 3.,
							1.74-bckg_fit, 191.5, 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.3-bckg_fit, 247., 10.,
							#11.2-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==88:
			idx_interval_dic = {'1':[147,174-1], '2':[174-0,214], '3':[241+1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							1.7-bckg_fit, 175., 2.,
							2.2-bckg_fit, 178.5, 3.5,
							1.7-bckg_fit, 181., 2.,
							1.58-bckg_fit, 184., 3.,
							1.74-bckg_fit, 191.5, 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.3-bckg_fit, 247., 10.,
							#11.2-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							#2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==87:
			idx_interval_dic = {'1':[147,174-1], '2':[174+1,214], '3':[241+1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.37-bckg_fit, 167., 3.5,
							2.18-bckg_fit, 171., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.8-bckg_fit, 176., 3.,
							2.4-bckg_fit, 178.5, 3.5,
							1.8-bckg_fit, 181., 2.,
							#1.83-bckg_fit, 186., 2.,
							1.74-bckg_fit, 191.5, 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.3-bckg_fit, 247., 10.,
							#11.2-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							#2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==86:
			idx_interval_dic = {'1':[147,174-1], '2':[174+1,214], '3':[241+1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							#2.37-bckg_fit, 167.5, 3.,
							2.21-bckg_fit, 169.5, 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.8-bckg_fit, 176., 3.,
							2.4-bckg_fit, 178.5, 3.5,
							1.8-bckg_fit, 181., 2.,
							1.83-bckg_fit, 186., 2.,
							1.74-bckg_fit, 191.5, 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.3-bckg_fit, 247., 10.,
							#11.2-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							#2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]

		if row==85:
			idx_interval_dic = {'1':[147,174-1], '2':[174+1,214], '3':[241+1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							#2.37-bckg_fit, 167.5, 3.,
							2.21-bckg_fit, 169.5, 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.8-bckg_fit, 176., 3.,
							2.4-bckg_fit, 178.5, 3.5,
							1.8-bckg_fit, 181., 2.,
							1.83-bckg_fit, 186., 2.,
							#1.74-bckg_fit, 191.5, 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.3-bckg_fit, 247., 10.,
							11.2-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							#2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if  row==84:
			idx_interval_dic = {'1':[147,174-1], '2':[174+1,214], '3':[241+1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							#2.37-bckg_fit, 167.5, 3.,
							2.21-bckg_fit, 169.5, 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.8-bckg_fit, 176., 3.,
							2.4-bckg_fit, 178.5, 3.5,
							1.8-bckg_fit, 181., 2.,
							1.83-bckg_fit, 186., 2.,
							1.74-bckg_fit, 191.5, 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.3-bckg_fit, 247., 10.,
							11.2-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							#2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if  row==82 or row==83:
			idx_interval_dic = {'1':[147,174-1], '2':[174+1,214], '3':[241+1,270-3], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							#2.37-bckg_fit, 167.5, 3.,
							2.21-bckg_fit, 169.5, 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.8-bckg_fit, 176., 3.,
							2.4-bckg_fit, 178.5, 3.5,
							#1.8-bckg_fit, 181., 2.,
							1.83-bckg_fit, 186., 2.,
							2.0-bckg_fit, 193., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.3-bckg_fit, 247., 10.,
							11.2-bckg_fit, 250., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==80 or row==81:
			idx_interval_dic = {'1':[147,174-1], '2':[174+1,214], '3':[241+1,270-1], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							#2.37-bckg_fit, 167.5, 3.,
							2.21-bckg_fit, 169.5, 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.8-bckg_fit, 176., 3.,
							2.4-bckg_fit, 178.5, 3.5,
							1.8-bckg_fit, 181., 2.,
							1.83-bckg_fit, 186., 2.,
							2.0-bckg_fit, 193., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							#2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==78 or row==79:
			idx_interval_dic = {'1':[147,174-1], '2':[174,214], '3':[241+1,270-1], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							#2.37-bckg_fit, 167.5, 3.,
							2.21-bckg_fit, 169.5, 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.8-bckg_fit, 176., 3.,
							2.4-bckg_fit, 178.5, 3.5,
							#1.8-bckg_fit, 181., 2.,
							1.83-bckg_fit, 186., 2.,
							2.0-bckg_fit, 193., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							#2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==76 or row==77:
			idx_interval_dic = {'1':[147,174-1], '2':[174,214], '3':[241+1,270-1], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							#2.37-bckg_fit, 167.5, 3.,
							2.21-bckg_fit, 169.5, 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.87-bckg_fit, 174., 3.,
							2.4-bckg_fit, 178.5, 3.5,
							1.9-bckg_fit, 181.5, 2.,
							1.83-bckg_fit, 186., 2.,
							2.0-bckg_fit, 193., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							#2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==72 or row==73 or row==74 or row==75:
			idx_interval_dic = {'1':[147,174-1], '2':[174,214], '3':[241+1,270-5], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							#2.37-bckg_fit, 167.5, 3.,
							2.21-bckg_fit, 169.5, 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.87-bckg_fit, 174., 3.,
							2.4-bckg_fit, 178.5, 3.5,
							1.9-bckg_fit, 181.5, 2.,
							1.83-bckg_fit, 186., 2.,
							2.0-bckg_fit, 193., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							#2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==71:
			idx_interval_dic = {'1':[147,174-1], '2':[174,214], '3':[241+1,270+1], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							#2.37-bckg_fit, 167.5, 3.,
							2.21-bckg_fit, 169.5, 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.87-bckg_fit, 174., 3.,
							2.4-bckg_fit, 178.5, 3.5,
							1.9-bckg_fit, 181.5, 2.,
							1.83-bckg_fit, 186., 2.,
							2.0-bckg_fit, 193., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							#2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==70:
			idx_interval_dic = {'1':[147,174-1], '2':[174,214], '3':[241+1,270+1], '4':[275,288+1], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							#2.37-bckg_fit, 167.5, 3.,
							2.21-bckg_fit, 169.5, 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.87-bckg_fit, 174., 3.,
							2.36-bckg_fit, 178.5, 3.5,
							2.-bckg_fit, 181.5, 3.,
							1.83-bckg_fit, 186., 2.,
							2.0-bckg_fit, 193., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							#2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==69:
			idx_interval_dic = {'1':[147,174-1], '2':[174,214], '3':[241+1,270+1], '4':[275,288+1], '5':[311+1,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							#2.37-bckg_fit, 167.5, 3.,
							2.21-bckg_fit, 169.5, 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.87-bckg_fit, 174., 3.,
							2.36-bckg_fit, 178.5, 3.5,
							2.-bckg_fit, 181.5, 3.,
							1.83-bckg_fit, 186., 2.,
							2.0-bckg_fit, 193., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							#2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if  row==67 or row==68:
			idx_interval_dic = {'1':[147-2,174-1], '2':[174,214], '3':[241+1,270+1], '4':[275,288+1], '5':[311+1,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							#2.37-bckg_fit, 167.5, 3.,
							2.21-bckg_fit, 169.5, 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.87-bckg_fit, 174., 3.,
							2.36-bckg_fit, 178.5, 3.5,
							#2.-bckg_fit, 181.5, 3.,
							1.83-bckg_fit, 185., 3.,
							2.0-bckg_fit, 193., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==65 or row==66 or row==67:
			idx_interval_dic = {'1':[147-2,174-1], '2':[174,214], '3':[241+1,270+1], '4':[275,288+1], '5':[311+1,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							#2.37-bckg_fit, 167.5, 3.,
							2.21-bckg_fit, 169.5, 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.87-bckg_fit, 174., 3.,
							2.36-bckg_fit, 178.5, 3.5,
							2.-bckg_fit, 181.5, 3.,
							1.83-bckg_fit, 185., 3.,
							2.0-bckg_fit, 193., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.78-bckg_fit, 279.5, 3.5,
							2.25-bckg_fit, 282., 3.,
							2.15-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==63 or row==64:
			idx_interval_dic = {'1':[147-2,174-1], '2':[174,214], '3':[241+1,270+1], '4':[275,288+1], '5':[311+1,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							#2.37-bckg_fit, 167.5, 3.,
							2.21-bckg_fit, 169.5, 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.87-bckg_fit, 174., 3.,
							2.36-bckg_fit, 178.5, 3.5,
							2.-bckg_fit, 181.5, 3.,
							1.83-bckg_fit, 185., 3.,
							2.0-bckg_fit, 193., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.98-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.41-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==62:
			idx_interval_dic = {'1':[147-2,174-1], '2':[174,214], '3':[241+1,270+1], '4':[275,288+1], '5':[311+1,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							#2.37-bckg_fit, 167.5, 3.,
							2.21-bckg_fit, 169.5, 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.87-bckg_fit, 174., 3.,
							2.36-bckg_fit, 178.5, 3.5,
							2.-bckg_fit, 181.5, 3.,
							1.83-bckg_fit, 185., 3.,
							2.0-bckg_fit, 193., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.98-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.41-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==61:
			idx_interval_dic = {'1':[147-2,174-1], '2':[175,214], '3':[241+1,270+1], '4':[275,288+1], '5':[311+1,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							#2.37-bckg_fit, 167.5, 3.,
							2.21-bckg_fit, 169.5, 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.87-bckg_fit, 174., 3.,
							2.36-bckg_fit, 178.5, 3.5,
							2.-bckg_fit, 181.5, 3.,
							1.83-bckg_fit, 185., 3.,
							2.0-bckg_fit, 193., 3.,
							#2.16-bckg_fit, 201., 3.,
							#1.87-bckg_fit, 196., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.05-bckg_fit, 200., 3.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.98-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.41-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==60:
			idx_interval_dic = {'1':[147-2,174-1], '2':[175+1,214], '3':[241+1,270+1], '4':[275,288+1], '5':[311+1,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.17-bckg_fit, 167., 3.,
							2.27-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.87-bckg_fit, 174., 3.,
							2.36-bckg_fit, 178.5, 3.5,
							#2.-bckg_fit, 181., 3.,
							1.81-bckg_fit, 185.4, 3.,
							2.0-bckg_fit, 193., 3.,
							2.16-bckg_fit, 201., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							#2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							1.98-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.98-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.41-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]

		if row==59:
			idx_interval_dic = {'1':[147-2,174-1], '2':[175-2,214], '3':[241+1,270+1], '4':[275,288+1], '5':[311+1,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.17-bckg_fit, 167., 3.,
							2.27-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							#1.77-bckg_fit, 182., 3.,
							1.87-bckg_fit, 186.5, 3.,
							1.86-bckg_fit, 193., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							2.18-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.98-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.41-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==58:
			idx_interval_dic = {'1':[147,174], '2':[175,214], '3':[241+1,270+1], '4':[275,288+1], '5':[311+1,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.17-bckg_fit, 167., 3.,
							2.27-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							#1.77-bckg_fit, 182., 3.,
							1.87-bckg_fit, 186.5, 3.,
							1.86-bckg_fit, 193., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							2.18-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.98-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.41-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==57:
			idx_interval_dic = {'1':[147,174], '2':[175,214], '3':[241+1,270+1], '4':[275,288+1], '5':[311+1,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.18-bckg_fit, 150.5, 3.,
							2.04-bckg_fit, 157., 3.,
							#2.5-bckg_fit, 159.5, 3.,
							2.5-bckg_fit, 163., 3.,
							2.17-bckg_fit, 167., 3.,
							2.27-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							1.77-bckg_fit, 182., 3.,
							1.87-bckg_fit, 186.5, 3.,
							1.86-bckg_fit, 193., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							2.18-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							13.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.6-bckg_fit, 259.5, 3.5,
							#2.26-bckg_fit, 261., 5.,
							2.58-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.98-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.41-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==56:
			idx_interval_dic = {'1':[147,174], '2':[175,214], '3':[241+1,270+1], '4':[275,288+1], '5':[311+1,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.45-bckg_fit, 150., 3.,
							2.34-bckg_fit, 155., 3.,
							2.34-bckg_fit, 159.5, 3.,
							2.91-bckg_fit, 163., 3.,
							#2.4-bckg_fit, 167., 3.,
							2.64-bckg_fit, 169., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							#2.05-bckg_fit, 181.5, 5.,
							1.87-bckg_fit, 186.5, 3.,
							1.86-bckg_fit, 193., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							2.18-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.4-bckg_fit, 259., 3.5,
							2.26-bckg_fit, 261., 5.,
							2.31-bckg_fit, 265., 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.98-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.41-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==53 or row==54 or row==55:
			idx_interval_dic = {'1':[147,174], '2':[175,214], '3':[241+1,270-2], '4':[275,288+1], '5':[311+1,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.45-bckg_fit, 150., 3.,
							2.34-bckg_fit, 155., 3.,
							2.34-bckg_fit, 159.5, 3.,
							2.91-bckg_fit, 163., 3.,
							#2.4-bckg_fit, 167., 3.,
							2.64-bckg_fit, 169., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							#2.05-bckg_fit, 181.5, 5.,
							1.87-bckg_fit, 186.5, 3.,
							1.86-bckg_fit, 193., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							2.18-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.98-bckg_fit, 280., 3.5,
							2.28-bckg_fit, 283., 3.,
							2.41-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==49 or row==50 or row==51 or row==52:
			idx_interval_dic = {'1':[147,174], '2':[175,214], '3':[241+1,270-2], '4':[275-1,288+1], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.45-bckg_fit, 150., 3.,
							2.34-bckg_fit, 155., 3.,
							2.34-bckg_fit, 159.5, 3.,
							2.91-bckg_fit, 163., 3.,
							#2.4-bckg_fit, 167., 3.,
							2.64-bckg_fit, 169., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							#2.05-bckg_fit, 181.5, 5.,
							1.87-bckg_fit, 186.5, 3.,
							1.86-bckg_fit, 193., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							2.18-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.98-bckg_fit, 280., 3.5,
							2.28-bckg_fit, 283., 3.,
							2.41-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==48:
			idx_interval_dic = {'1':[147,174], '2':[175,214], '3':[241+1,270-2], '4':[275-4,288+1], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.45-bckg_fit, 150., 3.,
							2.34-bckg_fit, 155., 3.,
							2.34-bckg_fit, 159.5, 3.,
							2.91-bckg_fit, 163., 3.,
							#2.4-bckg_fit, 167., 3.,
							2.64-bckg_fit, 169., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							#2.05-bckg_fit, 181.5, 5.,
							1.87-bckg_fit, 186.5, 3.,
							1.86-bckg_fit, 193., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							2.18-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.98-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.41-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==47:
			idx_interval_dic = {'1':[147,174], '2':[175,214], '3':[241+1,270-2], '4':[275-4,288+1], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.45-bckg_fit, 150., 3.,
							2.34-bckg_fit, 155., 3.,
							2.34-bckg_fit, 159.5, 3.,
							2.91-bckg_fit, 163., 3.,
							#2.4-bckg_fit, 167., 3.,
							2.64-bckg_fit, 169., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							#2.05-bckg_fit, 181.5, 5.,
							1.87-bckg_fit, 186.5, 3.,
							1.86-bckg_fit, 193., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							2.18-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.98-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.41-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==46:
			idx_interval_dic = {'1':[147,174], '2':[175,214], '3':[241+1,270-2], '4':[275-4,288+1], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.45-bckg_fit, 150., 3.,
							2.34-bckg_fit, 155., 3.,
							2.34-bckg_fit, 159.5, 3.,
							2.91-bckg_fit, 163., 3.,
							#2.4-bckg_fit, 167., 3.,
							2.64-bckg_fit, 169., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							#2.05-bckg_fit, 181.5, 5.,
							1.87-bckg_fit, 186.5, 3.,
							1.86-bckg_fit, 193., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							2.18-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.06-bckg_fit, 273., 2.5,
							3.98-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.41-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==45:
			idx_interval_dic = {'1':[147,174], '2':[175,214], '3':[241-3,270-2], '4':[275-4,288+1], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.45-bckg_fit, 150., 3.,
							2.34-bckg_fit, 155., 3.,
							2.34-bckg_fit, 159.5, 3.,
							2.91-bckg_fit, 163., 3.,
							#2.4-bckg_fit, 167., 3.,
							2.64-bckg_fit, 169., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							#2.05-bckg_fit, 181.5, 5.,
							1.87-bckg_fit, 186.5, 3.,
							1.86-bckg_fit, 193., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							2.18-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.06-bckg_fit, 273., 2.5,
							3.98-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.41-bckg_fit, 284.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==44:
			idx_interval_dic = {'1':[147,174], '2':[175,214], '3':[241-3,270-2], '4':[275-4,288+3], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.45-bckg_fit, 150., 3.,
							2.34-bckg_fit, 155., 3.,
							2.34-bckg_fit, 159.5, 3.,
							2.91-bckg_fit, 163., 3.,
							#2.4-bckg_fit, 167., 3.,
							2.64-bckg_fit, 169., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							#2.05-bckg_fit, 181.5, 5.,
							1.87-bckg_fit, 186.5, 3.,
							1.86-bckg_fit, 193., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							2.18-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.06-bckg_fit, 273., 2.5,
							3.59-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.27-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==43:
			idx_interval_dic = {'1':[147,174], '2':[175,214], '3':[241-3,270-2], '4':[275-4,288+3], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.45-bckg_fit, 150., 3.,
							2.34-bckg_fit, 155., 3.,
							2.34-bckg_fit, 159.5, 3.,
							2.91-bckg_fit, 163., 3.,
							#2.4-bckg_fit, 167., 3.,
							2.64-bckg_fit, 169., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							#2.05-bckg_fit, 181.5, 5.,
							1.87-bckg_fit, 186.5, 3.,
							1.86-bckg_fit, 193., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							2.18-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.06-bckg_fit, 273., 2.5,
							3.59-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.27-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==41 or row==42:
			idx_interval_dic = {'1':[147,174], '2':[175,214], '3':[241-3,270-2], '4':[275-4,288+3], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.45-bckg_fit, 150., 3.,
							2.34-bckg_fit, 155., 3.,
							2.34-bckg_fit, 159.5, 3.,
							2.91-bckg_fit, 163., 3.,
							#2.4-bckg_fit, 167., 3.,
							2.64-bckg_fit, 169., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							#2.05-bckg_fit, 181.5, 5.,
							1.87-bckg_fit, 186.5, 3.,
							1.86-bckg_fit, 193., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							2.18-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.06-bckg_fit, 273., 2.5,
							3.59-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.27-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==40:
			idx_interval_dic = {'1':[147,174], '2':[175,214], '3':[241-3,270-2], '4':[275-4,288+3], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.45-bckg_fit, 150., 3.,
							2.34-bckg_fit, 155., 3.,
							2.34-bckg_fit, 159.5, 3.,
							2.91-bckg_fit, 163., 3.,
							#2.4-bckg_fit, 167., 3.,
							2.64-bckg_fit, 169., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							#2.05-bckg_fit, 181.5, 5.,
							1.87-bckg_fit, 186.5, 3.,
							1.86-bckg_fit, 193., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							2.18-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.06-bckg_fit, 273., 2.5,
							3.59-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.27-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]

		if row==39:
			idx_interval_dic = {'1':[147,174], '2':[175,214], '3':[241-3,270-2], '4':[275-4,288+3], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.45-bckg_fit, 150., 3.,
							2.34-bckg_fit, 155., 3.,
							2.34-bckg_fit, 159.5, 3.,
							2.91-bckg_fit, 163., 3.,
							#2.4-bckg_fit, 167., 3.,
							2.64-bckg_fit, 169., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							2.-bckg_fit, 183., 5.,
							1.87-bckg_fit, 186.5, 3.,
							1.86-bckg_fit, 193., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							2.18-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.06-bckg_fit, 273., 2.5,
							3.59-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.27-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==38:
			idx_interval_dic = {'1':[147,174], '2':[175,214], '3':[241-3,270-2], '4':[275-4,288+3], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.45-bckg_fit, 150., 3.,
							2.34-bckg_fit, 155., 3.,
							2.34-bckg_fit, 159.5, 3.,
							2.91-bckg_fit, 163., 3.,
							#2.4-bckg_fit, 167., 3.,
							2.64-bckg_fit, 169., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							2.-bckg_fit, 183., 5.,
							1.87-bckg_fit, 186.5, 3.,
							1.86-bckg_fit, 193., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							2.18-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.06-bckg_fit, 273., 2.5,
							3.59-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.27-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==37:
			idx_interval_dic = {'1':[147,174], '2':[175,214], '3':[241-3,270-2], '4':[275,288], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.45-bckg_fit, 150., 3.,
							#2.34-bckg_fit, 158., 4.,
							#2.38-bckg_fit, 159.5, 4.,
							2.91-bckg_fit, 163., 3.,
							#2.4-bckg_fit, 167., 3.,
							2.64-bckg_fit, 169., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							2.-bckg_fit, 184., 5.,
							#1.9-bckg_fit, 186., 3.,
							1.86-bckg_fit, 193., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							2.12-bckg_fit, 200., 4.,
							#1.54-bckg_fit, 204., 2.,
							#1.92-bckg_fit, 205.5, 3.,
							2.18-bckg_fit, 210., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.6-bckg_fit, 274.5, 2.5,
							3.59-bckg_fit, 280., 3.5,
							#2.28-bckg_fit, 283., 3.,
							2.27-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==36:
			idx_interval_dic = {'1':[148,174], '2':[175,204], '3':[241-3,270-2], '4':[275,289+3], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.63-bckg_fit, 151., 3.,
							#2.27-bckg_fit, 155., 3.,
							2.38-bckg_fit, 159.5, 3.,
							3.73-bckg_fit, 164.5, 3.,
							#2.4-bckg_fit, 167., 3.,
							3.34-bckg_fit, 170.5, 5.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 179., 3.,
							2.-bckg_fit, 182., 3.5,
							1.9-bckg_fit, 186., 3.,
							1.95-bckg_fit, 192.5, 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							3.09-bckg_fit, 201.5, 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.6-bckg_fit, 274.5, 2.5,
							3.59-bckg_fit, 280., 3.5,
							2.28-bckg_fit, 283., 3.,
							2.27-bckg_fit, 285.5, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.8-bckg_fit, 316.5, 3.,
							2.4-bckg_fit, 322., 3.,
							2.0-bckg_fit, 327.5, 3.]


		if row==35:
			idx_interval_dic = {'1':[148,174], '2':[175,204], '3':[241-3,270-2], '4':[276,289+3], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.63-bckg_fit, 151., 3.,
							2.27-bckg_fit, 155., 3.,
							2.38-bckg_fit, 159.5, 3.,
							3.73-bckg_fit, 164.5, 3.,
							#2.4-bckg_fit, 167., 3.,
							3.34-bckg_fit, 170.5, 5.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.8-bckg_fit, 179.5, 3.,
							2.2-bckg_fit, 182.5, 2.,
							#1.91-bckg_fit, 188., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							3.09-bckg_fit, 201.5, 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.6-bckg_fit, 274.5, 2.5,
							4.65-bckg_fit, 281.5, 3.5,
							#2.2-bckg_fit, 281., 3.,
							2.43-bckg_fit, 287., 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.4-bckg_fit, 317.5, 3.,
							2.91-bckg_fit, 323., 3.,
							2.14-bckg_fit, 329.5, 2.]


		if row==34:
			idx_interval_dic = {'1':[148,174], '2':[175,204], '3':[241+3,270-2], '4':[276,289+3], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.63-bckg_fit, 151., 3.,
							2.27-bckg_fit, 155., 3.,
							2.38-bckg_fit, 159.5, 3.,
							3.73-bckg_fit, 164.5, 3.,
							#2.4-bckg_fit, 167., 3.,
							3.34-bckg_fit, 170.5, 5.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.8-bckg_fit, 179.5, 3.,
							2.2-bckg_fit, 182.5, 2.,
							#1.91-bckg_fit, 188., 3.,
							#1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							3.09-bckg_fit, 201.5, 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.6-bckg_fit, 274.5, 2.5,
							4.65-bckg_fit, 281.5, 3.5,
							#2.2-bckg_fit, 281., 3.,
							2.43-bckg_fit, 287., 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.4-bckg_fit, 317.5, 3.,
							2.91-bckg_fit, 323., 3.,
							2.14-bckg_fit, 329.5, 2.]


		if row==33:
			idx_interval_dic = {'1':[148,174], '2':[175,204], '3':[241+3,270-2], '4':[275-3,289+3], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.63-bckg_fit, 151., 3.,
							2.27-bckg_fit, 155., 3.,
							2.38-bckg_fit, 159.5, 3.,
							3.73-bckg_fit, 164.5, 3.,
							#2.4-bckg_fit, 167., 3.,
							3.34-bckg_fit, 170.5, 5.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.9-bckg_fit, 179.5, 3.,
							2.3-bckg_fit, 182., 2.,
							1.9-bckg_fit, 185., 3.,
							1.9-bckg_fit, 192., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							3.09-bckg_fit, 201.5, 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.6-bckg_fit, 274.5, 2.5,
							4.65-bckg_fit, 281.5, 3.5,
							#2.2-bckg_fit, 281., 3.,
							2.43-bckg_fit, 287., 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.4-bckg_fit, 317.5, 3.,
							2.91-bckg_fit, 323., 3.,
							2.14-bckg_fit, 329.5, 2.]


		if row==30 or row==31 or row==32:
			idx_interval_dic = {'1':[148,174], '2':[175,204], '3':[241+3,270-2], '4':[275-3,289+3], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.63-bckg_fit, 151.5, 3.,
							3.16-bckg_fit, 159., 3.,
							#3.28-bckg_fit, 160., 3.,
							3.73-bckg_fit, 164.5, 3.,
							#2.4-bckg_fit, 167., 3.,
							3.34-bckg_fit, 170.5, 5.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							3.85-bckg_fit, 179.5, 3.,
							#2.8-bckg_fit, 182., 3.,
							2.9-bckg_fit, 187., 3.,
							#2.68-bckg_fit, 193., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							3.09-bckg_fit, 201.5, 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.6-bckg_fit, 274.5, 2.5,
							4.65-bckg_fit, 281.5, 3.5,
							#2.2-bckg_fit, 281., 3.,
							2.43-bckg_fit, 287., 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.4-bckg_fit, 317.5, 3.,
							2.91-bckg_fit, 323., 3.,
							2.14-bckg_fit, 329.5, 2.]


		if row==29:
			idx_interval_dic = {'1':[148,174], '2':[175,204], '3':[241-3,270-2], '4':[275-3,289+3], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.63-bckg_fit, 151.5, 3.,
							3.16-bckg_fit, 159., 3.,
							#3.28-bckg_fit, 160., 3.,
							3.73-bckg_fit, 164.5, 3.,
							#2.4-bckg_fit, 167., 3.,
							3.34-bckg_fit, 170.5, 5.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							3.85-bckg_fit, 179.5, 3.,
							#2.8-bckg_fit, 182., 3.,
							2.9-bckg_fit, 187., 3.,
							#2.68-bckg_fit, 193., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							3.09-bckg_fit, 201.5, 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.6-bckg_fit, 274.5, 2.5,
							4.65-bckg_fit, 281.5, 3.5,
							#2.2-bckg_fit, 281., 3.,
							2.43-bckg_fit, 287., 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.4-bckg_fit, 317.5, 3.,
							2.91-bckg_fit, 323., 3.,
							2.14-bckg_fit, 329.5, 2.]


		if row==28:
			idx_interval_dic = {'1':[148,174], '2':[175,204], '3':[241-3,270-2], '4':[275-3,289+3], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.63-bckg_fit, 151.5, 3.,
							3.16-bckg_fit, 159., 3.,
							#3.28-bckg_fit, 160., 3.,
							3.73-bckg_fit, 164.5, 3.,
							#2.4-bckg_fit, 167., 3.,
							3.34-bckg_fit, 170.5, 5.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							3.85-bckg_fit, 179.5, 3.,
							#2.8-bckg_fit, 182., 3.,
							2.9-bckg_fit, 187., 3.,
							#2.68-bckg_fit, 193., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							3.09-bckg_fit, 201.5, 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.6-bckg_fit, 274.5, 2.5,
							4.65-bckg_fit, 281.5, 3.5,
							#2.2-bckg_fit, 281., 3.,
							2.43-bckg_fit, 287., 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.4-bckg_fit, 317.5, 3.,
							2.91-bckg_fit, 323., 3.,
							2.14-bckg_fit, 329.5, 2.]


		if row==27:
			idx_interval_dic = {'1':[148,174], '2':[175,204], '3':[241-3,270-2], '4':[275-3,289+3], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.63-bckg_fit, 151.5, 3.,
							3.16-bckg_fit, 159., 3.,
							#3.28-bckg_fit, 160., 3.,
							3.73-bckg_fit, 164.5, 3.,
							#2.4-bckg_fit, 167., 3.,
							3.34-bckg_fit, 170.5, 5.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							3.85-bckg_fit, 179.5, 3.,
							#2.8-bckg_fit, 182., 3.,
							2.9-bckg_fit, 187., 3.,
							#2.68-bckg_fit, 193., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							3.09-bckg_fit, 201.5, 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.6-bckg_fit, 274.5, 2.5,
							4.65-bckg_fit, 281.5, 3.5,
							#2.2-bckg_fit, 281., 3.,
							2.43-bckg_fit, 287., 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.4-bckg_fit, 317.5, 3.,
							2.91-bckg_fit, 323., 3.,
							2.14-bckg_fit, 329.5, 2.]


		if row==25 or row==26:
			idx_interval_dic = {'1':[148,174], '2':[175,204], '3':[241-3,270-2], '4':[275-3,289+3], '5':[311+2,331+1]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.63-bckg_fit, 151.5, 3.,
							3.16-bckg_fit, 159., 3.,
							#3.28-bckg_fit, 160., 3.,
							3.73-bckg_fit, 164.5, 3.,
							#2.4-bckg_fit, 167., 3.,
							3.34-bckg_fit, 170.5, 5.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							3.85-bckg_fit, 179.5, 3.,
							#2.8-bckg_fit, 182., 3.,
							2.9-bckg_fit, 187., 3.,
							#2.68-bckg_fit, 193., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							3.09-bckg_fit, 201.5, 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							15.-bckg_fit, 247., 10.,
							#12.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.6-bckg_fit, 274.5, 2.5,
							4.65-bckg_fit, 281.5, 3.5,
							#2.2-bckg_fit, 281., 3.,
							2.43-bckg_fit, 287., 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.4-bckg_fit, 317.5, 3.,
							2.91-bckg_fit, 323., 3.,
							2.14-bckg_fit, 329.5, 2.]


		if row==22 or row==23 or row==24:
			idx_interval_dic = {'1':[148,174], '2':[175,204], '3':[241+3,270-2], '4':[275-3,289], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.63-bckg_fit, 151.5, 3.,
							3.16-bckg_fit, 159., 3.,
							#3.28-bckg_fit, 160., 3.,
							3.73-bckg_fit, 164.5, 3.,
							#2.4-bckg_fit, 167., 3.,
							3.34-bckg_fit, 170.5, 5.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							3.85-bckg_fit, 179.5, 3.,
							#2.8-bckg_fit, 182., 3.,
							2.9-bckg_fit, 187., 3.,
							#2.68-bckg_fit, 193., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							3.09-bckg_fit, 201.5, 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#5.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.6-bckg_fit, 274.5, 2.5,
							4.65-bckg_fit, 281.5, 3.5,
							#2.2-bckg_fit, 281., 3.,
							2.43-bckg_fit, 287., 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.4-bckg_fit, 317.5, 3.,
							2.91-bckg_fit, 323., 3.]
							#2.14-bckg_fit, 329.5, 2.]


		if row==21:
			idx_interval_dic = {'1':[148,174], '2':[175,204], '3':[241+3,270-2], '4':[275-3,289], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.6-bckg_fit, 151., 3.,
							3.2-bckg_fit, 157., 3.,
							3.28-bckg_fit, 160., 3.,
							3.73-bckg_fit, 164.5, 3.,
							#2.4-bckg_fit, 167., 3.,
							3.41-bckg_fit, 170., 5.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							3.85-bckg_fit, 179.5, 3.,
							#2.8-bckg_fit, 182., 3.,
							2.9-bckg_fit, 187., 3.,
							#2.68-bckg_fit, 193., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							3.09-bckg_fit, 201.5, 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#5.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.6-bckg_fit, 274.5, 2.5,
							4.65-bckg_fit, 281.5, 3.5,
							#2.2-bckg_fit, 281., 3.,
							2.43-bckg_fit, 287., 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.93-bckg_fit, 317., 3.,
							3.59-bckg_fit, 322., 3.,
							2.91-bckg_fit, 328., 3.]

		if row==20:
			idx_interval_dic = {'1':[148,174], '2':[175,204], '3':[241+3,270-2], '4':[275,289], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.6-bckg_fit, 151., 3.,
							3.26-bckg_fit, 158.5, 3.,
							#2.-bckg_fit, 157.5, 4.,
							3.71-bckg_fit, 164., 3.,
							#2.4-bckg_fit, 167., 3.,
							3.85-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							3.85-bckg_fit, 179.5, 3.,
							#2.8-bckg_fit, 182., 3.,
							2.9-bckg_fit, 187., 3.,
							#2.68-bckg_fit, 193., 3.,
							#2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							3.09-bckg_fit, 201.5, 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#5.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.63-bckg_fit, 274., 2.5,
							4.65-bckg_fit, 281., 3.5,
							#2.2-bckg_fit, 281., 3.,
							2.54-bckg_fit, 287., 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.93-bckg_fit, 317., 3.,
							3.59-bckg_fit, 322., 3.,
							2.91-bckg_fit, 328., 3.]


		if row==19:
			idx_interval_dic = {'1':[148,174], '2':[175,204], '3':[241+3,270-2], '4':[275,289], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.6-bckg_fit, 151., 3.,
							3.26-bckg_fit, 158.5, 3.,
							#2.-bckg_fit, 157.5, 4.,
							3.71-bckg_fit, 164., 3.,
							#2.4-bckg_fit, 167., 3.,
							3.85-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							3.85-bckg_fit, 179.5, 3.,
							#2.8-bckg_fit, 182., 3.,
							2.9-bckg_fit, 187., 3.,
							#2.68-bckg_fit, 193., 3.,
							2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							3.09-bckg_fit, 201.5, 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							5.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#2.7-bckg_fit, 275., 3.5,
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.9-bckg_fit, 286.3, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.93-bckg_fit, 317., 3.,
							3.59-bckg_fit, 322., 3.,
							2.91-bckg_fit, 328., 3.]


		if row==18:
			idx_interval_dic = {'1':[148,174], '2':[175,204], '3':[241+3,270-2], '4':[275-3,289], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.6-bckg_fit, 151., 3.,
							3.26-bckg_fit, 158.5, 3.,
							#2.-bckg_fit, 157.5, 4.,
							3.71-bckg_fit, 164., 3.,
							#2.4-bckg_fit, 167., 3.,
							3.85-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							3.85-bckg_fit, 179.5, 3.,
							2.8-bckg_fit, 182., 3.,
							2.9-bckg_fit, 187., 3.,
							#2.68-bckg_fit, 193., 3.,
							2.68-bckg_fit, 198., 3.,
							#1.6-bckg_fit, 197., 3.,
							3.09-bckg_fit, 201.5, 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							5.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.7-bckg_fit, 275., 3.5,
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.9-bckg_fit, 286.3, 2.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.93-bckg_fit, 317., 3.,
							3.59-bckg_fit, 322., 3.,
							2.91-bckg_fit, 328., 3.]

		if row==17:
			idx_interval_dic = {'1':[148,174], '2':[175,204], '3':[241+3,270-2], '4':[275,289], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.6-bckg_fit, 151., 3.,
							3.26-bckg_fit, 158.5, 3.,
							#2.-bckg_fit, 157.5, 4.,
							3.71-bckg_fit, 164., 3.,
							#2.4-bckg_fit, 167., 3.,
							3.85-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.53-bckg_fit, 178., 3.,
							1.7-bckg_fit, 182., 3.5,
							1.54-bckg_fit, 186., 3.,
							1.68-bckg_fit, 191.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.85-bckg_fit, 200., 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							5.-bckg_fit, 251., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.93-bckg_fit, 317., 3.,
							3.59-bckg_fit, 322., 3.,
							2.91-bckg_fit, 328., 3.]

		if row==16:
			idx_interval_dic = {'1':[148,173], '2':[175,204], '3':[241-3,270], '4':[275,289], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 150.5, 3.,
							1.88-bckg_fit, 156., 3.,
							#2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.53-bckg_fit, 178., 3.,
							1.7-bckg_fit, 182., 3.5,
							1.54-bckg_fit, 186., 3.,
							1.68-bckg_fit, 191.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.85-bckg_fit, 200., 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.93-bckg_fit, 317., 3.,
							3.59-bckg_fit, 322., 3.,
							2.91-bckg_fit, 328., 3.]

		if row==15:
			idx_interval_dic = {'1':[148,173], '2':[175-1,204], '3':[241-3,270], '4':[275,289], '5':[311+2,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 150.5, 3.,
							1.88-bckg_fit, 156., 3.,
							#2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.53-bckg_fit, 178., 3.,
							#1.7-bckg_fit, 182., 3.5,
							2.92-bckg_fit, 184.5, 4.,
							2.67-bckg_fit, 192., 2.5,
							#1.6-bckg_fit, 197., 3.,
							1.85-bckg_fit, 200., 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.93-bckg_fit, 317., 3.,
							3.59-bckg_fit, 322., 3.,
							2.91-bckg_fit, 328., 3.]

		if row==13 or row==14:
			idx_interval_dic = {'1':[148,173], '2':[175,204], '3':[241-3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 150.5, 3.,
							1.88-bckg_fit, 156., 3.,
							#2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.53-bckg_fit, 178., 3.,
							1.7-bckg_fit, 182., 3.5,
							1.54-bckg_fit, 186., 3.,
							1.68-bckg_fit, 191.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.85-bckg_fit, 200., 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]

		if row==9 or row==10 or row==11 or row==12:
			idx_interval_dic = {'1':[148,173], '2':[175,204], '3':[241-3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							1.88-bckg_fit, 156., 3.,
							#2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.53-bckg_fit, 178., 3.,
							1.7-bckg_fit, 182., 3.5,
							1.54-bckg_fit, 186., 3.,
							1.68-bckg_fit, 191.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.85-bckg_fit, 200., 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==7 or row==8:
			idx_interval_dic = {'1':[148,173], '2':[175,204], '3':[241+3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							1.88-bckg_fit, 156., 3.,
							#2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.53-bckg_fit, 178., 3.,
							1.7-bckg_fit, 182., 3.5,
							1.54-bckg_fit, 186., 3.,
							1.68-bckg_fit, 191.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.85-bckg_fit, 200., 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]


		if row==6:
			idx_interval_dic = {'1':[148,173], '2':[175,204], '3':[241-3,270], '4':[275,289], '5':[311,331]}
			bckg_fit = 1.1
			init_parameters_dic = {}
			init_parameters_dic['1'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.14-bckg_fit, 149.5, 3.,
							1.88-bckg_fit, 156., 3.,
							#2.-bckg_fit, 157.5, 4.,
							2.35-bckg_fit, 162.5, 3.,
							2.4-bckg_fit, 167., 3.,
							2.14-bckg_fit, 170., 3.]
			init_parameters_dic['2'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							#1.76-bckg_fit, 175.5, 3.5,
							2.53-bckg_fit, 178., 3.,
							1.7-bckg_fit, 182., 3.5,
							#1.54-bckg_fit, 184.5, 3.,
							1.68-bckg_fit, 191.5, 3.5,
							#1.6-bckg_fit, 197., 3.,
							1.85-bckg_fit, 200., 3.]#,
							#1.54-bckg_fit, 204., 2.,
							#3.-bckg_fit, 205.5, 3.]
							#2.86-bckg_fit, 211., 2.5]
			init_parameters_dic['3'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							11.5-bckg_fit, 247., 10.,
							#8.-bckg_fit, 249., 5.,
							#2.5-bckg_fit, 255., 3.,
							2.3-bckg_fit, 260., 3.5,
							#1.9-bckg_fit, 261., 5.,
							2.4-bckg_fit, 264.5, 3.]
							#2.16-bckg_fit, 268., 2.]
			init_parameters_dic['4'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							3.49-bckg_fit, 279., 4.,
							#2.2-bckg_fit, 281., 3.,
							2.1-bckg_fit, 285., 3.5]
			init_parameters_dic['5'] = [bckg_fit, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
							2.53-bckg_fit, 316., 4.,
							2.51-bckg_fit, 321.26, 5.,
							1.86-bckg_fit, 327., 4.]
		
		"""
		print('#############################')
		print(f'slope_list["{row}"] = {slope_fit}')
		print(f'slope_unc_list["{row}"] = {slope_unc_fit}')
		print(f'intercept_list["{row}"] = {intercept_fit}')
		print(f'intercept_unc_list["{row}"] = {intercept_unc_fit}')
		print('#############################')
		"""
		
		
		if 'idx_interval_dic' in locals() and 'bckg_fit' in locals() and 'init_parameters_dic' in locals():
			def convert_to_serializable(obj):
				if isinstance(obj, np.integer):
					return int(obj)
				elif isinstance(obj, np.floating):
					return float(obj)
				elif isinstance(obj, (list, tuple)):
					return [convert_to_serializable(item) for item in obj]
				elif isinstance(obj, dict):
					return {k: convert_to_serializable(v) for k, v in obj.items()}
				return obj
			
			calibration_parameters_all_rows[str(row)] = {
				'idx_interval': convert_to_serializable(idx_interval_dic),
				'bckg_fit': convert_to_serializable(bckg_fit),
				'init_parameters': convert_to_serializable(init_parameters_dic)
			}
		
		##########################################################



	# The results are given as numpy.float64, I want them as floats, so:
	pixelscale_list = [float(x) for x in pixelscale_list_float64]
	pixelscale_unc_list = [float(x) for x in pixelscale_unc_list_float64]
	pixelscale_intercept_list = [float(x) for x in pixelscale_intercept_list_float64]
	pixelscale_intercept_unc_list = [float(x) for x in pixelscale_intercept_unc_list_float64]


	print('#############################')
	print('pixelscale_list =', pixelscale_list) 
	print('#############################')
	print('pixelscale_unc_list =', pixelscale_unc_list) 
	print('#############################')
	print('pixelscale_intercept_list =', pixelscale_intercept_list) 
	print('#############################')
	print('pixelscale_intercept_unc_list =', pixelscale_intercept_unc_list) 
	print('#############################')

	# Save calibration parameters to JSON
	if calibration_parameters_all_rows and save:
		json_output_path = Path(json_path) / 'calibration_parameters_wcal1.json'
		
		with open(json_output_path, 'w') as f:
			json.dump(calibration_parameters_all_rows, f, indent=2)
		
		print(f"\n✓ Calibration parameters saved to JSON:")
		print(f"  {json_output_path}")
		print(f"  Rows saved: {sorted([int(k) for k in calibration_parameters_all_rows.keys()])}")
		return
	if calibration_parameters_all_rows and not save:
		return 

if __name__ == "__main__":
	generate_init_parameters()


