"""
Script to estimate the H/L parameter. The idea is:

1 - Fit the lengths of horizontal (H) and vertical (L) fractures.
2 - Draw n random samples from the fitted distributions.
3 - Compute n ratios (H/L).
4 - Iterate m times and analyze:
    a. The distribution of the ratios.
    b. The distribution of the means of the ratios.

"""
from fracability import Entities, Statistics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

# INPUT - VERTICAL FRACTURE NETWORK AND INTEPRETATION BOUNDARY SHAPEFILES

vertical_fractures = Entities.Fractures(shp='Pontrelli/verticale/Set_2.shp', set_n=1)
vertical_boundary = Entities.Boundary(shp='Pontrelli/verticale/Interpretation_boundary.shp', group_n=1)

#INPUT - HORIZONTAL FRACTURE NETWORK AND INTERPRETATION BOUNDARY SHAPEFILES

horizontal_fractures = Entities.Fractures(shp='Pontrelli/orizzontale/FN_set_2.shp', set_n=1)
horizontal_boundary = Entities.Boundary(shp='Pontrelli/orizzontale/Interpretation_boundary.shp', group_n=1)

#TOPOLOGICAL ANALYSIS AND LENGTH DISTRIBUTION FITTING

vertical_fn = Entities.FractureNetwork()
vertical_fn.add_fractures(vertical_fractures)
vertical_fn.add_boundaries(vertical_boundary)

horizontal_fn = Entities.FractureNetwork()
horizontal_fn.add_fractures(horizontal_fractures)
horizontal_fn.add_boundaries(horizontal_boundary)

vertical_fn.calculate_topology()
horizontal_fn.calculate_topology()

vertical_fitter = Statistics.NetworkFitter(vertical_fn)
horizontal_fitter = Statistics.NetworkFitter(horizontal_fn)

vertical_fitter.fit('lognorm')
vertical_fitter.fit('expon')
vertical_fitter.fit('norm')
vertical_fitter.fit('gengamma')
vertical_fitter.fit('powerlaw')
vertical_fitter.fit('weibull_min')

horizontal_fitter.fit('lognorm')
horizontal_fitter.fit('expon')
horizontal_fitter.fit('norm')
horizontal_fitter.fit('gengamma')
horizontal_fitter.fit('powerlaw')
horizontal_fitter.fit('weibull_min')


vertical_dist = vertical_fitter.best_fit().distribution
horizontal_dist = horizontal_fitter.best_fit().distribution



scipy_vertical_dist = vertical_dist.distribution
scipy_horizontal_dist = horizontal_dist.distribution


n_samples = 100 #Set number of random samples
n_iter = 100 #Set number of iteration
print(f'\n')

ratio_values = np.zeros((n_iter, n_samples))
ratio_mean_values = np.zeros(n_iter)
regress_slope_values = np.zeros(n_iter)
regress_slope_values2 = np.zeros(n_iter)
regress_intercept_values = np.zeros(n_iter)
regress_intercept_values2 = np.zeros(n_iter)
regress_rvalue = np.zeros(n_iter)

horizontal_values = np.zeros((n_iter, n_samples))
vertical_values = np.zeros((n_iter, n_samples))


print(f'vertical model mean: {vertical_dist.mean}')
print(f'vertical model std: {vertical_dist.std}')
print(f'horizontal model mean: {horizontal_dist.mean}')
print(f'horizontal model std: {horizontal_dist.std}')
print('\n')



for i in range(n_iter):
    print(f'Iteration: {i}', end='\r')

    vertical_random_samples = np.sort(scipy_vertical_dist.rvs(n_samples))
    horizontal_random_samples = np.sort(scipy_horizontal_dist.rvs(n_samples))

    IQR = ss.iqr(vertical_random_samples)
    pos_cut_off = 6 # fixed cutoff
    pos_cut_off2 = vertical_dist.mean + IQR*0.5 # Adaptive cutoff: A variable cutoff that changes based on the random samples. I compute the mean of the vertical random sample minus half of the interquartile range (IQR) of the data (to reduce the effect of outliers).

    mask = (vertical_random_samples<=pos_cut_off)
    mask2 = (vertical_random_samples<=pos_cut_off2)

    regress = ss.linregress(horizontal_random_samples[mask],
                            vertical_random_samples[mask])

    regress2 = ss.linregress(horizontal_random_samples[mask2],
                            vertical_random_samples[mask2])


    regress_slope_values[i] = regress.slope
    regress_intercept_values[i] = regress.intercept
    regress_slope_values2[i] = regress2.slope
    regress_intercept_values2[i] = regress2.intercept
    regress_rvalue[i] = regress.rvalue

    horizontal_values[i, :] = horizontal_random_samples
    vertical_values[i, :] = vertical_random_samples

    #define confidence interval
    # Define fitted line
    y_est = regress.slope * horizontal_random_samples[mask] + regress.intercept
    # Calculate the confidence interval
    confidence_level = 0.95
    n = len(horizontal_random_samples[mask])  # Number of data points
    t_value = ss.t.ppf((1 + confidence_level) / 2, n - 2)  # t-value for confidence level
    # Residuals and standard error of the estimate
    residuals = vertical_random_samples[mask] - y_est
    s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))

    # Margin of error for confidence interval
    y_err = t_value * s_err * np.sqrt(1 / n + (horizontal_random_samples[mask] - horizontal_random_samples[mask].mean()) ** 2 / np.sum((horizontal_random_samples[mask] - horizontal_random_samples[mask].mean()) ** 2))

    # Upper and lower bounds for the confidence interval
    y_upper = y_est + y_err
    y_lower = y_est - y_err


    #Comment these lines if you don't want to see every iterations

    fig_ratio = plt.figure('Regression lines')
    plt.plot(horizontal_random_samples, vertical_random_samples, 'bo',zorder=1)
    plt.plot(horizontal_random_samples, horizontal_random_samples*regress.slope+regress.intercept, 'k-',zorder=2, label='Fixed')
    #plt.plot(horizontal_random_samples, horizontal_random_samples*regress2.slope+regress2.intercept, 'k--',zorder=2, label='Adaptive')
    plt.text(2,8, f'rsquare = {regress.rvalue:0.2f}')
    plt.fill_between(horizontal_random_samples[mask], y_lower, y_upper, color="blue", alpha=0.2,
                     label="95% Confidence Interval")
    plt.legend()

    plt.show()

    ################################################################

print(f'min H/L (fixed, adaptive): {np.min(regress_slope_values)}, {np.min(regress_slope_values2)}')
print(f'max H/L (fixed, adaptive): {np.max(regress_slope_values)}, {np.max(regress_slope_values2)}')
print(f'mean H/L (fixed, adaptive): {np.mean(regress_slope_values)}, {np.mean(regress_slope_values2)}')


# Slope frequency histogram
fig_ratio = plt.figure('Slope frequency')
sns.histplot(regress_slope_values, label='Fixed')
#sns.histplot(regress_slope_values2, label='Adaptive')

plt.show()

