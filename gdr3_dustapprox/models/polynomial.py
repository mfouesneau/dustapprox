r""" Polynomial approximation of extinction effect per passband.

In this library, we provide tools and pre-computed models to obtain the extinction coefficient
:math:`k_x = A_x / A_0` for various passbands.

Extinction coefficients depend primarily on the source spectral energy
distribution and on the extinction itself (e.g., Gordon et al., 2016; Jordi et
al., 2010) but also other parameters such as :math:`\log g`,
:math:`[\alpha/Fe]`, and :math:`[Fe/H]`.

We define the extinction coefficient :math:`k_x` as

.. math::

    k_x = A_x / A_0,

with :math:`A_0` the extinction parameter at :math:`550 nm`, and :math:`x` the passband.

We use a L1-regularized regression model (Lasso-Lars model) using BIC or AIC for
model/complexity selection.

The optimization objective for Lasso is:

.. math::

    \frac{1}{2 n_{samples}} \cdot \|y - X \cdot w\|^2_2 + \alpha \times \|w\|_1

AIC is the `Akaike information
criterion <https://en.wikipedia.org/wiki/Akaike_information_criterion>`_ and BIC
is the `Bayes Information criterion
<https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_.
AIC or BIC are useful criteria to select the value of the regularization parameter :math:`\alpha` by making a
trade-off between the goodness-of-fit and
the complexity of the model.

This allows is to follow the principle of parsimony (aka Occam's razor): a good
model should explain well the data while being simple.


.. plot::
    :caption: Statistics on polynomial approximation of extinction effect per passband.
              The top panel shows the residual statistics of the model to the grid values, while the bottom
              panel shows the meaningful coefficient amplitudes. (Grey pixels indicate values below :math:`10^{-5}`).

    import pylab as plt
    from gdr3_dustapprox.models import polynomial
    import pandas as pd

    r = pd.read_csv('models/precomputed/kurucs_gaiaedr3_small_a0_grid.csv')
    # polynomial.quick_plot_models(r, input_parameters='teff A0'.split())
    polynomial.quick_plot_models(r, input_parameters='teff A0 feh logg'.split())
    plt.show()

"""
from typing import Sequence
from pandas import DataFrame
import numpy as np
import warnings


def approx_model(r: DataFrame,
                 passband: str = 'GAIA_GAIA3.G',
                 degree: int = 3, interaction_only: bool = False,
                 input_parameters: Sequence[str] = None,
                 verbose=False) -> dict:
    """ Fit the passband grid data with a polynomial model.

    Parameters
    ----------
    r : DataFrame
        DataFrame with the passband grid data.
    passband : str
        The passband to fit.
    degree : int
        The degree of the polynomial model.
    interaction_only : bool
        If True, only the interaction terms are used.
    input_parameters : Sequence[str]
        The input parameters to use.
        If None, 'teff logg feh A0 alpha' parameters are used.
    verbose : bool
        If True, print the model parameters and statistics

    Returns
    -------
    dict
        Dictionary with the model parameters and statistics.

    """
    from sklearn.linear_model import LassoLarsIC
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import median_absolute_error, mean_squared_error

    if input_parameters is None:
        input_parameters = 'teff logg feh A0 alpha'.split()
    # input_parameters = 'teff A0'.split()
    predict_parameter = 'Ax'

    col_subset = [predict_parameter] + input_parameters
    subset = r[r.passband == passband][col_subset]
    subset = subset[subset['A0'] > 0]

    xdata = subset[input_parameters]
    # replace teff by teffnorm = teff / 5040K
    xdata['teffnorm'] = xdata['teff'] / 5040.
    xdata.drop(columns='teff', inplace=True)
    ydata = subset[predict_parameter]
    ydata /= subset['A0']

    # the common method
    poly = PolynomialFeatures(degree=degree,
                              interaction_only=interaction_only,
                              include_bias=True).fit(xdata)
    expand = poly.transform(xdata)
    coeff_names = poly.get_feature_names_out()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        regr = LassoLarsIC(fit_intercept=False, copy_X=False).fit(expand, ydata)
    pred = regr.predict(expand)

    mae = median_absolute_error(ydata, pred)
    rmse = mean_squared_error(ydata, pred, squared=False)
    stddev = np.std(pred-ydata)
    mean = np.mean(pred-ydata)

    named_coeffs = sorted([(k, v) for k, v in zip(coeff_names, regr.coef_)
                           if abs(v) > 1e-8], key=lambda x: x[1])

    if verbose:
        print('\n'.join(f"""
        --------------
        Band: {passband}
        Polynomial degree: {poly.degree}
        MAE = {mae:.3f},
        RMSE = {rmse:.3f},
        Mean = {mean:.3f},
        Stddev = {stddev:.3f}
        --------------
        """.splitlines()))

        [print("{0:15s} {1:0.3g}".format(k, v)) for k, v in named_coeffs]

    return {'features': coeff_names,
            'coefficients': regr.coef_,
            'mae': mae, 'rmse': rmse,
            'mean_residuals': mean, 'std_residuals': stddev}


def quick_plot_models(r: DataFrame, **kwargs) -> DataFrame:
    """ Plot diagnostic plots for the models.

    Parameters
    ----------
    r : DataFrame
        DataFrame with the passband grid data.

    Returns
    -------
    DataFrame
        DataFrame with the all model parameters and statistics.

    .. seealso::

        :func:`approx_model`
    """
    import pylab as plt

    names = r.passband.unique()
    data = []
    for name in names:
        res = approx_model(r, name, **kwargs)
        coeff_names = res['features']
        data.append([name] + list(res['coefficients']) +
                    [res['mae'], res['rmse'],
                    res['mean_residuals'], res['std_residuals']])

    res =  DataFrame(data, columns=['passband'] + list(coeff_names) + ['mae', 'rmse', 'mean', 'stddev'])

    image = res[res.columns[1:]].to_numpy().T

    fig = plt.figure()
    subdata = image[-4:]

    plt.plot(subdata[0], label='mae')
    plt.plot(subdata[1], label='rmse')
    plt.plot(subdata[2], label='mean')
    plt.plot(subdata[3], label='stddev')
    plt.legend(loc='best', frameon=False)
    plt.xticks(np.arange(len(names)), labels=names, rotation=90)
    plt.xlabel('passband')
    plt.ylabel('residual statistics')
    plt.tight_layout()

    image = np.ma.masked_where(np.abs(image) < 1e-5, image)
    subdata = image[:-4]
    shape = subdata.shape
    vmin = np.percentile(subdata, 10)
    vmax = np.percentile(subdata, 90)
    if (vmin < 0) and (vmax > 0):
        vmin = -1 * max(abs(vmin), vmax)
        vmax = - vmin
    cmap = plt.cm.RdYlBu
    cmap.set_bad('0.5', 1.)
    imshow_kwargs = dict(vmin=vmin, vmax=vmax, cmap=cmap,
                         interpolation='nearest', aspect='auto')
    if shape[0] < shape[1]:
        figsize=(len(names) * 0.8, len(coeff_names) * 0.5)
    else:
        figsize=(len(coeff_names) * 0.25, len(names) * 0.8)
    fig = plt.figure(figsize=(max(figsize[0], 6), max(figsize[1], 4)))
    if shape[0] < shape[1]:
        plt.imshow(subdata, **imshow_kwargs)
        plt.xticks(np.arange(len(names)), labels=names, rotation=90)
        plt.yticks(np.arange(len(coeff_names)), labels=coeff_names)
        plt.xlabel('passband')
        plt.ylabel('features')
    else:
        plt.imshow(subdata.T, **imshow_kwargs)
        plt.yticks(np.arange(len(names)), labels=names)
        plt.xticks(np.arange(len(coeff_names)), labels=coeff_names, rotation=90)
        plt.ylabel('passband')
        plt.xlabel('features')
    plt.colorbar(extend='both').set_label('coefficient')
    plt.tight_layout()

    return res
