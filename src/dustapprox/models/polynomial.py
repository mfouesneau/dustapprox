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
    from dustapprox.models import polynomial
    import pandas as pd

    r = pd.read_csv('models/precomputed/kurucs_gaiaedr3_small_a0_grid.csv')
    # polynomial.quick_plot_models(r, input_parameters='teff A0'.split())
    polynomial.quick_plot_models(r, input_parameters='teff A0 feh logg'.split())
    plt.show()

"""
from typing import Sequence, Union
from pandas import DataFrame, Series
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import warnings
from ..io import ecsv
from .basemodel import _BaseModel


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

class PolynomialModel(_BaseModel):
    """ A polynomial model object

    Attributes
    ----------
    meta: dict
        meta information about the model

    transformer_: PolynomialFeatures
        polynomial transformer

    coeffs_: pd.Series
        coefficients of the regression on the polynomial expended features
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transformer_ = None
        self.coeffs_ = None

    @property
    def feature_names(self) -> Sequence[str]:
        """ Input feature dimensions of the model """
        try:
            return self.meta['model']['feature_names']
        except KeyError:
            return None

    @property
    def degree_(self) -> int:
        """ Degree of the polynomial transformation """
        if self.transformer_:
            return self.transformer_.degree
        else:
            return None

    @property
    def name(self) -> str:
        """ Get the model name also stored in the coeffs series """
        if self.coeffs_ is not None:
            if not hasattr(self.coeffs_, 'name'):
                self.coeffs_ = Series(self.coeffs_,
                                         index=self.get_transformed_feature_names())
                self.coeffs_.name = None
            if (self.coeffs_.name is None) and (self.name_ is not None):
                self.coeffs_.name = self.name_
        if (self.coeffs_.name is not None) and (self.name_ is None):
            self.name_ = self.coeffs_.name
        if self.name_:
            return self.name_

    def _consolidate_named_data(self, X: Union[np.ndarray, DataFrame]) -> DataFrame:
        """ A convenient consolidation of input data to named data fields

        As we use the names internally to make the operations more readable, it
        makes it easier to also convert the data.

        Parameters
        ----------
        X: Union[np.ndarray, pd.DataFrame]
            input features

        Returns
        -------
        Xp: pd.DataFrame
            named data input
        """
        if isinstance(X, DataFrame) or hasattr(X, 'columns'):
            return X.copy()
        else:
            return DataFrame.from_records(np.atleast_2d(X),
                                          columns=self.feature_names)

    def __repr__(self) -> str:
        txt = """PolynomialModel: {0} \n{1:s}\n""".format(self.name, object.__repr__(self))
        txt += """   from: {0:s}""".format(', '.join(self.feature_names))
        txt += """   polynomial degree: {0:d}""".format(self.degree_)
        return txt

    def fit(self, df: DataFrame,
        features: Sequence[str] = None,
        label: str = 'Ax',
        degree: int = 3,
        interaction_only: bool = False
        ):
        """
        Parameters
        ----------
        df : DataFrame
            DataFrame with the passband grid data.
        features: Sequence[str]
            input features from df (note: if used, teff will be normalized to teff/5040)
        label: str
            which field contains the label values
        degree : int
            The degree of the polynomial model.
        interaction_only : bool
            If True, only the interaction terms are used.
        input_parameters : Sequence[str]
            The input parameters to use.
            If None, 'teff logg feh A0 alpha' parameters are used.
        """
        from sklearn.linear_model import LassoLarsIC
        from sklearn.metrics import median_absolute_error, mean_squared_error

        if features is None:
            # features = 'teff A0'.split()
            features = 'teff logg feh A0 alpha'.split()
        if label is None:
            label = 'Ax'

        if 'A0' not in features:
            raise AttributeError("field `A0` expected in the input data.")

        col_subset = [label] + features
        subset = df[col_subset]
        subset = subset[subset['A0'] > 0]

        xdata = subset[features]
        # replace teff by teffnorm = teff / 5040K
        xdata['teffnorm'] = xdata['teff'] / 5040.
        xdata.drop(columns='teff', inplace=True)
        ydata = subset[label]
        ydata /= subset['A0']

        # the common method
        poly = PolynomialFeatures(degree=degree,
                                interaction_only=interaction_only,
                                include_bias=True).fit(xdata)
        expand = poly.transform(xdata)


        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            regr = LassoLarsIC(fit_intercept=False, copy_X=False).fit(expand, ydata)
        pred = regr.predict(expand)

        mae = median_absolute_error(ydata, pred)
        rmse = mean_squared_error(ydata, pred, squared=False)
        stddev = np.std(pred - ydata)
        mean = np.mean(pred - ydata)

        self.transformer_ = poly
        self.coeffs_ = Series(regr.coef_,
                              index=self.get_transformed_feature_names())
        self.meta.update(df.attrs)
        self.meta['comment'] = 'teffnorm = teff / 5040; predicts kx = Ax / A0'
        self.meta['model'] = {'kind': 'polynomial',
                              'degree': degree,
                              'interaction_only': interaction_only,
                              'include_bias': True,
                              'feature_names': list(xdata.columns)}
        self.meta['mae'] = mae
        self.meta['rmse'] = rmse
        self.meta['std_residuals'] = stddev
        self.meta['mean_residuals'] = mean

        return self

    def predict(self, X: Union[np.ndarray, DataFrame]) -> np.array:
        """ Predict the extinction in the specific passband

        .. note::

            if X is a :class:`Dataframe`, `teffnorm` could be automatically added


        Parameters
        ----------
        X: Union[np.ndarray, pd.DataFrame]
            input features

        Return
        ------
        y: np.ndarray
            predicted values
        """
        transformer = self.transformer_
        X_ = self._consolidate_named_data(X)
        if ('teffnorm' in self.feature_names):
            if 'teffnorm' not in X_.columns:
                X_['teffnorm'] = X_['teff'] / 5040.
        X_ = X_[self.feature_names]
        coeffs = self.coeffs_[self.get_transformed_feature_names()]
        expand = transformer.transform(X_)
        return np.inner(coeffs, expand)

    def get_transformed_feature_names(self) -> Sequence[str]:
        """ get the feature names of the internal transformation """
        return self.transformer_.get_feature_names_out()

    def _set_transformer(self, degree: int = 2, interaction_only: bool = False,
                         include_bias: bool = True, order: str = 'C', **params):
        """ Setup the PolynomialFeature transformer

        Generate a new feature matrix consisting of all polynomial combinations
        of the features with degree less than or equal to the specified degree.
        For example, if an input sample is two dimensional and of the form
        [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

        .. seealso::

            :class:`PolynomialFeature` from sklearn

        Parameters
        ----------
        degree : int or tuple (min_degree, max_degree), default=2
            If a single int is given, it specifies the maximal degree of the
            polynomial features. If a tuple `(min_degree, max_degree)` is passed,
            then `min_degree` is the minimum and `max_degree` is the maximum
            polynomial degree of the generated features. Note that `min_degree=0`
            and `min_degree=1` are equivalent as outputting the degree zero term is
            determined by `include_bias`.

        interaction_only : bool, default=False
            If `True`, only interaction features are produced: features that are
            products of at most `degree` *distinct* input features, i.e. terms with
            power of 2 or higher of the same input feature are excluded:

                - included: `x[0]`, `x[1]`, `x[0] * x[1]`, etc.
                - excluded: `x[0] ** 2`, `x[0] ** 2 * x[1]`, etc.

        include_bias : bool, default=True
            If `True` (default), then include a bias column, the feature in which
            all polynomial powers are zero (i.e. a column of ones - acts as an
            intercept term in a linear model).

        order : {'C', 'F'}, default='C'
            Order of output array in the dense case. `'F'` order is faster to
            compute, but may slow down subsequent estimators.
        """
        # check that the model attributes match
        kind = params.pop('kind')
        if kind not in ('polynomial', ):
            raise NotImplementedError(kind, "Expecting a polynomial model definition")

        feature_names = self.feature_names
        # prepare transformer on fake data
        X = DataFrame.from_records(np.zeros((1, len(feature_names))),
                                   columns=feature_names)
        transformer = PolynomialFeatures(degree=degree,
                                         include_bias=include_bias,
                                         interaction_only=interaction_only,
                                         order=order).fit(X)
        self.transformer_ = transformer

    @classmethod
    def from_file(cls, filename: str, passband: str):
        """ Restore a model from a file

        Parameters
        ----------
        filename: str
            the ECSV filepath containing the model definition

            The file should contain the various parameters associated with the
            model in its `metadata`.

        passband: str
            name of the model to load (passband column in the ecsv file)

        Returns
        -------
        model: PolynomialModel
            model object
        """
        data = ecsv.read(filename).set_index('passband')
        name = data.attrs.pop('name', None)

        # setting model metadata
        model = cls(name=name, meta=data.attrs.copy())
        model_attrs = model.meta['model'].copy()
        model._set_transformer(**model_attrs)

        # get regression coefficients
        coeffs = data.loc[passband]
        model.coeffs_ = coeffs[model.get_transformed_feature_names()]

        # get stats if provided
        keys = 'mae,rmse,mean,stddev'.split(',')
        try:
            stats = data.loc[passband][keys]
            for key in keys:
                model.meta[key] = float(stats[key])
        except KeyError:
            pass
        return model

    def to_pandas(self) -> DataFrame:
        """ Export the model to a pandas array, useful for storage """
        # set name consistency
        self.name
        data = self.coeffs_.to_frame().T
        meta = self.meta.copy()
        keys = 'mae,rmse,mean_residuals,std_residuals'.split(',')
        for key in keys:
            data[key] = meta.pop(key, float('nan'))
        data.attrs.update(meta)
        return data

    def to_ecsv(self, fname: str, **meta):
        """ Export model into an ECSV file """
        df = self.to_pandas()
        meta = df.attrs
        meta.update(**meta)
        ecsv.write(df, fname, **meta)
