"""Base class for deriving various kinds of models."""


class BaseModel:
    """A model object other approximations derive from

    Attributes
    ----------
    meta: dict
        meta information about the model
    """

    def __init__(self, /, **kwargs):
        self.meta = kwargs.get("meta", {})
        self.name_ = kwargs.get("name", None)

    def fit(self, *args, **kwargs):
        """Fit the model to the data

        Parameters
        ----------
        *args:
            positional arguments
        **kwargs:
            keyword arguments
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """Predict the extinction in a given passband.

        Parameters
        ----------
        *args:
            positional arguments
        **kwargs:
            keyword arguments
        """
        raise NotImplementedError

    def to_pandas(self):
        """Export the model to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The model as a pandas DataFrame.
        """
        raise NotImplementedError
