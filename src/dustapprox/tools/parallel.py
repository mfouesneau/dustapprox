"""Parallel processing with joblib and tqdm."""
import contextlib
import joblib
from tqdm import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument

    Solution adapted from `Stackoverflow <https://stackoverflow.com/a/58936697>`_.

    .. code-block:: python

        import time
        from joblib import Parallel, delayed
        def some_method(wait_time):
            time.sleep(wait_time)

        with tqdm_joblib(tqdm(desc="My method", total=10)) as progress_bar:
            Parallel(n_jobs=2)(delayed(some_method)(0.2) for i in range(10))

    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()