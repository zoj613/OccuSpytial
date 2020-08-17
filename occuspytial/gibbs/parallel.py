from joblib import Parallel, delayed


def sample_parallel(class_, **kwargs):
    """Perform MCMC sampling in parallel.

    Parameters
    ----------
    class_ : object
        Sampler instance that implements a `sample`, `step` and `_run` methods.
    **kwargs
        Keyword arguments to pass to the sampler's `sample` / `_run` methods.

    Returns
    -------
    out : List[PosteriorParameter]
        posterior samplers
    """
    chains = kwargs.pop('chains')
    samplers = [class_]

    if chains > 1:
        samplers.extend([class_.copy() for _ in range(chains - 1)])
        backend = 'processes'
    else:
        # for a single chain the threading backend is selected so that the
        # coverage reported during tests is accurate. If the loky or multi-
        # processing backends are chosen then relevent parts of the codebase
        # are not accurately reported in code coverage reports since even for
        # 1 chain the multiprocessing backends copy the instance into a new
        # process before executing it inside joblib. Which means that in a
        # coverage report the relevent lines will show as never executed. This
        # also means that execution with 1 chain will be half the speed of
        # execution with 2 chains. This is not an issue when using other
        # backends, So I will need to find a workaround for the speed hit.
        backend = 'threads'

    out = Parallel(prefer=backend, n_jobs=-1)(
        delayed(sampler._run)(pos=pos, **kwargs)
        for pos, sampler in enumerate(samplers)
    )
    return out
