from copy import deepcopy

from joblib import Parallel, delayed


def sample_parallel(class_, **kwargs):
    chains = kwargs.pop('chains')

    if chains > 1:
        try:
            samplers = [class_.copy() for _ in range(chains)]
        except AttributeError:
            samplers = [deepcopy(class_) for _ in range(chains)]
    else:
        samplers = (class_,)

    out = Parallel(prefer='processes', n_jobs=-1)(
        delayed(sampler._run)(pos=pos, **kwargs)
        for pos, sampler in enumerate(samplers)
    )
    return out
