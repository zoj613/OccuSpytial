import argparse

import numpy as np

from demo_data import X, W, y, Q
from occuspytial.interface import Sampler

# Optionally define the hyper-parameter values. If not explicitly passed
# on to the model the ones defined below are used by default
num_x_cols = X.shape[1]
num_w_cols = W[0].shape[1]
total_sites = X.shape[0]

HYPERS = dict(
    alpha_mu=np.zeros(num_w_cols),
    alpha_prec=np.diag([1. / 1000] * num_w_cols),
    beta_mu=np.zeros(num_x_cols),
    beta_prec=np.diag([1. / 1000] * num_x_cols),
    shape=0.5,
    rate=0.0005
)

# Same with model initializing values
INITS = dict(
    alpha=np.zeros(num_w_cols),
    beta=np.zeros(num_x_cols),
    tau=10.,
    eta=np.random.uniform(-10, 10, size=total_sites)
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Demo script for running ICAR sampler'
    )
    parser.add_argument(
        '--model',
        help='name of the model to use',
        default='ICAR',
        type=str
    )
    parser.add_argument(
        '--chains',
        help='The number of chains to run in parallel during sampling',
        default=2,
        type=int
    )
    parser.add_argument(
        '--iters',
        help='The number of iterations',
        default=20000,
        type=int
    )
    parser.add_argument(
        '--burnin',
        help='The number of iterations to discard as burnout samples',
        default=15000,
        type=int
    )
    args = parser.parse_args()
    if args.model.lower() not in ("icar", "rsr"):
        raise ValueError("model should be one of {'icar', 'rsr}.")
    model_sampler = Sampler(
        X, W, y, Q, model=args.model, chains=args.chains, threshold=0.95
    )
    model_sampler.run(iters=args.iters, burnin=args.burnin)
    # Print the sampler summary
    print(model_sampler.summary)
    model_sampler.trace_plots(save=True, name=f'traces_{args.model}')
    model_sampler.corr_plots(save=True, name=f'corr_{args.model}')
