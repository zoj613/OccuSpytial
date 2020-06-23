import arviz as az
import numpy as np

az.style.use("arviz-darkgrid")


class PosteriorParameter:
    _exclude = ('z', 'eta')

    def __init__(self, *chains):
        self.data = self._create_inference_data(chains)

    def _create_inference_data(self, chains):
        self._excluded = dict.fromkeys(self._exclude)
        if len(chains) > 1:
            data = {
                name: np.stack([c[name] for c in chains])
                for name in chains[0]._names
                if name not in self._exclude
            }
            self._excluded = {
                name: np.stack([c[name] for c in chains])
                for name in chains[0]._names
                if name in self._exclude
            }
        else:
            data = {
                name: chains[0][name][None, :]
                for name in chains[0]._names
                if name not in self._exclude
            }
            self._excluded = {
                name: chains[0][name][None, :]
                for name in chains[0]._names
                if name in self._exclude
            }

        return az.convert_to_inference_data(data).posterior

    @property
    def summary(self):
        return az.summary(self.data)

    def plot_trace(self, **kwargs):
        return az.plot_trace(self.data, **kwargs)

    def plot_auto_corr(self, **kwargs):
        return az.plot_autocorr(self.data, **kwargs)

    def plot_pair(self, **kwargs):
        return az.plot_pair(self.data, **kwargs)

    def plot_density(self, **kwargs):
        return az.plot_posterior(self.data, **kwargs)

    def plot_ess(self, **kwargs):
        return az.plot_ess(self.data, **kwargs)

    def __getitem__(self, name):
        return self.data[name].data
