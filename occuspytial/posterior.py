import arviz as az
import numpy as np

az.style.use("arviz-darkgrid")


class PosteriorParameter:
    """Container to store posterior samples, produce plots and summaries.

    This object is returned by samplers so that posterior parameter samples
    can be easily accessed. It also provides several methods to perform basic
    inference on the posterior samples.

    Parameters
    ----------
    *chains
        instances of :class:`~occuspytial.chain.Chain`.

    Attributes
    ----------
    summary
    data : arviz.InferenceData
        Inference data object.
    """

    def __init__(self, *chains):
        self.data = self._create_inference_data(chains)

    def _create_inference_data(self, chains):
        if len(chains) > 1:
            data = {
                name: np.stack([c[name] for c in chains])
                for name in chains[0]._names
            }
        else:
            data = {name: chains[0][name][None] for name in chains[0]._names}

        return az.convert_to_inference_data(data).posterior

    @property
    def summary(self):
        """Return summary statistics of posterior parameter samples.

        Default statistics are: ``mean``, ``sd``, ``hdi_3%``, ``hdi_97%``,
        ``mcse_mean``, ``mcse_sd``, ``ess_bulk``, ``ess_tail``, and ``r_hat``.
        ``r_hat`` is only computed for traces with 2 or more chains.

        Returns
        -------
        pandas.DataFrame
            A dataframe of the summary.
        """
        return az.summary(self.data)

    def plot_trace(self, **kwargs):
        """Plot density and samples values of parameters.

        See arviz library documentation for a full list of legal parameters.

        Parameters
        ----------
        **kwargs
            Keyword arguments optionally passed to ``arviz.plot_trace``

        Returns
        -------
        axes : matplotlib axes
        """
        return az.plot_trace(self.data, **kwargs)

    def plot_auto_corr(self, **kwargs):
        """Plot the autocorrelation function of each posterior parameter.

        See arviz library documentation for a full list of legal parameters.

        Parameters
        ----------
        **kwargs
            Keyword arguments optionally passed to ``arviz.plot_autocorr``

        Returns
        -------
        axes : matplotlib axes
        """
        return az.plot_autocorr(self.data, **kwargs)

    def plot_pair(self, **kwargs):
        """Pair plots of the posterior parameters.

        See arviz library documentation for a full list of legal parameters.

        Parameters
        ----------
        **kwargs
            Keyword arguments optionally passed to ``arviz.plot_pair``.

        Returns
        -------
        axes : matplotlib axes
        """
        return az.plot_pair(self.data, **kwargs)

    def plot_density(self, **kwargs):
        """Plot Posterior densities in the style of John K. Kruschke’s book.

        See arviz library documentation for a full list of legal parameters.

        Parameters
        ----------
        **kwargs
            Keyword arguments optionally passed to `arviz.plot_posterior`

        Returns
        -------
        axes : matplotlib axes
        """
        return az.plot_posterior(self.data, **kwargs)

    def plot_ess(self, **kwargs):
        """Plot quantile, local or evolution of effective sample sizes (ESS).

        See arviz library documentation for a full list of legal parameters.

        Parameters
        ----------
        **kwargs
            Keyword arguments optionally passed to `arviz.plot_ess`

        Returns
        -------
        axes : matplotlib axes
        """
        return az.plot_ess(self.data, **kwargs)

    def __getitem__(self, name):
        return self.data[name].data
