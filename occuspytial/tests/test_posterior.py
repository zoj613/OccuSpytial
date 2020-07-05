from unittest.mock import patch

import pytest

from occuspytial.chain import Chain
from occuspytial.posterior import PosteriorParameter


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_posterior():
    c = Chain({'p1':  2, 'p2': 1}, size=2)
    p = PosteriorParameter(c)

    with patch('occuspytial.posterior.az') as az:
        p.summary
        assert az.summary.called is True

        p.plot_trace()
        assert az.plot_trace.called is True

        p.plot_auto_corr()
        assert az.plot_autocorr.called is True

        p.plot_pair()
        assert az.plot_pair.called is True

        p.plot_density()
        assert az.plot_posterior.called is True

        p.plot_ess()
        assert az.plot_ess.called is True

    with pytest.raises(KeyError):
        c['p3']

    assert c['p1'].shape == (0, 2)
