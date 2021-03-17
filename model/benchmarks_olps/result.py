import numpy as np
import pandas as pd
import hashlib
import matplotlib.pyplot as plt
import pickle
from model.benchmarks_olps import tools
import seaborn as sns
from statsmodels.api import OLS
from matplotlib.colors import ListedColormap


class PickleMixin(object):

    def save(self, filename):
        """ Save object as a pickle """
        with open(filename, 'wb') as f:
            pickle.dump(self, f, -1)

    @classmethod
    def load(cls, filename):
        """ Load pickled object. """
        with open(filename, 'rb') as f:
            return pickle.load(f)


class AlgoResult(PickleMixin):
    """ Results returned by algo's run method. The class containts useful
    metrics such as sharpe ratio, mean return, drawdowns, ... and also
    many visualizations.
    You can specify transactions by setting AlgoResult.fee. Fee is
    expressed in a percentages as a one-round fee.
    """

    def __init__(self, X, B):
        """
        :param X: Price relatives.
        :param B: Weights.
        """
        # set initial values
        self._fee = 0.
        self._B = B
        self.rf_rate = 0.
        self._X = X

        assert self.X.max().max() < np.inf

        # update logarithms, fees, etc.
        self._recalculate()

    def set_rf_rate(self, rf_rate):
        if isinstance(rf_rate, float):
            self.rf_rate = rf_rate
        else:
            self.rf_rate = rf_rate.reindex(self.X.index)
        self._recalculate()
        return self

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, _X):
        self._X = _X
        self._recalculate()

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, _B):
        self._B = _B
        self._recalculate()

    @property
    def fee(self):
        return self._fee

    @fee.setter
    def fee(self, value):
        """ Set transaction costs. Fees can be either float or Series
        of floats for individual assets with proper indices. """
        if isinstance(value, dict):
            value = pd.Series(value)
        if isinstance(value, pd.Series):
            missing = set(self.X.columns) - set(value.index)
            assert len(missing) == 0, 'Missing fees for {}'.format(missing)

        self._fee = value
        self._recalculate()

    def _recalculate(self):
        # calculate return for individual stocks
        r = (self.X - 1) * self.B
        self.asset_r = r + 1
        self.r = r.sum(axis=1) + 1

        # stock went bankrupt
        self.r[self.r < 0] = 0.

        # add risk-free asset
        self.r -= (self.B.sum(axis=1) - 1) * self.rf_rate / self.freq()

        # add fees
        if not isinstance(self._fee, float) or self._fee != 0:
            fees = (self.B.shift(-1).mul(self.r, axis=0) - self.B * self.X).abs()
            fees.iloc[0] = self.B.iloc[0]
            fees.iloc[-1] = 0.
            fees *= self._fee

            self.asset_r -= fees
            self.r -= fees.sum(axis=1)

        self.r_log = np.log(self.r)

    @property
    def weights(self):
        return self.B

    @property
    def equity(self):
        return self.r.cumprod()

    @property
    def equity_decomposed(self):
        """ Return equity decomposed to individual assets. """
        return self.asset_r.cumprod()

    @property
    def asset_equity(self):
        return self.X.cumprod()

    @property
    def total_wealth(self):
        return self.r.prod()

    @property
    def profit_factor(self):
        x = self.r_log
        up = x[x > 0].sum()
        down = -x[x < 0].sum()
        return up / down if down != 0 else np.inf

    @property
    def sharpe(self):
        """ Compute annualized sharpe ratio from log returns. If data does
        not contain datetime index, assume daily frequency with 252 trading days a year.
        """
        return tools.sharpe(self.r - 1, rf_rate=self.rf_rate, freq=self.freq())

    @property
    def sharpe_std(self):
        return tools.sharpe_std(self.r - 1, rf_rate=self.rf_rate, freq=self.freq())

    @property
    def ulcer(self):
        return tools.ulcer(self.r - 1, rf_rate=self.rf_rate, freq=self.freq())

    @property
    def growth_rate(self):
        return self.r_log.mean() * self.freq()

    @property
    def volatility(self):
        return np.sqrt(self.freq()) * self.r_log.std()

    @property
    def annualized_return(self):
        return np.exp(self.r_log.mean() * self.freq()) - 1

    @property
    def annualized_volatility(self):
        return np.exp(self.r_log).std() * np.sqrt(self.freq())

    @property
    def drawdown_period(self):
        ''' Returns longest drawdown perid. Stagnation is a drawdown too. '''
        x = self.equity
        period = [0.] * len(x)
        peak = 0
        for i in range(len(x)):
            # new peak
            if x[i] > peak:
                peak = x[i]
                period[i] = 0
            else:
                period[i] = period[i-1] + 1
        return max(period) * 252. / self.freq()

    @property
    def max_drawdown(self):
        ''' Returns highest drawdown in percentage. '''
        x = self.equity
        return max(1. - x / x.cummax())

    @property
    def winning_pct(self):
        x = self.r_log
        win = (x > 0).sum()
        all_trades = (x != 0).sum()
        return float(win) / all_trades

    @property
    def turnover(self):
        B = self.B
        X = self.X

        # equity increase
        E = (B * (X - 1)).sum(axis=1) + 1

        # required new assets
        R = B.shift(-1).multiply(E, axis=0) / X

        D = R - B

        # rebalancing
        return D.abs().sum().sum() / (len(B) / self.freq())

    def freq(self, x=None):
        """ Number of data items per year. If data does not contain
        datetime index, assume daily frequency with 252 trading days a year."""
        x = x or self.r
        return tools.freq(x.index)

    def summary(self, name=None):
        return f"""Summary{'' if name is None else ' for ' + name}:
    Profit factor: {self.profit_factor:.2f}
    Sharpe ratio: {self.sharpe:.2f} ± {self.sharpe_std:.2f}
    Ulcer index: {self.ulcer:.2f}
    Annualized return: {self.annualized_return:.2%}
    Annualized volatility: {self.annualized_volatility:.2%}
    Longest drawdown: {self.drawdown_period:.0f} days
    Max drawdown: {self.max_drawdown:.2%}
    Winning days: {self.winning_pct:.1%}
    Annual turnover: {self.turnover:.1f}
        """

    def plot(self, weights=True, assets=True, portfolio_label='PORTFOLIO', show_only_important=True, **kwargs):
        """ Plot equity of all assets plus our strategy.
        :param weights: Plot weights as a subplot.
        :param assets: Plot asset prices.
        :return: List of axes.
        """
        res = ListResult([self], [portfolio_label])
        if not weights:
            ax1 = res.plot(assets=assets, **kwargs)
            return [ax1]
        else:
            if show_only_important:
                ix = self.B.abs().sum().nlargest(n=20).index
                B = self.B.loc[:, ix].copy()
                assets = B.columns if assets else False
                if B.shape[1] > 20:
                    B['_others'] = self.B.drop(ix, 1).sum(1)
            else:
                B = self.B.copy()

            plt.figure(1)
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            res.plot(assets=assets, ax=ax1, **kwargs)
            ax2 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)

            # plot weights as lines
            if B.drop(['CASH'], 1, errors='ignore').values.min() < -0.01:
                B = B.sort_index(axis=1)
                B.plot(ax=ax2, ylim=(min(0., B.values.min()), max(1., B.values.max())),
                                          legend=False, color=_colors_hash(B.columns))
            else:
                B = B.drop('CASH', 1, errors='ignore')
                # fix rounding errors near zero
                if B.values.min() < 0:
                    pB = B - B.values.min()
                else:
                    pB = B
                pB.plot(ax=ax2, ylim=(0., max(1., pB.sum(1).max())),
                                           legend=False, color=_colors_hash(pB.columns), kind='area', stacked=True)
            plt.ylabel('weights')
            return [ax1, ax2]


    def plot_decomposition(self, **kwargs):
        """ Decompose equity into components of individual assets and plot
        them. Does not take fees into account. """
        ax = self.equity_decomposed.plot(**kwargs)
        return ax

    @property
    def importance(self):
        ws = self.weights.sum()
        return (ws / sum(ws)).order(ascending=False)

    def plot_total_weights(self):
        _, axes = plt.subplots(ncols=2)
        self.B.iloc[-1].sort_values(ascending=False).iloc[:15].plot(kind='bar', title='Latest weights', ax=axes[1])
        self.B.sum().sort_values(ascending=False).iloc[:15].plot(kind='bar', title='Total weights', ax=axes[0])


class ListResult(list, PickleMixin):
    """ List of AlgoResults. """

    def __init__(self, results=None, names=None):
        results = results if results is not None else []
        names = names if names is not None else []
        super().__init__(results)
        self.names = names

    def append(self, result, name):
        super(ListResult, self).append(result)
        self.names.append(name)

    def to_dataframe(self):
        """ Calculate equities for all results and return one dataframe. """
        eq = {}
        for result, name in zip(self, self.names):
            eq[name] = result.equity
        return pd.DataFrame(eq)

    def save(self, filename, **kwargs):
        # do not save it with fees
        #self.fee = 0.
        #self.to_dataframe().to_pickle(*args, **kwargs)

        with open(filename, 'wb') as f:
            pickle.dump(self, f, -1)

    @classmethod
    def load(cls, filename):
        # df = pd.read_pickle(*args, **kwargs)
        # return cls([df[c] for c in df], df.columns)

        with open(filename, 'rb') as f:
            return pickle.load(f)

    @property
    def fee(self):
        return {name: result.fee for result, name in zip(self, self.names)}

    @fee.setter
    def fee(self, value):
        for result in self:
            result.fee = value

    def summary(self):
        return '\n'.join([result.summary(name) for result, name in zip(self, self.names)])

    def plot(self, assets=False, **kwargs):
        """ Plot strategy equity.
        :param assets: Add asset prices.
        :param kwargs: Additional arguments for pd.DataFrame.plot
        """
        # NOTE: order of plotting is important because of coloring
        # plot portfolio
        d = self.to_dataframe()
        D = d.copy()

        # add individual assets
        if isinstance(assets, bool):
            if assets:
                assets = self[0].asset_equity.columns
            else:
                assets = []

        if list(assets):
            D = D.join(self[0].asset_equity)

        ax = D.plot(color=_colors_hash(D.columns), **kwargs)
        kwargs['ax'] = ax

        ax.set_ylabel('Total wealth')

        return ax


def _colors(n):
    return sns.color_palette(n_colors=n)

def _hash(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)

def _colors_hash(columns, n=19):
    palette = sns.color_palette(n_colors=n)
    return ['blue' if c == 'PORTFOLIO' else palette[_hash(c) % n] for c in columns]
