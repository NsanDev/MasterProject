from math import floor
from numpy import mean, multiply, ones


class RiskStatistics:

    @staticmethod
    def riskStatistics(values, weights=None, alpha=0.05):
        N = len(values)
        weights = ones(N) if weights == None else weights
        S = sorted(zip(values, weights), key=lambda x: x[0])
        index_alpha = 0
        cumulative_weights = 0
        while(index_alpha<N and cumulative_weights <1-alpha):
            cumulative_weights += S[index_alpha,1]
        VaR = S[index_alpha, 0]
        ExpectedShortfall = mean(multiply(S[index_alpha:, 0],S[index_alpha:, 1])/sum(S[index_alpha:, 1]) )
        Mean = mean(values)
        return [Mean, VaR, ExpectedShortfall]

    def VaR(values, alpha=0.05):
        S = sorted(values)
        N = len(S)
        result = S[floor((1 - alpha) * N) - 1]
        return result

    def ExpectedShortfall(values, alpha=0.05):
        S = sorted(values)
        N = len(S)
        VaR = S[floor((1 - alpha) * N) - 1]
        index_alpha = S.find(VaR)
        result = mean(S[index_alpha:, ])
        return result