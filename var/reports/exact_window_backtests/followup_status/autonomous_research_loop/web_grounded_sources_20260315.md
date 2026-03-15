# Web-grounded research intake (2026-03-15)

Primary-source directions reviewed for the autonomous research loop:

1. Time-series momentum
   - Moskowitz, Ooi, Pedersen (2012)
   - https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf
   - Direct relevance: crash-aware / regime-aware trend gating and momentum sleeves.

2. Carry across asset classes
   - Koijen, Moskowitz, Pedersen, Vrugt (NBER 19325)
   - https://www.nber.org/papers/w19325
   - Direct relevance: carry + momentum hybrid sleeves and cross-asset carry ranking.

3. Residual / factor-neutral momentum references
   - residual momentum cited in NBER 23394
   - https://www.nber.org/system/files/working_papers/w23394/w23394.pdf
   - Direct relevance: residualized cross-sectional momentum and market/beta-neutral top-cap ranking.

4. Crypto cross-sectional network / momentum structure
   - A Time-Varying Network for Cryptocurrencies
   - https://arxiv.org/abs/2108.11921
   - Direct relevance: inter-crypto momentum / network spillover features beyond plain top-cap rotation.

5. Joint time-series + cross-sectional momentum
   - Spatio-Temporal Momentum
   - https://arxiv.org/abs/2302.10175
   - Direct relevance: combining time-series and cross-sectional signal families rather than treating them independently.

6. Crypto multivariate market-neutral expansion
   - Optimal Market-Neutral Multivariate Pair Trading on the Cryptocurrency Platform
   - https://www.mdpi.com/2227-7072/12/3/77
   - Direct relevance: expanding pair spreads into multivariate / optimized market-neutral sleeves.

7. Dynamic pair scaling / adaptive pair control
   - Reinforcement Learning Pair Trading: A Dynamic Scaling approach
   - https://arxiv.org/abs/2407.16103
   - Direct relevance: dynamic hedge / scaling ideas for pair sleeves, even if RL itself is too heavy for direct adoption.

Current action from these sources:
- prioritize low-implementation-risk variants first:
  - residual/factor-neutral topcap momentum
  - carry+momentum hybrid ranking
  - multivariate/pair-state filters
  - crash-aware momentum gating
- defer high-complexity RL-only methods unless simpler deterministic approximations fail.

8. Residual momentum
   - Blitz, Huij, Martens (2011), Journal of Empirical Finance
   - DOI: 10.1016/j.jempfin.2011.01.003
   - Direct relevance: residualize momentum signals against common factors/beta before ranking.

9. Factor momentum
   - Ehsani, Linnainmaa (NBER Working Paper 25551)
   - https://www.nber.org/papers/w25551
   - Direct relevance: momentum crashes can be interpreted as factor-autocorrelation breaks; useful for crash-aware gating.

10. Slow momentum with fast reversion / changepoints
   - https://arxiv.org/abs/2105.13727
   - Direct relevance: use deterministic changepoint / reversal proxies to gate momentum sleeves when fast reversals dominate.

11. Kelly sizing / growth-optimal allocation
   - Kelly portfolio overview / multivariate references:
     - NBER factor-momentum context: https://www.nber.org/papers/w25551
     - multivariate Kelly reference: https://www.cs.miami.edu/home/burt/learning/mth649.191/docs/SSRN-id2259133.pdf
     - fractional Kelly discussion: https://www.edwardothorp.com/wp-content/uploads/2016/11/KellySimulationsNew.pdf
   - Practical interpretation for this loop:
     - full Kelly is usually too aggressive under parameter uncertainty;
     - fractional Kelly / capped Kelly-style scaling can be used as a sleeve-weighting or overlay benchmark;
     - only promote if locked-OOS return improves without blowing up drawdown.

12. Volatility-managed portfolios
   - Moreira, Muir (NBER Working Paper 22208)
   - https://www.nber.org/papers/w22208
   - Direct relevance: explicit volatility scaling / target-vol overlays can improve Sharpe when expected returns do not fully offset volatility spikes.

13. Optimal stop-loss / take-profit in mean reversion
   - Leung, Li (2014)
   - https://arxiv.org/abs/1411.5062
   - Direct relevance: stop-loss constraints interact with take-profit thresholds; useful for bounded pair-state / spread exit design.

14. Cut-loss / take-profit / trailing-high liquidation rationale
   - Xu, Zhou (2011)
   - https://arxiv.org/abs/1103.1755
   - Direct relevance: theoretical support for cut-loss, take-profit, and sell-on-percentage-of-historical-high style exit logic.
