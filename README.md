# EvalPortfolio
- Luke T. Bailey - <ltb33@pitt.edu>
- Andrew Y. Chen - <ayc23@pitt.edu>
- Anastasiia Rudenko - <anastasiia.rudenko@pitt.edu>

Idea: The Financial "Goodness" Calculator.

**"Don't just measure your wealth. Measure your wisdom."**

## The Big Idea

Most personal finance apps answer a simple question: *"How much money do I have?"*

But they fail to answer the far more important question: *"Is my portfolio actually __good__?"*

In finance, a +10% return is meaningless without context.
   * If inflation is 12%, you actually lost money.
   * If the S&P 500 went up 20% while you made 10%, you wasted your time picking stocks.
   * If you took on the risk of a chaotic penny stock just to make a safe 5% return, you made a bad bet.

**EvalPortfolio** bridges the gap between complex Wall Street theory and everyday investing. It helps the user to evaluate their stock selection and allocation of funds and quantifies the evaluation to make it digestible to the user. In simple words, EvalPortfolio is a calculator of a single **"Goodness Score" (0-100)** that is based on three first-principle pillars of investment: **Earning**, **Stability**, and **Risk**.


## The Core Logic: The 3 Pillars of "Goodness"

We don't just sum up your profits. We grade your portfolio on a curve against the harsh reality of the market.
Read more here: https://docs.google.com/document/d/1rXD0MWqgPYW27OnrxW_8jGhssJB9uKTWCmpM0AfSevA/edit?tab=t.4cxgxgjoe0jd.

Final score is between 0 and 100: 
  * score = 81+: EXCELLENT (Professional Grade)
  * score = 61..80: GOOD (Solid Strategy)
  * score = 41..60: MEDIOCRE (Needs Tuning)
  * score = 0..40: POOR (High Risk / Low Reward)

### The Survival Test (Real Return): 30 points/100
* **The Philosophy:** Money loses value over time due to inflation. If your portfolio grows by 2% but inflation is 3%, your "Nominal Return" is positive, but your "Real Return" is negative. You are getting poorer.
* **The Metric:** Comparison against the **Consumer Price Index (CPI)** target (approx. 3%).
* **The Score:** We award points only if you preserve and grow your purchasing power.

Survival gradient: if Inflation is 3% and you made 2.9%, you preserved almost all your wealth. That is infinitely better than losing -20%. So we set a "Floor" (0% return) and a "Ceiling" (a great return, e.g., 10% or Inflation + 7%).
* Floor (<0% Return): 0 Points. (You lost nominal money. This is bad.)
* Baseline (0% to Inflation): 1 to 15 Points. (You made money, but lost purchasing power. Better than nothing.)
* Target (> Inflation): 16 to 30 Points. (You grew your wealth in real terms.)

$$
\text{ScoreInflation} = ( R_p / 0.1 ) * 30pts
$$

### The Stability Test (Sharpe Ratio): 30 points/100
* **The Philosophy:** High returns are useless if they come with heart-attack-inducing volatility. A portfolio that goes +50% one month and -40% the next is "inefficient." You should be paid for every unit of risk you take.
* **The Metric:** **Sharpe Ratio** $$\frac{Return - RiskFree}{Volatility}$$
* **The Score:**
    * Measures return per unit of risk
    * **Sharpe < 1.0:** Poor. (Too much risk for too little reward).
    * **Sharpe > 1.0:** Good.
    * **Sharpe > 2.0:** Excellent. (A smooth, profitable ride).

### The Skill Test (Alpha-score): 40 points/100
**"Are you smart, or just lucky?"**
* **The Philosophy:** This is our secret sauce. Anyone can make money when the whole market is booming. That's not skill; that's "Beta" (market exposure).
* **The Metric:** **($\alpha$)**.
    * We calculate the *Expected Return* based on the risk you took (CAPM Model).
    * If you made *more* than the math predicted, you have **Positive Alpha**. You "beat the market."
    * If you made *less*, you have **Negative Alpha**. You underperformed your own risk profile.

Since beating the market is extremely hard, we don't penalize the score too hard for having an Alpha near 0.


## Under the Hood: The Math

We use **Modern Portfolio Theory (MPT)** to ensure our numbers are mathematically robust.

### Matrix Multiplication for Volatility
We do not simply average the volatility of individual stocks. We account for **Covariance** (how stocks move together). If Stock A zig-zags while Stock B zag-zigs, they cancel each other's risk out.

We calculate Portfolio Volatility ($\sigma_p$) using Linear Algebra:

$$
\sigma_p = \sqrt{W^T \cdot \Sigma \cdot W}
$$

* **$W$**: The Weights Vector (e.g., `[0.5, 0.3, 0.2]`).
* **$\Sigma$**: The Annualized Covariance Matrix of the assets.

### The Alpha Formula
We determine user skill using the Capital Asset Pricing Model (CAPM):

$$
\alpha = R_p - [ R_f + \beta \times (R_m - R_f) ]
$$

* **$R_p$**: User's Actual Return.
* **$R_f$**: Risk-Free Rate (10-Year Treasury Yield).
* **$R_m$**: Market Return (S&P 500).
* **$\beta$**: The portfolio's sensitivity to market movements.


## Tech Stack

* **Language:** Python 3.9+
* **Data Source:** `yfinance` (Yahoo Finance API) for real-time OHLCV market data.
* **Math Engine:** `numpy` for matrix operations and linear algebra.
* **Data Processing:** `pandas` for time-series manipulation and covariance calculations.
