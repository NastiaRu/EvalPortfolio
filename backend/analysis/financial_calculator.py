import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# --- CONFIGURATION (The "Financial Constants") ---
# These are the benchmarks we compare against.
RISK_FREE_RATE = 0.042  # 4.2% (Approx. current 10-Year Treasury Yield)
INFLATION_RATE = 0.03   # 3.0% (Target Inflation Benchmark)
MARKET_TICKER = "^GSPC" # S&P 500 Symbol

def fetch_data(tickers, period="1y"):
    """
    Fetches historical closing prices for user stocks + S&P 500.
    """
    print(f"\n[SYSTEM] Fetching data for {', '.join(tickers)} + S&P 500...")
    
    # We add the market ticker to the list to fetch everything at once
    all_tickers = tickers + [MARKET_TICKER]
    
    # Download data
    data = yf.download(all_tickers, period=period, progress=False)['Close'] # type: ignore
    
    # Drop any rows with missing values (clean data)
    data = data.dropna()
    
    return data

def calculate_metrics(data, user_tickers, weights):
    """
    The Core Financial Engine. Calculates Return, Volatility, Beta, Alpha, Sharpe.
    """
    # 1. Calculate Daily Returns (Percentage Change)
    daily_returns = data.pct_change().dropna()
    
    # 2. Separate Market Data from User Data
    market_returns = daily_returns[MARKET_TICKER]
    stock_returns = daily_returns[user_tickers]
    
    # 3. Calculate Portfolio Daily Returns
    # Multiply stock returns by their weights and sum them up for each day
    portfolio_daily_returns = (stock_returns * weights).sum(axis=1)
    
    # --- A. TOTAL RETURN (Annualized) ---
    # We use geometric mean for accuracy, but simple sum is okay for hackathons.
    # Simple Logic: Average daily return * 252 trading days
    avg_daily_return = portfolio_daily_returns.mean()
    portfolio_return_annual = avg_daily_return * 252
    
    market_return_annual = market_returns.mean() * 252

    # --- B. VOLATILITY (Annualized Risk) ---
    # Matrix Math: sqrt(Weights_Transpose * Covariance_Matrix * Weights)
    cov_matrix = stock_returns.cov() * 252
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(variance)

    # --- C. BETA (Correlation to Market) ---
    # Beta = Covariance(Portfolio, Market) / Variance(Market)
    covariance_with_market = portfolio_daily_returns.cov(market_returns)
    market_variance = market_returns.var()
    portfolio_beta = covariance_with_market / market_variance

    # --- D. SHARPE RATIO (Efficiency) ---
    # (Return - Risk_Free) / Volatility
    sharpe_ratio = (portfolio_return_annual - RISK_FREE_RATE) / portfolio_volatility

    # --- E. JENSEN'S ALPHA (Skill) ---
    # Actual Return - Expected Return based on Risk
    expected_return = RISK_FREE_RATE + portfolio_beta * (market_return_annual - RISK_FREE_RATE)
    alpha = portfolio_return_annual - expected_return

    return {
        "return": portfolio_return_annual,
        "market_return": market_return_annual,
        "volatility": portfolio_volatility,
        "beta": portfolio_beta,
        "sharpe": sharpe_ratio,
        "alpha": alpha
    }

def calculate_goodness_score(metrics):
    """
    Converts raw financial metrics into a 0-100 "Goodness Score".
    """
    # 1. Real Return Score (Max 30)
    # Target: 10% return. Floor: 0%.
    raw_return_score = (metrics['return'] / 0.10) * 30
    score_return = max(0, min(30, raw_return_score))
    
    # 2. Alpha Score (Max 40)
    # Base: 20 pts. +/- 5% alpha swings score to 0 or 40.
    raw_alpha_score = 20 + (metrics['alpha'] * 400)
    score_alpha = max(0, min(40, raw_alpha_score))
    
    # 3. Sharpe Score (Max 30)
    # Target: Sharpe 2.0.
    raw_sharpe_score = metrics['sharpe'] * 15
    score_sharpe = max(0, min(30, raw_sharpe_score))
    
    total_score = int(score_return + score_alpha + score_sharpe)
    
    return total_score, {
        "return_pts": score_return,
        "alpha_pts": score_alpha,
        "sharpe_pts": score_sharpe
    }

def print_report(metrics, score, breakdown, tickers, weights):
    """
    Outputs a beautiful ASCII report for the user.
    """
    print("\n" + "="*60)
    print(f"       PORTFOLIO ANALYSIS REPORT")
    print("="*60)
    print(f"STOCKS: {', '.join(tickers)}")
    print(f"WEIGHTS: {weights}")
    print("-" * 60)
    
    # 1. The Big Score
    print(f"\n>>> FINAL GOODNESS SCORE: {score}/100")
    
    if score > 80: print("    VERDICT: EXCELLENT (Professional Grade)")
    elif score > 60: print("    VERDICT: GOOD (Solid Strategy)")
    elif score > 40: print("    VERDICT: MEDIOCRE (Needs Tuning)")
    else: print("    VERDICT: POOR (High Risk / Low Reward)")

    print("\n" + "-"*60)
    
    # 2. The Detailed Benchmarks
    p_ret = metrics['return'] * 100
    m_ret = metrics['market_return'] * 100
    inf_ret = INFLATION_RATE * 100
    
    print(f"\n1. RETURN vs BENCHMARKS (Score: {int(breakdown['return_pts'])}/30)")
    print(f"   Your Return:      {p_ret:.2f}%")
    print(f"   S&P 500 Return:   {m_ret:.2f}%")
    print(f"   Inflation Target: {inf_ret:.2f}%")
    
    if p_ret > inf_ret:
        print(f"   [PASS] You beat inflation by {(p_ret - inf_ret):.2f}%")
    else:
        print(f"   [FAIL] You lost purchasing power.")
    
    # 3. Skill Analysis (Alpha)
    print(f"\n2. STRATEGY SKILL (Score: {int(breakdown['alpha_pts'])}/40)")
    print(f"   Beta (Risk Level): {metrics['beta']:.2f}")
    if metrics['beta'] > 1:
        print("   (Your portfolio is RISKIER than the market)")
    else:
        print("   (Your portfolio is SAFER than the market)")
        
    print(f"   Alpha (Excess Return): {metrics['alpha']*100:.2f}%")
    print(f"   (A positive alpha indicates outperformance of the benchmark index, S&P 500)")
    print(f"   (A negative alpha indicates underperformance of the benchmark index.)")
    if metrics['alpha'] > 0:
        print("   [SUCCESS] You are beating the market based on risk taken.")
    else:
        print("   [WARNING] You are underperforming relative to the risk taken.")

    # 4. Risk Analysis
    print(f"\n3. RISK PROFILE (Score: {int(breakdown['sharpe_pts'])}/30)")
    print(f"   Volatility: {metrics['volatility']*100:.2f}% (Lower is safer)")
    print(f"   (means the price could potentially drop or rise by {metrics['volatility']*100:.2f}% in one go)")
    print(f"   Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"   (measures return per unit of risk)")
    if metrics['sharpe'] > 1: 
        print(f"   >1.0 is good: you get smooth and steady returns given the risk taken.")
    else:
        print(f"   <1.0 is bad: you get volatile returns given the risk taken.")

    print("\n" + "="*60 + "\n")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # --- USER INPUT AREA ---
    # You can change these to test different portfolios!
    
    # portfolio - {allocations: {MSFT: 25, AAPL: 25, GOOGL: 25, AMZN: 25}} <- incoming from frontend, feed into this script
    
    # Example 1: The "Tech Bro" Portfolio (Risky)
    my_stocks = ['TSLA', 'NVDA', 'AMD']
    my_weights = np.array([0.9, 0.05, 0.05]) 
    
    # Example 2: The "Balanced" Portfolio
    # my_stocks = ['AAPL', 'MSFT', 'JNJ', 'KO', 'GOOGL']
    # my_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2]) # Must sum to 1.0
    
    try:
        # 1. Get Data
        market_data = fetch_data(my_stocks)
        
        # 2. Run Financial Math
        metrics = calculate_metrics(market_data, my_stocks, my_weights)
        
        # 3. Grade the Portfolio
        score, breakdown = calculate_goodness_score(metrics)
        
        # 4. Show Report
        print_report(metrics, score, breakdown, my_stocks, my_weights)
        
    except Exception as e:
        print(f"\n[ERROR] Something went wrong: {e}")
        print("Ensure tickers are correct and you have an internet connection.")