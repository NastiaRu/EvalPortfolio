import React, { useState } from 'react'
import Login from './pages/Login'
import StockSelection from './pages/StockSelection'

// Simple SVG Icons
const TrendingUpIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>
    <polyline points="17 6 23 6 23 12"/>
  </svg>
)

const CheckIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
)

const AlertIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"/>
    <line x1="12" y1="8" x2="12" y2="12"/>
    <line x1="12" y1="16" x2="12.01" y2="16"/>
  </svg>
)

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [user, setUser] = useState(null)
  const [portfolio, setPortfolio] = useState(null)

  const handleLoginSuccess = (userData) => {
    setIsAuthenticated(true)
    setUser(userData)
    sessionStorage.setItem('authToken', userData.token)
    sessionStorage.setItem('username', userData.username)
  }

  const handlePortfolioSubmit = (portfolioData) => {
    console.log('Portfolio data received:', portfolioData)
    setPortfolio(portfolioData)
  }

  const handleLogout = () => {
    setIsAuthenticated(false)
    setUser(null)
    setPortfolio(null)
    sessionStorage.removeItem('authToken')
    sessionStorage.removeItem('username')
  }

  // Helper function to display NaN as "N/A"
  const displayValue = (value, suffix = '') => {
    if (value === null || value === undefined || (typeof value === 'number' && isNaN(value))) {
      return 'N/A'
    }
    return typeof value === 'number' ? `${value.toFixed(2)}${suffix}` : value
  }

  // Show login if not authenticated
  if (!isAuthenticated) {
    return <Login onLoginSuccess={handleLoginSuccess} />
  }

  // Show stock selection if no portfolio yet
  if (!portfolio) {
    return <StockSelection onSubmitSuccess={handlePortfolioSubmit} />
  }

  // Define variables BEFORE using them
  const analysis = portfolio?.analysis
  
  if (!analysis) {
    return (
      <div style={{ padding: '2rem' }}>
        <h1>Error: No Analysis Data</h1>
        <pre>{JSON.stringify(portfolio, null, 2)}</pre>
        <button onClick={() => setPortfolio(null)}>Try Again</button>
      </div>
    )
  }

  const hasNaNValues = (
    isNaN(analysis.metrics?.return) ||
    isNaN(analysis.metrics?.alpha) ||
    isNaN(analysis.metrics?.beta) ||
    isNaN(analysis.metrics?.sharpe)
  )

  const getScoreColor = (score) => {
    if (score > 80) return '#10b981'
    if (score > 60) return '#3b82f6'
    if (score > 40) return '#f59e0b'
    return '#ef4444'
  }

  // Show results page with full analysis
  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f9fafb', padding: '2rem' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
          <div>
            <h1 style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '0.5rem', color: '#111827' }}>
              Portfolio Analysis Report
            </h1>
            <p style={{ color: '#6b7280' }}>
              Welcome, {user?.username}!
            </p>
          </div>
          <div style={{ display: 'flex', gap: '1rem' }}>
            <button 
              onClick={() => setPortfolio(null)}
              style={{ 
                width: 'auto',
                padding: '0.5rem 1.5rem',
                backgroundColor: '#667eea'
              }}
            >
              Edit Portfolio
            </button>
            <button 
              onClick={handleLogout}
              style={{ 
                width: 'auto',
                padding: '0.5rem 1.5rem',
                backgroundColor: '#ef4444'
              }}
            >
              Logout
            </button>
          </div>
        </div>

        {/* Warning Banner for NaN Values */}
        {hasNaNValues && (
          <div style={{
            backgroundColor: '#fef3c7',
            border: '2px solid #fbbf24',
            color: '#92400e',
            padding: '1rem',
            borderRadius: '0.5rem',
            marginBottom: '2rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem'
          }}>
            <AlertIcon />
            <div>
              <strong style={{ display: 'block', marginBottom: '0.25rem' }}>⚠️ Limited Data Available</strong>
              <span style={{ fontSize: '0.875rem' }}>
                Real-time market data unavailable (rate limited). Showing score based on portfolio structure. 
                Some metrics display as "N/A".
              </span>
            </div>
          </div>
        )}

        {/* Big Score Banner */}
        <div style={{
          backgroundColor: getScoreColor(analysis.score),
          color: 'white',
          padding: '3rem 2rem',
          borderRadius: '1rem',
          marginBottom: '2rem',
          textAlign: 'center',
          boxShadow: '0 10px 40px rgba(0,0,0,0.1)'
        }}>
          <div style={{ fontSize: '1rem', opacity: 0.9, marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '2px' }}>
            Final Goodness Score
          </div>
          <div style={{ fontSize: '5rem', fontWeight: 'bold', marginBottom: '0.5rem', lineHeight: 1 }}>
            {analysis.score}<span style={{ fontSize: '2rem', opacity: 0.8 }}>/100</span>
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
            {analysis.verdict}
          </div>
        </div>

        {/* Your Portfolio & Score Breakdown */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
          
          {/* Your Portfolio Card */}
          <div style={{ backgroundColor: 'white', padding: '2rem', borderRadius: '1rem', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
            <h3 style={{ fontWeight: 'bold', marginBottom: '1.5rem', fontSize: '1.25rem', color: '#111827' }}>
              📊 Your Portfolio
            </h3>
            <div style={{ marginBottom: '1.5rem' }}>
              {Object.entries(portfolio.allocations).map(([ticker, percentage]) => (
                <div key={ticker} style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  padding: '0.75rem 0',
                  borderBottom: '1px solid #e5e7eb'
                }}>
                  <span style={{ fontWeight: '600', color: '#374151' }}>{ticker}</span>
                  <span style={{ color: '#667eea', fontWeight: 'bold', fontSize: '1.125rem' }}>{percentage}%</span>
                </div>
              ))}
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                padding: '0.75rem 0',
                marginTop: '0.5rem',
                fontWeight: 'bold',
                fontSize: '1.125rem'
              }}>
                <span>Total</span>
                <span style={{ color: '#10b981' }}>100%</span>
              </div>
            </div>
          </div>

          {/* Score Breakdown Card */}
          <div style={{ backgroundColor: 'white', padding: '2rem', borderRadius: '1rem', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
            <h3 style={{ fontWeight: 'bold', marginBottom: '1.5rem', fontSize: '1.25rem', color: '#111827' }}>
              🎯 Score Breakdown
            </h3>
            <div style={{ marginBottom: '1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', alignItems: 'center' }}>
                <span style={{ color: '#6b7280' }}>Return Performance</span>
                <span style={{ fontWeight: 'bold', fontSize: '1.25rem', color: '#667eea' }}>
                  {analysis.breakdown.return_pts}/30 pts
                </span>
              </div>
              <div style={{ width: '100%', height: '8px', backgroundColor: '#e5e7eb', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ width: `${(analysis.breakdown.return_pts / 30) * 100}%`, height: '100%', backgroundColor: '#667eea', transition: 'width 0.5s' }} />
              </div>
            </div>

            <div style={{ marginBottom: '1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', alignItems: 'center' }}>
                <span style={{ color: '#6b7280' }}>Strategy Skill (Alpha)</span>
                <span style={{ fontWeight: 'bold', fontSize: '1.25rem', color: '#667eea' }}>
                  {analysis.breakdown.alpha_pts}/40 pts
                </span>
              </div>
              <div style={{ width: '100%', height: '8px', backgroundColor: '#e5e7eb', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ width: `${(analysis.breakdown.alpha_pts / 40) * 100}%`, height: '100%', backgroundColor: '#667eea', transition: 'width 0.5s' }} />
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', alignItems: 'center' }}>
                <span style={{ color: '#6b7280' }}>Risk Profile (Sharpe)</span>
                <span style={{ fontWeight: 'bold', fontSize: '1.25rem', color: '#667eea' }}>
                  {analysis.breakdown.sharpe_pts}/30 pts
                </span>
              </div>
              <div style={{ width: '100%', height: '8px', backgroundColor: '#e5e7eb', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ width: `${(analysis.breakdown.sharpe_pts / 30) * 100}%`, height: '100%', backgroundColor: '#667eea', transition: 'width 0.5s' }} />
              </div>
            </div>
          </div>
        </div>

        {/* Detailed Analysis Sections */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '2rem', marginBottom: '2rem' }}>
          
          {/* 1. Return vs Benchmarks */}
          <div style={{ backgroundColor: 'white', padding: '2rem', borderRadius: '1rem', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
              <div style={{ 
                width: '40px', 
                height: '40px', 
                backgroundColor: '#eff6ff', 
                borderRadius: '0.5rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#667eea'
              }}>
                <TrendingUpIcon />
              </div>
              <h3 style={{ fontWeight: 'bold', fontSize: '1.25rem', color: '#111827', margin: 0 }}>
                1. RETURN vs BENCHMARKS
              </h3>
              <div style={{ 
                marginLeft: 'auto',
                backgroundColor: '#eff6ff',
                color: '#667eea',
                padding: '0.25rem 0.75rem',
                borderRadius: '9999px',
                fontWeight: 'bold',
                fontSize: '0.875rem'
              }}>
                Score: {analysis.breakdown.return_pts}/30
              </div>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1rem' }}>
              <div style={{ textAlign: 'center', padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '0.5rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#6b7280', marginBottom: '0.5rem' }}>Your Return</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#111827' }}>
                  {displayValue(analysis.metrics.return, '%')}
                </div>
              </div>
              
              <div style={{ textAlign: 'center', padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '0.5rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#6b7280', marginBottom: '0.5rem' }}>S&P 500</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#6b7280' }}>
                  {displayValue(analysis.metrics.market_return, '%')}
                </div>
              </div>
              
              <div style={{ textAlign: 'center', padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '0.5rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#6b7280', marginBottom: '0.5rem' }}>Inflation</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#6b7280' }}>
                  {displayValue(analysis.metrics.inflation_rate, '%')}
                </div>
              </div>
            </div>
            
            {analysis.metrics.beats_inflation ? (
              <div style={{ padding: '0.75rem', backgroundColor: '#d1fae5', borderRadius: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <div style={{ color: '#065f46' }}><CheckIcon /></div>
                <span style={{ fontSize: '0.875rem', color: '#065f46', fontWeight: '600' }}>
                  Beat inflation by {displayValue(!isNaN(analysis.metrics.return) && !isNaN(analysis.metrics.inflation_rate) ? 
                    analysis.metrics.return - analysis.metrics.inflation_rate : 
                    NaN, '%')}
                </span>
              </div>
            ) : (
              <div style={{ padding: '0.75rem', backgroundColor: isNaN(analysis.metrics.return) ? '#fef3c7' : '#fee2e2', borderRadius: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <div style={{ color: isNaN(analysis.metrics.return) ? '#92400e' : '#991b1b' }}><AlertIcon /></div>
                <span style={{ fontSize: '0.875rem', color: isNaN(analysis.metrics.return) ? '#92400e' : '#991b1b', fontWeight: '600' }}>
                  {isNaN(analysis.metrics.return) ? 'Data unavailable' : 'Lost purchasing power'}
                </span>
              </div>
            )}
            
            <p style={{ fontSize: '0.875rem', color: '#6b7280', lineHeight: 1.6, padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '0.5rem' }}>
              {!isNaN(analysis.metrics.return) && analysis.metrics.beats_inflation ? (
                <>
                  <strong style={{ color: '#065f46' }}>[PASS]</strong> Your portfolio return of <strong>{displayValue(analysis.metrics.return, '%')}</strong> beats the inflation target of {displayValue(analysis.metrics.inflation_rate, '%')} by <strong>{displayValue(analysis.metrics.return - analysis.metrics.inflation_rate, '%')}</strong>. You are preserving purchasing power.
                </>
              ) : !isNaN(analysis.metrics.return) ? (
                <>
                  <strong style={{ color: '#991b1b' }}>[FAIL]</strong> Your portfolio return of <strong>{displayValue(analysis.metrics.return, '%')}</strong> is below the inflation target. You lost purchasing power.
                </>
              ) : (
                <>Portfolio performance data is currently unavailable. Showing cached score of <strong>{analysis.score}/100</strong>.</>
              )}
            </p>
          </div>

          {/* 2. Strategy Skill */}
          <div style={{ backgroundColor: 'white', padding: '2rem', borderRadius: '1rem', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
              <div style={{ 
                width: '40px', 
                height: '40px', 
                backgroundColor: '#eff6ff', 
                borderRadius: '0.5rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1.5rem'
              }}>
                🎯
              </div>
              <h3 style={{ fontWeight: 'bold', fontSize: '1.25rem', color: '#111827', margin: 0 }}>
                2. STRATEGY SKILL
              </h3>
              <div style={{ 
                marginLeft: 'auto',
                backgroundColor: '#eff6ff',
                color: '#667eea',
                padding: '0.25rem 0.75rem',
                borderRadius: '9999px',
                fontWeight: 'bold',
                fontSize: '0.875rem'
              }}>
                Score: {analysis.breakdown.alpha_pts}/40
              </div>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div style={{ padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '0.5rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#6b7280', marginBottom: '0.5rem' }}>Beta (Risk Level)</div>
                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: !isNaN(analysis.metrics.beta) && analysis.metrics.beta > 1 ? '#ef4444' : '#10b981' }}>
                  {displayValue(analysis.metrics.beta)}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.5rem' }}>
                  {!isNaN(analysis.metrics.beta) && analysis.metrics.beta > 1 ? 'Higher risk' : !isNaN(analysis.metrics.beta) ? 'Lower risk' : 'Calculating...'}
                </div>
              </div>
              
              <div style={{ padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '0.5rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#6b7280', marginBottom: '0.5rem' }}>Alpha (Skill)</div>
                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: !isNaN(analysis.metrics.alpha) && analysis.metrics.alpha > 0 ? '#10b981' : '#ef4444' }}>
                  {displayValue(analysis.metrics.alpha, '%')}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.5rem' }}>
                  {!isNaN(analysis.metrics.alpha) && analysis.metrics.alpha > 0 ? 'Outperforming' : !isNaN(analysis.metrics.alpha) ? 'Underperforming' : 'Calculating...'}
                </div>
              </div>
            </div>
            
            <p style={{ fontSize: '0.875rem', color: '#6b7280', lineHeight: 1.6, padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '0.5rem', marginBottom: '1rem' }}>
              {!isNaN(analysis.metrics.beta) ? (
                <>
                  <strong>Beta {displayValue(analysis.metrics.beta)}</strong> - Your portfolio is <strong>{analysis.metrics.beta > 1 ? 'RISKIER' : 'SAFER'}</strong> than the market.
                </>
              ) : 'Beta calculation requires market data.'}
            </p>
            
            <p style={{ fontSize: '0.875rem', color: '#6b7280', lineHeight: 1.6, padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '0.5rem' }}>
              {!isNaN(analysis.metrics.alpha) ? (
                <>
                  <strong>Alpha {displayValue(analysis.metrics.alpha, '%')}</strong> (Excess Return) - A positive alpha indicates outperformance of the benchmark index, S&P 500. A negative alpha indicates underperformance of the benchmark index.
                  <br /><br />
                  {analysis.metrics.alpha > 0 ? (
                    <><strong style={{ color: '#065f46' }}>[SUCCESS]</strong> You are beating the market based on risk taken.</>
                  ) : (
                    <><strong style={{ color: '#d97706' }}>[WARNING]</strong> You are underperforming relative to the risk taken.</>
                  )}
                </>
              ) : 'Alpha calculation requires market data.'}
            </p>
          </div>

          {/* 3. Risk Profile */}
          <div style={{ backgroundColor: 'white', padding: '2rem', borderRadius: '1rem', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
              <div style={{ 
                width: '40px', 
                height: '40px', 
                backgroundColor: '#eff6ff', 
                borderRadius: '0.5rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1.5rem'
              }}>
                ⚖️
              </div>
              <h3 style={{ fontWeight: 'bold', fontSize: '1.25rem', color: '#111827', margin: 0 }}>
                3. RISK PROFILE
              </h3>
              <div style={{ 
                marginLeft: 'auto',
                backgroundColor: '#eff6ff',
                color: '#667eea',
                padding: '0.25rem 0.75rem',
                borderRadius: '9999px',
                fontWeight: 'bold',
                fontSize: '0.875rem'
              }}>
                Score: {analysis.breakdown.sharpe_pts}/30
              </div>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div style={{ padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '0.5rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#6b7280', marginBottom: '0.5rem' }}>Volatility</div>
                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#111827' }}>
                  {displayValue(analysis.metrics.volatility, '%')}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.5rem' }}>
                  Lower is safer
                </div>
              </div>
              
              <div style={{ padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '0.5rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#6b7280', marginBottom: '0.5rem' }}>Sharpe Ratio</div>
                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: !isNaN(analysis.metrics.sharpe) && analysis.metrics.sharpe > 1 ? '#10b981' : '#f59e0b' }}>
                  {displayValue(analysis.metrics.sharpe)}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.5rem' }}>
                  {!isNaN(analysis.metrics.sharpe) && analysis.metrics.sharpe > 1 ? 'Good' : !isNaN(analysis.metrics.sharpe) ? 'Needs work' : 'Calculating...'}
                </div>
              </div>
            </div>
            
            <p style={{ fontSize: '0.875rem', color: '#6b7280', lineHeight: 1.6, padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '0.5rem', marginBottom: '1rem' }}>
              {!isNaN(analysis.metrics.volatility) ? (
                <>
                  <strong>Volatility {displayValue(analysis.metrics.volatility, '%')}</strong> (Lower is safer) - This means the price could potentially drop or rise by {displayValue(analysis.metrics.volatility, '%')} in one go.
                </>
              ) : 'Volatility calculation requires market data.'}
            </p>
            
            <p style={{ fontSize: '0.875rem', color: '#6b7280', lineHeight: 1.6, padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '0.5rem' }}>
              {!isNaN(analysis.metrics.sharpe) ? (
                <>
                  <strong>Sharpe Ratio {displayValue(analysis.metrics.sharpe)}</strong> (measures return per unit of risk) - 
                  {analysis.metrics.sharpe > 1 ? (
                    <> &gt;1.0 is <strong style={{ color: '#065f46' }}>good</strong>: you get smooth and steady returns given the risk taken.</>
                  ) : (
                    <> &lt;1.0 is <strong style={{ color: '#d97706' }}>not ideal</strong>: you get volatile returns given the risk taken.</>
                  )}
                </>
              ) : 'Sharpe ratio calculation requires market data.'}
            </p>
          </div>
        </div>

        {/* Footer */}
        <div style={{ marginTop: '2rem', textAlign: 'center' }}>
          <p style={{ fontSize: '0.875rem', color: '#6b7280', marginBottom: '1rem' }}>
            💡 Want to improve your score? Try adjusting your allocations or selecting different stocks.
          </p>
        </div>
      </div>
    </div>
  )
}

export default App