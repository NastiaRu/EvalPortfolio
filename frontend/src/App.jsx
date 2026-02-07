import React, { useState } from 'react'
import Login from './pages/Login'
import StockSelection from './pages/StockSelection'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [user, setUser] = useState(null)
  const [portfolio, setPortfolio] = useState(null)

  const handleLoginSuccess = (userData) => {
    setIsAuthenticated(true)
    setUser(userData)
    // Store token in sessionStorage
    sessionStorage.setItem('authToken', userData.token)
    sessionStorage.setItem('username', userData.username)
  }

  const handlePortfolioSubmit = (portfolioData) => {
    setPortfolio(portfolioData)
    console.log('Portfolio Saved:', portfolioData)
  }

  const handleLogout = () => {
    setIsAuthenticated(false)
    setUser(null)
    sessionStorage.removeItem('authToken')
    sessionStorage.removeItem('username')
  }

  // If not authenticated, show login
  if (!isAuthenticated) {
    return <Login onLoginSuccess={handleLoginSuccess} />
  }

  // show stock selection if no portfolio yet
  if (!portfolio) {
    return <StockSelection onSubmitSuccess={handlePortfolioSubmit}/>
  }

  // After login, show placeholder for other pages
  return (
    <div style={{ padding: '2rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <div>
          <h1 style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
            Stock Portfolio Analyzer
          </h1>
          <p style={{ color: '#6b7280' }}>
            Welcome, {user?.username}!
          </p>
        </div>
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

      <div style={{ 
        backgroundColor: '#f3f4f6', 
        padding: '3rem', 
        borderRadius: '1rem',
        textAlign: 'center'
      }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem' }}>
          ✅ Portfolio Submitted!
        </h2>
        <p style={{ color: '#6b7280', marginBottom: '2rem' }}>
          Porfolio successfully saved!
        </p>

        {/* Show Portfolio*/}

        <div style={{ 
          backgroundColor: 'white', 
          padding: '2rem', 
          borderRadius: '0.5rem',
          maxWidth: '600px',
          margin: '0 auto',
          textAlign: 'left'
        }}>
          <h3 style={{ fontWeight: 'bold', marginBottom: '1rem' }}>Your Allocations:</h3>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {Object.entries(portfolio).map(([ticker, percentage]) => (
              <li key={ticker} style={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                padding: '0.75rem',
                borderBottom: '1px solid #e5e7eb'
              }}>
                <span style={{ fontWeight: 'bold' }}>{ticker}</span>
                <span style={{ color: '#667eea', fontWeight: 'bold' }}>{percentage}%</span>
              </li>
            ))}
          </ul>
        </div>
        <p style={{ color: '#6b7280', marginTop: '2rem', fontSize: '0.875rem' }}>
          📊 Next: Build the Analysis Dashboard to show performance metrics
        </p>

        <button
          onClick={() => setPortfolio(null)}
          style={{
            marginTop: '1.5rem',
            width: 'auto',
            padding: '0.5rem 1.5rem'
          }}>
          Edit Portfolio
        </button>
        
      </div>
    </div>
  )
}

export default App