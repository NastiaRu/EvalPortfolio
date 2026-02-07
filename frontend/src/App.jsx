import React, { useState } from 'react'
import Login from './pages/Login'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [user, setUser] = useState(null)

  const handleLoginSuccess = (userData) => {
    setIsAuthenticated(true)
    setUser(userData)
    // Store token in sessionStorage
    sessionStorage.setItem('authToken', userData.token)
    sessionStorage.setItem('username', userData.username)
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
          🚧 Coming Soon
        </h2>
        <p style={{ color: '#6b7280', marginBottom: '2rem' }}>
          You'll build the following pages manually:
        </p>
        <ul style={{ 
          textAlign: 'left', 
          maxWidth: '600px', 
          margin: '0 auto',
          listStyle: 'none',
          padding: 0
        }}>
          <li style={{ padding: '0.75rem', backgroundColor: 'white', marginBottom: '0.5rem', borderRadius: '0.5rem' }}>
            ✅ Login Page (Complete!)
          </li>
          <li style={{ padding: '0.75rem', backgroundColor: 'white', marginBottom: '0.5rem', borderRadius: '0.5rem' }}>
            📝 Stock Selection Page (Choose 5 stocks, allocate percentages)
          </li>
          <li style={{ padding: '0.75rem', backgroundColor: 'white', marginBottom: '0.5rem', borderRadius: '0.5rem' }}>
            📊 Analysis Dashboard (Show results, comparisons, scores)
          </li>
          <li style={{ padding: '0.75rem', backgroundColor: 'white', marginBottom: '0.5rem', borderRadius: '0.5rem' }}>
            🤖 ML Suggestions Page (Display optimized allocations)
          </li>
        </ul>
      </div>
    </div>
  )
}

export default App