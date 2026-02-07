import React, { useState } from 'react'
import { Lock, User, TrendingUp, AlertCircle } from 'lucide-react'
import { login } from '../api/auth'

export default function Login({ onLoginSuccess }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)

  const handleLogin = async () => {
    // Reset states
    setError('')
    setLoading(true)

    // Basic validation
    if (!username || !password) {
      setError('Please enter both username and password')
      setLoading(false)
      return
    }

    try {
      // Call API
      const data = await login(username, password)
      
      // Success
      setSuccess(true)
      console.log('Login successful:', data)
      
      // Notify parent component
      setTimeout(() => {
        onLoginSuccess(data)
      }, 1000)
      
    } catch (err) {
      // Handle errors
      setError(err.message || 'Login failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !loading) {
      handleLogin()
    }
  }

  return (
    <div className="min-h-screen bg-gradient flex items-center justify-center" style={{ padding: '1rem' }}>
      <div className="w-full max-w-md">
        {/* Logo/Header */}
        <div className="text-center mb-8">
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '4rem',
            height: '4rem',
            backgroundColor: 'white',
            borderRadius: '50%',
            marginBottom: '1rem'
          }}>
            <TrendingUp style={{ width: '2rem', height: '2rem', color: '#667eea' }} />
          </div>
          <h1 className="text-white" style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
            Stock Portfolio Analyzer
          </h1>
          <p className="text-white" style={{ opacity: 0.9 }}>
            Sign in to optimize your investments
          </p>
        </div>

        {/* Login Card */}
        <div className="bg-white rounded-2xl shadow-2xl p-8">
          <div style={{ marginBottom: '1.5rem' }}>
            {/* Username Field */}
            <div className="mb-6">
              <label htmlFor="username" className="block text-sm font-medium text-gray-700 mb-2">
                Username
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <User style={{ height: '1.25rem', width: '1.25rem', color: '#9ca3af' }} />
                </div>
                <input
                  id="username"
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Enter your username"
                  disabled={loading}
                />
              </div>
            </div>

            {/* Password Field */}
            <div className="mb-6">
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-2">
                Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock style={{ height: '1.25rem', width: '1.25rem', color: '#9ca3af' }} />
                </div>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Enter your password"
                  disabled={loading}
                />
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="alert alert-error" style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                <AlertCircle style={{ width: '1.25rem', height: '1.25rem', flexShrink: 0, marginTop: '0.125rem' }} />
                <p style={{ fontSize: '0.875rem' }}>{error}</p>
              </div>
            )}

            {/* Success Message */}
            {success && (
              <div className="alert alert-success">
                <p style={{ fontSize: '0.875rem', fontWeight: 600 }}>
                  ✓ Login successful! Redirecting...
                </p>
              </div>
            )}

            {/* Submit Button */}
            <button
              onClick={handleLogin}
              disabled={loading}
            >
              {loading ? (
                <>
                  <div className="spinner" />
                  Signing in...
                </>
              ) : (
                'Sign In'
              )}
            </button>

            {/* Additional Links */}
            <div className="text-center" style={{ marginTop: '1.5rem' }}>
              <a href="#" style={{ fontSize: '0.875rem', display: 'block', marginBottom: '0.5rem' }}>
                Forgot password?
              </a>
              <p style={{ fontSize: '0.875rem', color: '#6b7280' }}>
                Don't have an account?{' '}
                <a href="#" style={{ fontWeight: 600 }}>
                  Sign up
                </a>
              </p>
            </div>
          </div>
        </div>

        {/* Demo Info */}
        <div style={{ 
          marginTop: '1.5rem', 
          padding: '1rem', 
          backgroundColor: 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(10px)',
          borderRadius: '0.5rem',
          border: '1px solid rgba(255, 255, 255, 0.2)'
        }}>
          <p style={{ fontSize: '0.75rem', color: 'white', fontWeight: 600, marginBottom: '0.5rem' }}>
            🔧 Demo Credentials
          </p>
          <p style={{ fontSize: '0.75rem', color: 'white', opacity: 0.9 }}>
            Username: <code style={{ backgroundColor: 'rgba(255, 255, 255, 0.2)', padding: '0.125rem 0.25rem', borderRadius: '0.25rem' }}>demo</code> | 
            Password: <code style={{ backgroundColor: 'rgba(255, 255, 255, 0.2)', padding: '0.125rem 0.25rem', borderRadius: '0.25rem' }}>password</code>
          </p>
        </div>
      </div>
    </div>
  )
}