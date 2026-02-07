import React, { useState, useEffect } from 'react'
import { submitPortfolio } from '../api/portfolio'

// Simple SVG icons
const TrendingUpIcon = () => (
  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>
    <polyline points="17 6 23 6 23 12"/>
  </svg>
)

const AlertIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"/>
    <line x1="12" y1="8" x2="12" y2="12"/>
    <line x1="12" y1="16" x2="12.01" y2="16"/>
  </svg>
)

// Available stocks (8 options)
const AVAILABLE_STOCKS = [
  { ticker: 'AAPL', name: 'Apple Inc.' },
  { ticker: 'MSFT', name: 'Microsoft Corporation' },
  { ticker: 'GOOGL', name: 'Alphabet Inc.' },
  { ticker: 'AMZN', name: 'Amazon.com Inc.' },
  { ticker: 'NVDA', name: 'NVIDIA Corporation' },
  { ticker: 'TSLA', name: 'Tesla Inc.' },
  { ticker: 'META', name: 'Meta Platforms Inc.' },
  { ticker: 'JPM', name: 'JPMorgan Chase & Co.' },
]

export default function StockSelection({ onSubmitSuccess }) {
  const [selectedStocks, setSelectedStocks] = useState([])
  const [allocations, setAllocations] = useState({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)

  // Calculate total percentage
  const totalPercentage = Object.values(allocations).reduce((sum, val) => sum + (parseFloat(val) || 0), 0)

  const handleStockToggle = (ticker) => {
    if (selectedStocks.includes(ticker)) {
      // Remove stock
      setSelectedStocks(selectedStocks.filter(t => t !== ticker))
      const newAllocations = { ...allocations }
      delete newAllocations[ticker]
      setAllocations(newAllocations)
    } else {
      // Add stock (max 4)
      if (selectedStocks.length < 4) {
        setSelectedStocks([...selectedStocks, ticker])
        setAllocations({ ...allocations, [ticker]: '' })
      }
    }
    setError('')
  }

  const handleAllocationChange = (ticker, value) => {
    // Only allow numbers and decimals
    if (value === '' || /^\d*\.?\d*$/.test(value)) {
      setAllocations({ ...allocations, [ticker]: value })
      setError('')
    }
  }

  const handleSubmit = async () => {
    setError('')
    setLoading(true)

    // Validation
    if (selectedStocks.length === 0) {
      setError('Please select at least one stock')
      setLoading(false)
      return
    }

    // Check all allocations are filled
    const hasEmptyAllocations = selectedStocks.some(ticker => !allocations[ticker] || allocations[ticker] === '')
    if (hasEmptyAllocations) {
      setError('Please enter allocation percentages for all selected stocks')
      setLoading(false)
      return
    }

    // Check sum is 100%
    if (Math.abs(totalPercentage - 100) > 0.01) {
      setError(`Allocations must sum to 100%. Current total: ${totalPercentage.toFixed(2)}%`)
      setLoading(false)
      return
    }

    try {
      // Build portfolio data
      const portfolio = {}
      selectedStocks.forEach(ticker => {
        portfolio[ticker] = parseFloat(allocations[ticker])
      })

      console.log('Submitting portfolio:', portfolio)

      // Call API
      const response = await submitPortfolio(portfolio)
      
      // Success
      setSuccess(true)
      console.log('Portfolio submitted successfully:', response)

      // Notify parent after short delay
      setTimeout(() => {
        if (onSubmitSuccess) {
          onSubmitSuccess(portfolio)
        }
      }, 1500)

    } catch (err) {
      setError(err.message || 'Failed to submit portfolio. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient flex items-center justify-center" style={{ padding: '2rem' }}>
      <div style={{ width: '100%', maxWidth: '800px' }}>
        {/* Header */}
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
            <div style={{ color: '#667eea' }}>
              <TrendingUpIcon />
            </div>
          </div>
          <h1 className="text-white" style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
            Build Your Portfolio
          </h1>
          <p className="text-white" style={{ opacity: 0.9 }}>
            Select up to 4 stocks and allocate your investment percentages
          </p>
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-2xl shadow-2xl p-8">
          {/* Instructions */}
          <div style={{ 
            backgroundColor: '#f3f4f6', 
            padding: '1rem', 
            borderRadius: '0.5rem',
            marginBottom: '1.5rem'
          }}>
            <p style={{ fontSize: '0.875rem', color: '#374151', marginBottom: '0.5rem' }}>
              <strong>Instructions:</strong>
            </p>
            <ul style={{ fontSize: '0.875rem', color: '#6b7280', paddingLeft: '1.5rem', margin: 0 }}>
              <li>Select up to 4 stocks by clicking on them</li>
              <li>Enter percentage allocation for each selected stock</li>
              <li>Total allocation must equal 100%</li>
            </ul>
          </div>

          {/* Stock Selection Table */}
          <div style={{ marginBottom: '1.5rem' }}>
            <h3 style={{ fontSize: '1.125rem', fontWeight: 'bold', marginBottom: '1rem', color: '#374151' }}>
              Select Stocks & Allocate Percentages
            </h3>

            <div style={{ 
              border: '2px solid #e5e7eb', 
              borderRadius: '0.5rem',
              overflow: 'hidden'
            }}>
              {/* Table Header */}
              <div style={{ 
                display: 'grid',
                gridTemplateColumns: '1fr 200px',
                backgroundColor: '#f9fafb',
                padding: '1rem',
                borderBottom: '2px solid #e5e7eb',
                fontWeight: 'bold',
                fontSize: '0.875rem',
                color: '#374151'
              }}>
                <div>Stock</div>
                <div style={{ textAlign: 'center' }}>Allocation (%)</div>
              </div>

              {/* Table Rows */}
              {AVAILABLE_STOCKS.map((stock) => {
                const isSelected = selectedStocks.includes(stock.ticker)
                
                return (
                  <div
                    key={stock.ticker}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 200px',
                      padding: '1rem',
                      borderBottom: '1px solid #e5e7eb',
                      backgroundColor: isSelected ? '#eff6ff' : 'white',
                      cursor: 'pointer',
                      transition: 'background-color 0.2s'
                    }}
                    onMouseEnter={(e) => {
                      if (!isSelected) e.currentTarget.style.backgroundColor = '#f9fafb'
                    }}
                    onMouseLeave={(e) => {
                      if (!isSelected) e.currentTarget.style.backgroundColor = 'white'
                    }}
                  >
                    {/* Stock Name Column */}
                    <div 
                      onClick={() => handleStockToggle(stock.ticker)}
                      style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}
                    >
                      <div style={{
                        width: '20px',
                        height: '20px',
                        border: '2px solid',
                        borderColor: isSelected ? '#667eea' : '#d1d5db',
                        borderRadius: '4px',
                        backgroundColor: isSelected ? '#667eea' : 'white',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexShrink: 0
                      }}>
                        {isSelected && (
                          <svg width="12" height="12" viewBox="0 0 12 12" fill="white">
                            <polyline points="2,6 5,9 10,3" stroke="white" strokeWidth="2" fill="none"/>
                          </svg>
                        )}
                      </div>
                      <div>
                        <div style={{ fontWeight: 'bold', color: '#374151' }}>{stock.ticker}</div>
                        <div style={{ fontSize: '0.875rem', color: '#6b7280' }}>{stock.name}</div>
                      </div>
                    </div>

                    {/* Allocation Input Column */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      {isSelected ? (
                        <div style={{ position: 'relative', width: '120px' }}>
                          <input
                            type="text"
                            value={allocations[stock.ticker] || ''}
                            onChange={(e) => handleAllocationChange(stock.ticker, e.target.value)}
                            placeholder="0.00"
                            disabled={loading}
                            style={{
                              width: '100%',
                              padding: '0.5rem',
                              paddingRight: '2rem',
                              border: '1px solid #d1d5db',
                              borderRadius: '0.375rem',
                              textAlign: 'right',
                              fontSize: '0.875rem'
                            }}
                          />
                          <span style={{
                            position: 'absolute',
                            right: '0.5rem',
                            top: '50%',
                            transform: 'translateY(-50%)',
                            color: '#6b7280',
                            fontSize: '0.875rem'
                          }}>
                            %
                          </span>
                        </div>
                      ) : (
                        <span style={{ color: '#9ca3af', fontSize: '0.875rem' }}>—</span>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Total Percentage Display */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '1rem',
            backgroundColor: totalPercentage === 100 ? '#f0fdf4' : '#fef2f2',
            borderRadius: '0.5rem',
            marginBottom: '1.5rem',
            border: '2px solid',
            borderColor: totalPercentage === 100 ? '#bbf7d0' : '#fecaca'
          }}>
            <span style={{ fontWeight: 'bold', color: '#374151' }}>Total Allocation:</span>
            <span style={{
              fontSize: '1.5rem',
              fontWeight: 'bold',
              color: totalPercentage === 100 ? '#166534' : '#991b1b'
            }}>
              {totalPercentage.toFixed(2)}%
            </span>
          </div>

          {/* Selected Count */}
          <div style={{ 
            fontSize: '0.875rem', 
            color: '#6b7280', 
            marginBottom: '1rem',
            textAlign: 'center'
          }}>
            {selectedStocks.length} of 4 stocks selected
          </div>

          {/* Error Message */}
          {error && (
            <div className="alert alert-error" style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', marginBottom: '1rem' }}>
              <div style={{ flexShrink: 0, marginTop: '0.125rem', color: '#991b1b' }}>
                <AlertIcon />
              </div>
              <p style={{ fontSize: '0.875rem' }}>{error}</p>
            </div>
          )}

          {/* Success Message */}
          {success && (
            <div className="alert alert-success" style={{ marginBottom: '1rem' }}>
              <p style={{ fontSize: '0.875rem', fontWeight: 600 }}>
                ✓ Portfolio submitted successfully! Redirecting...
              </p>
            </div>
          )}

          {/* Submit Button */}
          <button
            onClick={handleSubmit}
            disabled={loading || selectedStocks.length === 0}
            style={{
              width: '100%',
              padding: '0.75rem',
              backgroundColor: selectedStocks.length === 0 ? '#9ca3af' : '#667eea',
              opacity: loading ? 0.7 : 1
            }}
          >
            {loading ? (
              <>
                <div className="spinner" />
                Submitting...
              </>
            ) : (
              'Submit Portfolio'
            )}
          </button>
        </div>

        {/* Help Text */}
        <div style={{ 
          marginTop: '1.5rem', 
          padding: '1rem', 
          backgroundColor: 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(10px)',
          borderRadius: '0.5rem',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          textAlign: 'center'
        }}>
          <p style={{ fontSize: '0.75rem', color: 'white' }}>
            💡 Tip: Diversify your portfolio across different sectors for better risk management
          </p>
        </div>
      </div>
    </div>
  )
}