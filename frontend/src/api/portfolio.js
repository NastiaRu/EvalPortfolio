/**
 * Portfolio API functions
 * Communicates with Flask backend at /api/portfolio/*
 */

import api from './auth'

/**
 * Submit portfolio allocation
 * @param {Object} portfolio - { "AAPL": 40, "MSFT": 30, "GOOGL": 20, "AMZN": 10 }
 * @returns {Promise<Object>} Response data
 */
export const submitPortfolio = async (portfolio) => {
  try {
    console.log('Submitting portfolio to API:', portfolio) /* Clicked on the Submit Portfolio Button - POST Request is made */

    const response = await api.post('/portfolio/submit', { /* Waits for response - async method */
      allocations: portfolio
    })

    console.log('Portfolio submission response:', response.data)

    return response.data
  } catch (error) {
    console.error('Portfolio submission error:', error) /* Issue here */

    if (error.code === 'ECONNABORTED') {
      throw new Error('Request timeout. Please try again.')
    }

    if (error.code === 'ERR_NETWORK') {
      throw new Error('Cannot connect to server. Is the backend running?')
    }

    if (error.response) {
      const message = error.response.data?.message || 'Failed to submit portfolio'
      throw new Error(message)
    }

    throw new Error('An unexpected error occurred')
  }
}

/**
 * Get user's portfolio
 * @returns {Promise<Object>} Portfolio data
 */
export const getPortfolio = async () => {
  try {
    const response = await api.get('/portfolio')
    return response.data
  } catch (error) {
    console.error('Get portfolio error:', error)
    throw new Error('Failed to fetch portfolio')
  }
}

/**
 * Analyze portfolio
 * @param {Object} portfolio - Portfolio allocations
 * @returns {Promise<Object>} Analysis results
 */
export const analyzePortfolio = async (portfolio) => {
  try {
    const response = await api.post('/analyze', {
      stocks: Object.keys(portfolio),
      allocations: Object.values(portfolio)
    })
    return response.data
  } catch (error) {
    console.error('Analyze portfolio error:', error)
    throw new Error('Failed to analyze portfolio')
  }
}

export default {
  submitPortfolio,
  getPortfolio,
  analyzePortfolio
}