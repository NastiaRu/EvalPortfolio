/**
 * Authentication API functions
 * Communicates with Flask backend at /api/auth/*
 */

import axios from 'axios'

// Base API URL - uses Vite proxy in development
const API_BASE_URL = '/api'

// Create axios instance with defaults
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 100000, // 10 second timeout
})

// Request interceptor - add auth token to all requests
api.interceptors.request.use(
  (config) => {
    const token = sessionStorage.getItem('authToken')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor - handle common errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Unauthorized - clear session and redirect to login
      sessionStorage.removeItem('authToken')
      sessionStorage.removeItem('username')
      window.location.href = '/'
    }
    return Promise.reject(error)
  }
)

/**
 * Login user
 * @param {string} username
 * @param {string} password
 * @returns {Promise<Object>} User data with token
 */
export const login = async (username, password) => {
  try {
    const response = await api.post('/auth/login', {
      username,
      password,
    })

    console.log('Login API response:', response.data)

    // Return user data
    return {
      token: response.data.token,
      username: response.data.username,
      userId: response.data.userId,
    }
  } catch (error) {
    console.error('Login API error:', error)

    // Handle different error scenarios
    if (error.code === 'ECONNABORTED') {
      throw new Error('Request timeout. Please try again.')
    }

    if (error.code === 'ERR_NETWORK') {
      throw new Error('Cannot connect to server. Is the backend running?')
    }

    if (error.response) {
      // Server responded with error
      const message = error.response.data?.message || 'Invalid credentials'
      throw new Error(message)
    }

    // Unknown error
    throw new Error('An unexpected error occurred')
  }
}

/**
 * Logout user
 */
export const logout = () => {
  sessionStorage.removeItem('authToken')
  sessionStorage.removeItem('username')
}

/**
 * Check if user is authenticated
 * @returns {boolean}
 */
export const isAuthenticated = () => {
  return !!sessionStorage.getItem('authToken')
}

/**
 * Get current user info
 * @returns {Object|null}
 */
export const getCurrentUser = () => {
  const token = sessionStorage.getItem('authToken')
  const username = sessionStorage.getItem('username')

  if (!token || !username) return null

  return { token, username }
}

// Export axios instance for other API calls
export default api