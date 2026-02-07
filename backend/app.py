"""
Flask Backend API
Main application with authentication
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from datetime import datetime
import secrets

import numpy as np
from analysis import financial_calculator as fin_calc

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key")

# Mock user database (replace with real DB later)
MOCK_USERS = {
    "NastiaRu": {
        "id": 1,
        "username": "NastiaRu",
        "password": "password",  # In production: use bcrypt hashed passwords
        "email": "nastia@example.com",
    },
    "AndrewChen": {
        "id": 2,
        "username": "AndrewChen",
        "password": "password",
        "email": "andrew@example.com",
    },
    "LukeB": {
        "id": 3,
        "username": "LukeB",
        "password": "password",
        "email": "luke@example.com",
    },
}

# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Docker"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "backend",
            "version": "1.0.0",
        }
    ), 200


@app.route("/", methods=["GET"])
def root():
    """Root endpoint with API info"""
    return jsonify(
        {
            "message": "Stock Portfolio Analyzer API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "login": "/api/auth/login",
                "stocks": "/api/stocks",
                "analyze": "/api/analyze",
                "suggestions": "/api/suggestions",
                "portfolio": "/api/portfolio/submit",
            },
        }
    ), 200


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================


@app.route("/api/auth/login", methods=["POST"])
def login():
    """
    User login endpoint

    Request body:
    {
        "username": "demo",
        "password": "password"
    }

    Response:
    {
        "token": "generated_token",
        "username": "demo",
        "userId": 1,
        "message": "Login successful"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"message": "No data provided"}), 400

        username = data.get("username")
        password = data.get("password")

        # Validation
        if not username or not password:
            return jsonify({"message": "Username and password required"}), 400

        # Check if user exists
        user = MOCK_USERS.get(username)

        if not user:
            return jsonify({"message": "Invalid credentials"}), 401

        # Verify password (in production: use bcrypt.checkpw)
        if user["password"] != password:
            return jsonify({"message": "Invalid credentials"}), 401

        # Generate token (in production: use JWT)
        token = secrets.token_urlsafe(32)

        # Log successful login
        print(f"✅ Login successful: {username} at {datetime.now()}")

        # Return success response
        return jsonify(
            {
                "token": token,
                "username": user["username"],
                "userId": user["id"],
                "message": "Login successful",
            }
        ), 200

    except Exception as e:
        print(f"❌ Login error: {str(e)}")
        return jsonify({"message": "Internal server error"}), 500


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    """User logout endpoint"""
    # In production: invalidate token in database
    return jsonify({"message": "Logout successful"}), 200


# ============================================================================
# PORTFOLIO ENDPOINTS
# ============================================================================


@app.route("/api/portfolio/submit", methods=["POST"])
def submit_portfolio():
    """
    Submit portfolio allocation

    Request body:
    {
        "allocations": {
            "AAPL": 40,
            "MSFT": 30,
            "GOOGL": 20,
            "AMZN": 10
        }
    }
    """
    data = request.get_json()

    if not data or "allocations" not in data:
        return jsonify({"message": "Allocations data required"}), 400

    allocations = data["allocations"]

    # Validation
    if not allocations or not isinstance(allocations, dict):
        return jsonify({"message": "Invalid allocations format"}), 400

    # Check total is 100%
    total = sum(allocations.values())
    if abs(total - 100) > 0.01:
        return jsonify(
            {"message": f"Allocations must sum to 100%. Current: {total}%"}
        ), 400

    print(f"✅ Portfolio submitted: {allocations}")

    try:
        calculator_results = fin_calc.analyze_portfolio(data)
        
        print(f"Score: {calculator_results['score']}/100")
        
        # Return full results - ✅ jsonify handles the conversion
        return jsonify({
            'message': 'Portfolio analyzed successfully',
            'allocations': allocations,
            'total': total,
            'analysis': calculator_results,
        }), 200

        # Financial Calculator HERE
        # my_stocks = list(allocations.keys())
        # my_weights = np.array([allocations[ticker] / 100 for ticker in my_stocks])

        # # 1. Get Data
        # market_data = fin_calc.fetch_data(my_stocks)

        # # 2. Run Financial Math
        # metrics = fin_calc.calculate_metrics(market_data, my_stocks, my_weights)

        # # 3. Grade the Portfolio
        # score, breakdown = fin_calc.calculate_goodness_score(metrics)

        # # 4. Build response - ✅ Explicitly convert all NumPy types to Python types
        # response = {
        #     "score": int(score),  # ✅ Ensure it's Python int
        #     "verdict": str(fin_calc.get_verdict(score)),  # ✅ Ensure it's Python str
        #     "stocks": list(my_stocks),  # ✅ Ensure it's Python list
        #     "weights": dict(allocations),  # ✅ Ensure it's Python dict
        #     "metrics": {
        #         "return": float(metrics["return"] * 100),  # ✅ Explicit float conversion
        #         "market_return": float(metrics["market_return"] * 100),
        #         "inflation_rate": float(fin_calc.INFLATION_RATE * 100),
        #         "beats_inflation": bool(metrics["return"] > fin_calc.INFLATION_RATE),  # ✅ Explicit bool
        #         "alpha": float(metrics["alpha"] * 100),
        #         "beta": float(metrics["beta"]),
        #         "volatility": float(metrics["volatility"] * 100),
        #         "sharpe": float(metrics["sharpe"]),
        #     },
        #     "breakdown": {
        #         "return_pts": int(breakdown["return_pts"]),  # ✅ Already converted
        #         "alpha_pts": int(breakdown["alpha_pts"]),
        #         "sharpe_pts": int(breakdown["sharpe_pts"]),
        #     },
        # }
        # return response
    except Exception as e:
        print(f"Portfolio submission error: {str(e)}")
        return jsonify({"message": "Internal server error"}), 500


@app.route("/api/portfolio", methods=["GET"])
def get_portfolio():
    """Get user's portfolio (placeholder)"""
    # TODO: Get from database
    return jsonify({"message": "No portfolios yet"}), 200


# ============================================================================
# STOCK ENDPOINTS (Placeholder - Andrew to implement)
# ============================================================================


@app.route("/api/stocks", methods=["GET"])
def get_stocks():
    """
    Get available stocks
    TODO: Andrew - Fetch from database
    """
    # Mock data for now
    stocks = [
        {"id": 1, "ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
        {
            "id": 2,
            "ticker": "MSFT",
            "name": "Microsoft Corporation",
            "sector": "Technology",
        },
        {"id": 3, "ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
        {
            "id": 4,
            "ticker": "AMZN",
            "name": "Amazon.com Inc.",
            "sector": "Consumer Cyclical",
        },
        {
            "id": 5,
            "ticker": "NVDA",
            "name": "NVIDIA Corporation",
            "sector": "Technology",
        },
    ]
    return jsonify(stocks), 200


@app.route("/api/stocks/<ticker>", methods=["GET"])
def get_stock(ticker):
    """
    Get stock details
    TODO: Andrew - Implement with real data
    """
    return jsonify(
        {
            "ticker": ticker,
            "name": f"{ticker} Company",
            "current_price": 150.00,
            "ytd_return": 12.5,
        }
    ), 200


# ============================================================================
# PORTFOLIO ANALYSIS ENDPOINTS (Placeholder - Nastia to implement)
# ============================================================================


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Analyze portfolio

    Request body:
    {
        "stocks": ["AAPL", "MSFT", "GOOGL"],
        "allocations": [40, 35, 25]
    }

    TODO: Nastia - Implement actual analysis logic
    """
    data = request.get_json()

    # Mock response
    return jsonify(
        {
            "analysis": {
                "sp500_comparison": 5.2,
                "inflation_comparison": 2.1,
                "portfolio_volatility": 18.5,
                "risk_score": 45.3,
                "overall_score": 78.5,
            },
            "timestamp": datetime.now().isoformat(),
        }
    ), 200


# ============================================================================
# ML OPTIMIZATION ENDPOINTS (Placeholder - Luke to implement)
# ============================================================================


@app.route("/api/suggestions", methods=["POST"])
def get_suggestions():
    """
    Get ML optimization suggestions

    Request body:
    {
        "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    }

    TODO: Luke - Call ML service
    """
    data = request.get_json()
    stocks = data.get("stocks", [])

    # Mock response
    return jsonify(
        {
            "suggested_allocations": {
                stock: round(100 / len(stocks), 2) for stock in stocks
            },
            "confidence": 0.85,
            "expected_return": 12.5,
            "model_version": "v1",
        }
    ), 200


# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Flask backend server...")
    print(f"📍 Health check: http://localhost:5000/health")
    print(f"📍 API docs: http://localhost:5000/")
    print(f"🔑 Demo credentials: demo / password")

    app.run(host="0.0.0.0", port=5001, debug=True)
