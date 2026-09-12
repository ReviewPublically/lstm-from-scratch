"""
LSTM Time-Series Forecaster — Built From Scratch with NumPy
=============================================================
No TensorFlow. No PyTorch. No black-box model.fit().

This implements the actual LSTM forward pass, backpropagation through
time (BPTT), and gradient descent by hand, so every gate (forget,
input, output) and both memory paths (cell state, hidden state) are
visible and traceable rather than hidden inside a library call.

Task: forecast a synthetic time series with weekly seasonality, a
slow upward trend, and noise (modeled on something like website
traffic or search volume) — one step ahead.

Author: Khalid Hussain, ReviewPublically.com
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------------------------------------------------------
# 1. Generate a synthetic "website traffic" style time series
# ---------------------------------------------------------------
def generate_series(n_points=400):
    t = np.arange(n_points)
    trend = 0.02 * t
    weekly_seasonality = 3 * np.sin(2 * np.pi * t / 7)
    noise = np.random.normal(0, 0.6, n_points)
    series = 10 + trend + weekly_seasonality + noise
    return series

series = generate_series()
