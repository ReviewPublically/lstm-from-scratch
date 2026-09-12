# LSTM Time-Series Forecaster — Built From Scratch in NumPy
A Long Short-Term Memory (LSTM) network implemented entirely from scratch
using NumPy, no TensorFlow, PyTorch, or Keras. Every gate (forget, input,
output), the cell state update, the hidden state update, and the full
backpropagation-through-time (BPTT) gradient calculation are written out
explicitly.
This was built to genuinely understand how LSTM gates work internally
rather than treating model.fit() as a black box, and to support a
deep-dive article on Review Publically about how LSTMs solve the
vanishing gradient problem.
What it does
Forecasts a synthetic time series with weekly seasonality, an upward
trend, and noise (modeled after something like website traffic or
search interest) one step ahead, using the previous 14 time steps as
input.
Results
Training MSE dropped from 0.024 to 0.0037 over 120 epochs
Test MAE (denormalized): 0.70 on a series ranging roughly 10-22
The model correctly learned the weekly seasonal cycle and the
underlying upward trend on held-out data
�
�
Load image
Load image
Why from scratch
Every existing RNN/LSTM tutorial online calls a library function and
shows the result. None of them show what is actually happening inside
the cell at each time step. This implementation exposes:
The forget gate deciding what to erase from long-term memory (cell state)
The input gate deciding what new information to write
The output gate deciding what part of memory becomes the hidden state
Manual backpropagation through time, walking gradients backward
through every gate at every time step
Gradient clipping, included because untrained from-scratch LSTMs
are genuinely prone to exploding gradients, which is a good practical
demonstration of the problem the architecture is designed to reduce
Run it yourself
Bash
Files
lstm_forecaster.py — the full implementation and training loop
loss_curve.png — training loss over 120 epochs
prediction_vs_actual.png — held-out test predictions vs. real values
Author
Khalid Hussain, founder of Review Publically,
MSc Computer Science, Google Advanced Data Analytics certified.