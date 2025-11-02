import matplotlib.pyplot as plt
import numpy as np

complexity = np.linspace(0, 10, 100)
bias_sq = 4 / (1 + np.exp(complexity - 3))  # Decreasing sigmoid
variance = 1 / (1 + np.exp(-complexity + 3))  # Increasing sigmoid
mse = bias_sq + variance

plt.plot(complexity, bias_sq, label='Bias²')
plt.plot(complexity, variance, label='Variance')
plt.plot(complexity, mse, label='Total MSE', linewidth=2)
plt.axvline(x=3, color='r', linestyle='--', label='Optimal Complexity')
plt.legend()
plt.xlabel('Model Complexity')
plt.ylabel('Error')
plt.title('Bias-Variance Tradeoff')
plt.show()