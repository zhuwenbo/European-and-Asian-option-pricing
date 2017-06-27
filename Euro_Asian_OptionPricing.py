import matplotlib.pyplot as plt
import numpy as np
import math as mt
import statistics as stat

T = 1
mu = 0.1
sigma = 0.15
S0 = 15
dt = 1 / 24
K = 16
r = 0
N = round(T / dt)
t = np.linspace(0, T, N + 1)

Spaths = np.zeros((5, N + 1))
Spaths[:, 0] = S0
nudt = (r - 0.5 * sigma ** 2) * dt
sidt = sigma * np.sqrt(dt)

for i in range(5):
    for j in range(N):
        Spaths[i, j + 1] = Spaths[i, j] * np.exp(nudt + sidt * np.random.standard_normal(1))

plt.plot(t, Spaths[0, :], 'r', t, Spaths[1, :], 'b', t, Spaths[2, :], 'g', t, Spaths[3, :], 'y',t, Spaths[4, :], 'k')
plt.show()

# (b)
Euro_call_value = 0
Euro_put_value = 0
Asian_call_value = 0
call_values = []
put_values = []
asian_call_values = []
for i in range(10000):
    W = np.random.standard_normal(size=N)
    W = np.append([0], np.cumsum(W) * np.sqrt(dt))
    X = (r - 0.5 * sigma ** 2) * t + sigma * W
    S = S0 * np.exp(X)
    call_values.append(max(S[-1] - K, 0))
    put_values.append(max(K - S[-1], 0))
    asian_call_values.append(max(stat.mean(S) - K, 0))
    Euro_call_value += max(S[-1] - K, 0)
    Euro_put_value += max(K - S[-1], 0)
    Asian_call_value += max(stat.mean(S) - K, 0)

Euro_call_value /= 10000
Euro_put_value /= 10000
Asian_call_value /= 10000

Simulated_Euro_call_price = Euro_call_value * mt.exp(-r * T)
Simulated_Euro_put_price = Euro_put_value * mt.exp(-r * T)
Simulated_Asian_call_price = Asian_call_value * mt.exp(-r * T)

print('The European call option price is:', Simulated_Euro_call_price)
print('The European put option price is:', Simulated_Euro_put_price)
print('The Asian call option price is:', Simulated_Asian_call_price)

print('The Monte-Carlo variance of European Call Option pricing is:',stat.variance(call_values)/10000)
print('The Monte-Carlo variance of European Put Option pricing is:',stat.variance(put_values)/10000)
print('The Monte-Carlo variance of Asian Call Option pricing is:',stat.variance(asian_call_values)/10000)