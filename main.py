import numpy as np
import matplotlib.pyplot as plt

# Plot settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# Simulation parameters
T = 100  # Number of iterations
N = 100  # Population size

# Define strategies with fixed parameters
params = {
    "Strategy-1": {"gamma": 0.8, "theta": 0.3, "omega": 0.5, "mu": 0.1, "r": 0.9, "U": 1.2, "k": 0.4, "color": "green"},
    "Strategy-2": {"gamma": 0.4, "theta": 0.5, "omega": 0.7, "mu": 0.2, "r": 0.8, "U": 1.0, "k": 0.3, "color": "red"},
    "Strategy-3": {"gamma": 0.2, "theta": 0.6, "omega": 0.9, "mu": 0.3, "r": 0.7, "U": 0.8, "k": 0.2, "color": "blue"}
}

def calculate_payoff(share_ratio, gamma, theta, omega, mu, r, U, k):
    benefit = gamma * U * (1 - np.exp(-share_ratio * omega))
    cost = theta * share_ratio + mu * (1 - share_ratio)
    incentive_factor = r * np.log(1 + k * share_ratio)
    return benefit - cost + incentive_factor

histories = {}

def dynamic_adjustment(t, initial_value, max_value, min_value):
    return min_value + (max_value - min_value) * np.exp(-0.03 * t)  # Exponential decay for stability

for strategy, param in params.items():
    gamma, theta, omega, mu, r, U, k = param["gamma"], param["theta"], param["omega"], param["mu"], param["r"], param["U"], param["k"]
    share_ratio = 0.5
    history = []
    
    for t in range(T):
        dynamic_theta = dynamic_adjustment(t, theta, max_value=0.7, min_value=0.2)
        dynamic_gamma = dynamic_adjustment(t, gamma, max_value=0.9, min_value=0.3)
        
        payoff_sharing = calculate_payoff(share_ratio, dynamic_gamma, dynamic_theta, omega, mu, r, U, k)
        payoff_non_sharing = 0
        avg_payoff = share_ratio * payoff_sharing + (1 - share_ratio) * payoff_non_sharing
        share_ratio += 0.1 * share_ratio * (payoff_sharing - avg_payoff)
        share_ratio = max(0, min(1, share_ratio))
        
        history.append(share_ratio * 100)  # Convert proportion to percentage
    
    histories[strategy] = history

# Plot proportion of sharing members
plt.figure(figsize=(12, 7))
for strategy, param in params.items():
    plt.plot(histories[strategy], label=strategy, color=param["color"], linewidth=2)

plt.xlabel("Iterations")
plt.ylabel("Participants (%)")
plt.yticks(np.arange(0, 101, 10))  # Ensure rounded values from 0 to 100
plt.legend()
# plt.grid()
plt.show()

# Plot expected payoffs vs. sharing proportion
share_ratios = np.linspace(0, 1, 100)
plt.figure(figsize=(12, 7))
for strategy, param in params.items():
    payoffs = [calculate_payoff(sr, param["gamma"], param["theta"], param["omega"], param["mu"], param["r"], param["U"], param["k"]) for sr in share_ratios]
    plt.plot(share_ratios * 100, payoffs, label=strategy, color=param["color"], linewidth=2)

plt.xlabel("Proportion of Sharers (%)")
plt.ylabel("Expected Payoff")
plt.yticks(np.arange(int(min(payoffs)), int(max(payoffs)) + 1, 1))  # Ensure integer y-ticks
plt.legend()
# plt.grid()
plt.show()

# Impact of incentives on payoff
incentive_levels = np.linspace(0.2, 1.0, 5)
plt.figure(figsize=(12, 7))
for inc in incentive_levels:
    payoffs = [calculate_payoff(sr, inc, 0.4, 0.7, 0.1, 0.8, 1.1, 0.3) for sr in share_ratios]
    plt.plot(share_ratios * 100, payoffs, label=f"Incentive {inc:.2f}", linewidth=2)

plt.xlabel("Proportion of Sharers (%)")
plt.ylabel("Expected Payoff")
# plt.yticks(np.arange(int(min(payoffs)), int(max(payoffs)) + 1, 1))  # Ensure integer y-ticks
plt.legend()
# plt.grid()
plt.show()

# Comparative Evaluation of FB-CTISM vs. Traditional Methods
plt.figure(figsize=(12, 7))
for strategy, param in params.items():
    plt.plot(histories[strategy], label=f"FB-CTISM - {strategy}", color=param["color"])

traditional_strategy = [30 + 0.2*t for t in range(T)]  # Simulating traditional method
plt.plot(traditional_strategy, label="Without Strategy", color='black',linestyle='dashed', linewidth=2)

plt.xlabel("Iterations")
plt.ylabel("Participants (%)")
plt.yticks(np.arange(0, 101, 10))  # Ensure rounded values from 0 to 100
plt.legend()
# plt.grid()
plt.show()

colors=["#AA60C8","#D69ADE","#EABDE6"]
# Comparison graph: Participation percentage per strategy
plt.figure(figsize=(12, 7))
final_participation = [histories[strategy][-1] for strategy in params.keys()]  # Get final participation values
strategies = list(params.keys())

plt.bar(strategies, final_participation, color=colors)

# plt.xlabel("Strategies")
plt.ylabel("Participation (%)")
plt.ylim(0, 100)
# plt.title("Final Participation Rate Comparison")
# plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# Trust Evaluation using AI-driven Anomaly Detection (Placeholder for Deep Learning Integration)
# Future Work: Implement hybrid anomaly detection with Autoencoders + LSTMs for threat intelligence validation
print("Future Work: AI-driven anomaly detection integration to improve trust and participation.")
print("\nFinal Participation Results:")
for strategy, participation in zip(strategies, final_participation):
    print(f"{strategy}: {participation:.2f}% participants")