import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import networkx as nx
from scipy.stats import pearsonr

# Set random seed for reproducibility
np.random.seed(42)


class ComplexColliderSEM:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples

    def generate_data(self):
        """
        Generate data according to the DAG: a -> b <- c and a -> c
        """
        # Generate exogenous variable a
        a = np.random.normal(0, 1, self.n_samples)  # a ~ N(0,1)

        # Generate c: c = β_ac * a + ε_c (a has direct effect on c)
        beta_ac = 0.6  # direct effect of a on c
        epsilon_c = np.random.normal(0, 0.5, self.n_samples)
        c = beta_ac * a + epsilon_c

        # Generate b: b = β_ab * a + β_cb * c + ε_b (collider)
        beta_ab = 0.8  # effect of a on b
        beta_cb = 0.7  # effect of c on b
        epsilon_b = np.random.normal(0, 0.4, self.n_samples)
        b = beta_ab * a + beta_cb * c + epsilon_b

        # Create DataFrame
        data = pd.DataFrame({
            'a': a,
            'b': b,
            'c': c
        })

        true_params = {
            'beta_ac': beta_ac,
            'beta_ab': beta_ab,
            'beta_cb': beta_cb
        }

        return data, true_params

    def verify_independencies(self, data):
        """
        Verify the conditional independencies implied by the DAG
        """
        print("=== Conditional Independence Tests ===")

        # a and c should be correlated due to a->c
        corr_ac, p_ac = pearsonr(data['a'], data['c'])
        print(f"Correlation between a and c: {corr_ac:.3f} (p = {p_ac:.4f})")

        # Test collider effect: a and c should show different relationships when conditioning on b
        print("\n--- Collider Effect Analysis ---")

        # Split by b values to see how conditioning affects a-c relationship
        b_quantiles = data['b'].quantile([0.25, 0.5, 0.75])

        for i, (q_name, q_value) in enumerate(b_quantiles.items()):
            if i == 0:
                subset = data[data['b'] <= q_value]
                label = "Bottom 25% of b"
            elif i == 1:
                subset = data[(data['b'] > b_quantiles.iloc[0]) & (data['b'] <= q_value)]
                label = "25-50% of b"
            elif i == 2:
                subset = data[(data['b'] > b_quantiles.iloc[1]) & (data['b'] <= q_value)]
                label = "50-75% of b"
            else:
                subset = data[data['b'] > q_value]
                label = "Top 25% of b"

            corr, p_val = pearsonr(subset['a'], subset['c'])
            print(f"{label}: correlation(a,c) = {corr:.3f} (p = {p_val:.4f})")

        # Test using regression residuals
        print("\n--- Regression-based Tests ---")

        # Test if a and c are independent given b (they shouldn't be, due to a->c)
        model_ac_b = LinearRegression()
        model_ac_b.fit(data[['b']], data['a'])
        a_resid = data['a'] - model_ac_b.predict(data[['b']])
        corr_ac_given_b, p_ac_b = pearsonr(a_resid, data['c'])
        print(f"Correlation between a and c given b: {corr_ac_given_b:.3f} (p = {p_ac_b:.4f})")

        return {
            'ac_corr': corr_ac,
            'ac_given_b_corr': corr_ac_given_b
        }

    def estimate_parameters(self, data):
        """
        Estimate the structural parameters using regression
        """
        print("\n=== Parameter Estimation ===")

        results = {}

        # Estimate c ~ a (direct effect)
        model_c = LinearRegression()
        model_c.fit(data[['a']], data['c'])
        results['beta_ac'] = model_c.coef_[0]
        results['r2_c'] = model_c.score(data[['a']], data['c'])

        # Estimate b ~ a + c (collider structure)
        model_b = LinearRegression()
        model_b.fit(data[['a', 'c']], data['b'])
        results['beta_ab'] = model_b.coef_[0]
        results['beta_cb'] = model_b.coef_[1]
        results['r2_b'] = model_b.score(data[['a', 'c']], data['b'])

        # Try to estimate the total effect of a on c (should be just direct effect)
        print("Structural equation for c: c = {:.3f}*a + ε".format(results['beta_ac']))
        print("Structural equation for b: b = {:.3f}*a + {:.3f}*c + ε".format(
            results['beta_ab'], results['beta_cb']))

        # Print results
        for param, value in results.items():
            if param.startswith('beta'):
                print(f"Estimated {param}: {value:.3f}")
            elif param.startswith('r2'):
                print(f"{param}: {value:.3f}")

        return results

    def plot_dag(self):
        """Plot the DAG structure"""
        G = nx.DiGraph()
        G.add_edges_from([('a', 'b'), ('c', 'b'), ('a', 'c')])

        plt.figure(figsize=(8, 6))
        pos = {
            'a': (0, 0),
            'c': (2, 0),
            'b': (1, 1)
        }

        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue',
                font_size=16, font_weight='bold', arrowsize=20, arrows=True)
        plt.title("DAG: a → b ← c and a → c")
        plt.show()

    def plot_relationships(self, data):
        """Plot the key relationships"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # a -> c (direct effect)
        axes[0, 0].scatter(data['a'], data['c'], alpha=0.6, color='blue')
        axes[0, 0].set_xlabel('a')
        axes[0, 0].set_ylabel('c')
        axes[0, 0].set_title('a → c (direct effect)')

        # a -> b
        axes[0, 1].scatter(data['a'], data['b'], alpha=0.6, color='green')
        axes[0, 1].set_xlabel('a')
        axes[0, 1].set_ylabel('b')
        axes[0, 1].set_title('a → b')

        # c -> b
        axes[1, 0].scatter(data['c'], data['b'], alpha=0.6, color='orange')
        axes[1, 0].set_xlabel('c')
        axes[1, 0].set_ylabel('b')
        axes[1, 0].set_title('c → b')

        # Collider effect: a vs c stratified by b
        b_high = data[data['b'] > data['b'].median()]
        b_low = data[data['b'] <= data['b'].median()]

        axes[1, 1].scatter(b_high['a'], b_high['c'], alpha=0.6, color='red', label='High b')
        axes[1, 1].scatter(b_low['a'], b_low['c'], alpha=0.6, color='purple', label='Low b')
        axes[1, 1].set_xlabel('a')
        axes[1, 1].set_ylabel('c')
        axes[1, 1].set_title('a vs c stratified by b (collider effect)')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def analyze_causal_paths(self, data, true_params):
        """
        Analyze the different causal paths and their effects
        """
        print("\n=== Causal Path Analysis ===")

        # Total effect of a on c (only direct path a->c)
        total_effect_ac = true_params['beta_ac']
        print(f"Total effect of a on c (a→c): {total_effect_ac:.3f}")

        # The path a->b<-c is blocked (collider), so no indirect effect
        print("Indirect effect through b (a→b←c): BLOCKED (collider)")

        # Test if we can detect the collider structure
        print("\n--- Collider Detection ---")
        print("If we naively regress c on a and b (controlling for the collider):")

        model_naive = LinearRegression()
        model_naive.fit(data[['a', 'b']], data['c'])
        naive_beta_a = model_naive.coef_[0]
        naive_beta_b = model_naive.coef_[1]

        print(f"c ~ a + b gives: beta_a = {naive_beta_a:.3f}, beta_b = {naive_beta_b:.3f}")
        print("This is biased because we're controlling for a collider!")

        return {
            'total_effect_ac': total_effect_ac,
            'naive_beta_a': naive_beta_a,
            'naive_beta_b': naive_beta_b
        }


# Generate data
n_samples = 2000
sem = ComplexColliderSEM(n_samples)
data, true_params = sem.generate_data()

print("True parameters:")
for param, value in true_params.items():
    print(f"{param}: {value}")

print(f"\nGenerated {n_samples} samples")
print("\nFirst 5 rows of data:")
print(data.head())

# Plot the DAG
sem.plot_dag()

# Verify conditional relationships
independencies = sem.verify_independencies(data)

# Estimate parameters
estimated_params = sem.estimate_parameters(data)

# Plot relationships
sem.plot_relationships(data)

# Correlation matrix
print("\n=== Correlation Matrix ===")
corr_matrix = data.corr()
print(corr_matrix.round(3))

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True,
            fmt='.3f', cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix - a → b ← c and a → c')
plt.tight_layout()
plt.show()

# Analyze causal paths
path_analysis = sem.analyze_causal_paths(data, true_params)

# Additional analysis: Show what happens when we misinterpret the collider
print("\n=== Common Mistake: Controlling for Collider ===")
print("What happens if we mistakenly control for b when estimating a→c:")

# Correct model: c ~ a
model_correct = LinearRegression()
model_correct.fit(data[['a']], data['c'])
correct_effect = model_correct.coef_[0]

# Incorrect model: c ~ a + b (controlling for collider)
model_incorrect = LinearRegression()
model_incorrect.fit(data[['a', 'b']], data['c'])
incorrect_effect = model_incorrect.coef_[0]

print(f"Correct estimate (c ~ a): {correct_effect:.3f}")
print(f"Incorrect estimate (c ~ a + b): {incorrect_effect:.3f}")
print(f"Bias: {abs(correct_effect - incorrect_effect):.3f}")

# Simulation: Show how the bias changes with different parameter values
print("\n=== Sensitivity Analysis ===")
print("How does the bias from controlling for collider change with parameters?")

beta_ac_values = [0.3, 0.6, 0.9]
beta_ab_values = [0.5, 0.8, 1.1]
beta_cb_values = [0.5, 0.7, 0.9]

for beta_ac in beta_ac_values:
    for beta_ab in beta_ab_values:
        for beta_cb in beta_cb_values:
            # Generate data with these parameters
            a = np.random.normal(0, 1, 1000)
            c = beta_ac * a + np.random.normal(0, 0.5, 1000)
            b = beta_ab * a + beta_cb * c + np.random.normal(0, 0.4, 1000)

            # Estimate effects
            correct = LinearRegression().fit(a.reshape(-1, 1), c).coef_[0]
            incorrect = LinearRegression().fit(np.column_stack([a, b]), c).coef_[0]
            bias = abs(correct - incorrect)

            if bias > 0.1:  # Only show cases with significant bias
                print(f"β_ac={beta_ac}, β_ab={beta_ab}, β_cb={beta_cb}: bias={bias:.3f}")