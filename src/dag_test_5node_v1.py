import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import networkx as nx

# Set random seed for reproducibility
np.random.seed(42)


class ColliderChainSEM:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples

    def generate_data(self):
        """
        Generate data according to the DAG: a -> b <- c -> d -> e
        """
        # Generate exogenous variables a and c
        a = np.random.normal(0, 1, self.n_samples)  # a ~ N(0,1)
        c = np.random.normal(0, 1, self.n_samples)  # c ~ N(0,1)

        # Generate b: b = β_ab * a + β_cb * c + ε_b (collider)
        beta_ab = 0.8  # effect of a on b
        beta_cb = 0.7  # effect of c on b
        epsilon_b = np.random.normal(0, 0.4, self.n_samples)
        b = beta_ab * a + beta_cb * c + epsilon_b

        # Generate d: d = β_cd * c + ε_d
        beta_cd = 0.6  # effect of c on d
        epsilon_d = np.random.normal(0, 0.5, self.n_samples)
        d = beta_cd * c + epsilon_d

        # Generate e: e = β_de * d + ε_e
        beta_de = 0.9  # effect of d on e
        epsilon_e = np.random.normal(0, 0.6, self.n_samples)
        e = beta_de * d + epsilon_e

        # Create DataFrame
        data = pd.DataFrame({
            'a': a,
            'b': b,
            'c': c,
            'd': d,
            'e': e
        })

        true_params = {
            'beta_ab': beta_ab,
            'beta_cb': beta_cb,
            'beta_cd': beta_cd,
            'beta_de': beta_de
        }

        return data, true_params

    def verify_independencies(self, data):
        """
        Verify the conditional independencies implied by the DAG
        """
        print("=== Conditional Independence Tests ===")

        # a and c should be independent (no direct connection)
        corr_ac = np.corrcoef(data['a'], data['c'])[0, 1]
        print(f"Correlation between a and c: {corr_ac:.3f}")

        # a and d should be independent (no open paths)
        corr_ad = np.corrcoef(data['a'], data['d'])[0, 1]
        print(f"Correlation between a and d: {corr_ad:.3f}")

        # a and e should be independent (no open paths)
        corr_ae = np.corrcoef(data['a'], data['e'])[0, 1]
        print(f"Correlation between a and e: {corr_ae:.3f}")

        # Test collider effect: a and c should become dependent when conditioning on b
        model_ac_b = LinearRegression()
        model_ac_b.fit(data[['b']], data['a'])
        a_resid = data['a'] - model_ac_b.predict(data[['b']])
        corr_ac_given_b = np.corrcoef(a_resid, data['c'])[0, 1]
        print(f"Correlation between a and c given b: {corr_ac_given_b:.3f}")

        # c and e should be independent given d
        model_ce_d = LinearRegression()
        model_ce_d.fit(data[['d']], data['c'])
        c_resid = data['c'] - model_ce_d.predict(data[['d']])
        corr_ce_given_d = np.corrcoef(c_resid, data['e'])[0, 1]
        print(f"Correlation between c and e given d: {corr_ce_given_d:.3f}")

        return {
            'ac': corr_ac,
            'ad': corr_ad,
            'ae': corr_ae,
            'ac_given_b': corr_ac_given_b,
            'ce_given_d': corr_ce_given_d
        }

    def estimate_parameters(self, data):
        """
        Estimate the structural parameters using regression
        """
        print("\n=== Parameter Estimation ===")

        results = {}

        # Estimate b ~ a + c (collider structure)
        model_b = LinearRegression()
        model_b.fit(data[['a', 'c']], data['b'])
        results['beta_ab'] = model_b.coef_[0]
        results['beta_cb'] = model_b.coef_[1]
        results['r2_b'] = model_b.score(data[['a', 'c']], data['b'])

        # Estimate d ~ c (controlling for potential confounders)
        model_d = LinearRegression()
        model_d.fit(data[['c', 'a']], data['d'])  # control for a to be safe
        results['beta_cd'] = model_d.coef_[0]
        results['r2_d'] = model_d.score(data[['c', 'a']], data['d'])

        # Estimate e ~ d (controlling for c and a)
        model_e = LinearRegression()
        model_e.fit(data[['d', 'c', 'a']], data['e'])
        results['beta_de'] = model_e.coef_[0]
        results['r2_e'] = model_e.score(data[['d', 'c', 'a']], data['e'])

        # Print results
        print("Structural equation for b: b = {:.3f}*a + {:.3f}*c + ε".format(
            results['beta_ab'], results['beta_cb']))
        print("Structural equation for d: d = {:.3f}*c + ε".format(results['beta_cd']))
        print("Structural equation for e: e = {:.3f}*d + ε".format(results['beta_de']))

        for param, value in results.items():
            if param.startswith('beta'):
                print(f"Estimated {param}: {value:.3f}")
            elif param.startswith('r2'):
                print(f"{param}: {value:.3f}")

        return results

    def plot_dag(self):
        """Plot the DAG structure"""
        G = nx.DiGraph()
        G.add_edges_from([('a', 'b'), ('c', 'b'), ('c', 'd'), ('d', 'e')])

        plt.figure(figsize=(10, 6))
        pos = {
            'a': (0, 1),
            'c': (0, -1),
            'b': (2, 0),
            'd': (4, -1),
            'e': (6, -1)
        }

        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue',
                font_size=16, font_weight='bold', arrowsize=20, arrows=True)
        plt.title("DAG: a → b ← c → d → e")
        plt.show()

    def plot_relationships(self, data):
        """Plot the key relationships"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # a -> b
        axes[0, 0].scatter(data['a'], data['b'], alpha=0.6)
        axes[0, 0].set_xlabel('a')
        axes[0, 0].set_ylabel('b')
        axes[0, 0].set_title('a → b')

        # c -> b
        axes[0, 1].scatter(data['c'], data['b'], alpha=0.6, color='orange')
        axes[0, 1].set_xlabel('c')
        axes[0, 1].set_ylabel('b')
        axes[0, 1].set_title('c → b')

        # c -> d
        axes[0, 2].scatter(data['c'], data['d'], alpha=0.6, color='green')
        axes[0, 2].set_xlabel('c')
        axes[0, 2].set_ylabel('d')
        axes[0, 2].set_title('c → d')

        # d -> e
        axes[1, 0].scatter(data['d'], data['e'], alpha=0.6, color='red')
        axes[1, 0].set_xlabel('d')
        axes[1, 0].set_ylabel('e')
        axes[1, 0].set_title('d → e')

        # a vs c (should be independent)
        axes[1, 1].scatter(data['a'], data['c'], alpha=0.6, color='purple')
        axes[1, 1].set_xlabel('a')
        axes[1, 1].set_ylabel('c')
        axes[1, 1].set_title('a and c (independent)')

        # Collider effect: a vs c given b
        # Show how conditioning on b creates correlation
        b_high = data[data['b'] > data['b'].median()]
        b_low = data[data['b'] <= data['b'].median()]

        axes[1, 2].scatter(b_high['a'], b_high['c'], alpha=0.6, color='blue', label='High b')
        axes[1, 2].scatter(b_low['a'], b_low['c'], alpha=0.6, color='cyan', label='Low b')
        axes[1, 2].set_xlabel('a')
        axes[1, 2].set_ylabel('c')
        axes[1, 2].set_title('a vs c stratified by b')
        axes[1, 2].legend()

        plt.tight_layout()
        plt.show()


# Generate data
n_samples = 2000
sem = ColliderChainSEM(n_samples)
data, true_params = sem.generate_data()

print("True parameters:")
for param, value in true_params.items():
    print(f"{param}: {value}")

print(f"\nGenerated {n_samples} samples")
print("\nFirst 5 rows of data:")
print(data.head())

# Plot the DAG
sem.plot_dag()

# Verify conditional independencies
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
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True,
            fmt='.3f', cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix - a → b ← c → d → e')
plt.tight_layout()
plt.show()

# Additional analysis: Show collider effect more clearly
print("\n=== Collider Effect Analysis ===")
print("When we condition on different values of b:")

# Split data by b quantiles
quantiles = data['b'].quantile([0.25, 0.5, 0.75])
for i, (q_name, q_value) in enumerate(quantiles.items()):
    if i == 0:
        subset = data[data['b'] <= q_value]
        label = "Bottom 25% of b"
    elif i == 1:
        subset = data[(data['b'] > quantiles.iloc[0]) & (data['b'] <= q_value)]
        label = "25-50% of b"
    elif i == 2:
        subset = data[(data['b'] > quantiles.iloc[1]) & (data['b'] <= q_value)]
        label = "50-75% of b"
    else:
        subset = data[data['b'] > q_value]
        label = "Top 25% of b"

    corr = np.corrcoef(subset['a'], subset['c'])[0, 1]
    print(f"{label}: correlation(a,c) = {corr:.3f}")

# Path analysis
print("\n=== Path Analysis ===")
print("Total correlation between c and e:", np.round(corr_matrix.loc['c', 'e'], 3))
print("Direct effect (c→d→e):", np.round(true_params['beta_cd'] * true_params['beta_de'], 3))
print("Indirect through b (c→b←a): 0 (blocked by collider)")

data1 = data.loc[:, ['b', 'c', 'a', 'd', 'e']]
sigma1 = np.cov(data1.T)
L1 = np.linalg.cholesky(sigma1)
Gamma1 = np.linalg.inv(L1).T
Gamma1_normalized = np.round(Gamma1 / np.diag(Gamma1), 6)
print(Gamma1_normalized)
