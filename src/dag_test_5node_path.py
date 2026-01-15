import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import networkx as nx

# Set random seed for reproducibility
np.random.seed(42)


class ChainSEM:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples

    def generate_data(self):
        """
        Generate data according to the DAG: a -> b -> c -> d -> e
        """
        # Generate exogenous variable a
        a = np.random.normal(0, 1, self.n_samples)  # a ~ N(0,1)

        # Generate b: b = β_ab * a + ε_b
        beta_ab = 0.8
        epsilon_b = np.random.normal(0, 0.3, self.n_samples)
        b = beta_ab * a + epsilon_b

        # Generate c: c = β_bc * b + ε_c
        beta_bc = 0.7
        epsilon_c = np.random.normal(0, 0.4, self.n_samples)
        c = beta_bc * b + epsilon_c

        # Generate d: d = β_cd * c + ε_d
        beta_cd = 0.6
        epsilon_d = np.random.normal(0, 0.5, self.n_samples)
        d = beta_cd * c + epsilon_d

        # Generate e: e = β_de * d + ε_e
        beta_de = 0.9
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
            'beta_bc': beta_bc,
            'beta_cd': beta_cd,
            'beta_de': beta_de
        }

        return data, true_params

    def verify_independencies(self, data):
        """
        Verify the conditional independencies implied by the chain DAG
        """
        print("=== Conditional Independence Tests ===")

        # a and c should be independent given b
        model_ac_b = LinearRegression()
        model_ac_b.fit(data[['b']], data['a'])
        a_resid = data['a'] - model_ac_b.predict(data[['b']])
        corr_ac_given_b = np.corrcoef(a_resid, data['c'])[0, 1]
        print(f"Correlation between a and c given b: {corr_ac_given_b:.3f}")

        # a and d should be independent given b, c
        model_ad_bc = LinearRegression()
        model_ad_bc.fit(data[['b', 'c']], data['a'])
        a_resid2 = data['a'] - model_ad_bc.predict(data[['b', 'c']])
        corr_ad_given_bc = np.corrcoef(a_resid2, data['d'])[0, 1]
        print(f"Correlation between a and d given b,c: {corr_ad_given_bc:.3f}")

        # b and d should be independent given c
        model_bd_c = LinearRegression()
        model_bd_c.fit(data[['c']], data['b'])
        b_resid = data['b'] - model_bd_c.predict(data[['c']])
        corr_bd_given_c = np.corrcoef(b_resid, data['d'])[0, 1]
        print(f"Correlation between b and d given c: {corr_bd_given_c:.3f}")

        # c and e should be independent given d
        model_ce_d = LinearRegression()
        model_ce_d.fit(data[['d']], data['c'])
        c_resid = data['c'] - model_ce_d.predict(data[['d']])
        corr_ce_given_d = np.corrcoef(c_resid, data['e'])[0, 1]
        print(f"Correlation between c and e given d: {corr_ce_given_d:.3f}")

        return {
            'ac_given_b': corr_ac_given_b,
            'ad_given_bc': corr_ad_given_bc,
            'bd_given_c': corr_bd_given_c,
            'ce_given_d': corr_ce_given_d
        }

    def estimate_parameters(self, data):
        """
        Estimate the structural parameters using regression
        """
        print("\n=== Parameter Estimation ===")

        # Estimate each relationship with appropriate controls
        results = {}

        # b ~ a
        model_b = LinearRegression()
        model_b.fit(data[['a']], data['b'])
        results['beta_ab'] = model_b.coef_[0]
        results['r2_b'] = model_b.score(data[['a']], data['b'])

        # c ~ b (controlling for a to block backdoor paths)
        model_c = LinearRegression()
        model_c.fit(data[['b', 'a']], data['c'])
        results['beta_bc'] = model_c.coef_[0]  # effect of b on c
        results['r2_c'] = model_c.score(data[['b', 'a']], data['c'])

        # d ~ c (controlling for a, b)
        model_d = LinearRegression()
        model_d.fit(data[['c', 'b', 'a']], data['d'])
        results['beta_cd'] = model_d.coef_[0]  # effect of c on d
        results['r2_d'] = model_d.score(data[['c', 'b', 'a']], data['d'])

        # e ~ d (controlling for a, b, c)
        model_e = LinearRegression()
        model_e.fit(data[['d', 'c', 'b', 'a']], data['e'])
        results['beta_de'] = model_e.coef_[0]  # effect of d on e
        results['r2_e'] = model_e.score(data[['d', 'c', 'b', 'a']], data['e'])

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
        G.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')])

        plt.figure(figsize=(10, 3))
        pos = {'a': (0, 0), 'b': (1, 0), 'c': (2, 0), 'd': (3, 0), 'e': (4, 0)}
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue',
                font_size=16, font_weight='bold', arrowsize=20)
        plt.title("DAG: a → b → c → d → e")
        plt.show()

    def plot_causal_chain(self, data):
        """Plot the causal chain relationships"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # a -> b
        axes[0, 0].scatter(data['a'], data['b'], alpha=0.6)
        axes[0, 0].set_xlabel('a')
        axes[0, 0].set_ylabel('b')
        axes[0, 0].set_title('a → b')

        # b -> c
        axes[0, 1].scatter(data['b'], data['c'], alpha=0.6, color='orange')
        axes[0, 1].set_xlabel('b')
        axes[0, 1].set_ylabel('c')
        axes[0, 1].set_title('b → c')

        # c -> d
        axes[1, 0].scatter(data['c'], data['d'], alpha=0.6, color='green')
        axes[1, 0].set_xlabel('c')
        axes[1, 0].set_ylabel('d')
        axes[1, 0].set_title('c → d')

        # d -> e
        axes[1, 1].scatter(data['d'], data['e'], alpha=0.6, color='red')
        axes[1, 1].set_xlabel('d')
        axes[1, 1].set_ylabel('e')
        axes[1, 1].set_title('d → e')

        plt.tight_layout()
        plt.show()


# Generate data
n_samples = 2000
sem = ChainSEM(n_samples)
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

# Plot causal relationships
sem.plot_causal_chain(data)

# Correlation matrix
print("\n=== Correlation Matrix ===")
corr_matrix = data.corr()
print(corr_matrix.round(3))

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True,
            fmt='.3f', cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix - Chain Structure')
plt.tight_layout()
plt.show()

# Additional analysis: Show how correlations decrease with distance
print("\n=== Correlation by Distance in Chain ===")
distances = {
    'adjacent (e.g., a-b)': corr_matrix.loc['a', 'b'],
    '2 steps (e.g., a-c)': corr_matrix.loc['a', 'c'],
    '3 steps (e.g., a-d)': corr_matrix.loc['a', 'd'],
    '4 steps (a-e)': corr_matrix.loc['a', 'e']
}

for distance, corr in distances.items():
    print(f"{distance}: {corr:.3f}")

data1 = data.loc[:, ['c','b',  'a', 'd', 'e']]
sigma1 = np.cov(data1.T)
L1 = np.linalg.cholesky(sigma1)
Gamma1 = np.linalg.inv(L1).T
Gamma1_normalized = np.round(Gamma1 / np.diag(Gamma1), 6)
print(Gamma1_normalized)
