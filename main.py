import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Veri setini oku
df = pd.read_csv('mcs_ds_edited_iter_shuffled.csv')
df = df.iloc[:, 0:5]
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

# Evren aralıkları
anchor_ratio_range = np.linspace(10, 30, 100)
trans_range = np.linspace(12, 25, 100)
node_density_range = np.linspace(100, 300, 100)
iteration_range = np.linspace(14, 100, 100)
ale_range = np.linspace(0, 2.6, 100)

# Üyelik fonksiyonları
def create_triangular_mfs(x):
    low = fuzz.trimf(x, [x[0], x[0], np.mean(x)])
    medium = fuzz.trimf(x, [x[0], np.mean(x), x[-1]])
    high = fuzz.trimf(x, [np.mean(x), x[-1], x[-1]])
    return {'low': low, 'medium': medium, 'high': high}

def create_gaussian_mfs(x):
    sigma = (x[-1]-x[0])/6
    low = fuzz.gaussmf(x, x[0], sigma)
    medium = fuzz.gaussmf(x, np.mean(x), sigma)
    high = fuzz.gaussmf(x, x[-1], sigma)
    return {'low': low, 'medium': medium, 'high': high}

# Üyelik fonksiyonları grafikleri
def plot_mfs(x, mfs, title):
    plt.figure(figsize=(7, 4))
    for label, mf in mfs.items():
        plt.plot(x, mf, label=label.capitalize())
    plt.title(f"{title} Üyelik Fonksiyonları")
    plt.xlabel(title)
    plt.ylabel("Üyelik Derecesi")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

for var_range, var_name in zip(
    [anchor_ratio_range, trans_range, node_density_range, iteration_range, ale_range],
    ['Anchor Ratio', 'Transmission Range', 'Node Density', 'Iteration Count', 'ALE']):
    mfs_tri = create_triangular_mfs(var_range)
    mfs_gauss = create_gaussian_mfs(var_range)
    plot_mfs(var_range, mfs_tri, f"{var_name} (Triangular)")
    plot_mfs(var_range, mfs_gauss, f"{var_name} (Gaussian)")

# Çıkış (ALE) üyelik fonksiyonları
ale_mfs = {
    'low': fuzz.trimf(ale_range, [0, 0, 1.0]),
    'medium': fuzz.trimf(ale_range, [0.5, 1.3, 2.0]),
    'high': fuzz.trimf(ale_range, [1.5, 2.6, 2.6])
}

# 25 kural
rules = [
    ('low', 'medium', 'high', 'medium', 'low'),
    ('medium', 'low', 'medium', 'high', 'low'),
    ('medium', 'medium', 'high', 'high', 'low'),
    ('medium', 'low', 'medium', 'high', 'low'),
    ('low', 'low', 'medium', 'high', 'low'),
    ('high', 'low', 'high', 'medium', 'low'),
    ('medium', 'medium', 'low', 'low', 'medium'),
    ('medium', 'low', 'low', 'medium', 'medium'),
    ('high', 'low', 'low', 'medium', 'medium'),
    ('low', 'medium', 'low', 'low', 'medium'),
    ('low', 'medium', 'high', 'low', 'medium'),
    ('low', 'low', 'medium', 'high', 'low'),
    ('low', 'medium', 'high', 'high', 'low'),
    ('low', 'low', 'low', 'medium', 'medium'),
    ('medium', 'medium', 'medium', 'high', 'medium'),
    ('medium', 'high', 'medium', 'medium', 'low'),
    ('medium', 'high', 'high', 'medium', 'low'),
    ('high', 'medium', 'high', 'medium', 'low'),
    ('high', 'high', 'high', 'medium', 'low'),
    ('high', 'low', 'low', 'medium', 'medium'),
    ('medium', 'high', 'medium', 'high', 'low'),
    ('low', 'high', 'medium', 'high', 'low'),
    ('low', 'medium', 'medium', 'high', 'low'),
    ('medium', 'medium', 'high', 'medium', 'low'),
    ('low', 'medium', 'high', 'medium', 'low')
]

# Fuzzify
def fuzzify(value, universe, mfs):
    degrees = {}
    for label in ['low', 'medium', 'high']:
        deg = fuzz.interp_membership(universe, mfs[label], value)
        degrees[label] = deg
    return degrees

# Kural aktivasyonu
def rule_activation(fuzz_vals, rule):
    ar = fuzz_vals['anchor_ratio'][rule[0]]
    tr = fuzz_vals['trans_range'][rule[1]]
    nd = fuzz_vals['node_density'][rule[2]]
    it = fuzz_vals['iteration_count'][rule[3]]
    return np.min([ar, tr, nd, it]), rule[4]

# Çıktı birleştirme
def aggregate_outputs(rule_activations, ale_mfs, ale_range):
    aggregated = np.zeros_like(ale_range)
    for activation, out_label in rule_activations:
        cut_mf = np.fmin(activation, ale_mfs[out_label])
        aggregated = np.fmax(aggregated, cut_mf)
    return aggregated

# Defuzzification
def defuzz_cos(ale_range, aggregated, rule_activations):
    centers, weights = [], []
    for label in ['low', 'medium', 'high']:
        centroid = fuzz.defuzz(ale_range, ale_mfs[label], 'centroid')
        max_act = max([act for act, out_label in rule_activations if out_label == label], default=0)
        centers.append(centroid)
        weights.append(max_act)
    numerator = np.sum(np.array(centers)*np.array(weights))
    denominator = np.sum(weights)
    return numerator / denominator if denominator != 0 else np.nan

def defuzz_wam(ale_range, aggregated, rule_activations):
    centers, weights = [], []
    for label in ['low', 'medium', 'high']:
        max_act = max([act for act, out_label in rule_activations if out_label == label], default=0)
        centroid = fuzz.defuzz(ale_range, ale_mfs[label], 'centroid')
        centers.append(centroid)
        weights.append(max_act)
    numerator = np.sum(np.array(weights)*np.array(centers))
    denominator = np.sum(weights)
    return numerator / denominator if denominator != 0 else np.nan

# Ana simülasyon
def run_simulation(mf_type='triangular', defuzz_method='cos'):
    if mf_type == 'triangular':
        ar_mfs = create_triangular_mfs(anchor_ratio_range)
        tr_mfs = create_triangular_mfs(trans_range)
        nd_mfs = create_triangular_mfs(node_density_range)
        it_mfs = create_triangular_mfs(iteration_range)
    else:
        ar_mfs = create_gaussian_mfs(anchor_ratio_range)
        tr_mfs = create_gaussian_mfs(trans_range)
        nd_mfs = create_gaussian_mfs(node_density_range)
        it_mfs = create_gaussian_mfs(iteration_range)

    predictions = []
    for i in range(len(X)):
        fuzz_vals = {
            'anchor_ratio': fuzzify(X[i][0], anchor_ratio_range, ar_mfs),
            'trans_range': fuzzify(X[i][1], trans_range, tr_mfs),
            'node_density': fuzzify(X[i][2], node_density_range, nd_mfs),
            'iteration_count': fuzzify(X[i][3], iteration_range, it_mfs),
        }
        rule_activations = [rule_activation(fuzz_vals, rule) for rule in rules]
        aggregated = aggregate_outputs(rule_activations, ale_mfs, ale_range)

        if defuzz_method == 'cos':
            pred = defuzz_cos(ale_range, aggregated, rule_activations)
        else:
            pred = defuzz_wam(ale_range, aggregated, rule_activations)
        predictions.append(pred)

    preds = np.array(predictions)
    mask = ~np.isnan(preds)
    y_clean, preds_clean = y[mask], preds[mask]
    mae = mean_absolute_error(y_clean, preds_clean)
    rmse = np.sqrt(mean_squared_error(y_clean, preds_clean))
    return mae, rmse

# Tüm kombinasyonları çalıştır
results = {}
for mf in ['triangular', 'gaussian']:
    for dfz in ['cos', 'wam']:
        mae, rmse = run_simulation(mf, dfz)
        label = f"{mf.capitalize()} + {dfz.upper()}"
        results[label] = [mae, rmse]
        print(f"{label}: MAE={mae:.4f}, RMSE={rmse:.4f}")

# Sonuç grafiği
df_results = pd.DataFrame(results, index=["MAE", "RMSE"]).T
df_results.plot(kind="bar", figsize=(10, 6))
plt.title("4 Kombinasyonun Performansı (ALE Tahmini)")
plt.ylabel("Değer")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
