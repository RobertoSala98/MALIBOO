from scipy.stats import mannwhitneyu
import pandas as pd
from numpy import mean
import json

dim = 8
allData = [['ML type', 'epsilon', 'ML model', 'MAPR [%]', 'feasibility [%]', 'time [s]']]

print("\n")

# Varying ML type
results = {}

ML_types = ['indicator', 'probability']

for _ in range(len(ML_types)):

    ML_type = ML_types[_]

    results[ML_type] = []
    for idx in range(30):
        df = pd.read_csv(f"outputs/Schwefel_d={dim}/True/True/True/{ML_type}/Ridge/{ML_type}/Ridge/0.1/{idx}/results.csv")
        df['feasible'] = df['feasible'].astype(bool)
        feasible_rows = df[df['feasible'] == True]
        max_target_row = feasible_rows.loc[feasible_rows['target'].idxmax()]

        results[ML_type].append(float(-max_target_row.target))

    for idx_ in range(_):

        stat, p_value = mannwhitneyu(results[ML_types[idx_]], results[ML_type], alternative='two-sided')
        print(ML_types[idx_], ML_type, "p-value:", p_value)

    json_file = f"outputs/Schwefel_d={dim}/True/True/True/{ML_type}/Ridge/{ML_type}/Ridge/0.1/output.json"
    with open(json_file, "r") as file:
        data = json.load(file)
    allData.append([ML_type,0.1,'Ridge',round(data["MAPR [%]"],3), round(data["feasibility [%]"],3), round(data["time [s]"],3)])

print("\n")



# Varying epsilon-greedy
results = {}

eps_values = [0.01, 0.025, 0.05, 0.1, 0.2]

for _ in range(len(eps_values)):

    eps = eps_values[_]

    results[eps] = []
    for idx in range(30):
        df = pd.read_csv(f"outputs/Schwefel_d={dim}/True/True/True/indicator/Ridge/indicator/Ridge/{eps}/{idx}/results.csv")
        df['feasible'] = df['feasible'].astype(bool)
        feasible_rows = df[df['feasible'] == True]
        max_target_row = feasible_rows.loc[feasible_rows['target'].idxmax()]

        results[eps].append(float(-max_target_row.target))

    for idx_ in range(_):

        stat, p_value = mannwhitneyu(results[eps_values[idx_]], results[eps], alternative='two-sided')
        print(eps_values[idx_], eps, "p-value:", p_value)

    if eps != 0.1:
        json_file = f"outputs/Schwefel_d={dim}/True/True/True/indicator/Ridge/indicator/Ridge/{eps}/output.json"
        with open(json_file, "r") as file:
            data = json.load(file)
        allData.append(['indicator',eps,'Ridge',round(data["MAPR [%]"],3), round(data["feasibility [%]"],3), round(data["time [s]"],3)])

print("\n")



# Varying ml_model
results = {}

model = ['Ridge', 'RandomForest', 'NeuralNetwork', 'XGBoost']

for _ in range(len(model)):

    model_ = model[_]

    results[model_] = []
    for idx in range(30):
        df = pd.read_csv(f"outputs/Schwefel_d={dim}/True/True/True/indicator/{model_}/indicator/{model_}/0.1/{idx}/results.csv")
        df['feasible'] = df['feasible'].astype(bool)
        feasible_rows = df[df['feasible'] == True]
        max_target_row = feasible_rows.loc[feasible_rows['target'].idxmax()]

        results[model_].append(float(-max_target_row.target))

    for idx_ in range(_):

        stat, p_value = mannwhitneyu(results[model[idx_]], results[model_], alternative='two-sided')
        print(model[idx_], model_, "p-value:", p_value)

    if model != 'Ridge':
        json_file = f"outputs/Schwefel_d={dim}/True/True/True/indicator/{model_}/indicator/{model_}/0.1/output.json"
        with open(json_file, "r") as file:
            data = json.load(file)
        allData.append(['indicator',0.1,model_,round(data["MAPR [%]"],3), round(data["feasibility [%]"],3), round(data["time [s]"],3)])

print("\n\n")

df = pd.DataFrame(allData[1:], columns=allData[0])
df.to_csv("ablation_study.csv", index=False)



allData = [['dim', 'MAPR [%]', 'feasibility [%]', 'time [s]']]

print("\n")

# Varying the dimension
results = {}

dims = [4, 8, 12, 16]
feas = []
time_ = []

for _ in range(len(dims)):

    dim = dims[_]

    results[dim] = []
    for idx in range(30):
        df = pd.read_csv(f"outputs/Schwefel_d={dim}/True/True/True/indicator/Ridge/indicator/Ridge/0.1/{idx}/results.csv")
        df['feasible'] = df['feasible'].astype(bool)
        feasible_rows = df[df['feasible'] == True]
        max_target_row = feasible_rows.loc[feasible_rows['target'].idxmax()]

        results[dim].append(float(-max_target_row.target))

    json_file = f"outputs/Schwefel_d={dim}/True/True/True/indicator/Ridge/indicator/Ridge/0.1/output.json"
    with open(json_file, "r") as file:
        data = json.load(file)
    allData.append([dim,round(data["MAPR [%]"],3), round(data["feasibility [%]"],3), round(data["time [s]"],3)])

    feas.append(data["feasibility [%]"])
    time_.append(data["time [s]"])

df = pd.DataFrame(allData[1:], columns=allData[0])
df.to_csv("scalability_analysis.csv", index=False)


import matplotlib.pyplot as plt

# Common figure size and DPI
figsize = (6, 4)
dpi = 300
xticks = [4, 8, 12, 16]
fontsize = 24

# First plot: Feasibility
fig, ax = plt.subplots(figsize=figsize)
ax.plot(dims, feas, marker='o')
#ax.plot([2, 18], [20, 20], 'r--')  # horizontal line
ax.set_xlim(3.5, 16.5)

ax.set_xticks(xticks)
ax.set_yticks([16,18,20,22,24])  # keep default y-ticks
ax.tick_params(axis='both', labelsize=fontsize)

ax.set_xlabel(r"$d$", fontsize=fontsize)
ax.set_ylabel("Feasib. rate [%]", fontsize=fontsize)

ax.grid(True, which='both', linestyle='--', linewidth=0.7)  # add grid

plt.tight_layout()
fig.savefig("feasibility.png", dpi=dpi, bbox_inches='tight')
fig.savefig("feasibility.pdf", dpi=dpi, bbox_inches='tight')
plt.close(fig)


# Second plot: Optimization time
fig, ax = plt.subplots(figsize=figsize)
ax.plot(dims, time_, marker='o')

ax.set_xticks(xticks)
ax.set_yticks([1000, 1200, 1400, 1600, 1800])
ax.tick_params(axis='both', labelsize=fontsize)

ax.set_xlabel(r"$d$", fontsize=fontsize)
ax.set_ylabel(r"Time $[s]$", fontsize=fontsize)

ax.grid(True, which='both', linestyle='--', linewidth=0.7)  # add grid

plt.tight_layout()
fig.savefig("optimization_time.png", dpi=dpi, bbox_inches='tight')
fig.savefig("optimization_time.pdf", dpi=dpi, bbox_inches='tight')
plt.close(fig)





