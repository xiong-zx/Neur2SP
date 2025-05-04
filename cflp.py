# %%
from types import SimpleNamespace
import torch
import numpy as np
import pandas as pd
import gurobipy as gp

import nsp.params as params
from nsp.approximator.cflp import FacilityLocationProblemApproximator
from nsp.model2mip.net2mip import Net2MIPExpected, Net2MIPPerScenario
from nsp.model2mip.lr2mip import LR2MIP
from nsp.two_sp.cflp import FacilityLocationProblem
from nsp.utils.cflp import get_path as cflp_get_path
from nsp.utils import load_instance
# %%

args = SimpleNamespace(
    problem="cflp_10_10",
    n_scenarios=100,
    model_type="nn_e",
    time_limit=3600,
    mip_gap=0.01,
    mip_threads=1,
    n_procs=1,
    test_set=0
    )
problem_str = f"s{args.n_scenarios}_ts{args.test_set}"
cfg = getattr(params, args.problem)
inst = load_instance(args, cfg)
two_sp = FacilityLocationProblem(inst)
model_path = cflp_get_path(data_path=cfg.data_path, cfg=cfg, ptype=args.model_type, suffix=".pt")
trained_model = torch.load(model_path)
trained_model.dropout = False
if args.model_type == "nn_e":
    mipper = Net2MIPExpected
elif args.model_type == "nn_p":
    mipper = Net2MIPPerScenario
elif args.model_type == "lr":
    mipper = LR2MIP

approximator = FacilityLocationProblemApproximator(two_sp=two_sp, model=trained_model, model_type=args.model_type, mipper=mipper)
log_dir = cflp_get_path(data_path=cfg.data_path, cfg=cfg, ptype=f"grb_log_{args.model_type}_{problem_str}", suffix=".log", as_str=True)

first_stage_mip = approximator.get_master_mip()  # gurobi model
first_stage_vars = approximator.get_first_stage_variables(first_stage_mip)  # gurobi variables
scenarios = np.array(approximator.two_sp.get_scenarios(args.n_scenarios, args.test_set))
x_scen = torch.from_numpy(scenarios).float()
x_scen = torch.reshape(x_scen, (1, x_scen.shape[0], x_scen.shape[1]))
scenario_embedding = approximator.model.embed_scenarios(x_scen).detach().numpy().reshape(-1)


instantiated_mipper = approximator.mipper(
    first_stage_mip=first_stage_mip, 
    first_stage_vars=first_stage_vars, 
    network=trained_model, 
    scenario_embedding=scenario_embedding
    )

second_stage_surrogate = instantiated_mipper.get_second_stage()
master_surrogate = instantiated_mipper.get_mip()

second_stage_surrogate.setAttr("ModelName", f"{args.model_type}_{args.problem}_second_stage_surrogate")
second_stage_surrogate.update()
master_surrogate.setAttr("ModelName", f"{args.model_type}_{args.problem}_master_surrogate")
master_surrogate.update()

# %%
# analyze surrogate model statisitcs

import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gp


def reindex_variables_by_type(surrogate_model: gp.Model):
    binary_vars = []
    continuous_vars = []
    
    for var in surrogate_model.getVars():
        if var.VType == gp.GRB.BINARY:
            binary_vars.append(var)
        elif var.VType == gp.GRB.CONTINUOUS:
            continuous_vars.append(var)
    
    reindexed_vars = binary_vars + continuous_vars
    
    var_index_map = {var.VarName: i for i, var in enumerate(reindexed_vars)}
    return var_index_map, len(binary_vars), len(continuous_vars)


def visualize_nnz_by_vtype(surrogate_model: gp.Model):
    var_index_map, num_binary, num_continuous = reindex_variables_by_type(surrogate_model)
    num_constraints = surrogate_model.numConstrs
    num_vars = len(var_index_map)
    
    binary_matrix = np.zeros((num_constraints, num_vars))
    continuous_matrix = np.zeros((num_constraints, num_vars))

    for i, constr in enumerate(surrogate_model.getConstrs()):
        row = surrogate_model.getRow(constr)
        for j in range(row.size()):
            var = row.getVar(j)
            var_index = var_index_map[var.VarName]
            if var.VType == gp.GRB.BINARY:
                binary_matrix[i, var_index] = 1
            elif var.VType == gp.GRB.CONTINUOUS:
                continuous_matrix[i, var_index] = 1

    plt.figure(figsize=(12, 10))
    plt.spy(binary_matrix, markersize=2, color='red', label='Binary Variables')
    plt.spy(continuous_matrix, markersize=2, color='blue', label='Continuous Variables')
    plt.xlabel('Variables')
    plt.ylabel('Constraints')
    plt.title(f'NNZ Pattern of {surrogate_model.ModelName}')
    plt.legend(loc='upper right')
    plt.show()

# visualize_nnz_by_vtype(master_surrogate)
# visualize_nnz_by_vtype(second_stage_surrogate)
# %%
# optimize the master / second-stage surrogate model
mip_model = second_stage_surrogate.copy()

# reset the variable types and bounds
fs_vars = [var for var in mip_model.getVars() if "x_in" in var.VarName]
for var in fs_vars:
    var.VType = gp.GRB.CONTINUOUS


mip_model.setParam("OutputFlag", 0)
mip_model.update()

mip_model.reset()
mip_model.optimize()
optimal_solution = {var.VarName: var.X for var in mip_model.getVars() if "x_in" in var.VarName}
print("Optimal Solution:", optimal_solution)

# %%
perturbations = [0.01, 0.05, -0.01, -0.05]
unique_obj_vals = set()
piecewise_linear_regions = {}

for var_name, var_value in optimal_solution.items():
    for perturb in perturbations:
        perturbed_solution = optimal_solution.copy()
        perturbed_solution[var_name] = var_value + perturb

        # Fix variables to the perturbed values
        for v in fs_vars:
            v.lb = v.ub = perturbed_solution[v.VarName]

        # Re-optimize the model with the perturbed solution
        mip_model.reset()
        mip_model.optimize()

        # Store the objective value and check if it defines a new piece
        obj_val = mip_model.ObjVal
        unique_obj_vals.add(obj_val)
        
        x_fs = torch.tensor(list(perturbed_solution.values()), dtype=torch.float32).unsqueeze(0)
        nn_output = trained_model(x_fs, x_scen).detach().item()
        
        piecewise_linear_regions[obj_val] = {
            "nn_output": nn_output,
            "milp_output": obj_val
        }
        
for obj_val, outputs in piecewise_linear_regions.items():
    print(f"NN Output: {outputs['nn_output']:.3f}, MILP Output: {outputs['milp_output']:.3f}, Diff: {outputs['nn_output'] - outputs['milp_output']:.3f}")

# %%
n_tries = 10
results = {}
for _ in range(n_tries):
    random_values = np.random.uniform(low=0.0, high=1.0, size=len(fs_vars))
    for var, value in zip(fs_vars, random_values):
        var.lb = var.ub = value

    # Re-optimize the model with the perturbed solution
    mip_model.reset()
    mip_model.optimize()

    # Store the objective value and check if it defines a new piece
    obj_val = mip_model.ObjVal
    
    x_fs = torch.tensor(random_values, dtype=torch.float32).unsqueeze(0)
    nn_output = trained_model(x_fs, x_scen).detach().item()
    
    results[obj_val] = {
        "nn_output": nn_output,
        "milp_output": obj_val
    }
    print(f"NN Output: {nn_output:.3f}, MILP Output: {obj_val:.3f}, Diff: {nn_output - obj_val:.3f}")

# %%
for v in mip_model.getVars():
    print(v.VarName, v.X)
# %%
# # find pool solutions around the optimal solution
# mip_model.setParam("PoolSolutions", 100)
# mip_model.setParam("PoolSearchMode", 2)
# num_solutions = mip_model.SolCount
# print(f"Number of solutions found: {num_solutions}")

# binary_vars = [var for var in mip_model.getVars() if var.VType == gp.GRB.BINARY]
# num_binary_vars = len(binary_vars)
# solutions_array = np.zeros((num_solutions, num_binary_vars+1))
# for i in range(num_solutions):
#     mip_model.setParam("SolutionNumber", i)
#     for j, var in enumerate(binary_vars):
#         solutions_array[i, j] = var.Xn
#     solutions_array[i, -1] = mip_model.PoolObjVal

# cols = [var.VarName for var in binary_vars] + ["ObjVal"]

# df = pd.DataFrame(solutions_array, columns=cols)
# for col in cols:
#     if col == "ObjVal":
#         df[col] = df[col].astype(float)
#     else:
#         df[col] = df[col].astype(int)
# activation_count = df.iloc[:, 10:-1].sum(axis=1)
# activation_count.name = "ActivationCount"
# joined_df = pd.concat([df, activation_count], axis=1)
# joined_df.sort_values(by="ActivationCount", ascending=True, inplace=True)
# joined_df.to_csv("mip_model.csv", index=False)

# activation_pattern = df.iloc[:, 10:-1].copy()
# duplicated_activation = activation_pattern[activation_pattern.duplicated()]
# duplicated_activation

# input_pattern = df.iloc[:, 0:10].copy()
# duplicated_input = input_pattern[input_pattern.duplicated()]
# duplicated_input
# %%
