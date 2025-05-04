# Neur2SP

Main steps:

1. Generating instances
2. Generating training data: solve
   1. NN-P
   2. NN-E
3. Training neural networks
4. Evaluating models in extensive form

## 1 Generating instances

### 1.1 Code for generating instances

This will generate instances and save them in `data/[PROBLEM]/inst_*.pkl`.

This will only generate **instances**, which is a `dict`. Scenarios are not generated.

```shell
python -m nsp.scripts.run_dm --mode GEN_INSTANCE --problem [PROBLEM]
```

`CFLP`

- `inst_f10_c10_r2.0_iss1_bt1_nsp10000_nse5000_sd7.pkl`
  - `f`: facilities
  - `c`: customers
  - `r`: ratio of customers to facilities
  - `iss`: integer second stage
  - `bt`: bound tightening
  - `nsp`: number of samples for NN-P
  - `nse`: number of samples for NN-E (**each sample is averaged over at most 100 scenarios**)
  - `sd`: seed

`INVP`

- `inst_fvC_svB_n1_tmi0_sd777.pkl`
  - `fv`: first-stage variables
  - `sv`: second-stage variables
  - `n`: number of instances

`PP`

- `inst_n1_sd7.pkl`
  - `n`: number of instances
  - `sd`: seed

### 1.2 Details

`nsp/two_sp/`

- `inst` is a dict of (first-stage) problem data

`nsp/dm/`

## 2 Generating training data: solving optimization models

This will create `data/[PROBLEM]/ml_data_*.pkl`.

### 2.1 Code for generating training data

Generating NN-P data

```bash
python -m nsp.scripts.run_dm --mode GEN_DATASET_P --n_procs {n_cpus} --problem {problem}
```

Generating NN-E data

```bash
python -m nsp.scripts.run_dm --mode GEN_DATASET_E --n_procs {n_cpus} --problem {problem}
```

### 2.2

`nsp/two_sp/`

## 3 Training

```bash
python runner.py --problems {problem} --train_nn_p 1
python runner.py --problems {problem} --train_nn_e 1
```

### Linear regression model

```bash
python -m nsp.scripts.train_model --model lr --problem {problem} 
```

### NN-P model

```bash
python -m nsp.scripts.train_model --model_type nn_p --problem {problem} \
--hidden_dims \
--lr \
--dropout \
--optimizer \
--batch_size \
--wt_lasso \
--wt_ridge \
--log_freq \
--n_epochs \
--use_wanbd \
```

### NN-E model

```bash
python -m nsp.scripts.train_model --model_type nn_e --problem {problem} \
--embed_hidden_dim \
--embed_dim1 \
--embed_dim2 \
--relu_hidden_dim \
--agg_type \
--lr \
--dropout \
--optimizer \
--batch_size \
--wt_lasso \
--wt_ridge \
--log_freq \
--n_epochs \
--use_wanbd \
```

### Training data explanation

#### CFLP

- `ml_data_p_*.pkl` is a dict with keys `tr_data`, `val_data`, `data`, `total_time`. `*_data` is a list of samples. Each sample is a `dict` with keys:
  - `x`: `dict` of first-stage solutions, length is n_customers
  - `obj`: objective value
  - `features`: feature vector of first-stage decisions and scenarios. For example, for CLFP, the scenario is the the random demands
  - `demands`: an `ndarray` with shape (n_customers,), representing the demands in one scenario
  - `time`: time to solve the scenario
- `ml_data_e_*.pkl` is a dict with keys `tr_data`, `val_data`, `data`, `total_time`, `mp_time`. `*_data` is a list of samples. Each sample is a `dict` with keys:
  - `x`: `dict` of first-stage solutions, length is n_customers
  - `obj_vals`: `list` of objective values, length is the number of scenarios
  - `obj_mean`: mean of `obj_vals`
  - `demands`: `list` of n_scenarios `ndarray`, each array has shape (n_customers,)
  - `time`: sum of `times`
  - `times`: `list` of n_scenarios `float`, each element is the time to solve the scenario

#### PP

- `ml_data_e_*.pkl` is a dict with 4 keys `tr_data`, `val_data`, `data`, `total_time`. `*_data` is a list of samples. Each sample is a `dict` with keys:
  - `sol`
  - `scenario`
  - `n_scenarios`
  - `obj_mean`
  - `obj_vals`
  - `scenario_ids`: `list` of `int`
  - `obj_probs`: `ndarray` of `float` with shape (n_scenarios,)
  - `time`
  - `times`
- `ml_data_p_*.pkl` is a dict with 4 keys `tr_data`, `val_data`, `data`, `total_time`. `*_data` is a list of samples. Each sample is a `dict` with keys:
  - `sol`: `dict` of first-stage solutions
  - `obj`: `float`
  - `time`: `float`
  - `scenario_id`: `int`
  - `features`: `list` with length of 19

## 4 Get best model

```bash
python -m nsp.scripts.get_best_model --problem {problem} --model [nn_e|nn_p]
```

## 5 Evaluation

### 5.1 Generate scenario sets and test sets

```python
scenarios, test_sets = get_scenario_and_test_sets(problem)
```

### 5.2 Evaluate neural network models

#### Load instances (`dict` of problem data)

#### Create 2SP problem

#### Load trained model

#### Get MIP approximator

`FacilityLocationProblemApproximator` as an exmaple

- get_master_mip
  - create the first-stage problem: 10 binary variables and the objective with no constraints

- get_first_stage_variables

get_scenario_embedding

get_first_stage_solution

mipper: `Net2MIPExpected` and `Net2MIPPerScenario`

converting nn to mip

get_mip

### 5.3 Evaluate linear regression model

```bash
python -m nsp.scripts.evaluate_model --problem {problem} --model lr --n_scenarios {scenario} --test_set {test_set} --n_procs {args.n_cpus}
```
