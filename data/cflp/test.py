# %%
import pickle
with open("ml_data_p_f10_c10_r2.0_iss1_bt1_nsp10000_nse5000_sd7.pkl", "rb") as f:
    p_data = pickle.load(f)

with open("ml_data_e_f10_c10_r2.0_iss1_bt1_nsp10000_nse5000_sd7.pkl", "rb") as f:
    e_data = pickle.load(f)
# %%
import pickle as pkl
import matplotlib.pyplot as plt

model = "nn_e"

with open(f"{model}_results_f10_c10_r2.0_iss1_bt1_nsp10000_nse5000_sd7.pkl", "rb") as f:
    results_e = pkl.load(f)
    
epochs = list(range(len(results_e['tr_results']['mse'])))
tr_mse = results_e['tr_results']['mse']
tr_mae = results_e['tr_results']['mae']
tr_mape = results_e['tr_results']['mape']
val_mse = results_e['val_results']['mse']
val_mae = results_e['val_results']['mae']
val_mape = results_e['val_results']['mape']

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# mse
axes[0].plot(epochs, tr_mse, label='tr_mse')
axes[0].plot(epochs, val_mse, label='val_mse')
axes[0].set_ylabel('MSE')
axes[0].set_yscale('log')
axes[0].legend()

# mae
axes[1].plot(epochs, tr_mae, label='tr_mae')
axes[1].plot(epochs, val_mae, label='val_mae')
axes[1].set_ylabel('MAE')
axes[1].set_yscale('log')
axes[1].legend()

# mape
axes[2].plot(epochs, tr_mape, label='tr_mape')
axes[2].plot(epochs, val_mape, label='val_mape')
axes[2].set_xlabel('Epochs')
axes[2].set_ylabel('MAPE')
axes[2].set_yscale('log')
axes[2].legend()

axes[0].set_title(f'{model.upper()} Training and Validation Results',loc="center")
plt.tight_layout()
plt.show()
# %%
