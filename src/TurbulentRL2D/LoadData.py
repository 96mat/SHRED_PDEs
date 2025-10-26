# %%
from pyshred import DataManager, SHRED, SHREDEngine, SINDy_Forecaster
import torch
# %%
from the_well.benchmark.metrics import VRMSE
from the_well.data import WellDataset
from the_well.utils.download import well_download

device = "cuda"
base_path = "./datasets"  # path/to/storage
#%%
#well_download(base_path=base_path, dataset="turbulent_radiative_layer_2D", split="train")

# %%
#well_download(base_path=base_path, dataset="turbulent_radiative_layer_2D", split="valid")

# %%
dataset = WellDataset(
    well_base_path="DataSets/datasets/datasets",
    well_dataset_name="turbulent_radiative_layer_2D",
    well_split_name="train",
    n_steps_input=101,
    n_steps_output=0,
    use_normalization=True,
    normalization_type='RMS',
    full_trajectory_mode=False,
)
# %%
item = dataset[0]

list(item.keys())
# %%
item["input_fields"].shape
# %%
#item["output_fields"].shape
# %%
dataset.metadata.field_names
field_names = [
    name for group in dataset.metadata.field_names.values() for name in group
]
field_names
# %%
window_size = dataset.n_steps_input + dataset.n_steps_output

total_windows = 0
for i in range(dataset.metadata.n_files):
    windows_per_trajectory = (
        dataset.metadata.n_steps_per_trajectory[i] - window_size + 1
    )
    total_windows += (
        windows_per_trajectory * dataset.metadata.n_trajectories_per_file[i]
    )

print(total_windows)
print(len(dataset))
# %%
F = dataset.metadata.n_fields
# %%
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
x = dataset[42]["input_fields"]
x = rearrange(x, "T Lx Ly F -> F T Lx Ly")

fig, axs = plt.subplots(F, 4, figsize=(4 * 2.4, F * 1.2))

for field in range(F):
    vmin = np.nanmin(x[field])
    vmax = np.nanmax(x[field])

    axs[field, 0].set_ylabel(f"{field_names[field]}")

    for t in range(4):
        axs[field, -t].imshow(
            x[field, -t], cmap="RdBu_r", interpolation="none", vmin=vmin, vmax=vmax
        )
        axs[field, -t].set_xticks([])
        axs[field, -t].set_yticks([])

        axs[0, -t].set_title(f"$x_{t}$")


plt.tight_layout()
# %%
fig, axs = plt.subplots(F, 4, figsize=(4 * 2.4, F * 1.2))
for field in range(F):
    vmin = np.nanmin(x[field])
    vmax = np.nanmax(x[field])
    axs[field, 0].set_ylabel(f"{field_names[field]}")
    for t in range(4):
        im = axs[field, -t].imshow(x[field, -t], cmap="RdBu_r",
                                   interpolation="none", vmin=vmin, vmax=vmax)
        axs[field, -t].set_xticks([]); axs[field, -t].set_yticks([])
    # add colorbar to the rightmost subplot of this row
        fig.colorbar(im, ax=axs[field, -t], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.plot()
# %%
# To DO: usare SHRED sul dataset!!!
import os
print("Current working directory:", os.getcwd())
print("File exists:", os.path.exists('DataSets/datasets/datasets/turbulent_radiative_layer_2D/data/train/turbulent_radiative_layer_tcool_0.03.hdf5'))
# %%
import h5py
hf = h5py.File('/home/orion/SHRED_tests/DataSets/datasets/datasets/turbulent_radiative_layer_2D/data/train/turbulent_radiative_layer_tcool_0.03.hdf5', 'r')
data = np.array(hf)
print(list(hf.keys()))


# %%
grp = hf['t1_fields']
print(list(grp.keys()))
sub = grp['velocity']   # o 'velocity_x'
print(np.shape(list(sub)))  # N_sim x Nt x Nx x Ny x v_x,v_y
# %%
item["input_fields"].shape
# %%
data=torch.tensor(item["input_fields"])
data.size()
data[0,0,0,:] #density x pressione x v_x x v_y (per ogni punto 2D della mesh e ogni instante temporale)
# %%
manager = DataManager(
    lags=101, #       # 1 year of weekly history as input
    train_size=0.8,   # 0.68 + 0.22 + 0.10 = 1.0
    val_size=0.1,
    test_size=0.1,
)
# %%
manager.add_data(
    data=data,         # 3D array (time, lat, lon); time must be on axis 0
    id="TRL2D",          # Unique identifier for the dataset
    random=100,          # Randomly select 3 sensor locations
    stationary=[(100,140,1),(80,135,1),(60,140,1),\
                (40,135,1),(20,135,1),(70,135,1),\
                (90,150,1),(80,200,1),(60,140,1),\
                (40,135,1),(20,170,1),(70,160,1)],
    compress=False    # Keep original spatial resolution (no compression)
)

# %%

manager.sensor_summary_df
# %%
pos=manager.sensor_summary_df.iloc[:,3].to_numpy() # Retrieve position of random sensors
# %%
manager.sensor_measurements_df
# %%
train_dataset, val_dataset, test_dataset= manager.prepare()
# %%
import torch
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
from pyshred.models.latent_forecaster_models.lstm import *

LSTM_Custom = LSTM_Forecaster(
    lags=10,          # numero di step temporali di memoria
    hidden_size=64,  # neuroni per layer
    num_layers=10,    # profondità LSTM
)

shred = SHRED(
    sequence_model="LSTM",
    decoder_model="MLP",
    latent_forecaster=LSTM_Custom
)

"""shred.to(device)
print("Device:", device)
print(type(shred))"""
# %%
from time import perf_counter
start=perf_counter()

val_errors = shred.fit(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    num_epochs=200,
    patience=50,
    batch_size=128,
    lr=1e-4,
    verbose=True
)

#torch.from_numpy().cuda()

stop=perf_counter()
print(f'elapsed time = {stop-start:.3f}[s]')
print(val_errors.shape)
#.cpu().numpy()
plt.figure(figsize = (8,5))
plt.plot(val_errors, 'orange', linewidth = 3, label = 'Validation error')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
# %%
train_mse = shred.evaluate(dataset=train_dataset)
val_mse = shred.evaluate(dataset=val_dataset)
test_mse = shred.evaluate(dataset=test_dataset)
print(f"Train MSE: {train_mse:.3f}")
print(f"Val   MSE: {val_mse:.3f}")
print(f"Test  MSE: {test_mse:.3f}")
# %%
engine = SHREDEngine(manager, shred)
print(type(engine))
# %%
test_latent_from_sensors = engine.sensor_to_latent(manager.test_sensor_measurements)
print(f"to be sent to latent = {np.shape(manager.test_sensor_measurements)}")
print(np.shape(test_latent_from_sensors))
# %%
# generate latent states from validation sensor measurements
val_latents = engine.sensor_to_latent(manager.val_sensor_measurements)
print(f"val_latents = {np.shape(val_latents)}")
# seed the forecaster with the final `seed_length` latent states from validation
init_latents = val_latents[-shred.latent_forecaster.seed_length:] # seed forecaster with final lag timesteps of latent space from val
print(shred.latent_forecaster.seed_length)
print(f"init_latents = {np.shape(init_latents)}")
print(init_latents.shape)
# set forecast horizon to match the length of the test dataset
h = len(manager.test_sensor_measurements)
print(h)
# forecast latent states for the test horizon
test_latent_from_forecaster = engine.forecast_latent(h=h, init_latents=init_latents)
# %%
# decode latent space generated from sensor measurements (generated using engine.sensor_to_latent())
test_reconstruction = engine.decode(test_latent_from_sensors)

# decode latent space generated by the latent forecaster (generated using engine.forecast_latent())
test_forecast = engine.decode(test_latent_from_forecaster)

# %%
# ---------------- Train Evaluation ----------------
t_train = len(manager.train_sensor_measurements)
train_Y = {"TRL2D": data[0:t_train]}  # Ground truth segment
train_error = engine.evaluate(manager.train_sensor_measurements, train_Y)

# ---------------- Validation Evaluation ----------------
t_val = len(manager.val_sensor_measurements)
val_Y = {"TRL2D": data[t_train:t_train + t_val]}
val_error = engine.evaluate(manager.val_sensor_measurements, val_Y)

# ---------------- Test Evaluation ----------------
t_test = len(manager.test_sensor_measurements)
test_Y = {"TRL2D": data[-t_test:]}
test_error = engine.evaluate(manager.test_sensor_measurements, test_Y)

# ---------------- Print Results ----------------
print("---------- TRAIN ----------")
print(train_error)

print("\n---------- VALIDATION ----------")
print(val_error)

print("\n---------- TEST ----------")
print(test_error)# %%

# %%
import numpy as np

# Final ground truth frame from the test set
truth = data[-1]

# Extract final reconstructed frame (from sensor-based latents)
reconstructions = test_reconstruction["TRL2D"]
reconstruction = reconstructions[h - 1]

# Extract final forecasted frame (from forecasted latents)
forecasts = test_forecast["TRL2D"]
forecast = forecasts[h - 1]
print(f"shape: {forecast.shape}")
vmin_original=np.min(forecast)
vmax_original=np.max(forecast)
fig,ax=plt.subplots(1,1,figsize=(20,4));
im=ax.imshow(forecast[:,:,0],cmap='turbo')
ax.set_title("Density Field")
ax.set_ylabel("y-coordinate")
ax.set_xlabel("x-coordinate")
for i in range(len(manager.sensor_summary_df)):
    ax.scatter(pos[i][1],pos[i][0])

fig.colorbar(im,ax=ax,label='[A.U]',shrink=0.8,\
            )
plt.show()
# %%
