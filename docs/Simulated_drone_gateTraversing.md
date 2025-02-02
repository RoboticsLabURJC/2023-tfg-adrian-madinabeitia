# Gate traversing application

## Generate world
For creating a new gate circuit you can modify the next script with some functions for creating new gate circuits:

```bash
python3 drone_sim_driver/src/generateGateWorld.py
```

Then launch the world with:

```bash
ros2 launch drone_sim_driver as2_sim_gates.launch.py
```

## RemotePilot

For launching the remote pilot:

```bash
ros2 launch drone_sim_driver remoteContol.launch.py net_dir:=../../models_tfg_drone_simulation/gateTravesing/gate_constant_altitude_v1.tar dp_dir:=../../models_tfg_drone_simulation/gateTravesing/gate_full_control_v1.tar
```
The arguments are: 
* **out_dir:** Directory were the logs will be stored.
* **net_dir:** Pilot net neural network path
* **dp_dir:** Deep pilot (control in Z) neural network path

If net_dir is none, the remote controller won't have neural control. If the dp_dir is not set, the altitude controll will be manual. 

<div align="center">
<img width=700px src="https://roboticslaburjc.github.io/2023-tfg-adrian-madinabeitia/assets/images/post22/controller.png" alt="explode"></a>
</div>


## Rosbags to general dataset 

Once you have the raw dataset with rosbags, the next step is going to **/drone_sim_driver/src/features** and run the next script: 

```bash
python3 rosbag2generalDataset.py --rosbags_path ROSBAGS_PATH
```

This will take all the rosbags in the folder and create two directories with a standard format dataset.

- **Frontal images** It contains all the images captured with the drone.
- **Labels:** It contains the velocities associated to each image.

**Note:**
It's recommended see the dataset images to delete failures or compensate the dataset 

## Train the neural network
Once the dataset is generated, the next step is train the neural network:

```bash
## model: Selects the model the user wants to train
## resume: Resume the training or creates a new one
## network: Network path
## Supports 4 dataset paths (--dp1, --dp2, --dp3, --dp4)

python3 drone_sim_driver/src/models/train.py --model [pilotNet|deepPilot] --resume [bool] --network PATH_TO_NEW_NETWORK.tar --dp1 STD_DATASET_PATH1
```

While the training script is running is recommended to execute the script **netCheckpointSaver.sh** for saving different wights distributions.

```bash
drone_sim_driver/utils/netCheckpointSaver.sh MODEL_PATH.tar
```

Once the training is done you can test the trained neural network with:

