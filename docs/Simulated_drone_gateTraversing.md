## Follow line application

You can try the expert pilot with the following command once the world is launched.

```bash
ros2 launch drone_sim_driver expertPilot.launch.py
```

If you want to generate a full dataset, go to the path **/drone_sim_diver/utils** and run:

```bash
tmux # First init a tmux session 
./generateDataset.sh <record time> <output directory>
```

This command will run the expert pilot in different circuits, recording with rosbags all the needed measures. The record time is the time the drone will be racing in each circuit and the output directory where the data will be storaged. \\

Once you have the raw dataset with rosbags, the next step is going to **/drone_sim_driver/src/features** and run the next script: 

```bash
python3 rosbag2generalDataset.py --rosbags_path ROSBAGS_PATH
```

This will take all the rosbags in the folder and create two directories with a standard format dataset.

- **Frontal images** It contains all the images captured with the drone.
- **Labels:** It contains the velocities associated to each image.

**Note:**
It's recommended see the dataset images to delete failures or compensate the dataset 

Once the dataset is generated, the next step is train the neural network:

```bash
## model: Selects the model the user wants to train
## resume: Resume the training or creates a new one
## network: Network path
## Supports 4 dataset paths (--dp1, --dp2, --dp3, --dp4)

python3 models/train.py --model [pilotNet|deepPilot] --resume [bool] --network PATH_TO_NEW_NETWORK.tar --dp1 STD_DATASET_PATH1
```

While the training script is running is recommended to execute the script **netCheckpointSaver.sh** for saving different wights distributions.

```bash
./utils/netCheckpointSaver.sh MODEL_PATH.tar
```

Once the training is done you can test the trained neural network with:

```bash
ros2 launch drone_sim_driver neuralPilot.launch.py trace=[BOOL] out_dir=[OUT_DIR] network_path=[NET_PATH]
```

If you want to run this automatically you can run:

```bash
tmux 
./utils/trainAndCheck.sh [dataset_dir] [output_dir] [model_name]
```

Where:

* **Dataset dir:** Is the path where your dataset is stored 
* **Output dir:** Where the logs will be saved and the paths followed by the different models trained. 
* **Model name:** The trained models name which will be saved.

If you want to make ALL the process run:

```bash
tmux
./utils/generateNeuralPilot.sh [output_dir] [model_name]
```
### Getting the best model

To speed up the model selection process for this application, the following script was created. It executes a folder containing 𝑛 models and generates an image of the result for each on as the images below:

<div align="center">
<img width=350px src="https://roboticslaburjc.github.io/2023-tfg-adrian-madinabeitia/assets/images/post17/result.png" alt="explode"></a>
<img width=340px src="https://roboticslaburjc.github.io/2023-tfg-adrian-madinabeitia/assets/images/post18/montmelo.png" alt="explode"></a>
</div>