# 2023-tfg-adrian-madinabeitia

<div align="center">
<img width=400px src="https://roboticslaburjc.github.io/2023-tfg-adrian-madinabeitia/assets/images/readme/drone.png" alt="explode">
</div>

<br>
<div align="center">
<img width=100px src="https://img.shields.io/badge/lenguage-python-orange" alt="explode"></a>
<img width=100px src="https://img.shields.io/badge/status-In Progress-yellow" alt="explode"></a>

<img width=100px src="https://img.shields.io/badge/license-Apache-blue" alt="explode">

</div>

This work involves the application of neural networks to program drones and thus test their performance in such systems. It will be implemented in two applications: initially, a line-following application, where the drone must follow a line in various circuits with the least possible error; and simultaneously, it will have to cross windows, which will consist of a mixed control where the drone will be teleoperated, but if the pilot wishes, the neural network will take control of the drone and attempt to cross the gates that appear in its field of vision.

Usually, classical programming algorithms are used for drone control. The control of these drones is typically done with cameras and various sensors that require intensive processing. This means that the drone needs a processing unit as a payload to handle all the data, or alternatively, the data must be processed at the ground station, which adds a delay in communications and potential errors that could affect a system that is critical to operate in real-time.

## Index

- [2023-tfg-adrian-madinabeitia](#2023-tfg-adrian-madinabeitia)
  - [Index](#index)
  - [Repository distribution](#repository-distribution)
  - [Drone platforms](#drone-platforms)
  - [Training](#training)
  - [Follow line application](#follow-line-application)
    - [Algorithmic expert pilot](#algorithmic-expert-pilot)
    - [Getting the best model](#getting-the-best-model)
  - [Gate travesing application](#gate-travesing-application)
    - [1. Constant Altitude](#1-constant-altitude)
    - [2. Variable Altitude](#2-variable-altitude)
    - [Remote pilot control](#remote-pilot-control)
  - [Licencia](#licencia)

---

---

## Repository distribution

This project has two packages: one contains all the drone platforms, and the other contains the behaviors programmed into the drone_behaviors packages. The drone is able to perform two applications: the line-following application and the gate-traversing application.

We follow the following [repository distribution](https://medium.com/analytics-vidhya/folder-structure-for-machine-learning-projects-a7e451a8caaa) for machine learning projects, along with the basic ROS structure.

---

---

## Drone platforms

We used de [aerostack2](https://github.com/aerostack2/aerostack2) platforms to utilize all the programmed behaviors in multiple drones. The platform launchers are in the package drone_platforms This package was created to separate the platform used from the behavior, thus allowing the developed software to be used in real drones in the future.

1. For launching follow line world:

```bash
cd /PATH_TO_PACKAGE/drone_platforms/launch

ros2 launch drone_platforms as2_sim_circuit.launch.py world:=/PATH_TO_WORLD/NAME.world yaw:=3.14
```

2. For launching the gates world:

```bash
cd /PATH_TO_PACKAGE/drone_platforms/src

# Execute if you want to change the scenario
python3 python3 generateGateWorld.py

ros2 ros2 launch drone_platforms as2_sim_gates.launch.py
```

---
---

## Training

Bot applications follows the same training scripts so that is how the basic training script is called:

0. Access to the src drone_behaviors directory:

```bash
cd /PATH_TO_PACKAGE/drone_behaviors/src/
```

1. Converting rosbags to standard format dataset:

```bash
python3 features/rosbag2generalDataset.py --rosbags_path ROSBAGS_PATH
```

1. Train a neural network with the standard format dataset, this script will receive the path of the dataset and will train the model specified in PATH_TO_NEW_NETWORK.tar:

```bash

## model: Selects the model the user wants to train
## resume: Resume the training or creates a new one
## network: Network path
## Supports 4 dataset paths (--dp1, --dp2, --dp3, --dp4)

python3 models/train.py --model [pilotNet|deepPilot] --resume [true or false] --network PATH_TO_NEW_NETWORK.tar --dp1 STD_DATASET_PATH1
```

2. Back-up script for saving different weights distribution in one training:

```bash
./utils/netCheckpointSaver.sh MODEL_PATH.tar
```

---
---

## Follow line application

For the imitation learning validation in this exercise, we collected a dataset with an algorithmic pilot, the dataset is available in the next [repository](https://huggingface.co/datasets/Adrimapo/tfg_2024_drone_simulation/tree/main). Clicking the next image you can watch the demo video:

[![Youtube video](https://img.youtube.com/vi/jJ4Xdin1gg4/0.jpg)](https://www.youtube.com/watch?v=jJ4Xdin1gg4)

### Algorithmic expert pilot

For this application, the [Pilot_Net](https://github.com/lhzlhz/PilotNet) network was used to control linear and angular velocity. The expert pilot is launched as:

```bash
#! First launch the platform

## out_dir = Path where all the execution data will be stored.
## trace = Shows the filtered image in another window 
ros2 launch drone_behaviors expertPilot.launch.py out_dir:=PATH trace_arg:=BOOL
```

If you want to launch the neural pilot instead the expert pilot:

```bash
#! First launch the platform

## out_dir = Path where all the execution data will be stored.
## trace = Shows the filtered image in another window 
## network_path = Model used for the velocity inference 
ros2 launch drone_behaviors neuralPilot.launch.py out_dir:=PATH trace_arg:=BOOL network_path:=NET_PATH
```

In this application, automation scripts were created to facilitate training. To repeatedly launch the expert pilot while recording the dataset:

```bash
## TIME_RECORDING = Integer that sets the recording time for each circuit.
## OUT_DIR = Path where all the execution data will be stored.

cd /PATH_TO_PACKAGE/drone_behaviors/utils
./generateDataset.sh TIME_RECORDING OUT_DIR

```

After that you can use the automate training:

```bash
./generateNeuralPilot OUT_DIR MODEL_DIR 
```

For forcing the exit of a tmux session:

```bash
./end_tmux.sh SESSION_NAME
```

### Getting the best model

To speed up the model selection process for this application, the following script was created. It executes a folder containing 𝑛 models and generates an image of the result for each on as the images below:

<div align="center">
<img width=350px src="https://roboticslaburjc.github.io/2023-tfg-adrian-madinabeitia/assets/images/post17/result.png" alt="explode"></a>
<img width=340px src="https://roboticslaburjc.github.io/2023-tfg-adrian-madinabeitia/assets/images/post18/montmelo.png" alt="explode"></a>
</div>

---

---

## Gate travesing application

In this application we collected two datasets, one more generic for the pilotNet training (available in the next [repository](https://huggingface.co/datasets/Adrimapo/tfg_2024_drone_simulation/tree/main) where it's all the verified dataset of the project)

### 1. Constant Altitude

In this video, you can see the drone maintaining a constant altitude:

[![Watch the video](https://img.youtube.com/vi/l-WyA2C-4I4/0.jpg)](https://www.youtube.com/watch?v=l-WyA2C-4I4)

### 2. Variable Altitude

In this video, you can see the drone adjusting its altitude:

[![Watch the video](https://img.youtube.com/vi/Q1zBNXdW7Ns/0.jpg)](https://www.youtube.com/watch?v=Q1zBNXdW7Ns)

### Remote pilot control

Since this expert pilot will be autopiloted, the following control configuration was chosen to teleoperate the drone:

<div align="center">
<img width=700px src="https://roboticslaburjc.github.io/2023-tfg-adrian-madinabeitia/assets/images/post22/controller.png" alt="explode"></a>
</div>

Launch this expert pilot with the following command:

```bash
#! First launch the platform

## out_dir = Path where all the execution data will be stored.
## net_dir = PilotNet model path 
## deep_dir = DeepPilot model path

ros2 launch drone_behaviors remoteControl.launch.py out_dir:=PATH net_dir:=PILOT_NET_MODEL deep_dir:=DEEP_PILOT_MODEL
```

---

---

## Licencia

<a rel="license" href=<https://www.apache.org/licenses/LICENSE-2.0>><img alt="Apache License" style="border-width:0" src=<https://www.apache.org/img/asf-estd-1999-logo.jpg> /></a><br/> </a><br/>This work is licensed under a <a rel="license" href=<https://www.apache.org/licenses/LICENSE-2.0>Apache> license 2.0
