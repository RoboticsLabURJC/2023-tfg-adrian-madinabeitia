# 2023-tfg-adrian-madinabeitia

<div align="center">
    <img width=100px src="https://img.shields.io/badge/lenguage-%20python-blue" alt="Python">
    <img width=100px src="https://img.shields.io/badge/status-in%20process-orange" alt="In process">
</div>

## Index

- [2023-tfg-adrian-madinabeitia](#2023-tfg-adrian-madinabeitia)
  - [Index](#index)
  - [Folder distribution](#folder-distribution)
  - [Manual](#manual)
  - [Automation scripts](#automation-scripts)
    - [Dataset generator](#dataset-generator)
    - [Training script](#training-script)

---
---

## Folder distribution
We follow the following [repository distribution](https://medium.com/analytics-vidhya/folder-structure-for-machine-learning-projects-a7e451a8caaa) for machine learning projects, along with the basic ROS structure.

The **drone_sim_driver** contains nodes and scripts related to the training and usage of the neural network for a simulated environment, while **tello_driver** contains the nodes for the DJI Tello drone.

---
---

## Manual

1. For launch only the world

```bash
ros2 launch drone_driver as2_sim_circuit.launch.py world:=../worlds/simple_circuit.world yaw:=1.0
```

2. Launch the expert pilot

```bash
ros2 ros2 launch drone_driver expertPilot.launch.py out_dir:=TRACES_DIR trace:=BOOL
```

## Automation scripts

### Dataset generator

1. Go to utils directory

```bash
cd PATH_TO_WORKSPACE/src/drone_driver/utils
```

2. Open a tmux session

```bash
tmux
```

3. Run the script

```bash
./generateDataset.sh TIME_RECORDING DATASET_PATH
```

* **Time recording:** Int -> Time recording the topics of the drone.
* **Dataset path** String -> Path where the dataset and profiling data are located.will be save.

---

### Training script

1. Go to utils directory

```bash
cd PATH_TO_WORKSPACE/src/drone_driver/utils
```

2. Open a tmux session

```bash
tmux
```

3. Run the script

```bash
/trainAndCheck.sh DATASET_PATH PROFILING_PATH MODEL_NAME
```

* **Dataset path:** String -> Route where the dataset is located.
* **Profiling path:** String -> Path where the dataset and profiling data are located.
* **Model path:** String -> Route where the network model will be saved.
