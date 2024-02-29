# 2023-tfg-adrian-madinabeitia

## Index

---
---

<!-- ## Installation
Versions
Ros Humble
Gazebo 11.10.2 -->

<!-- Instalación de tmux -->

<!-- ---
--- -->
## Manually
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

2. Open a tmux sesion

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

2. Open a tmux sesion

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