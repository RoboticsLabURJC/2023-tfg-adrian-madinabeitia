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
  - [Drone scenarios](#drone-scenarios)
  - [Remote pilot control](#remote-pilot-control)
  - [Follow line simulation](#follow-line-simulation)
  - [Gate traversing simulation](#gate-traversing-simulation)
    - [1. Constant Altitude](#1-constant-altitude)
    - [2. Variable Altitude](#2-variable-altitude)
  - [License](#license)



## Repository distribution

This project has two packages: one contains all the drone platforms, and the other contains the behaviors programmed into the drone_behaviors packages. The drone is able to perform two applications: the line-following application and the gate-traversing application.

We follow the following [repository distribution](https://medium.com/analytics-vidhya/folder-structure-for-machine-learning-projects-a7e451a8caaa) for machine learning projects, along with the basic ROS structure.

The models and the dataset occupy significant space, so it was decided to use [hugging Face](https://huggingface.co/) to host all this data. Thus, a repository was created for the [dataset](https://huggingface.co/datasets/Adrimapo/dataset_tfg_drone_simulation) and another for the [models](https://huggingface.co/Adrimapo/models_tfg_drone_simulation)



## Drone scenarios

We used de [aerostack2](https://github.com/aerostack2/aerostack2) platforms to utilize all the programmed behaviors in multiple drones. The platform launchers are in the package drone_platforms This package was created to separate the platform used from the behavior, thus allowing the developed software to be used in real drones.

1. For launching follow line world:

```bash
# Launch default circuit
ros2 launch drone_sim_driver as2_sim_circuit.launch.py

# Launching with arguments
ros2 launch drone_sim_driver as2_sim_circuit.launch.py world:=/PATH_TO_WORLD/NAME.world yaw:=3.14
```

2. For launching the gates world:

```bash
# Execute if you want to change to a random scenario
python3 python3 generateGateWorld.py

ros2 ros2 launch drone_sim_driver as2_sim_gates.launch.py
```

## Remote pilot control

Since this expert pilot will be autopiloted, the following control configuration was chosen to teleoperate the drone:

<div align="center">
<img width=700px src="https://roboticslaburjc.github.io/2023-tfg-adrian-madinabeitia/assets/images/post22/controller.png" alt="explode"></a>
</div>


```bash
#! First launch the platform

## out_dir = Path where all the execution data will be stored.
## net_dir = PilotNet model path 
## deep_dir = DeepPilot model path

ros2 launch drone_behaviors remoteControl.launch.py out_dir:=PATH net_dir:=PILOT_NET_MODEL deep_dir:=DEEP_PILOT_MODEL
```


## Follow line simulation

Click the nex [link](./docs/Simulated_drone_followLine.md) for the guide of the follow line application.

For the imitation learning validation in this exercise, we collected a dataset with an algorithmic pilot. Clicking the next image you can watch the demo video:

[![Youtube video](https://img.youtube.com/vi/jJ4Xdin1gg4/0.jpg)](https://www.youtube.com/watch?v=jJ4Xdin1gg4)

---
## Gate traversing simulation

In this application we collected two datasets, one more generic for the pilotNet training. The results of this application where successful too: 

### 1. Constant Altitude

In this video, you can see the drone maintaining a constant altitude:

[![Watch the video](https://img.youtube.com/vi/l-WyA2C-4I4/0.jpg)](https://www.youtube.com/watch?v=l-WyA2C-4I4)

### 2. Variable Altitude

In this video, you can see the drone adjusting its altitude:

[![Watch the video](https://img.youtube.com/vi/Q1zBNXdW7Ns/0.jpg)](https://www.youtube.com/watch?v=Q1zBNXdW7Ns)

---
---

## License

<a rel="license" href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="Apache License" style="border-width:0" src="https://www.apache.org/img/asf-estd-1999-logo.jpg" /></a><br/> </a><br/>This work is licensed under a <a rel="license" href="https://www.apache.org/licenses/LICENSE-2.0">Apache license 2.0