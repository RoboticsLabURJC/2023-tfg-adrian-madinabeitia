# 2023-tfg-adrian-madinabeitia

<div align="center">
<img width=700px src="https://roboticslaburjc.github.io/2023-tfg-adrian-madinabeitia/assets/images/readme/drone.png" alt="explode">
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
  - [Follow line application](#follow-line-application)
    - [Algorithmic expert pilot](#algorithmic-expert-pilot)
    - [Getting the best model](#getting-the-best-model)
    - [Video](#video)
  - [Gate travesing application](#gate-travesing-application)
    - [Remote pilot control](#remote-pilot-control)
    - [Validation](#validation)
  - [Licencia](#licencia)

  
---

---

## Repository distribution

This project has two packages: one contains all the drone platforms [drone_plataforms](), and the other contains the behaviors programmed into the drone [drone_behaviors](). The drone is able to perform two applications: the line-following application and the gate-traversing application.

 
We follow the following [repository distribution](https://medium.com/analytics-vidhya/folder-structure-for-machine-learning-projects-a7e451a8caaa) for machine learning projects, along with the basic ROS structure.

 

---

---

 

## Drone platforms

We used de [aerostack2](https://github.com/aerostack2/aerostack2) platforms to utilize all the programmed behaviors in multiple drones. The platform launchers are in the package [drone_plataforms](). This package was created to separate the platform used from the behavior, thus allowing the developed software to be used in real drones in the future.

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

## Follow line application

### Algorithmic expert pilot

For the imitation learning validation in this exercise, we collected a dataset with an algorithmic pilot, the dataset is available in the next [link]().

n this application, the [Pilot_Net](https://github.com/lhzlhz/PilotNet) network was used to control linear and angular velocity. The expert pilot is launched as:

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


### Getting the best model

To speed up the model selection process for this application, the following script was created. It executes a folder containing 𝑛 models and generates an image of the result for each on as the images below: 

<div align="center">
<img width=350px src="https://roboticslaburjc.github.io/2023-tfg-adrian-madinabeitia/assets/images/post17/result.png" alt="explode"></a>
<img width=340px src="https://roboticslaburjc.github.io/2023-tfg-adrian-madinabeitia/assets/images/post18/montmelo.png" alt="explode"></a>
</div>

### Video
In this video you can see the neural network piloting the drone: 

<iframe width="560" height="315" src="https://www.youtube.com/embed/jJ4Xdin1gg4?si=dHWRwDyc4kTha1dJ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

---

## Gate travesing application

In this application we collected two datasets, one more generic for the pilotNet training (available in the next [link]()) and another one, more specialized in altitude movements for the deepPilot training (available in the next [link]())



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
 

### Validation
The only difference between the first application and this one was the expert pilot so the procedure in training was the same. This was the final result: 

1. Constant altitude:
<iframe width="560" height="315" src="https://www.youtube.com/embed/l-WyA2C-4I4?si=zG3eUutcG0e2m87q" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

2. Variable altitude:
<iframe width="560" height="315" src="https://www.youtube.com/embed/Q1zBNXdW7Ns?si=_ehmQ87xtfdJHUgK" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

---

 

## Licencia

<a rel="license" href=https://www.apache.org/licenses/LICENSE-2.0><img alt="Apache License" style="border-width:0" src=https://www.apache.org/img/asf-estd-1999-logo.jpg /></a><br/> </a><br/>This work is licensed under a <a rel="license" href=https://www.apache.org/licenses/LICENSE-2.0>Apache license 2.0