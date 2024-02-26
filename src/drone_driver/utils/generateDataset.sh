#!/bin/bash

worlds=("montreal_line" "simple_circuit" "montmelo_line")
orientations=(0.0 3.14)


tmux new-window -n drone_window

# Dividir la ventana horizontalmente en tres paneles
tmux split-window -v
tmux split-window -v



for world in "${worlds[@]}"; do
    for yaw in "${orientations[@]}"; do
        # Ejecutar cada comando en un panel separado
        tmux send-keys -t 0 "ros2 launch drone_driver as2_sim_circuit.launch.py world:=../worlds/$world.world yaw:=$yaw" C-m

        sleep 5
        tmux send-keys -t 1 "ros2 launch drone_driver expertPilot.launch.py | tee temp_output.txt" C-m


        # Waits till the drone is following the line
        while ! grep -q "Following" temp_output.txt; do
            sleep 1
        done

        # Starts recording the topics
        sleep 3
        tmux send-keys -t 2 "ros2 bag record -o folderName /drone0/sensor_measurements/frontal_camera/image_raw /drone0/motion_reference/twist" C-m

        # Record time
        sleep 40

        # Sends ctrl+C to end the simulation
        tmux send-keys -t 0 C-c
        tmux send-keys -t 1 C-c
        tmux send-keys -t 2 C-c

        # Waits untill all is cleaned
        sleep 2
    done
done



# Closes the tmux sesion
tmux kill-session -t drone_window

tmux kill-server