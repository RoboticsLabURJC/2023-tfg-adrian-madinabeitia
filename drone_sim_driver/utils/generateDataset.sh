#!/bin/bash

# Gets the arguments
record_time=$1
output_directory=$2

if [ -z "$record_time" ] || [ -z "$output_directory" ]; then
    echo "Usage: $0 <record_time> <output_directory>"
    exit 1
fi

worlds=( "simple_circuit" "montmelo_line" "montreal_line")
orientations=(0.0 3.14)

tmux new-window -n drone_recorder

# Split the window into four panes
tmux split-window -v
tmux split-window -v

iteration=0

for world in "${worlds[@]}"; do
    for yaw in "${orientations[@]}"; do
    
        # Executes the simulator
        tmux send-keys -t 0 "ros2 launch drone_sim_driver as2_sim_circuit.launch.py world:=../worlds/$world.world yaw:=$yaw" C-m
        sleep 5
        
        touch temp_output.txt
        tmux send-keys -t 1 "ros2 launch drone_sim_driver expertPilot.launch.py trace:=False out_dir:=$output_directory/profiling/$world$iteration | tee ./temp_output.txt" C-m
        
        sleep 5

        # Waits untill the drone is flying
        while ! grep -q "Following" ./temp_output.txt; do
            sleep 1
        done
        rm ./temp_output.txt
        sleep 5

        # Starts recording the topics
        tmux send-keys -t 2 "ros2 bag record -o $output_directory/$world$iteration /drone0/sensor_measurements/frontal_camera/image_raw /drone0/commanded_vels /tf" C-m
        sleep "$record_time"

        # Sends ctrl+C to end the simulation
        tmux send-keys -t 0 C-c
        tmux send-keys -t 1 C-c
        tmux send-keys -t 2 C-c

        # Waits untill all is cleaned
        sleep 2
        iteration=$((iteration + 1))
        
    done
done

# Kills the other panels
tmux kill-pane -t 1
tmux kill-window -t drone_recorder

# Converts the rosbags to a general dataset
python3 ../src/rosbag2generalDataset.py --rosbags_path $output_directory

# tmux kill-server