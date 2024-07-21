#!/bin/bash

# Gets the arguments
dataset_dir=$1
output_directory=$2
model_name=$3

testTime=240

if [ -z "$dataset_dir" ] || [ -z "$output_directory" ] || [ -z "$model_name" ]; then
    echo "Usage: $0 <dataset_dir> <output_directory> <model_name>"
    exit 1
fi

test_worlds=("nurburgring_line")
perfect_pilots=("../src/plotFunctions/positionRecords/perfectNurburGingTF")


orientations=(0.0)

names=("Neural" "Expert")

python3 ../src/train.py --dataset_path $dataset_dir --network_path $output_directory

tmux new-window -n drone_test

# Split the window into four panes
tmux split-window -v
tmux split-window -v

iteration=0
worldIter=0

for world in "${test_worlds[@]}"; do
    pilots=(
    "ros2 launch drone_behaviors neuralPilot.launch.py trace:=False out_dir:=${output_directory}$world/profiling/neural_$world network_path:=$model_name"
    "ros2 launch drone_behaviors expertPilot.launch.py trace:=False out_dir:=${output_directory}$world/profiling/expert_$world"
    )
    for pilot in "${pilots[@]}"; do
    
        # Executes the simulator
        tmux send-keys -t 0 "ros2 launch drone_platforms as2_sim_circuit.launch.py world:=../worlds/$world.world yaw:=3.14" C-m
        sleep 5

        tmux send-keys -t 1 "$pilot | tee temp_output.txt" C-m

        # Waits untill the drone is flying
        while ! grep -q "Following" temp_output.txt; do
            sleep 1
        done
        rm ./temp_output.txt
        sleep 2

        # Starts recording the topics
        name="${names[iteration]}"
        tmux send-keys -t 2 "ros2 bag record -o ${output_directory}$world/test$name /tf /drone0/sensor_measurements/frontal_camera/image_raw /drone0/commanded_vels" C-m
        sleep "$testTime"

        # Sends ctrl+C to end the simulation
        tmux send-keys -t 0 C-c
        tmux send-keys -t 1 C-c
        tmux send-keys -t 2 C-c

        # Waits untill all is cleaned
        sleep 2
        iteration=$((iteration + 1))
        
    done
    echo ${perfect_pilots[$worldIter]} 
    python3 ../src/plotFunctions/graphPath.py --perfect_path ${perfect_pilots[$worldIter]} --expert_path ${output_directory}$world/test${names[1]} --neural_path ${output_directory}$world/test${names[0]} --file_path ${output_directory}$world --show_vels "False"
    worldIter=$((worldIter + 1))
done

# Kills the other panels
tmux kill-window -t drone_test