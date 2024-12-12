#!/bin/bash

# Gets the arguments
output_directory=$1
model_folder=$2

testTime=180

if [ -z "$output_directory" ] || [ -z "$model_folder" ]; then
    echo "Usage: $0 <output_directory> <model_folder>"
    exit 1
fi

test_worlds=("nurburgring_line")
orientations=(0.0)

tar_files=$(ls "$model_folder" | grep '\.tar$')

for filename in $tar_files; do
    # Remove the .tar extension and add the filename to the array
    tar_files_no_extension+=("${filename%.tar}")
done

tmux new-window -n drone_test
# Split the window into four panes
tmux split-window -v
tmux split-window -v
tmux split-window -v

iteration=0
mkdir -p $output_directory/results

for name in "${tar_files_no_extension[@]}"; do
    for world in "${test_worlds[@]}"; do

        # Executes the simulator
        tmux send-keys -t 0 "ros2 launch drone_sim_driver as2_sim_circuit.launch.py world:=../worlds/$world.world yaw:=0.0" C-m
        sleep 5

        tmux send-keys -t 1 "ros2 launch drone_sim_driver neuralPilot.launch.py trace:=False out_dir:=$output_directory/profiling/$world$iteration network_path:=$model_folder/${name}.tar | tee temp_output.txt" C-m

        # Waits until the drone is flying
        while ! grep -q "Following" temp_output.txt; do
            sleep 1
        done
        rm temp_output.txt
        sleep 2

        # Starts recording the topics
        tmux send-keys -t 2 "ros2 bag record -o $output_directory/test$name /tf /drone0/sensor_measurements/frontal_camera/image_raw /drone0/self_localization/twist" C-m
        sleep "$testTime"

        # Sends ctrl+C to end the simulation
        tmux send-keys -t 0 C-c
        tmux send-keys -t 1 C-c
        tmux send-keys -t 2 C-c

        sleep 2
        iteration=$((iteration + 1))

        # Saves the results image
        tmux send-keys -t 3 "python3 ../src/plotFunctions/graphPath.py --perfect_path ../src/plotFunctions/positionRecords/perfectNurburGingTF/ --expert_path ../src/plotFunctions/positionRecords/EPNumbirngTF --neural_path $output_directory/test$name --file_name "$output_directory/results/$name.png"" C-m
        
    done
done

# Kills the other panels
tmux kill-pane -t 1
tmux kill-pane -t 2
tmux kill-window -t drone_test

