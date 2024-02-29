#!/bin/bash

# Gets the arguments
dataset_dir=$1
output_directory=$2
model_name=$3

testTime=120

if [ -z "$dataset_dir" ] || [ -z "$output_directory" ]; then
    echo "Usage: $0 <dataset_dir> <model_dir>"
    exit 1
fi

test_worlds=("nurburgring_line")
orientations=(0.0)
pilots=(
  "ros2 launch drone_driver neuralPilot.launch.py trace:=False out_dir:=$output_directory/profiling/$world$iteration network_path:=$model_name"
  "ros2 launch drone_driver expertPilot.launch.py trace:=False out_dir:=$output_directory/profiling/$world$iteration"
)
names=("Neural" "Expert")

python3 ../src/train.py --dataset_path $dataset_dir --network_path $output_directory


tmux new-window -n drone_test
# Split the window into four panes
tmux split-window -v
tmux split-window -v

iteration=0

for pilot in "${pilots[@]}"; do
    for world in "${test_worlds[@]}"; do

        # Executes the simulator
        tmux send-keys -t 0 "ros2 launch drone_driver as2_sim_circuit.launch.py world:=../worlds/$world.world yaw:=0.0" C-m
        sleep 5

        tmux send-keys -t 1 "$pilot | tee temp_output.txt" C-m

        # Waits untill the drone is flying
        while ! grep -q "Following" temp_output.txt; do
            sleep 1
        done
        rm temp_output.txt
        sleep 2

        # Starts recording the topics
        name="${names[iteration]}"
        tmux send-keys -t 2 "ros2 bag record -o $output_directory/test$name /tf /drone0/sensor_measurements/frontal_camera/image_raw /drone0/self_localization/twist" C-m
        sleep "$testTime"

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
tmux kill-window -t drone_test

# Compare the two executions with the ideal one
python3 ../src/plotFunctions/graphPath.py --perfect_path ../src/plotFunctions/positionRecords/perfectNurburGingTF/ --expert_path $output_directory/test${names[1]} --neural_path $output_directory/test${names[0]}