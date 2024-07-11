#!/bin/bash

# Gets the arguents
network_path=$1

iteration=0

if [ -z "$network_path" ]; then
    echo "Usage: $0 <network_path>"
    exit 1
fi

while true; do
    new_network_path="${network_path%.tar}-$iteration.tar"
    
    cp "$network_path" "$new_network_path"

    echo Copy $new_network_path created
    sleep 7200
    ((iteration++))
done
