import json
import sys

def update_world_path(json_file_path, new_world_path):
    # Load the JSON data from the file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Check if the "world" key exists in the JSON data
    if "world" in data:
        world_path = data["world"]
        print(f"Original World path: {world_path}")

        # Update the "world" key with the new path
        data["world"] = new_world_path

        # Write the updated JSON data back to the file
        with open(json_file_path, 'w') as updated_json_file:
            json.dump(data, updated_json_file, indent=4)

        print(f"Updated World path: {new_world_path}")
    else:
        print("The 'world' key does not exist in the JSON data.")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <json_file_path> <new_world_path>")
    else:
        json_file_path = sys.argv[1]
        new_world_path = sys.argv[2]
        update_world_path(json_file_path, new_world_path)
