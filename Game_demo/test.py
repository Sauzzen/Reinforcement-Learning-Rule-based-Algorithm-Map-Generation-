import os
import json

# Check what files exist
print("Current directory:", os.getcwd())
print("\nFiles in current directory:")
for item in os.listdir('.'):
    print(f"  {item}")

if os.path.exists('maps'):
    print("\nFiles in maps directory:")
    for item in os.listdir('maps'):
        print(f"  {item}")

# Test loading the JSON directly
test_files = [
    "maps/final_map_seed0.json",
    "final_map_seed0.json", 
    "1758502386424_final_map_seed0.json"
]

for test_file in test_files:
    print(f"\nTesting: {test_file}")
    print(f"  Exists: {os.path.exists(test_file)}")
    if os.path.exists(test_file):
        try:
            with open(test_file, 'r') as f:
                data = json.load(f)
            print(f"  Valid JSON: Yes")
            print(f"  Width: {data.get('width', 'Missing')}")
            print(f"  Height: {data.get('height', 'Missing')}")
        except Exception as e:
            print(f"  Error reading: {e}")