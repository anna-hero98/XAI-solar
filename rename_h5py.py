import h5py

# Open the HDF5 file to explore its structure
def explore_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        # Print all top-level keys (datasets or groups)
        print("Keys in HDF5 file:", list(f.keys()))
        
        # Optionally, explore nested groups or datasets further by inspecting the keys recursively
        def explore_group(group):
            print(f"Exploring group: {group}")
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Group):
                    explore_group(item)  # Recursively explore nested groups
                else:
                    print(f"Dataset: {key} (shape: {item.shape})")
                    
        # Recursively explore the root group
        explore_group(f)

# Call the function to explore your Results_CNN.h5 file
explore_h5_file('Results_CNN.h5')
