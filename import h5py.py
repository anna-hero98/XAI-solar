import h5py

# Open the .h5 file
with h5py.File('SOLETE_Pombo_1min.h5', 'r') as file:
    # List all the top-level keys (groups)
    print("Keys:", list(file.keys()))

    # Access the 'DATA' group
    data_group = file['DATA']

    # List all sub-items (datasets or groups) within 'DATA'
    print("Sub-items in 'DATA':", list(data_group.keys()))
    
    # Iterate over the sub-items and print them
    for key in data_group:
        sub_item = data_group[key]
        
        # Check if the sub-item is a dataset
        if isinstance(sub_item, h5py.Dataset):
            print(f"Dataset '{key}':")
            print(sub_item[:])  # Print the entire dataset
        elif isinstance(sub_item, h5py.Group):
            print(f"Group '{key}':")
            # If it's a group, list its contents
            print(list(sub_item.keys()))


    # Access the 'DATA' group
    data_group = file['DATA']

    # List all sub-items (datasets or groups) within 'DATA'
    print("Sub-items in 'DATA':", list(data_group.keys()))