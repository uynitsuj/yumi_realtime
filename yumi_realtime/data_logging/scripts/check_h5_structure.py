import h5py

with h5py.File('/home/xi/HDD1/success/yumi_put_the_white_cup_on_the_coffee_machine/robot_trajectory_2025_04_14_20_45_43.h5', 'r') as f:
    # Print top-level keys
    print("Top-level keys:", list(f.keys()))
    
    # If you want to explore nested structure
    def print_structure(name, obj):
        print(name, type(obj))
    
    f.visititems(print_structure)