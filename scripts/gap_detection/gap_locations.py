import json
from itertools import combinations

def find_gap_locations(xyz_a, xyz_b): # top brick a, bottom brick b
    x_a, y_a, z_a, length_a = xyz_a
    x_b, y_b, z_b, length_b = xyz_b

    if y_a != y_b or z_a != z_b: return []
    
    gap_locations = []
    if x_a >= x_b:
        for x in range(x_a, x_b+1):
            if x < x_a + length_a:
                gap_locations.append((x, y_a, z_a))
    else:
        for x in range(x_b, x_a+1):
            if x < x_b + length_b:
                gap_locations.append((x, y_a, z_a))
    
    return gap_locations


if __name__ == '__main__':
    assembly_json = '/home/mfi/repos/ros1_ws/src/ruixuan/Robotic_Lego_Manipulation/config/assembly_tasks/tower.json'
    main_arm = 0
    
    # TODO
    width = 4
    height = 2

    with open(assembly_json, 'r') as f:
        data = json.load(f)

    coords_list = []
    for lego in data: # '1', '2', '3'
        node = data[lego]
        if main_arm == 0 and node['ori'] == 0:
            coords = (node['x'], node['y'], node['z'], width)
        elif main_arm == 0 and node['ori'] == 1:
            coords = (node['x'], node['y'], node['z'], height)
        elif main_arm == 1 and node['ori'] == 0:
            coords = (node['x'] + width - 1, node['y'] + height - 1, node['z'], width)
        elif main_arm == 1 and node['ori'] == 1:
            coords = (node['x'] + height - 1, node['y'] + width - 1, node['z'], height)
        coords_list.append(coords) # x, y, z, length

    for combination in combinations(coords, 2):
        gap_location = find_gap_locations(coords[0], coords[1])
        if gap_location is not None:
            print('gap:', gap_location)