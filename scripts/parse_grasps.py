import numpy as np
import json
from ros_utils import ros_qt_to_rt


def parse_grasps(filename):

    with open(filename, 'r') as f:
        data = json.load(f)
    grasps = data['grasps']
    
    n = len(grasps)
    poses_grasp = np.zeros((n, 4, 4), dtype=np.float32)
    for i in range(n):
        pose = grasps[i]['pose']
        rot = pose[3:]
        trans = pose[:3]
        RT = ros_qt_to_rt(rot, trans)
        poses_grasp[i, :, :] = RT
    return poses_grasp
    
    
def extract_grasps(graspit_grasps, gripper_name, obj_offset):

    # counting
    n = 0
    index = []
    for i in range(len(graspit_grasps)):
        if graspit_grasps[i]['gripper'] == gripper_name:
            n += 1
            index.append(i)
            
    # get grasps
    poses_grasp = np.zeros((n, 4, 4), dtype=np.float32)
    for i in range(n):
        ind = index[i]
        pose = graspit_grasps[ind]['pose']
        rot = pose[3:]
        trans = pose[:3]
        RT = ros_qt_to_rt(rot, trans)
        
        RT_offset = np.eye(4, dtype=np.float32)
        RT_offset[:3, 3] = -obj_offset
        
        poses_grasp[i, :, :] = RT_offset @ RT
    return poses_grasp    


if __name__ == "__main__":
    """
    Main function to run the code
    """
    
    filename = 'refined_003_cracker_box_google_16k_textured_scale_1000-fetch_gripper.json'
    poses_grasp = parse_grasps(filename)
    print(poses_grasp)
