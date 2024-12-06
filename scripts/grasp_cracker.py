#!/usr/bin/env python3

"""
CS 6301 Homework 5 Programming
Robot Control for Grasping
"""

import sys, os
import actionlib
import rospy
import numpy as np
import copy
import moveit_commander
import geometry_msgs.msg
import moveit_msgs.msg
import tf2_ros
import tf.transformations as tra

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from control_msgs.msg import PointHeadAction, PointHeadGoal
from geometry_msgs.msg import PoseStamped, Point
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf.transformations import quaternion_from_euler, quaternion_matrix
from ros_utils import ros_pose_to_rt, rt_to_ros_qt, rt_to_ros_pose, set_axes_equal
from parse_grasps import parse_grasps, extract_grasps
from gripper import Gripper
from gazebo_msgs.srv import GetModelState


# Send a trajectory to controller
class FollowTrajectoryClient(object):

    def __init__(self, name, joint_names):
        self.client = actionlib.SimpleActionClient("%s/follow_joint_trajectory" % name,
                                                   FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for %s..." % name)
        self.client.wait_for_server()
        self.joint_names = joint_names

    def move_to(self, positions, duration=5.0):
        if len(self.joint_names) != len(positions):
            print("Invalid trajectory position")
            return False
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        trajectory.points.append(JointTrajectoryPoint())
        trajectory.points[0].positions = positions
        trajectory.points[0].velocities = [0.0 for _ in positions]
        trajectory.points[0].accelerations = [0.0 for _ in positions]
        trajectory.points[0].time_from_start = rospy.Duration(duration)
        follow_goal = FollowJointTrajectoryGoal()
        follow_goal.trajectory = trajectory

        self.client.send_goal(follow_goal)
        self.client.wait_for_result()


# Point the head using controller
class PointHeadClient(object):

    def __init__(self):
        self.client = actionlib.SimpleActionClient("head_controller/point_head", PointHeadAction)
        rospy.loginfo("Waiting for head_controller...")
        self.client.wait_for_server()

    def look_at(self, x, y, z, frame, duration=1.0):
        """
        Turning head to look at x,y,z
        :param x: x location
        :param y: y location
        :param z: z location
        :param frame: the frame of reference
        :param duration: given time for operation to calcualte the motion plan
        :return:
        """
        goal = PointHeadGoal()
        goal.target.header.stamp = rospy.Time.now()
        goal.target.header.frame_id = frame
        goal.target.point.x = x
        goal.target.point.y = y
        goal.target.point.z = z
        goal.min_duration = rospy.Duration(duration)
        self.client.send_goal(goal)
        self.client.wait_for_result()
        
        
'''
quat is with format (x, y, z, w) for a quanternion
trans is the 3D translation (x, y, z)
group is the moveit group interface
'''
def plan_to_pose(group, quat, trans):

    ################ TO DO ##########################
    # use moveit to plan to trajectory towards the gripper pose defined by (quat, trans)
    # refer to https://ros-planning.github.io/moveit_tutorials/doc/move_group_python_interface/move_group_python_interface_tutorial.html

    # Get the current robot state
    group.set_start_state_to_current_state()

    # Set the target pose for the end-effector
    target_pose = geometry_msgs.msg.Pose()
    target_pose.orientation.x = quat[0]
    target_pose.orientation.y = quat[1]
    target_pose.orientation.z = quat[2]
    target_pose.orientation.w = quat[3]
    target_pose.position.x = trans[0]
    target_pose.position.y = trans[1]
    target_pose.position.z = trans[2]

    group.set_pose_target(target_pose)
    ################ TO DO ##########################
    
    plan = group.plan()
    return plan
        

'''
RT_grasps_base is with shape (50, 4, 4): 50 grasps in the robot base frame
The plan_grasp function tries to plan a trajectory to each grasp. It stops when a plan is found.
A standoff is a gripper pose with a short distance along x-axis of the gripper frame before grasping the object.
'''       
def plan_grasp(group, RT_grasps_base, grasp_index):
    
    # number of grasps
    n = RT_grasps_base.shape[0]
    reach_tail_len = 10
    # define the standoff distance as 10cm
    standoff_dist = 0.10
    
    # compute standoff pose
    offset = -standoff_dist * np.linspace(0, 1, reach_tail_len, endpoint=False)[::-1]
    offset = np.append(offset, [0.04])
    
    reach_tail_len += 1
    pose_standoff = np.tile(np.eye(4), (reach_tail_len, 1, 1))    
    pose_standoff[:, 0, 3] = offset
    
    # for each grasp    
    for i in range(n):
        RT_grasp = RT_grasps_base[i]
        grasp_idx = grasp_index[i]
    
        standoff_grasp_global = np.matmul(RT_grasp, pose_standoff)
    
        # Calling `stop()` ensures that there is no residual movement
        group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        group.clear_pose_targets()

        # plan to the standoff
        quat, trans = rt_to_ros_qt(standoff_grasp_global[0, :, :])  # xyzw for quat          
        plan = plan_to_pose(group, quat, trans)
        trajectory = plan[1]
        if plan[0]:
            print('find a plan for grasp')
            print(RT_grasp)
            print('grasp idx', grasp_idx)
            print('grasp index', grasp_index)
            break
        else:
            print('no plan for grasp %d with index %d' % (i, grasp_idx))

    if not plan[0]:
        print('no plan found')
        return None, -1
            
    return RT_grasp, grasp_idx    


# first plan to the standoff pose, then move the the grasping pose
def grasp(gripper, group, scene, object_name, RT_grasp):
    
    reach_tail_len = 10
    standoff_dist = 0.10
    
    # compute standoff pose
    offset = -standoff_dist * np.linspace(0, 1, reach_tail_len, endpoint=False)[::-1]
    offset = np.append(offset, [0.04])
    
    reach_tail_len += 1
    pose_standoff = np.tile(np.eye(4), (reach_tail_len, 1, 1))    
    pose_standoff[:, 0, 3] = offset

    # plan to grasp
    standoff_grasp_global = np.matmul(RT_grasp, pose_standoff)
    
    # Calling `stop()` ensures that there is no residual movement
    group.stop()
    # It is always good to clear your targets after planning with poses.
    # Note: there is no equivalent function for clear_joint_value_targets()
    group.clear_pose_targets()

    # plan to the standoff
    quat, trans = rt_to_ros_qt(standoff_grasp_global[0, :, :])  # xyzw for quat
    plan = plan_to_pose(group, quat, trans)
    trajectory = plan[1]
    if not plan[0]:
        print('no plan found')
        return

    input('execute?')
    group.execute(trajectory, wait=True)
    group.stop()
    group.clear_pose_targets()
    
    # remove the target from the planning scene for grasping
    scene.remove_world_object(object_name)
    
    waypoints = []
    wpose = group.get_current_pose().pose
    for i in range(1, reach_tail_len):
        wpose = rt_to_ros_pose(wpose, standoff_grasp_global[i])
        print(wpose)
        waypoints.append(copy.deepcopy(wpose))
    (plan_standoff, fraction) = group.compute_cartesian_path(
                                   waypoints,   # waypoints to follow
                                   0.01,        # eef_step
                                   True)         # jump_threshold
    trajectory = plan_standoff
    
    input('execute?')    
    group.execute(trajectory, wait=True)
    group.stop()
    group.clear_pose_targets()
    
    # close gripper
    print('close gripper')
    gripper.close()
    rospy.sleep(2)
   
    # lift object
    print('lifting object')
    lift_arm(group)
    

# lift the robot arm
def lift_arm(group):

    # lift the object
    offset = -0.2
    rospy.loginfo("lift object")
    pose = group.get_current_joint_values()
    pose[1] += offset
    group.set_joint_value_target(pose)
    plan = group.plan()
    
    if not plan[0]:
        print('no plan found in lifting')
        sys.exit(1)
    
    input('execute?')
    trajectory = plan[1]
    group.execute(trajectory, wait=True)
    group.stop()
    group.clear_pose_targets()

    
# Query pose of frames from the Gazebo environment
def get_pose_gazebo(model_name, relative_entity_name=''):

    def gms_client(model_name, relative_entity_name):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp1 = gms(model_name, relative_entity_name)
            return resp1
        except (rospy.ServiceException, e):
            print("Service call failed: %s" % e)
           
    # query the object pose in Gazebo world T_wo
    res = gms_client(model_name, relative_entity_name) 
    T_wo = ros_pose_to_rt(res.pose)  
    
    # query fetch base link pose in Gazebo world T_wb
    res = gms_client(model_name='fetch', relative_entity_name='base_link')
    T_wb = ros_pose_to_rt(res.pose)
    
    ################ TO DO ##########################
    # compute the object pose in robot base link T_bo
    # use your code from homework 2

    T_bw = np.linalg.inv(T_wb)
    T_bo = np.dot(T_bw, T_wo)

    ################ TO DO ##########################
    return T_bo
    

# sort grasps according to distances to gripper
def sort_grasps(RT_obj, RT_gripper, RT_grasps):
    # transform grasps to robot base
    n = RT_grasps.shape[0]
    RT_grasps_base = np.zeros_like(RT_grasps)
    distances = np.zeros((n, ), dtype=np.float32)
    for i in range(n):
        RT_g = RT_grasps[i]
        # transform grasp to robot base
        RT = RT_obj @ RT_g
        RT_grasps_base[i] = RT
        d = np.linalg.norm(RT_gripper[:3, 3] - RT[:3, 3])
        distances[i] = d
    
    # sort according to distances
    index = np.argsort(distances)
    RT_grasps_base = RT_grasps_base[index]
    print(distances)
    print(index)
    return RT_grasps_base, index


if __name__ == "__main__":
    """
    Main function to run the code
    """
    
    # set hyper-parameters
    object_name = '003_cracker_box'
        
    # Create a node
    rospy.init_node("fetch_grasping")

    # # Setup clients
    torso_action = FollowTrajectoryClient("torso_controller", ["torso_lift_joint"])
    head_action = PointHeadClient()
    gripper = Gripper()

    # Raise the torso using just a controller
    rospy.loginfo("Raising torso...")
    torso_action.move_to([0.4, ])

    # --------- initialize moveit components ------
    moveit_commander.roscpp_initialize(sys.argv)
    group = moveit_commander.MoveGroupCommander("arm")
    scene = moveit_commander.PlanningSceneInterface()
    scene.clear()
    robot = moveit_commander.RobotCommander()
        
    # look at table    
    head_action.look_at(0.7, 0, 0.75, "base_link")

    # get object pose
    RT_obj = get_pose_gazebo(model_name=object_name)
    trans = RT_obj[:3, 3]
    
    # # sleep before adding objects
    # # dimension of each default(1,1,1) box is 1x1x1m
    # -------- planning scene set-up -------
    rospy.loginfo("adding table object into planning scene")
    print("adding table object into planning scene")    
    rospy.sleep(1.0)
    p = PoseStamped()
    p.header.frame_id = robot.get_planning_frame()
    p.pose.position.x = 0.9
    p.pose.position.y = 0
    p.pose.position.z = trans[2] - 0.5 - 0.1
    scene.add_box("table", p, (1, 5, 1))
    
    # add a box for robot base
    p.pose.position.x = 0
    p.pose.position.y = 0
    p.pose.position.z = 0.18
    scene.add_box("base", p, (0.56, 0.56, 0.4))    
            
    # load grasps
    '''
    RT_grasps is with shape (50, 4, 4)
    It contains 50 grasps of the cracker box.
    Each grasp is a 4x4 transformation matrix defining the gripper pose in the object frame.
    '''    
    filename = 'data/refined_%s_google_16k_textured_scale_1000-fetch_gripper.json' % object_name
    RT_grasps = parse_grasps(filename)
    print(RT_grasps.shape)
        
    # current gripper pose
    RT_gripper = get_pose_gazebo(model_name='fetch', relative_entity_name='wrist_roll_link')        

    # add the target to the planning scene
    p.pose = rt_to_ros_pose(p.pose, RT_obj)
    scene.add_mesh(object_name, p, 'data/' + object_name + '.ply')
        
    # sort grasps according to distances to gripper
    # RT_grasps_base contains all the grasps in the robot base frame
    RT_grasps_base, grasp_index = sort_grasps(RT_obj, RT_gripper, RT_grasps)
        
    # grasp planning
    RT_grasp, grasp_num = plan_grasp(group, RT_grasps_base, grasp_index)
        
    # grasp object
    grasp(gripper, group, scene, object_name, RT_grasp)

    input("Open Gripper??")
    gripper.open()
