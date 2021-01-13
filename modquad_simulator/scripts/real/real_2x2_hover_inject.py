#!/usr/bin/env python3

import numpy as np
import time
import matplotlib.pyplot as plt
import math

import rospy
import tf2_ros

from geometry_msgs.msg  import Twist
from std_msgs.msg       import Int8MultiArray
from modquad_simulator.msg import ModquadCoexecutorTime
from std_srvs.srv       import Empty, EmptyRequest
from crazyflie_driver.srv import UpdateParams

from tf.transformations import euler_from_quaternion

from modsim.datatype.structure import Structure

from modsim.controller  import position_controller
from modsim.trajectory  import min_snap_trajectory

from modsim             import params
from modsim.params      import RunType
from modsim.util.state  import init_state, state_to_quadrotor
from modsim.util.thrust import convert_thrust_newtons_to_pwm

from modsim.util.fault_detection import fault_exists_real,      \
                                        real_find_suspects,     \
                                        get_faulty_quadrant_rotors_real,\
                                        update_ramp_rotors, update_ramp_factors, \
                                        form_groups, update_rotmat

from dockmgr.datatype.PoseManager import PoseManager
#from dockmgr.datatype.ImuManager import ImuManager

from modquad_sched_interface.interface import convert_modset_to_struc, \
                                              convert_struc_to_mat,    \
                                              rotpos_to_mat

import modquad_sched_interface.waypt_gen     as waypt_gen
import modquad_sched_interface.structure_gen as structure_gen

from modquad_sched_interface.simple_scheduler import lin_assign

from modsim.util.sim_coexecutor import setup_sim_node, sim_once

# Set up for Structure Manager
structure = None
t = 0.0
traj_func = min_snap_trajectory
start_id = 16 # 1 indexed

logs = {
	't':     [], 'sthrust': [],                 # time, desired thrust
	'x':     [], 'y':       [], 'z':        [], # measured position
	'vx':    [], 'vy':      [], 'vz':       [], # measured linear velocity
	'roll':  [], 'pitch':   [], 'yaw':      [], # measured attitude
	'vroll': [], 'vpitch':  [], 'vyaw':     [], # measured angular velocity
	'sroll': [], 'spitch':  [], 'syawrate': [], # desired roll, pitch, yaw rate
	'desx':  [], 'desy':    [], 'desz':     [], # desired position
	'desvx': [], 'desvy':   [], 'desvz':    [], # desired linear velocity
	'pred_x':  [], 'pred_y':  [], 'pred_z':  [], # predicted position
	'pred_vx': [], 'pred_vy': [], 'pred_vz': [], # predicted linear velocity
	'pred_roll':  [], 'pred_pitch':  [], 'pred_yaw':  [], # predict attitude
	'pred_vroll': [], 'pred_vpitch': [], 'pred_vyaw': [], # predict ang vel
}

def switch_estimator_to_kalman_filter(start_id, num_robot):
    srv_name_set = ['/modquad{:02d}/switch_to_kalman_filter'.format(mid) for mid in range(start_id, start_id+num_robot)]
    switch_set = [ rospy.ServiceProxy(srv_name, Empty) for srv_name in srv_name_set ]
    rospy.loginfo('Wait for all switch_to_kalman_filter services')
    [rospy.wait_for_service(srv_name) for srv_name in srv_name_set]
    rospy.loginfo('Found all switch_to_kalman_filter services')
    msg = EmptyRequest()
    [switch(msg) for switch in switch_set]

def update_att_ki_gains(start_id, num_robot):
    # Zero the attitude I-Gains
    srv_name_set = ['/modquad{:02d}/zero_att_i_gains'.format(mid) for mid in range(start_id, start_id+num_robot)]
    zero_att_i_gains_set = [ rospy.ServiceProxy(srv_name, Empty) 
                             for srv_name in srv_name_set
                           ]
    rospy.loginfo('Wait for all zero_att_gains services')
    [rospy.wait_for_service(srv_name) for srv_name in srv_name_set]
    rospy.loginfo('Found all zero_att_gains services')
    msg = EmptyRequest()
    [zero_att_i_gains(msg) for zero_att_i_gains in zero_att_i_gains_set]

def update_logs(t, state_vector, desired_state, 
                thrust, roll, pitch, yawrate, pred_state):
    global logs

    # Add to logs
    logs['t'].append(t)

    logs['x'].append(state_vector[0])
    logs['y'].append(state_vector[1])
    logs['z'].append(state_vector[2])

    #vel = structure.state_vector[3:6]
    logs['vx'].append(state_vector[3])
    logs['vy'].append(state_vector[4])
    logs['vz'].append(state_vector[5])


    # Orientation for a single rigid body is constant throughout the body
    euler = euler_from_quaternion(state_vector[6:10])
    logs['roll'].append(math.degrees(euler[0]))
    logs['pitch'].append(math.degrees(euler[1]))
    logs['yaw'].append(math.degrees(euler[2]))

    # Measured angular velocities 
    logs['vroll'].append(state_vector[-3])
    logs['vpitch'].append(state_vector[-2])
    logs['vyaw'].append(state_vector[-1])

    # Add to logs
    logs['desx'].append(desired_state[0][0])
    logs['desy'].append(desired_state[0][1])
    logs['desz'].append(desired_state[0][2])

    logs['desvx'].append(desired_state[1][0])
    logs['desvy'].append(desired_state[1][1])
    logs['desvz'].append(desired_state[1][2])

    logs['sthrust'].append(thrust)
    logs['sroll'].append(roll)
    logs['spitch'].append(pitch)
    logs['syawrate'].append(yawrate)

    pred_euler = euler_from_quaternion(pred_state[6:10])
    logs['pred_x'     ].append(pred_state[0])
    logs['pred_y'     ].append(pred_state[1])
    logs['pred_z'     ].append(pred_state[2])
    logs['pred_vx'    ].append(pred_state[3])
    logs['pred_vy'    ].append(pred_state[4])
    logs['pred_vz'    ].append(pred_state[5])
    logs['pred_roll'  ].append(math.degrees(pred_euler[0]))
    logs['pred_pitch' ].append(math.degrees(pred_euler[1]))
    logs['pred_yaw'   ].append(math.degrees(pred_euler[2]))
    logs['pred_vroll' ].append(pred_state[10])
    logs['pred_vpitch'].append(pred_state[11])
    logs['pred_vyaw'  ].append(pred_state[12])
                      
def init_params(speed):
    #rospy.set_param("kalman/resetEstimation", 1)
    rospy.set_param("fault_det_time_interval",    5.0) # Time interval for fault detection groups
    rospy.set_param("fdd_group_type",         "indiv") # FDD Group Size = 1 Robot
    rospy.set_param('opmode',                'normal')
    rospy.set_param('structure_speed',          speed)
    rospy.set_param('rotor_map',                    3) # So that modquad_torque_control knows which mapping to use
    rospy.set_param('is_modquad_sim',           False) # For controller.py
    rospy.set_param('is_modquad_bottom_framed', False)
    rospy.set_param('is_modquad_unframed',      False)
    rospy.set_param('is_strong_rots',           False) # For controller.py
    rospy.loginfo("!!READY!!")
    np.set_printoptions(precision=2)

def update_state(pose_mgr, structure, freq): # freq computed as 1 / (t - prev_t)
    # Convert the individual new states into structure new state
    # As approximant, we let the first modules state be used
    new_states = pose_mgr.get_new_states() # From OPTITRACK
    new_pos = np.array([state[:3] for state in new_states])
    new_pos = np.mean(new_pos, axis=0).tolist()

    if np.all(structure.prev_state_vector == 0):
        structure.prev_state_vector = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
    else:
        structure.prev_state_vector = structure.state_vector

    # Update position to be centroid of structure
    structure.state_vector = np.array(pose_mgr.get_new_state(1))
    structure.state_vector[0] = new_pos[0]
    structure.state_vector[1] = new_pos[1]
    structure.state_vector[2] = new_pos[2]

    # compute instantaneous velocities
    vels = (structure.state_vector[:3] - structure.prev_state_vector[:3]) / (1.0 / freq)
    structure.state_vector[3] = vels[0] # np.mean(velbuf[:, 0]) # vx
    structure.state_vector[4] = vels[1] # np.mean(velbuf[:, 1]) # vy
    structure.state_vector[5] = vels[2] # np.mean(velbuf[:, 2]) # vz

    # compute instantaneous angular velocities
    vels = [0.0, 0.0, 0.0]
    if np.all(structure.prev_state_vector == 0):
        # compute euler angles - RPY roll pitch yaw
        prev_angs = euler_from_quaternion(structure.prev_state_vector[6:10])
        curr_angs = euler_from_quaternion(structure.state_vector[6:10])

        prev_angs = np.array([prev_angs[0], prev_angs[1], prev_angs[2]])
        curr_angs = np.array([curr_angs[0], curr_angs[1], curr_angs[2]])

        # compute angular velocities
        vels = (curr_angs - prev_angs) / (1.0 / freq)
    
    # Update state vector with smoothed linear/angular velocities
    structure.state_vector[-3] = vels[0] # np.mean(velbuf[:, 3]) # vroll 
    structure.state_vector[-2] = vels[1] # np.mean(velbuf[:, 4]) # vpitch
    structure.state_vector[-1] = vels[2] # np.mean(velbuf[:, 5]) # vyaw

def check_to_inject_fault(t, fault_injected, structure, inject_time):
    # Test fault injection
    if t > 20.0 and not fault_injected:
        fault_injected = True
        inject_time = t
        rid = 1
        mid = 3
        structure.single_rotor_toggle(
            [(structure.ids[mid], structure.xx[mid], structure.yy[mid], rid)],
            rot_thrust_cap=0.7
        )
        rospy.loginfo("INJECT FAULT")
    return fault_injected, inject_time

def apply_ramp_factors(ramp_rotor_set, ramp_factors, structure):
    # Test fault injection
    for i, ramp_set in enumerate(ramp_rotor_set):
        for rotor in ramp_set:
            mid = rotor[0]
            rid = rotor[1]
            structure.single_rotor_toggle(
                [(structure.ids[mid], structure.xx[mid], structure.yy[mid], rid)],
                rot_thrust_cap=ramp_factors[i]
            )
    return

def run(traj_vars, t_step=0.01, speed=1):
    global logs
    global t, traj_func, start_id, structure

    freq = 100.0  # 100hz
    rate = rospy.Rate(freq)
    t = 0
    ind = 0
    num_robot = 4

    worldFrame = rospy.get_param("~worldFrame", "/world")
    init_params(speed) # Prints "!!READY!!" to log

    pose_mgr = PoseManager(num_robot, '/modquad', start_id=start_id-1) #0-indexed
    pose_mgr.subscribe()
 
    """ Publish here to control
    crazyflie_controller/src/controller.cpp has been modified to subscribe to
    this topic, and if we are in the ModQuad state, then the Twist message
    from mq_cmd_vel will be passed through to cmd_vel
    TODO: modify so that we publish to all modules in the struc instead of
    single hardcoded one """
    publishers = [ rospy.Publisher('/modquad{:02d}/mq_cmd_vel'.format(mid), 
                    Twist, queue_size=100) 
                    for mid in range (start_id, start_id + num_robot) ]
    time_pub   = rospy.Publisher('/modquad_structure/time', 
                                    ModquadCoexecutorTime, queue_size=100)

    # Publish to robot
    time_msg = ModquadCoexecutorTime()
    msg = Twist()

    # First few msgs will be zeros
    msg.linear.x  = 0 # roll [-30, 30] deg
    msg.linear.y  = 0 # pitch [-30, 30] deg
    msg.linear.z  = 0 # Thrust ranges 10000 - 60000
    msg.angular.z = 0 # yaw rate

    # Initialize the time message
    time_msg.current_time  = 0
    time_msg.previous_time = 0

    # shutdown
    rospy.on_shutdown(_simple_landing)

    # Update pose
    rospy.sleep(1)

    # Start by sending NOPs so that we have known start state
    # Useful for debugging and safety
    t = 0
    while t < 3:
        t += 1.0 / freq
        [ p.publish(msg) for p in publishers ]
        if round(t, 2) % 1.0 == 0:
            rospy.loginfo("Sending zeros at t = {}".format(round(t,2)))
        rate.sleep()
        update_state(pose_mgr, structure, freq)

    # Update for the 2x2 structure
    update_att_ki_gains(start_id, num_robot)
    structure.update_firmware_params()
    #import pdb; pdb.set_trace()
    #switch_estimator_to_kalman_filter(start_id, num_robot)

    fault_injected = False
    fault_detected = False
    suspects_initd = False

    # Setup the simulated coexecutor node
    setup_sim_node(structure, traj_func, traj_vars, pose_mgr)
    rospy.set_param('shutdown_coexecutor', False)
    rospy.loginfo('CALLED SIM_COEXECUTOR SETUP FUNCTION')

    for mid in range(start_id, start_id + num_robot):
        rospy.loginfo(
            "setup complete for /modquad{:02d}".format(mid))


    _takeoff(pose_mgr, structure, freq, publishers)

    """
    THIS WILL NOT AUTOMATICALLY CAUSE THE ROBOT TO DO ANYTHING!!
    YOU MUST PAIR THIS WITH MODIFIED CRAZYFLIE_CONTROLLER/SRC/CONTROLLER.CPP
    AND USE JOYSTICK TO SWITCH TO MODQUAD MODE FOR THESE COMMANDS TO WORK
    """
    state_pred = [0,0,0, 0,0,0, 0,0,0,0, 0,0,0]
    tstart = rospy.get_time()
    t = 0
    prev_t = t
    inject_time = 0
    presence_detect_time = 0
    diagnose_mode = False
    next_diag_t = 0
    diag_time = 3 # sec
    ramp_rotor_set = []
    ramp_factors = [1.0, 0.0]
    groups = []
    ramp_rotor_set_idx = 0
    while not rospy.is_shutdown() and t < 10.0:
        # Update time
        prev_t = t
        t = rospy.get_time() - tstart
        dt = t - prev_t

        time_msg.current_time = t 
        time_msg.previous_time = prev_t

        update_state(pose_mgr, structure, 1.0/dt)

        # Get new desired state
        desired_state = traj_func(t, speed, traj_vars)

        # Get new control inputs
        [thrust, roll, pitch, yaw] = \
                position_controller(structure, desired_state, dt)

        des_pos  = np.array(desired_state[0])
        is_pos   = structure.state_vector[:3]
        residual = des_pos - is_pos

        update_logs(t, structure.state_vector, desired_state,
                    thrust, roll, pitch, yaw, state_pred)

        # Perform a sim step to predict next state
        state_pred = sim_once(structure, desired_state, 
                                [thrust, roll, pitch, yaw], freq)
 
        # Update message content
        msg.linear.x  = pitch  # pitch [-30, 30] deg
        msg.linear.y  = roll   # roll [-30, 30] deg
        msg.linear.z  = thrust # Thrust ranges 10000 - 60000
        msg.angular.z = yaw    # yaw rate

        if round(t, 2) % 0.5 == 0:
            rospy.loginfo("[{}] {}".format(round(t, 1), 
                                        np.array([thrust, roll, pitch, yaw])))
            rospy.loginfo("     Des={}, Is={}".format(
                                            np.array(desired_state[0]), 
                                            np.array(structure.state_vector[:3])))
            rospy.loginfo("     Fpwm={}".format( thrust ))
            rospy.loginfo("")

        # Send real robot control msg and coexecutor time update
        time_pub.publish(time_msg)
        [ p.publish(msg) for p in publishers ]

        # The sleep preserves sending rate
        rate.sleep()

        # fault_injected, inject_time = \
        #     check_to_inject_fault(t, fault_injected, structure, inject_time)

        # # The New Stuff goes here
        # if not fault_detected:
        #     fault_detected = fault_exists_real(logs)
        #     #fault_detected = True
        #     if fault_detected:
        #         presence_detect_time = t
        #         #break # TODO: Remove temporary break
        # elif not diagnose_mode:
        #     next_diag_t = t + diag_time
        #     diagnose_mode = True
        #     quadrant = get_faulty_quadrant_rotors_real(logs, structure)
        #     print(quadrant)
        #     rotmat = rotpos_to_mat(structure, quadrant, start_id=start_id)
        #     rospy.loginfo("FAULT IS DETECTED, SUSPECTS BELOW")
        #     print(rotmat)
        #     groups = form_groups(quadrant, rotmat)
        #     ramp_rotor_set = [[], groups[0]]
        # else: # Already detected fault presennce and quadrant
        #     if t >= next_diag_t: # Update rotor set
        #         next_diag_t += diag_time
        #         if ramp_rotor_set_idx == len(groups) - 1:
        #             rospy.loginfo("Not implemented actual detection yet.")
        #             break

        #         ramp_rotor_set, ramp_rotor_set_idx = \
        #             update_ramp_rotors(
        #                 structure, t, next_diag_t, groups, 
        #                 ramp_rotor_set_idx, rotmat, ramp_rotor_set
        #             )
        #     else:
        #         ramp_factors = update_ramp_factors(t, next_diag_t, ramp_factors)
        #         apply_ramp_factors(ramp_rotor_set, ramp_factors, structure)

    rospy.loginfo("SHUTDOWN PREPARATIONS")
    rospy.set_param('shutdown_coexecutor', True)
    time_pub.publish(time_msg)
    rospy.loginfo("SHUTDOWN COEXECUTOR")
    _landing(pose_mgr, structure, freq, publishers, msg.linear.z)
    rospy.loginfo("LAND")

    make_plots(inject_time, presence_detect_time)
    rospy.loginfo("DONE")

def make_plots(inject_time, presence_detect_time):
    global logs

    plt.figure()
    ax0 = plt.subplot(3,3,1)
    ax1 = ax0.twinx()
    ax0.plot(logs['t'], logs['spitch'], 'c', label='cmd pitch')
    ax0.plot(logs['t'], logs['pred_pitch'], 'g--', label='pred_pitch')
    ax1.plot(logs['t'], logs['x'], 'r-.', label='xpos')
    ax1.plot(logs['t'], logs['desx'], 'k', label='desx')
    ax1.plot(logs['t'], logs['pred_x'], 'b--', label='predx')
    ax1.set_ylabel("X (m)")
    ax1.set_ylim(-15, 15)
    ax0.set_ylim(-5, 5)
    ax0.set_ylabel("Pitch (deg)")
    ax1.axvline(inject_time, color='grey', linewidth=2.0)
    ax1.axvline(presence_detect_time, color='yellow', linewidth=2.0)
    ax1.legend(loc='lower right')
    ax0.legend(loc='upper left')

    ax2 = plt.subplot(3,3,2)
    ax3 = ax2.twinx()
    ax2.plot(logs['t'], logs['spitch'], 'c', label='cmd pitch')
    ax2.plot(logs['t'], logs['pred_pitch'], 'g--', label='pred_pitch')
    ax3.plot(logs['t'], logs['vx'], 'r-.', label='xvel')
    ax3.plot(logs['t'], logs['desvx'], 'k', label='desvx')
    ax3.plot(logs['t'], logs['pred_vx'], 'b--', label='predvx')
    ax3.set_ylabel("X (m/s)")
    ax2.set_ylabel("Pitch (deg)")
    ax3.set_ylim(-200, 200)
    ax2.set_ylim(-5, 5)
    ax3.axvline(inject_time, color='grey', linewidth=2.0)
    ax3.axvline(presence_detect_time, color='yellow', linewidth=2.0)
    ax2.legend(loc='upper left')
    ax3.legend(loc='lower right')

    ax4 = plt.subplot(3,3,3)
    ax5 = ax4.twinx()
    ax4.plot(logs['t'], logs['spitch'], 'c', label='cmd pitch')
    ax4.plot(logs['t'], logs['pred_pitch'], 'g--', label='pred_pitch')
    ax4.set_ylabel("Cmd Pitch (deg), cyan")
    ax5.plot(logs['t'], logs['pitch'], 'm', label='meas pitch')
    ax5.set_ylabel("Meas Pitch (deg), magenta")
    ax4.set_ylim(-10, 10)
    ax5.set_ylim(-10, 10)
    ax5.axvline(inject_time, color='grey', linewidth=2.0)
    ax5.axvline(presence_detect_time, color='yellow', linewidth=2.0)
    ax5.legend(loc='lower right')
    ax4.legend(loc='upper left')

    ax6 = plt.subplot(3,3,4)
    ax7 = ax6.twinx()
    ax6.plot(logs['t'], logs['sroll'], 'c', label='cmd roll')
    ax6.plot(logs['t'], logs['pred_roll'], 'g--', label='pred_roll')
    ax7.plot(logs['t'], logs['y'], 'r-.', label='ypos')
    ax7.plot(logs['t'], logs['desy'], 'k', label='desy')
    ax7.plot(logs['t'], logs['pred_y'], 'b--', label='predy')
    ax7.set_ylabel("Y (m)")
    ax6.set_ylabel("Roll (deg)")
    ax6.set_ylim(-5, 5)
    ax7.set_ylim(-15, 15)
    ax7.axvline(inject_time, color='grey', linewidth=2.0)
    ax7.axvline(presence_detect_time, color='yellow', linewidth=2.0)
    ax6.legend(loc='upper left')
    ax7.legend(loc='lower right')

    ax8 = plt.subplot(3,3,5)
    ax9 = ax8.twinx()
    ax8.plot(logs['t'], logs['sroll'], 'c', label='cmd roll')
    ax8.plot(logs['t'], logs['pred_roll'], 'g--', label='pred_roll')
    ax9.plot(logs['t'], logs['vy'], 'r-.', label='vy')
    ax9.plot(logs['t'], logs['desvy'], 'k', label='des_vy')
    ax9.plot(logs['t'], logs['pred_vy'], 'b--', label='pred_vy')
    ax9.set_ylabel("Y (m/s)")
    ax8.set_ylabel("Roll (deg)")
    ax8.set_ylim(-5, 5)
    ax9.set_ylim(-200, 200)
    ax9.axvline(inject_time, color='grey', linewidth=2.0)
    ax9.axvline(presence_detect_time, color='yellow', linewidth=2.0)
    ax9.legend(loc='lower right')
    ax8.legend(loc='upper left')

    ax10 = plt.subplot(3,3,6)
    ax10.plot(logs['t'], logs['sroll'], 'c', label='cmd roll')
    ax10.plot(logs['t'], logs['pred_roll'], 'g--', label='pred_roll')
    ax10.set_ylabel("Cmd Roll (deg), cyan")
    ax11 = ax10.twinx()
    ax11.plot(logs['t'], logs['roll'], 'm', label='meas_roll')
    ax11.set_ylabel("Meas Roll (deg), magenta")
    ax10.set_ylim(-10, 10)
    ax11.set_ylim(-10, 10)
    ax11.axvline(inject_time, color='grey', linewidth=2.0)
    ax11.axvline(presence_detect_time, color='yellow', linewidth=2.0)
    ax11.legend(loc='lower right')
    ax10.legend(loc='upper left')

    ax12 = plt.subplot(3,3,7)
    ax13 = ax12.twinx()
    ax12.plot(logs['t'], logs['sthrust'], 'c', label='cmd thrust')
    ax13.plot(logs['t'], logs['z'], 'r-.', label='zpos')
    ax13.plot(logs['t'], logs['desz'], 'k', label='desz')
    ax13.plot(logs['t'], logs['pred_z'], 'b--', label='pred_z')
    ax13.set_ylabel("Z (m)")
    ax12.set_ylabel("Thrust (PWM)")
    ax12.set_ylim(5000, 61000)
    ax13.set_ylim(-0.1, 2.5)
    ax12.set_xlabel('Time (sec)')
    ax13.axvline(inject_time, color='grey', linewidth=2.0)
    ax13.axvline(presence_detect_time, color='yellow', linewidth=2.0)
    ax12.legend(loc='upper left')
    ax13.legend(loc='lower right')

    ax14 = plt.subplot(3,3,8)
    ax15 = ax14.twinx()
    ax14.plot(logs['t'], logs['sthrust'], 'c', label='cmd thrust')
    ax15.plot(logs['t'], logs['vz'], 'r-.', label='vz')
    ax15.plot(logs['t'], logs['desvz'], 'k', label='desvz')
    ax15.plot(logs['t'], logs['pred_vz'], 'b--', label='pred_vz')
    ax15.set_ylabel("Z (m/s)")
    ax14.set_ylabel("Thrust (PWM)")
    ax14.set_ylim(5000, 61000)
    ax15.set_ylim(-200, 200)
    ax14.set_xlabel('Time (sec)')
    ax15.axvline(inject_time, color='grey', linewidth=2.0)
    ax15.axvline(presence_detect_time, color='yellow', linewidth=2.0)
    ax14.legend(loc='upper left')
    ax15.legend(loc='lower right')

    ax16 = plt.subplot(3,3,9)
    ax17 = ax16.twinx()
    ax16.plot(logs['t'], [0 for _ in range(len(logs['t']))], 'k' )
    ax16.plot(logs['t'], logs['syawrate'], 'c', label=r'Sent $\dot{\psi}$')
    ax16.plot(logs['t'], logs['vyaw'], 'm', label=r'Meas $\dot{\psi}$')
    ax16.plot(logs['t'], logs['pred_vyaw'], 'g--', label=r'pred $\dot{\psi}$')
    ax17.plot(logs['t'], logs['yaw'], 'r-.', label='meas yaw')
    ax17.plot(logs['t'], logs['pred_yaw'], 'b--', label='pred yaw')
    ax16.set_ylabel("Cmd Yaw Rate (deg/s), cyan")
    ax17.set_ylabel("Meas Yaw (deg), magenta")
    ax16.set_ylim(-210, 210)
    ax17.set_ylim(-210, 210)
    ax16.set_xlabel('Time (sec)')
    ax17.axvline(inject_time, color='grey', linewidth=2.0)
    ax17.axvline(presence_detect_time, color='yellow', linewidth=2.0)
    ax16.legend(loc='upper left')
    ax17.legend(loc='lower right')

    plt.subplots_adjust(left=0.11, bottom=0.09, right=0.9, top=0.99, wspace=0.36, hspace=0.26)

    plt.show()
    plt.close()

def _takeoff(pose_mgr, structure, freq, publishers):
    global start_id

    rate = rospy.Rate(freq)

    # Publish to robot
    msg = Twist()

    # TAKEOFF
    taken_off = False

    # Message init for takeoff
    msg.linear.x  = 0  # pitch [-30, 30] deg
    msg.linear.y  = 0  # roll [-30, 30] deg
    msg.linear.z  = 0  # Thrust ranges 10000 - 60000
    msg.angular.z = 0  # yaw rate
    pidz_ki = 3500
    rospy.loginfo("Start Control")
    rospy.loginfo("TAKEOFF")
    while not taken_off:
        update_state(pose_mgr, structure, freq)

        if structure.state_vector[2] > 0.05 or msg.linear.z > 50000:
            #msg.linear.z = 0
            structure.pos_accumulated_error = msg.linear.z / pidz_ki
            taken_off = True

        # Convert thrust to PWM range
        msg.linear.z += 10000 * (1.0/freq)

        [ p.publish(msg) for p in publishers ]

        # The sleep preserves sending rate
        rate.sleep()
    rospy.loginfo("COMPLETED TAKEOFF")

def _landing(pose_mgr, structure, freq, publishers, cur_thrust):
    global start_id

    rospy.loginfo("LANDING")
    rate = rospy.Rate(freq)

    # Publish to robot
    msg = Twist()

    # LANDING
    landed = False

    # Message init for takeoff
    msg.linear.x  = 0  # pitch [-30, 30] deg
    msg.linear.y  = 0  # roll [-30, 30] deg
    msg.angular.z = 0  # yaw rate
    msg.linear.z  = cur_thrust
    while not landed:
        update_state(pose_mgr, structure, freq)

        if structure.state_vector[2] <= 0.02 or msg.linear.z < 11000:
            landed = True

        msg.linear.z -= 10000 * (1.0/freq)

        [ p.publish(msg) for p in publishers ]

        # The sleep preserves sending rate
        rate.sleep()

    msg.linear.z = 0
    [ p.publish(msg) for p in publishers ]

def _simple_landing():
    global start_id

    rospy.loginfo("SIMPLE LANDING")
    freq = 50.0 # Hz
    rate = rospy.Rate(freq)

    prefix = 'modquad'
    num_robot = rospy.get_param("num_robots", 4)

    publishers = [ rospy.Publisher('/modquad{:02d}/mq_cmd_vel'.format(mid), Twist, queue_size=100) for mid in range (start_id, start_id + num_robot) ]

    # Publish to robot
    msg = Twist()

    # LANDING
    landed = False

    # Message init for takeoff
    msg.linear.x  = 0  # pitch [-30, 30] deg
    msg.linear.y  = 0  # roll [-30, 30] deg
    msg.angular.z = 0  # yaw rate
    while not landed:

        if msg.linear.z < 11000:
            landed = True

        msg.linear.z -= 1000 * (1.0/freq)

        [ p.publish(msg) for p in publishers ]

        # The sleep preserves sending rate
        rate.sleep()

    msg.linear.z = 0
    [ p.publish(msg) for p in publishers ]

def test_shape_with_waypts(num_struc, wayptset, speed=1, test_id="", 
        doreform=False, max_fault=1, rand_fault=False):

    global traj_func, t, start_id, structure
    # Need to call before reset_docking to ensure it gets right start_id
    start_mod_id = start_id-1 # 0-indexed
    rospy.set_param('start_mod_id', start_mod_id) # 0-indexed

    traj_func = min_snap_trajectory
    traj_vars = traj_func(0, speed, None, wayptset)
    loc=[0,0,0]
    state_vector = init_state(loc, 0)

    mset = structure_gen.rect(2, 2)
    lin_assign(mset, start_id=start_mod_id, reverse=True) # 0-indexed
    #print(mset.pi)
    structure = convert_modset_to_struc(mset, start_mod_id)
    structure.state_vector = state_vector
    structure.traj_vars = traj_vars

    # Verify this is the correct structure
    pi = convert_struc_to_mat(structure.ids, structure.xx, structure.yy)
    print("Structure Used: ")
    print("{}".format(pi.astype(np.int64)))

    rospy.on_shutdown(_simple_landing)

    rospy.init_node('modrotor_simulator')

    time.sleep(2)

    run(speed=speed, traj_vars=traj_vars)

if __name__ == '__main__':
    print("starting simulation")

    # The place, according to mocap, where robot will start
    x = -0.92 # 6.68 # 4.9#  6.3
    y =  5.85 # 0.64 #-0.9# -1.0
    z =   0.0 # 0.0  # 0.5

    num_struc = 1
    results = test_shape_with_waypts(
                       num_struc, 
                       #waypt_gen.zigzag_xy(2.0, 1.0, 6, start_pt=[x-1,y-0.5,0.5]),
                       #waypt_gen.helix(radius=0.5, 
                       #                rise=1.5, 
                       #                num_circ=4, 
                       #                start_pt=[x, y, 0.0]),
                       waypt_gen.waypt_set([[x-0.0  , y+0.00  , 0.0],
                                            [x-0.0  , y+0.00  , 0.1],
                                            [x-0.0  , y+0.00  , 0.2],
                                            [x-0.0  , y+0.00  , 0.5],
                                            [x-0.0  , y+0.00  , 0.6],
                                            [x-0.0  , y+0.00  , 0.5],
                                            [x-0.0  , y+0.00  , 0.6],
                                            [x-0.0  , y+0.00  , 0.4]
                                            #[x+1  , y    , 0.5]
                                           ]),
                       #waypt_gen.waypt_set([[x    , y    , 0.0],
                       #                     [x    , y    , 0.1],
                       #                     [x    , y    , 0.3],
                       #                     [x-1.0, y-0.2, 0.5],
                       #                     [x-1.0, y+0.2, 0.5],
                       #                     [x+1.0, y+0.2, 0.5],
                       #                     [x+1.0, y-0.2, 0.5],
                       #                     [x-1.0, y-0.2, 0.5],
                       #                     [x-1.0, y+0.2, 0.5],
                       #                     [x+1.0, y+0.2, 0.5],
                       #                     [x+1.0, y-0.2, 0.5],
                       #                     [x-1.0, y-0.2, 0.5],
                       #                     [x-1.0, y+0.2, 0.5],
                       #                     [x+1.0, y+0.2, 0.5],
                       #                     [x+1.0, y-0.2, 0.5],
                       #                     [x    , y    , 0.5],
                       #                     [x    , y    , 0.2]
                       #                    ]
                       #                   ),
                       speed=0.1, test_id="controls", 
                       doreform=True, max_fault=1, rand_fault=False)
    print("---------------------------------------------------------------")
