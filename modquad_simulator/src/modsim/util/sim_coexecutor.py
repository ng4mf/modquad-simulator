#!/usr/bin/env python

"""
This is not meant to be run for the simulation-only experiments.
This file runs alongside the real robots.
This takes the newly measured state and next desired state and 
    predict the next state.
This is useful in fault detection, when we want to compare where the
    structure should have ended up and where at actually did end up.
"""

import rospy
import tf2_ros
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from modquad_simulator.msg import ModquadCoexecutorTime
import numpy as np
from numpy import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import sys
import json
from tf.transformations import euler_from_quaternion
import math

from modsim.controller import position_controller, modquad_torque_control
from modsim.trajectory import circular_trajectory, simple_waypt_trajectory, \
    min_snap_trajectory

from modsim import params
from modsim.attitude import attitude_controller
# from modsim.plot.drawer_vispy import Drawer

from modsim.datatype.structure import Structure

# Functions to publish odom, transforms, and acc
from modsim.util.comm import publish_acc, publish_odom, \
                             publish_transform_stamped, \
                             publish_odom_relative,     \
                             publish_transform_stamped_relative, \
                             publish_structure_acc,     \
                             publish_acc_for_attached_mods, \
                             publish_structure_odometry, \
                             publish_odom_for_attached_mods

from modsim.util.state import init_state, state_to_quadrotor
from modquad_simulator.srv import Dislocation, DislocationResponse
from modsim.simulation.ode_integrator import simulation_step

# Fault detection functions
from modsim.util.fault_detection import fault_exists,               \
                                        get_faulty_quadrant_rotors, \
                                        update_ramp_rotors,         \
                                        update_ramp_factors,        \
                                        form_groups,                \
                                        update_rotmat

from modsim.util.fault_injection import inject_faults
from modsim.util.thrust import convert_thrust_pwm_to_newtons

from modquad_sched_interface.interface import convert_modset_to_struc, \
                                              convert_struc_to_mat   , \
                                              rotpos_to_mat

import modquad_sched_interface.waypt_gen as waypt_gen
import modquad_sched_interface.structure_gen as structure_gen

#from scheduler.gsolver import gsolve
from modquad_sched_interface.simple_scheduler import lin_assign

sim_coexec_data = {
    'structure' : None,
    'traj_func' : None,
    'traj_vars' : None,
    'pose_mgr'  : None,
    'tf_broad'  : None,
    'odom_pub'  : None
}

logs = {
	't':     [], 'sthrust': [],                 # time, desired thrust
	'x':     [], 'y':       [], 'z':        [], # measured position
	'vx':    [], 'vy':      [], 'vz':       [], # measured linear velocity
	'roll':  [], 'pitch':   [], 'yaw':      [], # measured attitude
	'vroll': [], 'vpitch':  [], 'vyaw':     [], # measured angular velocity
	'sroll': [], 'spitch':  [], 'syawrate': [], # desired roll, pitch, yaw rate
	'desx':  [], 'desy':    [], 'desz':     [], # desired position
	'desvx': [], 'desvy':   [], 'desvz':    []  # desired linear velocity
}

def setup_sim_node(structure, traj_func, traj_vars, pose_mgr):
    global sim_coexec_data

    sim_coexec_data['structure'] = structure
    sim_coexec_data['traj_func'] = traj_func
    sim_coexec_data['traj_vars'] = traj_vars
    sim_coexec_data['pose_mgr' ] = pose_mgr
    sim_coexec_data['tf_broad' ] = tf2_ros.TransformBroadcaster()
    sim_coexec_data['odom_pub' ] = rospy.Publisher('/sim_coexecutor/odom', \
                                                    Odometry, queue_size=1)

    rospy.set_param('sim_coexecutor_is_setup', True)
    rospy.loginfo("Sim coexec SETUP COMPLETE")

def _extract_time_data(time_msg):
    return time_msg.t, time_msg.t - time_msg.prev_t

def _publish_prediction(odom_pub, state):
    """
    Publishes an Odometry message that predicts the next state
    Convert predicted state into an odometry message and publish it.
    :param x:
    :param pub:
    """
    publish_odom(state, odom_pub, child_frame='modquad_coexec')

def update_logs(t, state_vector, desired_state, thrust, roll, pitch, yawrate):
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

def sim_once(structure, desired_state, pos_ctrl, freq):
    global sim_coexec_data
    #pose_mgr        = sim_coexec_data[ 'pose_mgr'  ]
    tf_broadcaster  = sim_coexec_data[ 'tf_broad'  ]
    #structure       = sim_coexec_data[ 'structure' ]
    #traj_func       = sim_coexec_data[ 'traj_func' ]
    #traj_vars       = sim_coexec_data[ 'traj_vars' ]
    odom_pub        = sim_coexec_data[ 'odom_pub'  ]
    en_motor_sat    = True
    thrust_pwm      = pos_ctrl[0]
    roll            = pos_ctrl[1]
    pitch           = pos_ctrl[2]
    yawrate         = pos_ctrl[3]
    yaw_des         = 0
    dt              = 1.0 / freq

    thrust_newtons = convert_thrust_pwm_to_newtons(thrust_pwm)

    # Simulate what runs onboard the real robots
    F_single, M_single = attitude_controller(structure,
                            (thrust_newtons, roll, pitch, yawrate), yaw_des)

    # Distribute amongst all modules in structure
    F_structure, M_structure, rotor_forces = \
        modquad_torque_control( F_single, M_single, structure, en_motor_sat )

    # Get the new state - this is a prediction of state at time t+dt
    state_pred = simulation_step( structure, structure.state_vector,
                                  F_structure, M_structure, dt      )

    # Publish prediction, which is used to generate residual
    #_publish_prediction(odom_pub, state_pred)
    return state_pred

def sim_thread():
    global sim_coexec_data

    odom_pub = None
    odom_pub = rospy.Publisher('/sim_coexecutor/odom', Odometry, queue_size=1)

    # Don't start execution until the setup is complete
    rospy.loginfo("Waiting for sim_coexecutor setup completion")

    rate = rospy.Rate(10) # Hz
    sim_coexecutor_is_setup =  rospy.get_param('sim_coexecutor_is_setup', False)
    while not sim_coexecutor_is_setup:
        sim_coexecutor_is_setup = rospy.get_param('sim_coexecutor_is_setup')
        rate.sleep()

    rate = rospy.Rate(100)
        
    pose_mgr        = sim_coexec_data[ 'pose_mgr'  ]
    tf_broadcaster  = sim_coexec_data[ 'tf_broad'  ]
    structure       = sim_coexec_data[ 'structure' ]
    traj_func       = sim_coexec_data[ 'traj_func' ]
    traj_vars       = sim_coexec_data[ 'traj_vars' ]
    speed           = rospy.get_param('structure_speed')
    time_sub        = rospy.Subscriber( '/modquad_structure/time',
                                        ModquadCoexecutorTime,
                                        _extract_time_data       )

    assert (odom_pub       is not None), "odom_pub is None!?"
    assert (structure      is not None), "structure is None!?"
    assert (tf_broadcaster is not None), "tf_broadcaster is None!?"
    assert (traj_func      is not None), "traj_func is None!?"
    assert (traj_vars      is not None), "traj_vars is None!?"

    en_motor_sat = True
    shutdown_param_set = rospy.get_param('shutdown_coexecutor', False)
    rospy.loginfo("STARTING SIMULATED COEXECUTOR")
    
    while not rospy.is_shutdown() or shutdown_param_set:
        
        rospy.loginfo("Coexecutor wait for time msg")
        t_data = rospy.wait_for_message(time_topic, time_msg)
        t, dt = None, None
        try:
            t, dt  = _extract_time_data(t_data)
        except:
            rospy.loginfo("Time msg had a problem")

        desired_state = traj_func(t, speed, traj_vars)

        # Get new control input
        [thrust_pwm, roll, pitch, yawrate] = \
            position_controller(structure, desired_state, dt)
        yaw_des = 0

        thrust_newtons = convert_thrust_pwm_to_newtons(thrust_pwm)

        # Simulate what runs onboard the real robots
        F_single, M_single = attitude_controller(structure,
                                (thrust_newtons, roll, pitch, yawrate), yaw_des)

        # Distribute amongst all modules in structure
        F_structure, M_structure, rotor_forces = \
            modquad_torque_control( F_single, M_single, structure, en_motor_sat )

        # Get the new state - this is a prediction of state at time t+dt
        state_pred = simulation_step( structure, structure.state_vector,
                                      F_structure, M_structure, dt      )

        update_logs(t, state_pred, desired_state, thrust, roll, pitch, yawrate)

        # Publish prediction, which is used to generate residual
        _publish_prediction(odom_pub, state_pred)

        # Check if real robot script has finished running
        shutdown_param_set = rospy.get_param('shutdown_coexecutor')

        # To prevent faster than 100 Hz execution of co-executor
        # Since we wait for time message, this is unneeded
        #rate.sleep()
    rospy.loginfo("Coexecutor shutting down")

    make_plots()
    rospy.loginfo("DONE")

def make_plots():
    global logs

    plt.figure()
    ax0 = plt.subplot(3,3,1)
    ax1 = ax0.twinx()
    ax0.plot(logs['t'], logs['spitch'], 'c')
    ax1.plot(logs['t'], logs['x'], 'r', label='xpos')
    ax1.plot(logs['t'], logs['desx'], 'k', label='desx')
    ax1.set_ylabel("X (m)")
    ax1.legend(loc='lower right')
    ax1.set_ylim(-10, 10)
    ax0.set_ylim(-5, 5)
    ax0.set_ylabel("Pitch (deg)")

    ax2 = plt.subplot(3,3,2)
    ax3 = ax2.twinx()
    ax2.plot(logs['t'], logs['spitch'], 'c')
    ax3.plot(logs['t'], logs['vx'], 'r', label='xvel')
    ax3.plot(logs['t'], logs['desvx'], 'k', label='desvx')
    ax3.legend(loc='lower right')
    ax3.set_ylabel("X (m/s)")
    ax2.set_ylabel("Pitch (deg)")
    #ax3.set_ylim(-200, 200)
    ax2.set_ylim(-5, 5)

    ax4 = plt.subplot(3,3,3)
    ax5 = ax4.twinx()
    ax4.plot(logs['t'], logs['spitch'], 'c', label='des pitch')
    ax4.legend(loc='lower right')
    ax4.set_ylabel("Cmd Pitch (deg), cyan")
    ax5.plot(logs['t'], logs['pitch'], 'm')
    ax5.set_ylabel("Meas Pitch (deg), magenta")
    ax4.set_ylim(-10, 10)
    ax5.set_ylim(-10, 10)

    ax6 = plt.subplot(3,3,4)
    ax7 = ax6.twinx()
    ax6.plot(logs['t'], logs['sroll'], 'c' )
    ax7.plot(logs['t'], logs['y'], 'g', label='ypos')
    ax7.plot(logs['t'], logs['desy'], 'k', label='desy')
    ax7.legend(loc='lower right')
    ax7.set_ylabel("Y (m)")
    ax6.set_ylabel("Roll (deg)")
    ax6.set_ylim(-5, 5)
    ax7.set_ylim(-10, 10)

    ax8 = plt.subplot(3,3,5)
    ax9 = ax8.twinx()
    ax8.plot(logs['t'], logs['sroll'], 'c' )
    ax9.plot(logs['t'], logs['vy'], 'g', label='vy')
    ax9.plot(logs['t'], logs['desvy'], 'k', label='des_vy')
    ax9.legend(loc='lower right')
    ax9.set_ylabel("Y (m/s)")
    ax8.set_ylabel("Roll (deg)")
    ax8.set_ylim(-5, 5)
    #ax9.set_ylim(-200, 200)

    ax10 = plt.subplot(3,3,6)
    ax10.plot(logs['t'], logs['sroll'], 'c', label='des_roll')
    ax10.set_ylabel("Cmd Roll (deg), cyan")
    ax11 = ax10.twinx()
    ax11.plot(logs['t'], logs['roll'], 'm', label='meas_roll')
    ax11.set_ylabel("Meas Roll (deg), magenta")
    ax10.set_ylim(-10, 10)
    ax11.set_ylim(-10, 10)

    ax12 = plt.subplot(3,3,7)
    ax13 = ax12.twinx()
    ax12.plot(logs['t'], logs['sthrust'], 'c' )
    ax13.plot(logs['t'], logs['z'], 'b', label='zpos')
    ax13.plot(logs['t'], logs['desz'], 'k', label='desz')
    ax13.legend(loc='lower right')
    ax13.set_ylabel("Z (m)")
    ax12.set_ylabel("Thrust (PWM)")
    ax12.set_ylim(5000, 61000)
    ax13.set_ylim(-0.1, 1.0)
    ax12.set_xlabel('Time (sec)')

    ax14 = plt.subplot(3,3,8)
    ax15 = ax14.twinx()
    ax14.plot(logs['t'], logs['sthrust'], 'c' )
    ax15.plot(logs['t'], logs['vz'], 'b', label='vz')
    ax15.plot(logs['t'], logs['desvz'], 'k', label='desvz')
    ax15.legend(loc='lower right')
    ax15.set_ylabel("Z (m/s)")
    ax14.set_ylabel("Thrust (PWM)")
    ax14.set_ylim(5000, 61000)
    #ax15.set_ylim(-20, 20)
    ax14.set_xlabel('Time (sec)')

    ax16 = plt.subplot(3,3,9)
    ax17 = ax16.twinx()
    ax16.plot(logs['t'], [0 for _ in range(len(logs['t']))], 'k' )
    ax16.plot(logs['t'], logs['syawrate'], 'c', label=r'Sent $\dot{\psi}$')
    ax16.plot(logs['t'], logs['vyaw'], 'm', label=r'Meas $\dot{\psi}$')
    ax17.plot(logs['t'], logs['yaw'], 'b')
    ax16.set_ylabel("Cmd Yaw Rate (deg/s), cyan")
    ax17.set_ylabel("Meas Yaw (deg), magenta")
    ax16.set_ylim(-210, 210)
    ax17.set_ylim(-210, 210)
    ax16.set_xlabel('Time (sec)')
    ax17.legend(loc='lower right')

    plt.subplots_adjust(left=0.11, bottom=0.09, right=0.9, top=0.99, wspace=0.36, hspace=0.26)

    plt.show()
    plt.close()

if __name__ == '__main__':
    #rospy.init_node('sim_coexecuter')
    rospy.set_param('sim_coexecutor_is_setup', False) # Initialize
    rospy.loginfo('Coexecutor node is running')
    #sim_thread()
