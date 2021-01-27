import rospy
import modsim.params as params
from math import sin, cos
import numpy as np
from math import sqrt
import math
from tf.transformations import euler_from_quaternion
from modsim.util.thrust import convert_thrust_newtons_to_pwm
from modsim.params import ATTITUDE_UPDATE_DT, Gainset, Gain
from transforms3d.euler import quat2euler

class CtrlVars:
    """
    This class of variables is based on firmware file controller_pid.c
    These variables are all static-declared at top of file
    """
    des_att      = [0.0, 0.0, 0.0] # Roll, pitch, yaw
    des_att_rate = [0.0, 0.0, 0.0] # Roll, pitch, yaw rates
    act_thrust   = 0.0             # Thrust (N?)

    cmd_thrust   = 0.0
    cmd_roll     = 0.0
    cmd_pitch    = 0.0
    cmd_yaw      = 0.0

    r_roll       = 0.0
    r_pitch      = 0.0
    r_yaw        = 0.0

    accel_z      = 0.0

class ControlSetpoint: # Mimics the control_t struct in firmware
    roll    = 0.0
    pitch   = 0.0
    yawdot  = 0.0
    thrust  = 0.0

class FirmwarePosVelCtrllerParams:
    """
    Values come from firmware file position_controller_pid.c
        static struct this_s this = {...}
    """
    dt = 0.0 # Need to set this?
    vx = Gainset(25.0, 0.0,  1.0)
    vy = Gainset(25.0, 0.0,  1.0)
    vz = Gainset(25.0, 0.0, 15.0)
    x  = Gainset( 2.0, 0.0,  0.0)
    y  = Gainset( 2.0, 0.0,  0.0)
    z  = Gainset( 2.0, 0.0,  0.5)

    roll       = Gainset(   6.0,   3.0, 0.00,  20.0 )
    roll_rate  = Gainset( 250.0, 500.0, 2.50,  33.3 )
    pitch      = Gainset(   6.0,   3.0, 0.00,  20.0 )
    pitch_rate = Gainset( 250.0, 500.0, 2.50,  33.3 )
    yaw        = Gainset(   6.0,   1.0, 0.35, 360.0 )
    yaw_rate   = Gainset( 120.0,  16.7, 0.00, 166.7 )

    thrust_base  = 36000
    thrust_min   = 20000
    thrust_scale = 1000.0

    # prev errors
    prev_v_err = [0.0, 0.0, 0.0]
    prev_p_err = [0.0, 0.0, 0.0]
    prev_att_err = [0.0, 0.0, 0.0]
    prev_att_rate_err = [0.0, 0.0, 0.0]

    # Accumulated errors
    accum_v_err = [0.0, 0.0, 0.0]
    accum_p_err = [0.0, 0.0, 0.0]
    accum_att_err = [0.0, 0.0, 0.0]
    accum_att_rate_err = [0.0, 0.0, 0.0]

"""
void controllerPid(control_t *control, setpoint_t *setpoint,
                                         const sensorData_t *sensors,
                                         const state_t *state,
                                         const uint32_t tick)
"""


# Make class instances to init the constant gains
pos_vel_params = FirmwarePosVelCtrllerParams()
ctrl_vars = CtrlVars()

# Make arrays for easier computation in PID loops
att_p = np.array( [ pos_vel_params.roll.p.k, 
                    pos_vel_params.pitch.p.k, 
                    pos_vel_params.yaw.p.k    ]   )
att_i = np.array( [ pos_vel_params.roll.i.k, 
                    pos_vel_params.pitch.i.k, 
                    pos_vel_params.yaw.i.k    ]   )
att_d = np.array( [ pos_vel_params.roll.d.k, 
                    pos_vel_params.pitch.d.k, 
                    pos_vel_params.yaw.d.k    ]   )

att_rate_p = np.array( [ pos_vel_params.roll_rate.p.k, 
                         pos_vel_params.pitch_rate.p.k, 
                         pos_vel_params.yaw_rate.p.k    ]   )
att_rate_i = np.array( [ pos_vel_params.roll_rate.i.k, 
                         pos_vel_params.pitch_rate.i.k, 
                         pos_vel_params.yaw_rate.i.k    ]   )
att_rate_d = np.array( [ pos_vel_params.roll_rate.d.k, 
                         pos_vel_params.pitch_rate.d.k, 
                         pos_vel_params.yaw_rate.d.k    ]   )

def cf2_firmware_controller(ctrl_setpt, structure, t, att_rate, pos_rate):
    global ctrl_vars
    """
    This is meant to mirror the controllerPid function in the firmware file
        controller_pid.c
    :param att_rate: Frq of attitude controller
    :param pos_rate: Frq of pos controller
    """
    update_rate_ratio = float(att_rate) / pos_rate
    print(update_rate_ratio, int(update_rate_ratio))
    assert update_rate_ratio == int(update_rate_ratio)

    dt = 1.0 / att_rate

    # 1) Update attitudeDesired.yaw based on ctrl_setpt yawrate
    ctrl_vars.des_att[2] += ctrl_setpt.yawdot * ATTITUDE_UPDATE_DT

    # 2) Call the firmware positionController, which internally calls
    #       velocityController
    if t % update_rate_ratio == 0:
        # Position part of controller does nothing
        # Velocity always gets desired vel = [0,0,0]
        # Affects the desired attitude (in ctrl_vars)
        cf2_firmware_pos_controller(ctrl_vars, ctrl_setpt, structure)

    # 3) mode.x = mode.y = mode.z = modeDisabled, so...
    ctrl_vars.act_thrust = ctrl_setpt.thrust
    ctrl_vars.des_att[0] = ctrl_setpt.roll
    ctrl_vars.des_att[1] = ctrl_setpt.pitch
    ctrl_vars.des_att[2] = 0 # Never want to yaw

    # 4) Attitude PID
    #    attitudeControllerCorrectAttitudePID(
	#    	state->attitude.roll, state->attitude.pitch, state->attitude.yaw,
    #       attitudeDesired.roll, attitudeDesired.pitch, attitudeDesired.yaw,
    #          &rateDesired.roll,    &rateDesired.pitch,    &rateDesired.yaw );
    #    ctrl_vars.des_att_rate is modified in this function
    cf2_firmware_att_controller(structure, ctrl_vars, pos_vel_params, dt)

    # 5) Skip lines conditional on mode.roll/pitch being in modeVelocity
    #    --> They are in modeAbs

    # 6) Attitude Rate PID
	#     attitudeControllerCorrectRatePID(
	#   	sensors->gyro.x, -sensors->gyro.y, sensors->gyro.z,
    #       rateDesired.roll, rateDesired.pitch, rateDesired.yaw);
	# This is the function that tells us the control outputs for RPY
	# Since actuatorControllerGetActuatorOutput(...) just copies output 
	#	of this function to control, we can do that directly
    cmd_att = cf2_firmware_att_rate_controller(
                                    structure, ctrl_vars, pos_vel_params, dt)

    # 8) Update commands
    #ctrl_vars.cmd_roll = cmd_att[0]
    #ctrl_vars.cmd_pitch = cmd_att[1]
    #ctrl_vars.cmd_yaw = cmd_att[2]
    #ctrl_vars.cmd_thrust = ctrl_vars.act_thrust

    return [cmd_att[0], cmd_att[1], cmd_att[2], ctrl_vars.act_thrust]

def cf2_firmware_att_rate_controller(structure, ctrl_vars, pos_vel_params, dt):
    """
    Mimics firmware function 
    void attitudeControllerCorrectRatePID(
       float rollRateActual, float pitchRateActual, float yawRateActual,
       float rollRateDesired, float pitchRateDesired, float yawRateDesired)
    """
    state_att_rate = structure.state_vector[-3:] # Units?
    err = ctrl_vars.des_att_rate - state_att_rate
    deriv = (err - pos_vel_params.prev_att_rate_err) / dt
    pos_vel_params.accum_att_rate_err += err
    pos_vel_params.accum_att_rate_err[0] = \
        max(min(pos_vel_params.accum_att_rate_err[0], 
                pos_vel_params.roll_rate.i_lim
               ),
                -pos_vel_params.roll_rate.i_lim
           )
    pos_vel_params.accum_att_rate_err[1] = \
        max(min(pos_vel_params.accum_att_rate_err[1], 
                pos_vel_params.pitch_rate.i_lim
               ),
                -pos_vel_params.pitch_rate.i_lim
           )
    pos_vel_params.accum_att_rate_err[2] = \
        max(min(pos_vel_params.accum_att_rate_err[2], 
                pos_vel_params.yaw_rate.i_lim
               ),
                -pos_vel_params.yaw_rate.i_lim
           )
    integ = pos_vel_params.accum_att_rate_err

    return att_rate_p * err + att_rate_d * deriv + att_rate_i * integ


def cf2_firmware_att_controller(structure, ctrl_vars, pos_vel_params, dt):
    """
    Mimics firmware function attitudeControllerCorrectAttitudePID(...)
        called in firmware file controller_pid.c
    ----
    # attitudeControllerCorrectAttitudePID(
    #     state->attitude.roll, state->attitude.pitch, state->attitude.yaw,
    #     attitudeDesired.roll, attitudeDesired.pitch, attitudeDesired.yaw,
    #       &rateDesired.roll,    &rateDesired.pitch,    &rateDesired.yaw );
    """
    global att_p, att_d, att_i

    # 1) Find desired roll rate as output of PID on roll
    # 2) Find desired pitch rate as output of PID on pitch
    # 3) Find desired yaw rate as output of PID on yaw

    euler = [math.degrees(i) for i in quat2euler(structure.state_vector[6:10])]

    err = np.array(ctrl_vars.des_att) - np.array(euler)
    deriv = (err - np.array(pos_vel_params.prev_att_err)) / dt
    pos_vel_params.accum_att_err += err
    pos_vel_params.accum_att_err[0] = max(min(pos_vel_params.accum_att_err[0], 
                                              pos_vel_params.roll.i_lim
                                             ),
                                              -pos_vel_params.roll.i_lim
                                         )
    pos_vel_params.accum_att_err[1] = max(min(pos_vel_params.accum_att_err[1], 
                                              pos_vel_params.pitch.i_lim
                                             ),
                                              -pos_vel_params.pitch.i_lim
                                         )
    pos_vel_params.accum_att_err[2] = max(min(pos_vel_params.accum_att_err[2], 
                                              pos_vel_params.yaw.i_lim
                                             ),
                                              -pos_vel_params.yaw.i_lim
                                         )
    integ = pos_vel_params.accum_att_err

    # Bound yaw error
    if err[2] > 180.0:
       err[2] -= 360.0
    elif err[2] < -180:
       err[2] += 360.0

    # Run the PID computation
    ctrl_vars.des_att_rate = att_p * err    + \
                             att_d * deriv  + \
                             att_i * integ

def cf2_firmware_pos_controller(ctrl_vars, ctrl_setpt, structure):
    """
    Mimics the function positionController(...) called by controllerPid(...) in
        firmware file controller_pid.c
    """
    # Since the X,Y,Z control are in modeDisabled in firmware,
    #   this does not actually do much
    # However, we call velocity controller from in this function because
    #   if X,Y,Z control NOT in modeDisabled, this modified ctrl_setpt
    # TODO: Implement other modes

    cf2_firmware_vel_controller(ctrl_vars, ctrl_setpt, structure)

def cf2_firmware_vel_controller(ctrl_vars, ctrl_setpt, structure):
    global pos_vel_params
    """
    Mimics the function velocityController(...) called by
        positionController(...) called by controllerPid(...) in firmware file
        controller_pid.c
    """

    #rollRaw  = runPid(state->velocity.x, &this.pidVX, setpoint->velocity.x, DT);

    dt = 1 / 100.0 # Frq of position controller in firmware is 100 Hz

    raw_roll, raw_pitch = \
        firmware_pid_update_rp(structure, pos_vel_params, ctrl_setpt, dt)

    euler = np.array(quat2euler(structure.state_vector[6:10]))
    yaw_rad = euler[2]

    # Roll
    ctrl_vars.des_att[0] = - (raw_roll  * math.cos(yaw_rad))   \
                           - (raw_pitch * math.sin(yaw_rad))
    # Pitch
    ctrl_vars.des_att[1] = - (raw_pitch * math.cos(yaw_rad)) \
                           + (raw_roll  * math.sin(yaw_rad))

    # Thrust
    state_vel_z = structure.state_vector[5]
    setpt_vel_z = 0 # Verified with Vicon-in-loop trajectory test

    err_vel_z = setpt_vel_z - state_vel_z
    prev_err_vel_z = pos_vel_params.prev_v_err[2]
    deriv_vel_z = (err_vel_z - prev_err_vel_z) / dt

    pos_vel_params.accum_v_err[2] += err_vel_z

    thrust_raw = pos_vel_params.vz.p.k * err_vel_z + \
                 pos_vel_params.vz.d.k * deriv_vel_z + \
                 pos_vel_params.vz.i.k * pos_vel_params.accum_v_err[2]

    ctrl_vars.act_thrust = \
       max(thrust_raw * pos_vel_params.thrust_scale + pos_vel_params.thrust_base,
          pos_vel_params.thrust_min)

def firmware_pid_update_rp(structure, pos_vel_params, ctrl_setpt, dt):
    """
    Mimics the runPid function called in firmware file position_controller_pid.c
    """
    # To compute PID outputs we need
    #   float rollRaw  = runPid(state->velocity.x, &this.pidVX, setpoint->velocity.x, DT);
    #   float pitchRaw = runPid(state->velocity.y, &this.pidVY, setpoint->velocity.y, DT);

    state_vel_x = structure.state_vector[3]
    state_vel_y = structure.state_vector[4]

    #pid_vx = pos_vel_params.vx
    #pid_vy = pos_vel_params.vy

    # Verified with CF2 flying a trajectory using Vicon-in-loop
    setpt_vel_x = 0 # Seems to be the case based on some simple tests
    setpt_vel_y = 0 # Seems to be the case based on some simple tests

    error = np.array([setpt_vel_x - state_vel_x, setpt_vel_y - state_vel_y, 0])

    deriv = (error - pos_vel_params.prev_v_err) / dt

    pos_vel_params.accum_v_err += error

    raw_roll = pos_vel_params.vx.p.k * error[0] + \
               pos_vel_params.vx.d.k * deriv[0] + \
               pos_vel_params.vx.i.k * pos_vel_params.accum_v_err[0]

    raw_pitch = pos_vel_params.vy.p.k * error[1] + \
                pos_vel_params.vy.d.k * deriv[1] + \
                pos_vel_params.vy.i.k * pos_vel_params.accum_v_err[1]

    return raw_roll, raw_pitch

# Import here to avoid circ dependency issue
from modsim.datatype.structure import Structure
