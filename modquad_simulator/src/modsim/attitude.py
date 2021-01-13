import math

from modsim.util.state import state_to_quadrotor
from modsim.params import Gain, Gainset
import numpy as np

en_att_i_gain = False # Used in attitude_controller(...)

# Since desired yaw is always 0 deg for us
persist_yaw_des = 0.0 # Used in cf2_attitude_controller(...)

# TODO: Need to update for sim to 100 Hz?
# Used in cf2_attitude_controller(...)
ATTITUDE_UPDATE_DT = 1.0 / 100.0 #1.0 / 500.0 # 1/500th of a second, i.e. 500 Hz update rate

class AttGains:
    """
    PID Gains from firmware file {firmware}/src/modules/interface/pid.h
    GAINS IN FIRMWARE ARE ON UNITS OF [deg] AND [deg/s] !!!
    """
    #                        KP     KD    KI  I_LIM
    #roll       = Gainset(   6.0,   3.0, 0.00,  20.0 )
    #roll_rate  = Gainset( 250.0, 500.0, 0.00,  33.3 )
    #pitch      = Gainset(   6.0,   3.0, 0.00,  20.0 )
    #pitch_rate = Gainset( 250.0, 500.0, 0.00,  33.3 )
    #yaw        = Gainset(   6.0,   1.0, 0.00, 360.0 )
    #yaw_rate   = Gainset( 120.0,  16.7, 0.00, 166.7 )

    # The below values are defaults encoded in firmware
    # #                        KP     KD    KI  I_LIM
    roll       = Gainset(   6.0,   3.0, 0.00,  20.0 )
    roll_rate  = Gainset( 250.0, 500.0, 2.50,  33.3 )
    pitch      = Gainset(   6.0,   3.0, 0.00,  20.0 )
    pitch_rate = Gainset( 250.0, 500.0, 2.50,  33.3 )
    yaw        = Gainset(   6.0,   1.0, 0.35, 360.0 )
    yaw_rate   = Gainset( 120.0,  16.7, 0.00, 166.7 )

# Inits gains based on firmware values (but no I gains)
att_gains = AttGains() # Used in cf2_attitude_controller(...)

#------------------------------------------------------------------------------------

def update_accum_att_err(structure, att_err):
    global att_gains
    structure.att_accumulated_error += att_err
    structure.att_accumulated_error[0] = \
        max(
            min(structure.att_accumulated_error[0], att_gains.roll.i_lim),
            -att_gains.roll.i_lim
        )
    structure.att_accumulated_error[1] = \
        max(
            min(structure.att_accumulated_error[1], att_gains.pitch.i_lim),
            -att_gains.pitch.i_lim
        )
    structure.att_accumulated_error[2] = \
        max(
            min(structure.att_accumulated_error[2], att_gains.yaw.i_lim),
            -att_gains.yaw.i_lim
        )

def update_accum_att_rate_err(structure, rate_err):
    global att_gains
    structure.att_rate_accumulated_error += rate_err
    structure.att_rate_accumulated_error[0] = \
        max(
            min(structure.att_rate_accumulated_error[0], att_gains.roll_rate.i_lim),
            -att_gains.roll_rate.i_lim
        )
    structure.att_rate_accumulated_error[1] = \
        max(
            min(structure.att_rate_accumulated_error[1], att_gains.pitch_rate.i_lim),
            -att_gains.pitch_rate.i_lim
        )
    structure.att_rate_accumulated_error[2] = \
        max(
            min(structure.att_rate_accumulated_error[2], att_gains.yaw_rate.i_lim),
            -att_gains.yaw_rate.i_lim
        )

def cf2_attitude_controller(structure, control_in, yaw_des):
    global en_att_i_gain, persist_yaw_des, att_gains
    """
    Attitude controller for CF2.x, receiving pwm as input.
    This controller mimics the dual PID controller in the robot firmware.
    The output are forces and moments. F_newtons in Newtons
    :type control_in: tuple defined as (F_newtons, roll_des, pitch_des, yawdot_des)
    :param x:
    :return:
    """
    """
    Sequence of events:
    1) Get new state by state estimator
    2) Get new setpoint (which is initialized to state from Step 1)
    3) Update setpoint based on "current situation"
        a) sitAwPostStateUpdateCallOut(sensorData, state); ] Relates to free 
        b) sitAwPreThrustUpdateCallOut(setpoint);          ] fall detection, etc.
    4) Call controller(...) function, of which below is a part of

    MODES: mode.roll = mode.pitch = modeAbs, mode.yaw = modeVelocity
        -> Thus, only yaw uses the PID_RATE variables 
    
    Once we have updated setpoint, the following occurs in controllerPid(...)
    As per firmware, roll+pitch are in modeAbs and yaw is in modeVelocity
    1) attitudeDesired.yaw += setpoint->attitudeRate.yaw * ATTITUDE_UPDATE_DT;
    2) attitudeDesired.yaw = capAngle(attitudeDesired.yaw);
    3) positionController(&actuatorThrust, &attitudeDesired, setpoint, state);
    4) attitudeControllerCorrectAttitudePID(state->attitude.roll, state->attitude.pitch, state->attitude.yaw,
                                            attitudeDesired.roll, attitudeDesired.pitch, attitudeDesired.yaw,
                                            &rateDesired.roll, &rateDesired.pitch, &rateDesired.yaw);
    5) attitudeControllerCorrectRatePID(sensors->gyro.x, -sensors->gyro.y, sensors->gyro.z,
                             rateDesired.roll, rateDesired.pitch, rateDesired.yaw);
    6) attitudeControllerGetActuatorOutput(&control->roll, &control->pitch, &control->yaw);

    The setpoint_t* is always set to be the last-measured-state, somewhat unintuitively

    Seemingly, in the mode we use only the yawRate gains are used, not the pitchRate or rollRate ones
        => Hence why below is simpler than the firmware
    """

    #global accumulated_error
    x = structure.state_vector
    F_newtons = control_in[0]
    roll_des = control_in[1]
    pitch_des = control_in[2]
    yawdot_des = control_in[3]

    # For why this is written, see src/modules/src/crtp_commander_rpyt.c Lines 192-231
    pitch_rate_des = 0.0 # Because roll and pitch operate in modeAbs
    roll_rate_des  = 0.0 # Because roll and pitch operate in modeAbs
    yaw_rate_des   = yawdot_des # Because yaw operates in modeVelocity

    persist_yaw_des += yaw_rate_des * ATTITUDE_UPDATE_DT
    # Cap the angle
    while persist_yaw_des > 180.0:
        persist_yaw_des -= 360.0
    while persist_yaw_des < -180.0:
        persist_yaw_des += 360.0

    # Quaternion to angles (in radians, seems like)
    quad_state = state_to_quadrotor(x)

    # Errors are stored in degrees
    # math.degrees(...) needed because state_to_quadrotor(...) calls
    #  transforms3d.quad2euler(quat), which converts to radians
    att_err  = [roll_des        - math.degrees(quad_state.euler[0]),
                pitch_des       - math.degrees(quad_state.euler[1]),
                persist_yaw_des - math.degrees(quad_state.euler[2]) ]

    # TODO: Verify omega is already in deg/s
    rate_err = [roll_rate_des  - quad_state.omega[0],
                pitch_rate_des - quad_state.omega[1],
                yaw_rate_des   - quad_state.omega[2] ]

    # Update accumulated errors taking integral caps into account
    update_accum_att_err     (structure, np.array(att_err ))
    update_accum_att_rate_err(structure, np.array(rate_err))

    #import pdb; pdb.set_trace()

    # Compute the moments
    Mx = att_gains.pitch.p.k * att_err[0] + \
         att_gains.pitch.d.k * (0 - quad_state.omega[0]) + \
         att_gains.pitch.i.k * structure.att_accumulated_error[0]

    My = att_gains.roll.p.k  * att_err[1] + \
         att_gains.roll.d.k  * (0 - quad_state.omega[1]) + \
         att_gains.roll.i.k  * structure.att_accumulated_error[1]

    Mz = att_gains.yaw.p.k   * att_err[2] + \
         att_gains.yaw.d.k   * (0 - quad_state.omega[2]) + \
         att_gains.yaw.i.k   * structure.att_accumulated_error[2]

    #import pdb; pdb.set_trace()

    # F_newtons is unchanged from the input control_in
    return F_newtons, [Mx, My, 0]

#------------------------------------------------------------------------------

def enable_attitude_i_gain():
    global en_att_i_gain
    en_att_i_gain = True

def disable_attitude_i_gain():
    global en_att_i_gain
    en_att_i_gain = False

def attitude_controller(structure, control_in, yaw_des):
    global en_att_i_gain
    """
    Attitude controller for crazyflie, receiving pwm as input.
    the output are forces and moments. F_newtons in Newtons
    This is NOT the same as the firmware controller (see prev function for that)
    :type control_in: tuple defined as (F_newtons, roll_des, pitch_des, yawdot_des)
    :param x:
    :return:
    """
    #global accumulated_error
    x = structure.state_vector
    F_newtons = control_in[0]
    roll_des = control_in[1]
    pitch_des = control_in[2]
    yawdot_des = control_in[3]

    ### Moments
    # Quaternion to angles
    quad_state = state_to_quadrotor(x)


    # Where are these numbers from? -> From Nanokontrol Kumar lab repo
    #kpx, kdx, kix = 1.43e-5 * 250, 1.43e-5 * 60, .0002 # ORIGINAL
    if en_att_i_gain:
        num_mod = len(structure.xx)
        kpx, kdx, kix = 1.43e-5 * 250, 1.43e-5 * 60, .0002 / (num_mod/2.5)
    else:
        kpx, kdx, kix = 1.43e-5 * 250, 1.43e-5 * 60, .0000

    e = [max(min(math.radians(roll_des)  - quad_state.euler[0], 0.05), -0.05),
         max(min(math.radians(pitch_des) - quad_state.euler[1], 0.05), -0.05),
         max(min(math.radians(yaw_des)   - quad_state.euler[2], 0.05), -0.05)]

    structure.att_accumulated_error += e
    #print(accumulated_error[0])

    Mx = kpx * e[0] + \
         kdx * (0 - quad_state.omega[0]) + \
         kix * structure.att_accumulated_error[0]
    My = kpx * e[1] + \
         kdx * (0 - quad_state.omega[1]) + \
         kix * structure.att_accumulated_error[1]
    #print(F_newtons, Mx, My)
    #print('---')
    return F_newtons, [Mx, My, 0]
#------------------------------------------------------------------------------
