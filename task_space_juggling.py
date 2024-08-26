"""
    Example of kinematic planning for toss juggling.
    Cartesian space trajectories are tracked by free floating hands.

    mail@kaiploeger.net
"""

from pathlib import Path
import time

import casadi as cas
import numpy as np

import mujoco as mj
import mujoco.viewer  # needs to get imported explicitly

from math_helpers import Rx, rotation_matrix_to_axis_angle, cas_cross_product
from mujoco_wrappers import Hand, Ball



# simulation settings

PLANNER_TIMESTEP = 0.02
NUM_SUBSTEPS = 10
SIM_TIMESTEP = PLANNER_TIMESTEP/NUM_SUBSTEPS
XML_PATH = (Path(__file__).parent / "two_cones.xml").as_posix()
GRAVITY = np.array([0, 0, -9.81])


# juggling pattern

N_BALLS = 7            # number of balls
K_CYCLE = 22            # duration between two throws
K_DWELL = 11            # duration a ball spends in a hand

K_FLIGHT = int(K_CYCLE * N_BALLS / 2 - K_DWELL)              # flight and vacant time are dependent
K_VACANT = int((K_FLIGHT+K_DWELL) * 2 / N_BALLS - K_DWELL)   # according to Shannon's juggling theorem

HAND_HEIGHT = 1.0               # level of catching plane
HAND_WIDTH = 0.9                # left-right distance between target catching positions
CARRY_DISTANCE = 0.15           # how far to carry each ball inward
HAND_TILT =  11 / 180 * np.pi   # positive = inward

TARGET_BALL_POS_TOUCHTDOWN_LEFT = np.array([0, -HAND_WIDTH/2, HAND_HEIGHT])    # ball touch down pos left
TARGET_BALL_POS_TOUCHDOWN_RIGHT = np.array([0, HAND_WIDTH/2, HAND_HEIGHT])     # ball touch down pos right
TARGET_BALL_POS_TAKEOFF_LEFT = TARGET_BALL_POS_TOUCHTDOWN_LEFT + np.array([0, CARRY_DISTANCE, 0])     # ball take off pos left
TARGET_BALL_POS_TAKEOFF_RIGHT = TARGET_BALL_POS_TOUCHDOWN_RIGHT + np.array([0, -CARRY_DISTANCE, 0])   # ball take off pos right

TARGET_BALL_VEL_TOUCHDOWN_LEFT = (TARGET_BALL_POS_TOUCHTDOWN_LEFT-TARGET_BALL_POS_TAKEOFF_RIGHT)/K_FLIGHT/PLANNER_TIMESTEP + 0.5*GRAVITY*K_FLIGHT*PLANNER_TIMESTEP      # ball touch down vel left
TARGET_BALL_VEL_TOUCHDOWN_RIGHT = (TARGET_BALL_POS_TOUCHDOWN_RIGHT-TARGET_BALL_POS_TAKEOFF_LEFT)/K_FLIGHT/PLANNER_TIMESTEP + 0.5*GRAVITY*K_FLIGHT*PLANNER_TIMESTEP      # ball touch down vel right
TARGET_BALL_VEL_TAKEOFF_LEFT = (TARGET_BALL_POS_TOUCHDOWN_RIGHT-TARGET_BALL_POS_TAKEOFF_LEFT)/K_FLIGHT/PLANNER_TIMESTEP - 0.5*GRAVITY*K_FLIGHT*PLANNER_TIMESTEP         # ball take off vel left
TARGET_BALL_VEL_TAKEOFF_RIGHT = (TARGET_BALL_POS_TOUCHTDOWN_LEFT-TARGET_BALL_POS_TAKEOFF_RIGHT)/K_FLIGHT/PLANNER_TIMESTEP - 0.5*GRAVITY*K_FLIGHT*PLANNER_TIMESTEP       # ball take off vel right


# planner options

NUM_TIMESTEPS_POST_TAKEOFF_CONSTRAINT = 2    # keep ball on hand's symmetry axis after takeoff
NUM_TIMESTEPS_PRE_TOUCHDOWN_CONSTRAINT = 2   # keep ball on hand's symmetry axis before touchdown
MAX_JERK = 50000                             # limt jerk of hand movement for smoothness
SLINGSHOT_FACTOR = 1.007                     # compensate potential energy stored in ball-hand spring contact
SOLVER_OPTIONS = {'print_time': 0,           # HSL MA57 linear solver works better for this problem, but defaults are fine.
                  'ipopt.print_level': 0}


# viewer options

VIEWER_SPPED = 1                                 # use to adjust speed of the simulation 
SLEEP_DURATION = SIM_TIMESTEP / VIEWER_SPPED     # simulation speed ignores computation time

CAMERA_DISTANCE = 3.25
CAMERA_LOOKAT_HEIGHT = 2.25
CAMERA_ELEVATION_ANGLE = -20



def get_touchdown(ball, touch_down_height):
    """ Computes when, where and with what velocity the ball will reach the touch_down_height """
    t_td = - ball.vel[2]/GRAVITY[2] + np.sqrt(ball.vel[2]**2/GRAVITY[2]**2 -2*(ball.pos[2]-touch_down_height)/GRAVITY[2])
    pos_td = ball.pos + ball.vel*t_td + 0.5*GRAVITY*t_td**2
    vel_td = ball.vel + GRAVITY*t_td
    return t_td, pos_td, vel_td


def plan_throw(pos_start, vel_start, acc_start, hand_rot,
               time_touchdown, pos_touchdown, vel_touchdown,
               pos_take_off, pos_target_touchdown):
    """ Computes one cycle of a juggling movement, starting and ending at the moment of takeoff """

    # NLP
    optim = cas.Opti()

    # decision variables
    num_steps = K_CYCLE
    cjerk = optim.variable(3, num_steps)

    # dynamics / integration
    param_pos_start = optim.parameter(3, 1)
    param_vel_start = optim.parameter(3, 1)
    param_acc_start = optim.parameter(3, 1)
    cpos = param_pos_start
    cvel = param_vel_start
    cacc = param_acc_start
    for k in range(0, num_steps):
        cpos = cas.horzcat(cpos, cpos[:,-1] + cvel[:,-1]*PLANNER_TIMESTEP + 1/2*cacc[:,-1]*PLANNER_TIMESTEP**2 + 1/6*cjerk[:,k]*PLANNER_TIMESTEP**3)
        cvel = cas.horzcat(cvel, cvel[:,-1] + cacc[:,-1]*PLANNER_TIMESTEP + 1/2*cjerk[:,k]*PLANNER_TIMESTEP**2)
        cacc = cas.horzcat(cacc, cacc[:,-1] + cjerk[:,k]*PLANNER_TIMESTEP)

    # objective
    cost = cas.sum1(cas.sum2(cacc**2)) / num_steps  # minimize averace squared accelerations
    optim.minimize(cost)

    # constraints...
    # ...match initial state
    optim.set_value(param_pos_start, pos_start)
    optim.set_value(param_vel_start, vel_start)
    optim.set_value(param_acc_start, acc_start)

    # ...detach from ball without sideways movement
    for i in range(NUM_TIMESTEPS_POST_TAKEOFF_CONSTRAINT):
        optim.subject_to(cas_cross_product(cacc[:, i+1]-GRAVITY, hand_rot[:,2]) == 0)

    # ...catch
    # ......match position at interpolated time
    k_td = int(np.floor(time_touchdown/PLANNER_TIMESTEP))
    Dt = time_touchdown - k_td*PLANNER_TIMESTEP
    cpos_touchdown = cpos[:, k_td] + cvel[:,k_td]*Dt + 1/2*cacc[:,k_td]*Dt**2 + 1/6*cjerk[:,k_td]*Dt**3
    optim.subject_to(cpos_touchdown==pos_touchdown)

    # ......be in line before touch-down -> collinear ball and hand velocities
    for i in range(NUM_TIMESTEPS_PRE_TOUCHDOWN_CONSTRAINT):
        k = k_td-1-i
        vel_ball = vel_touchdown - k*PLANNER_TIMESTEP*GRAVITY
        optim.subject_to(cas_cross_product(vel_ball, cvel[:,k])==0)

    # ...throw
    # ......match position
    optim.subject_to(cpos[:,-1]==pos_take_off)
    # ......match velocity
    vel_takeoff = (pos_target_touchdown - pos_take_off) / (K_FLIGHT*PLANNER_TIMESTEP) - 1/2*GRAVITY*K_FLIGHT*PLANNER_TIMESTEP
    optim.subject_to(cvel[:,-1]==vel_takeoff/SLINGSHOT_FACTOR)
    # ......match acceleration
    optim.subject_to(cacc[:,-1]==GRAVITY)

    # ...limit max jerk for smooth trajectory
    optim.subject_to(optim.bounded(-MAX_JERK, cas.vec(cjerk), MAX_JERK))

    # solver
    optim.solver('ipopt', SOLVER_OPTIONS)
    sol = optim.solve()
    return sol.value(cpos), sol.value(cvel), sol.value(cacc), sol.value(cjerk)


def initialize_balls_in_cascade(model, data, hands):
    """ Initializes the balls in a running cascade pattern. """

    ball_vel_takeoff_right = ( TARGET_BALL_POS_TOUCHTDOWN_LEFT - TARGET_BALL_POS_TAKEOFF_RIGHT - 0.5*GRAVITY*(K_FLIGHT*PLANNER_TIMESTEP)**2 ) / (K_FLIGHT*PLANNER_TIMESTEP)
    ball_vel_takeoff_left = ( TARGET_BALL_POS_TOUCHDOWN_RIGHT - TARGET_BALL_POS_TAKEOFF_LEFT - 0.5*GRAVITY*(K_FLIGHT*PLANNER_TIMESTEP)**2 ) / (K_FLIGHT*PLANNER_TIMESTEP)

    all_balls = []
    incoming_balls = {'right': [], 'left': []}
    balls_in_hand = {'left': [], 'right': []}

    for k_ball in np.arange(N_BALLS):

        k = k_ball*K_CYCLE

        # ball in right hand
        if k == 0:
            ball = Ball(model, data, k_ball)
            ball.pos = hands[1].pos + np.array([0, 0, 0.1])
            ball.vel = hands[1].vel
            all_balls += [ball]
            balls_in_hand['right'] += [ball]
                
        # ball in air from right to left
        elif k < K_FLIGHT:  
            time_in_air = k * PLANNER_TIMESTEP
            ball = Ball(model, data, k_ball)
            ball.pos = TARGET_BALL_POS_TAKEOFF_RIGHT + ball_vel_takeoff_right*time_in_air + 0.5*GRAVITY*time_in_air**2
            ball.vel = ball_vel_takeoff_right + GRAVITY*time_in_air
            all_balls += [ball]
            incoming_balls['left'] = [ball] + incoming_balls['left']

        # ball in left hand
        elif k < K_FLIGHT + K_DWELL:
            ball = Ball(model, data, k_ball)
            ball.pos = hands[0].pos + np.array([0, 0, 0.1])
            ball.vel = hands[0].vel
            all_balls += [ball]
            balls_in_hand['left'] += [ball]

        # ball in air again, but right to left
        else: 
            time_in_air = (k-K_FLIGHT-K_DWELL) * PLANNER_TIMESTEP
            ball = Ball(model, data, k_ball)
            ball.pos = TARGET_BALL_POS_TAKEOFF_LEFT + ball_vel_takeoff_left*time_in_air + 0.5*GRAVITY*time_in_air**2
            ball.vel = ball_vel_takeoff_left + GRAVITY*time_in_air
            all_balls += [ball]
            incoming_balls['right'] = [ball] + incoming_balls['right']

    incoming_balls['left'] = balls_in_hand['left'] + incoming_balls['left']
    incoming_balls['right'] = balls_in_hand['right'] + incoming_balls['right']

    return all_balls, incoming_balls, balls_in_hand


def juggle():

    model = mj.MjModel.from_xml_path(XML_PATH)
    model.opt.gravity = GRAVITY
    data = mj.MjData(model)

    with mj.viewer.launch_passive(model, data) as viewer:

        # move camera to view from front
        viewer.cam.distance = CAMERA_DISTANCE
        viewer.cam.lookat[2] = CAMERA_LOOKAT_HEIGHT
        viewer.cam.elevation = CAMERA_ELEVATION_ANGLE
        viewer.cam.azimuth = -180

        # get an initial trajectories
        pos_l, vel_l, acc_l, jer_l = plan_throw(pos_start=TARGET_BALL_POS_TAKEOFF_LEFT,
                                                vel_start=TARGET_BALL_VEL_TAKEOFF_LEFT,
                                                acc_start=GRAVITY,
                                                hand_rot=Rx(-HAND_TILT),
                                                time_touchdown=K_VACANT*PLANNER_TIMESTEP,
                                                pos_touchdown=TARGET_BALL_POS_TOUCHTDOWN_LEFT,
                                                vel_touchdown=TARGET_BALL_VEL_TOUCHDOWN_LEFT,
                                                pos_take_off=TARGET_BALL_POS_TAKEOFF_LEFT,
                                                pos_target_touchdown=TARGET_BALL_POS_TOUCHDOWN_RIGHT)

        pos_r, vel_r, acc_r, jer_r = plan_throw(pos_start=TARGET_BALL_POS_TAKEOFF_RIGHT,
                                                vel_start=TARGET_BALL_VEL_TAKEOFF_RIGHT,
                                                acc_start=GRAVITY,
                                                hand_rot=Rx(HAND_TILT),
                                                time_touchdown=K_VACANT*PLANNER_TIMESTEP,
                                                pos_touchdown=TARGET_BALL_POS_TOUCHDOWN_RIGHT,
                                                vel_touchdown=TARGET_BALL_VEL_TOUCHDOWN_RIGHT,
                                                pos_take_off=TARGET_BALL_POS_TAKEOFF_RIGHT,
                                                pos_target_touchdown=TARGET_BALL_POS_TOUCHTDOWN_LEFT)

        # initialize hands
        hands = [Hand(model, data, 'hand0', 'hands/hand0'), Hand(model, data, 'hand1', 'hands/hand1')]
        hands[1].pos = pos_r[:,K_CYCLE]
        hands[1].vel = vel_r[:,K_CYCLE]
        hands[1].R = Rx(HAND_TILT)
        hands[1].angvel = np.zeros(3)
        hands[0].pos = pos_l[:,int(K_CYCLE/2)]
        hands[0].vel = vel_l[:,int(K_CYCLE/2)]
        hands[0].R = Rx(-HAND_TILT)
        hands[0].angvel = np.zeros(3)

        # initialize balls
        balls, incoming_balls, balls_in_hand = initialize_balls_in_cascade(model, data, hands)

        # desired ball states
        balls_pos_des = np.zeros((N_BALLS, 3))
        balls_vel_des = np.zeros((N_BALLS, 3))
        for ball in balls:
            balls_pos_des[ball.ball_id] = ball.pos
            balls_vel_des[ball.ball_id] = ball.vel

        # simulation loop
        time_step = 0
        while viewer.is_running():

            # individual phase variables for each hand
            phase_step_left = int((time_step+K_CYCLE/2) % K_CYCLE)
            phase_step_right = int(time_step % K_CYCLE)

            # take off left -> plan next trajectory
            if phase_step_left == 0:
                # update queues
                balls_in_hand0 = []
                thrown_ball = incoming_balls['left'].pop(0)
                incoming_balls['right'].append(thrown_ball)

                # update desired ball positions
                balls_pos_des[thrown_ball.ball_id,:] = pos_l[:,K_CYCLE]
                balls_vel_des[thrown_ball.ball_id,:] = (TARGET_BALL_POS_TOUCHDOWN_RIGHT-pos_l[:,K_CYCLE])/K_FLIGHT/PLANNER_TIMESTEP - 0.5*GRAVITY*K_FLIGHT*PLANNER_TIMESTEP

                # plan next throw
                if len(incoming_balls['left']) > 0:
                    time_touch_down, pos_touchdown, vel_touchdown = get_touchdown(incoming_balls['left'][0], HAND_HEIGHT)
                else:
                    time_touch_down, pos_touchdown, vel_touchdown = K_VACANT * PLANNER_TIMESTEP, TARGET_BALL_POS_TOUCHTDOWN_LEFT, TARGET_BALL_VEL_TOUCHDOWN_LEFT

                pos_l, vel_l, acc_l, jer_l = plan_throw(pos_start=pos_l[:,K_CYCLE],
                                                        vel_start=vel_l[:,K_CYCLE],
                                                        acc_start=acc_l[:,K_CYCLE],
                                                        hand_rot=Rx(-HAND_TILT),
                                                        time_touchdown=time_touch_down, pos_touchdown=pos_touchdown, vel_touchdown=vel_touchdown,
                                                        pos_take_off=TARGET_BALL_POS_TAKEOFF_LEFT,
                                                        pos_target_touchdown=TARGET_BALL_POS_TOUCHDOWN_RIGHT)

            # take off right -> plan next trajectory
            if phase_step_right == 0: 
                # update queues
                balls_in_hand1 = []
                thrown_ball = incoming_balls['right'].pop(0)
                incoming_balls['left'].append(thrown_ball)

                # update desired ball positions
                balls_pos_des[thrown_ball.ball_id,:] = pos_r[:,K_CYCLE]
                balls_vel_des[thrown_ball.ball_id,:] = (TARGET_BALL_POS_TOUCHTDOWN_LEFT-pos_r[:,K_CYCLE])/K_FLIGHT/PLANNER_TIMESTEP - 0.5*GRAVITY*K_FLIGHT*PLANNER_TIMESTEP

                # plan next throw
                if len(incoming_balls['right']) > 0:
                    time_touch_down, pos_touchdown, vel_touchdown = get_touchdown(incoming_balls['right'][0], HAND_HEIGHT)
                else:
                    time_touch_down, pos_touchdown, vel_touchdown = K_VACANT * PLANNER_TIMESTEP, TARGET_BALL_POS_TOUCHDOWN_RIGHT, TARGET_BALL_VEL_TOUCHDOWN_RIGHT

                pos_r, vel_r, acc_r, jer_r = plan_throw(pos_start=pos_r[:,K_CYCLE],
                                                        vel_start=vel_r[:,K_CYCLE],
                                                        acc_start=acc_r[:,K_CYCLE],
                                                        hand_rot=Rx(HAND_TILT),
                                                        time_touchdown=time_touch_down, pos_touchdown=pos_touchdown, vel_touchdown=vel_touchdown,
                                                        pos_take_off=TARGET_BALL_POS_TAKEOFF_RIGHT,
                                                        pos_target_touchdown=TARGET_BALL_POS_TOUCHTDOWN_LEFT)

            # touch down left
            if phase_step_left == K_VACANT: 
                caught_ball = incoming_balls['left'][0]
                balls_in_hand0 = [caught_ball]

            # touch down right
            if phase_step_right == K_VACANT: 
                caught_ball = incoming_balls['right'][0]
                balls_in_hand1 = [caught_ball]

            # simulation substeps
            for k_sub in range(NUM_SUBSTEPS):

                # interpolate the desired state from plan
                del_t = k_sub * SIM_TIMESTEP
                pos_r_ = pos_r[:,phase_step_right] + vel_r[:,phase_step_right]*del_t + 1/2*acc_r[:,phase_step_right]*del_t**2 + 1/6*jer_r[:,phase_step_right]*del_t**3
                vel_r_ = vel_r[:,phase_step_right] + acc_r[:,phase_step_right]*del_t + 1/2*jer_r[:,phase_step_right]*del_t**2
                pos_l_ = pos_l[:,phase_step_left] + vel_l[:,phase_step_left]*del_t + 1/2*acc_l[:,phase_step_left]*del_t**2 + 1/6*jer_l[:,phase_step_left]*del_t**3
                vel_l_ = vel_l[:,phase_step_left] + acc_l[:,phase_step_left]*del_t + 1/2*jer_l[:,phase_step_left]*del_t**2

                # track hand velocity, while correcting position error
                hands[0].vel = vel_l_ + 100 * np.eye(3) @ (pos_l_ - hands[0].pos)
                hands[1].vel = vel_r_ + 100 * np.eye(3) @ (pos_r_ - hands[1].pos)

                # keep hand orientation
                hands[0].angvel = 10 * np.eye(3) @ rotation_matrix_to_axis_angle(Rx(-HAND_TILT) @ hands[0].R.T)
                hands[1].angvel = 10 * np.eye(3) @ rotation_matrix_to_axis_angle(Rx(HAND_TILT) @ hands[1].R.T)

                # advance simulation
                mj.mj_step(model, data)
                viewer.sync()
                time.sleep(SLEEP_DURATION)

                # update desired ball positions...
                balls_pos_des += balls_vel_des * SIM_TIMESTEP + np.tile(GRAVITY,(N_BALLS,1)) * SIM_TIMESTEP**2
                balls_vel_des += np.tile(GRAVITY,(N_BALLS,1)) * SIM_TIMESTEP
                if balls_in_hand0:
                    balls_pos_des[balls_in_hand0[0].ball_id] = hands[0].pos
                    balls_vel_des[balls_in_hand0[0].ball_id] = hands[0].vel
                if balls_in_hand1:
                    balls_pos_des[balls_in_hand1[0].ball_id] = hands[1].pos
                    balls_vel_des[balls_in_hand1[0].ball_id] = hands[1].vel
                for ball in balls:
                    ball.set_mocap(balls_pos_des[ball.ball_id])

            time_step += 1


if __name__ == '__main__':
    juggle()
