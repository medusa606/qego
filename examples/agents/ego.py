import math

import numpy as np

import reporting
from examples.agents.dynamic_body import make_body_state
from examples.agents.template import RandomAgent
from library import geometry
from library.bodies import Pedestrian
from library.bodies import DynamicBodyState
from library.geometry import Point
from examples.constants import M2PX


from icecream import ic

TARGET_ERROR = 0.000000000000001
ACTION_ERROR = 0.000000000000001


class QLearningEgoAgent(RandomAgent):
    def __init__(self, q_learning_config, body, pedestrians, time_resolution, num_opponents, num_actions, width, height, road_polgon, **kwargs):
        super().__init__(noop_action=body.noop_action, epsilon=q_learning_config.epsilon, **kwargs)

        self.road_polgon = road_polgon # GC added
        self.pedestrians = pedestrians
        self.width = width
        self.height = height


        self.target_alpha = q_learning_config.alpha.stop
        self.alphas = iter(np.linspace(start=q_learning_config.alpha.start, stop=self.target_alpha, num=q_learning_config.alpha.num_steps, endpoint=True))

        self.alpha = next(self.alphas, self.target_alpha)  # learning rate (should decrease over time)
        self.gamma = q_learning_config.gamma  # discount factor (should be fixed over time?)
        self.feature_config = q_learning_config.features

        self.body = body
        self.time_resolution = time_resolution

        self.opponent_indexes = list(range(1, num_opponents + 1)) # list of the pedestrians

        # for just speed control use:
        self.available_actions = [[throttle_action, self.noop_action[1]] for throttle_action in np.linspace(start=self.body.constants.min_throttle, stop=self.body.constants.max_throttle, num=num_actions, endpoint=True)]
        #for speed and steering control use:
        # self.available_actions = [[throttle_action, steering_action] for throttle_action in np.linspace(start=self.body.constants.min_throttle,
        #     stop=self.body.constants.max_throttle, num=num_actions, endpoint=True) for steering_action in np.linspace(start=self.body.constants.min_steering_angle,
        #     stop=self.body.constants.max_steering_angle,num=num_actions, endpoint=True)]

        self.log_file = None
        if q_learning_config.log is not None:
            self.log_file = reporting.get_agent_file_logger(q_learning_config.log)

        # which features are enabled?
        # ic(self.feature_config)

        # bounds are used to normalise features
        self.feature_bounds = dict()

        # self.feature_bounds["lane_position"] = (0,1)

        self.feature_bounds["safety_binary"] = (0,1)
        # self.feature_bounds["safety_time"] = (0,1)
        self.feature_bounds["safe_pass_behind"] = (0, 1)
        # self.feature_bounds["safety_pass_time"] = (0, 1)

        self.feature_bounds["goal_distance_x"] = (0,1)
        # self.feature_bounds["goal_distance_y"] = (0,1)
        # self.feature_bounds["goal_distance_euc"] = (0,1)


        # self.feature_bounds["distance_angle"] = (0,1)
        # self.feature_bounds["ego_speed"] = (0,self.body.constants.max_velocity)
        # self.feature_bounds["speed_heading"] = (0,1)
        # self.feature_bounds["speed_d_angle"] = (0,1)

        # self.feature_bounds["beam_long_L"] = (0,1)
        # self.feature_bounds["beam_long_R"] = (0,1)
        # self.feature_bounds["beam_med"] = (0,1)
        # self.feature_bounds["beam_short"] = (0,1)
        # self.feature_bounds["ped_crossing"] = (0,1)
        # self.feature_bounds["beam_x_crossing"] = (0,1)

        # self.feature_bounds["lidar_left"] = (0,1)
        # self.feature_bounds["lidar_right"] = (0, 1)

        # self.feature_bounds["lidar_box_L1"] = (0,1)
        # self.feature_bounds["lidar_box_L2"] = (0,1)
        # self.feature_bounds["lidar_box_L3"] = (0,1)
        # self.feature_bounds["lidar_box_R1"] = (0, 1)
        # self.feature_bounds["lidar_box_R2"] = (0, 1)
        # self.feature_bounds["lidar_box_R3"] = (0, 1)
        #
        # self.feature_bounds["pedx_lidar_L1"] = (0, 1)
        # self.feature_bounds["pedx_lidar_L2"] = (0, 1)
        # self.feature_bounds["pedx_lidar_L3"] = (0, 1)
        # self.feature_bounds["pedx_lidar_R1"] = (0, 1)
        # self.feature_bounds["pedx_lidar_R2"] = (0, 1)
        # self.feature_bounds["pedx_lidar_R3"] = (0, 1)

        # self.feature_bounds["speed_lidarL"] = (0, 1)
        # self.feature_bounds["speed_lidarR"] = (0, 1)

        x_mid = M2PX * 16  # 0 < x_mid < self.x_max
        y_mid = 0.5  # 0 < y_mid < 1
        self.x_max = self.feature_bounds["distance"][1] if "distance" in self.feature_bounds else math.sqrt((width ** 2) + (height ** 2))
        self.n = math.log(1 - y_mid) / math.log(x_mid / self.x_max)


        if self.feature_config.distance_x:
            self.feature_bounds["distance_x"] = (-float(width), float(width))
        if self.feature_config.distance_y:
            self.feature_bounds["distance_y"] = (-float(height), float(height))
        if self.feature_config.distance:
            self.feature_bounds["distance"] = (0.0, math.sqrt((width ** 2) + (height ** 2)))
        if self.feature_config.relative_angle:
            self.feature_bounds["relative_angle"] = (0, math.pi)
        if self.feature_config.heading:
            self.feature_bounds["heading"] = (0.0, math.pi)
        if self.feature_config.on_road:
            self.feature_bounds["on_road"] = (0.0, 1.0)
        if self.feature_config.inverse_distance:
            self.feature_bounds["inverse_distance"] = (0.0, 4.0)

        # set a zero weight for each feature against each opponent
        self.feature_weights = {index: {feature: 0.0 for feature in self.feature_bounds.keys()} for index in self.opponent_indexes}
        # ic(self.feature_weights)

        #store the q-values and actions for monitoring
        self.store_q_values = []
        self.store_action = []
        self.store_safety_time = []

        # create a set of features for each opponent (pedestrian)
        self.enabled_features = {index: sorted(self.feature_bounds.keys()) for index in self.opponent_indexes}
        if self.log_file:
            labels = [f"{feature}{index}" for index, features in self.enabled_features.items() for feature in features]
            self.log_file.info(f"{','.join(map(str, labels))}")

    def reset(self):
        pass

    def choose_action(self, state, action_space, info=None):
        if self.epsilon_valid():
            action = self.available_actions[self.np_random.choice(range(len(self.available_actions)))]
        else:
            best_actions = list()  # there may be multiple actions with max Q value
            max_q_value = -math.inf
            for action in self.available_actions:
                q_value = self.q_value(state, action)
                if q_value > max_q_value:
                    best_actions = [action]
                    max_q_value = q_value
                elif q_value == max_q_value:
                    best_actions.append(action)
            assert best_actions, "no best action(s) found"
            # if multiple best actions occur then a random is chosen
            action = best_actions[0] if len(best_actions) == 1 else best_actions[self.np_random.choice(range(len(best_actions)))]
            # if len(best_actions) == 1:
            #     action = best_actions[0] #GC Edit
            # #store q-values
            self.store_q_values = q_value
            self.store_action.append(action[0]) #store ego action history
            # if action[0] == (-144 or -72):
            #     ic(action[0])
        return action

    def process_feedback(self, previous_state, action, state, reward):
        difference = (reward + self.gamma * max(self.q_value(state, action_prime) for action_prime in self.available_actions)) - self.q_value(previous_state, action)
        for index, opponent_features in self.features(previous_state, action).items():
            for feature, feature_value in opponent_features.items():
                # approximate Q-value update
                self.feature_weights[index][feature] = self.feature_weights[index][feature] + self.alpha * difference * feature_value
                # distance = 3.5
                # relative angle = 9.5
                # self.feature_weights[index][feature] = 10
                # ic(self.feature_weights[1]['safety_time'])

        if self.log_file:
            weights = [self.feature_weights[index][feature] for index, features in self.enabled_features.items() for feature in features]
            self.log_file.info(f"{','.join(map(str, weights))}")

        self.alpha = next(self.alphas, self.target_alpha)
        # print("R %6.3f " % reward)
        # # print("R %6.3f F %6.3f W %6.3f" % (reward, self.features(state, action), self.feature_weights[0][0]))
        # # print("R %6.3f F %6.3f W %6.3f" % (reward, self.features(state, action), self.feature_weights[0][0]))

    def q_value(self, state, action):
        feature_values = self.features(state, action)
        # ic(feature_values)
        q_value = sum(feature_value * self.feature_weights[index][feature] for index, opponent_feature_values in feature_values.items() for feature, feature_value in opponent_feature_values.items())
        return q_value

    def features(self, state, action):
        return {index: self.features_opponent(state, action, index) for index in self.opponent_indexes}

    def features_opponent(self, state, action, opponent_index):
        self_state = make_body_state(state, self.index)
        opponent_state = make_body_state(state, opponent_index)

        #GC added:
        #ped_body = Pedestrian(opponent_state, self.body.constants) # TODO need to change for ped constants, not self.body

        def one_step_lookahead(body_state, throttle):  # one-step lookahead with no steering
            distance_velocity = body_state.velocity * self.time_resolution
            return DynamicBodyState(
                position=Point(
                    x=body_state.position.x + distance_velocity * math.cos(body_state.orientation),
                    y=body_state.position.y + distance_velocity * math.sin(body_state.orientation)
                ),
                velocity=max(self.body.constants.min_velocity, min(self.body.constants.max_velocity, body_state.velocity + (throttle * self.time_resolution))),
                orientation=body_state.orientation
            )

        # *****************************
        def n_step_lookahead(body_state, throttle, n=50):
            next_body_state = body_state
            for _ in range(n):
                next_body_state = one_step_lookahead(next_body_state, throttle)
            return next_body_state

        throttle_action, _ = action

        self_state = n_step_lookahead(self_state, throttle_action)
        opponent_state = n_step_lookahead(opponent_state, 0.0)

        def normalise(value, min_bound, max_bound):
            if value < min_bound:
                return 0.0
            elif value > max_bound:
                return 1.0
            else:
                return (value - min_bound) / (max_bound - min_bound)

        unnormalised_values = dict()
        # unnormalised_values["goal_distance_x"] = (self_state.position.x)/self.width # GC changed to be distance from target position
        rel_width  = (self.width - self_state.position.x)/self.width
        rel_height = (29 - self_state.position.y)/self.height #center of lane
        goal_euc = math.sqrt((rel_width**2) + (rel_height**2))
        unnormalised_values["goal_distance_x"] = rel_width # GC changed to be distance from target position
        # unnormalised_values["goal_distance_y"] = rel_height
        # unnormalised_values["goal_distance_euc"] = goal_euc
        # ic(goal_euc)


        # *************** distance to ped x relative angle
        # GC This should return a high value if a ped is close and in front of the ego
        dist_to_opponent = self_state.position.distance(opponent_state.position)
        inv_dist_to_opponent = 1 - (dist_to_opponent / self.x_max)
        upper_bound = math.sqrt((self.width ** 2) + (self.height ** 2))
        norm_inv_dist_to_opponent = normalise(inv_dist_to_opponent, 0 , upper_bound)
        rel_ang = abs(geometry.normalise_angle(geometry.Line(start=self_state.position, end=opponent_state.position).orientation() - self_state.orientation))
        norm_rel_angle = normalise(rel_ang,0,math.pi/2)
        distance_angle = 1.0 * norm_inv_dist_to_opponent * norm_rel_angle
        # ic(norm_inv_dist_to_opponent)
        # ic(norm_rel_angle)
        # ic(distance_angle)
        # unnormalised_values["distance_angle"] = distance_angle

        # *************** speed x distance angle
        ego_speed = self_state.velocity
        # ic(ego_speed)
        # unnormalised_values["ego_speed"] = ego_speed

        # *************** speed x angle
        ego_speed = self_state.velocity
        norm_ego_speed = normalise(ego_speed, 0, 144)
        # ic(norm_ego_speed)
        # unnormalised_values["speed_heading"] = norm_ego_speed * norm_rel_angle


        # *************** speed x distance_angle
        ego_speed = self_state.velocity
        # unnormalised_values["speed_d_angle"] = 1000 * norm_ego_speed * distance_angle
        # ic(unnormalised_values["speed_d_angle"])

        # ************** driving lidar LEFT
        # lidar_long = geometry.make_circle_segment(432, math.pi / 24, anchor=Point(self.body.constants.length / 2, 0),
        #           angle_left_offset=0.9).transform(self_state.orientation,self_state.position)
        # lidar_q1 = geometry.make_circle_segment(300, math.pi / 24, anchor=Point(0, 0),
        #             angle_left_offset=0.5).transform(self_state.orientation + math.pi / 22,self_state.position)
        # lidar_q2 = geometry.make_circle_segment(200, math.pi / 16, anchor=Point(0, 0),
        #             angle_left_offset=0.5).transform(self_state.orientation + math.pi / 16,self_state.position)
        # lidar_q3 = geometry.make_circle_segment(100, math.pi / 5, anchor=Point(0, 0),
        #             angle_left_offset=0.5).transform(self_state.orientation + math.pi / 10, self_state.position)
        # lidar_side = geometry.make_circle_segment(60, math.pi - 0.001, anchor=Point(0, 0),
        #           angle_left_offset=0.50).transform(self_state.orientation + math.pi / 2, self_state.position)

        # unnormalised_values["lidar_long"] = 1 if (any(ped_body.bounding_box().intersects(lidar_long) for ped_body in self.pedestrians)) else 0
        # unnormalised_values["lidar_q1"] = 1 if (any(ped_body.bounding_box().intersects(lidar_q1) for ped_body in self.pedestrians)) else 0
        # unnormalised_values["lidar_q2"] = 1 if (any(ped_body.bounding_box().intersects(lidar_q2) for ped_body in self.pedestrians)) else 0
        # unnormalised_values["lidar_q3"] = 1 if (any(ped_body.bounding_box().intersects(lidar_q3) for ped_body in self.pedestrians)) else 0
        # unnormalised_values["lidar_side"] = 1 if (any(ped_body.bounding_box().intersects(lidar_side) for ped_body in self.pedestrians)) else 0
        # unnormalised_values["lidar_left"] = 1 if (
        #             any(ped_body.bounding_box().intersects(lidar_long) for ped_body in self.pedestrians)
        #             or any(ped_body.bounding_box().intersects(lidar_q1) for ped_body in self.pedestrians)
        #             or any(ped_body.bounding_box().intersects(lidar_q2) for ped_body in self.pedestrians)
        #             or any(ped_body.bounding_box().intersects(lidar_q3) for ped_body in self.pedestrians)
        #             or any(ped_body.bounding_box().intersects(lidar_side) for ped_body in self.pedestrians)) else 0
        # ic(unnormalised_values["lidar_long"])
        # ic(unnormalised_values["lidar_q1"])
        # ic(unnormalised_values["lidar_q2"])
        # ic(unnormalised_values["lidar_q3"])
        # ic(unnormalised_values["lidar_side"])

        # ************** driving lidar RIGHT
        # lidar_longL = geometry.make_circle_segment(432, math.pi / 24, anchor=Point(self.body.constants.length / 2, 0),
        #           angle_left_offset=0.1).transform(self_state.orientation,self_state.position)
        # lidar_q1L = geometry.make_circle_segment(300, math.pi / 24, anchor=Point(0, 0),
        #             angle_left_offset=0.5).transform(self_state.orientation - math.pi / 22,self_state.position)
        # lidar_q2L = geometry.make_circle_segment(200, math.pi / 16, anchor=Point(0, 0),
        #             angle_left_offset=0.5).transform(self_state.orientation - math.pi / 16,self_state.position)
        # lidar_q3L = geometry.make_circle_segment(100, math.pi / 5, anchor=Point(0, 0),
        #             angle_left_offset=0.5).transform(self_state.orientation - math.pi / 10, self_state.position)
        # lidar_sideL = geometry.make_circle_segment(60, math.pi - 0.001, anchor=Point(0, 0),
        #           angle_left_offset=0.50).transform(self_state.orientation - math.pi / 2, self_state.position)

        # unnormalised_values["lidar_right"] = 1 if (any(ped_body.bounding_box().intersects(lidar_longL) for ped_body in self.pedestrians)
        #            or any(ped_body.bounding_box().intersects(lidar_q1L) for ped_body in self.pedestrians)
        #            or any(ped_body.bounding_box().intersects(lidar_q2L) for ped_body in self.pedestrians)
        #            or any(ped_body.bounding_box().intersects(lidar_q3L) for ped_body in self.pedestrians)
        #            or any(ped_body.bounding_box().intersects(lidar_sideL) for ped_body in self.pedestrians)) else 0
        # # ic(unnormalised_values["lidar_right"])

        # unnormalised_values["speed_lidarL"] = norm_ego_speed * unnormalised_values["lidar_left"]
        # unnormalised_values["speed_lidarR"] = norm_ego_speed * unnormalised_values["lidar_right"]
        # ic(unnormalised_values["speed_lidarL"])
        # ic(unnormalised_values["speed_lidarR"])
        lidar_range = 500
        lidar_width = 54
        # lidar_box_L = geometry.make_rectangle(lidar_range, lidar_width).transform(self_state.orientation,
        #     geometry.Point(self_state.position.x -self.body.constants.length / 2 + lidar_range / 2,
        #              self_state.position.y + lidar_width / 2))
        # lidar_box_R = geometry.make_rectangle(lidar_range, lidar_width).transform(self_state.orientation,
        #     geometry.Point(self_state.position.x - self.body.constants.length / 2 + lidar_range / 2,
        #     self_state.position.y - lidar_width / 2))

        # ************** box lidar split into 3 sections
        # lidar_box_L1 = geometry.make_rectangle(self.body.constants.length, lidar_width).transform(self_state.orientation,
        #     geometry.Point(self_state.position.x,self_state.position.y + lidar_width / 2))
        # lidar_box_L2 = geometry.make_rectangle(lidar_range / 2, lidar_width).transform(self_state.orientation,
        #     geometry.Point(self_state.position.x + self.body.constants.length / 2 + lidar_range / 4,self_state.position.y + lidar_width / 2))
        # lidar_box_L3 = geometry.make_rectangle(lidar_range / 2, lidar_width).transform(self_state.orientation,
        #     geometry.Point(self_state.position.x + self.body.constants.length / 2 + (3 * lidar_range / 4),self_state.position.y + lidar_width / 2))

        # L2 = 1 if (any(ped_body.bounding_box().intersects(lidar_box_L2) for ped_body in self.pedestrians)) else 0
        # unnormalised_values["lidar_box_L1"] = 1 if (any(ped_body.bounding_box().intersects(lidar_box_L1) for ped_body in self.pedestrians)) else 0
        # unnormalised_values["lidar_box_L2"] = 1 if (any(ped_body.bounding_box().intersects(lidar_box_L2) for ped_body in self.pedestrians)) else 0
        # unnormalised_values["lidar_box_L3"] = 1 if (any(ped_body.bounding_box().intersects(lidar_box_L3) for ped_body in self.pedestrians)) else 0
        # lidar_box_R1 = geometry.make_rectangle(self.body.constants.length, lidar_width).transform(self_state.orientation,
        #     geometry.Point(self_state.position.x,self_state.position.y - lidar_width / 2))
        # lidar_box_R2 = geometry.make_rectangle(lidar_range / 2, lidar_width).transform(self_state.orientation,
        #     geometry.Point(self_state.position.x + self.body.constants.length / 2 + lidar_range / 4,self_state.position.y - lidar_width / 2))
        # lidar_box_R3 = geometry.make_rectangle(lidar_range / 2, lidar_width).transform(self_state.orientation,
        #     geometry.Point(self_state.position.x + self.body.constants.length / 2 + (3 * lidar_range / 4),self_state.position.y - lidar_width / 2))
        # R2 = 1 if (any(ped_body.bounding_box().intersects(lidar_box_R2) for ped_body in self.pedestrians)) else 0
        # unnormalised_values["lidar_box_R1"] = 1 if (any(ped_body.bounding_box().intersects(lidar_box_R1) for ped_body in self.pedestrians)) else 0
        # unnormalised_values["lidar_box_R2"] = 1 if (any(ped_body.bounding_box().intersects(lidar_box_R2) for ped_body in self.pedestrians)) else 0
        # unnormalised_values["lidar_box_R3"] = 1 if (any(ped_body.bounding_box().intersects(lidar_box_R3) for ped_body in self.pedestrians)) else 0

        # ************** ped orientation
        # ped_xdown = 1 if (-1.6 <= opponent_state.orientation <= -1.4) else 0
        # ped_xup = 1 if (1.6 <= opponent_state.orientation <= 1.4) else 0
        # ************** opponent orientation & lidar_detection
        # unnormalised_values["pedx_lidar_L1"] = 1 if (unnormalised_values["lidar_box_L1"] == 1 and ped_xdown == 1) else 0
        # unnormalised_values["pedx_lidar_L2"] = 1 if (L2 == 1 and ped_xdown == 1) else 0
        # unnormalised_values["pedx_lidar_L3"] = 1 if (unnormalised_values["lidar_box_L3"] == 1 and ped_xdown == 1) else 0
        # unnormalised_values["pedx_lidar_R1"] = 1 if (unnormalised_values["lidar_box_R1"] == 1 and ped_xup == 1) else 0
        # unnormalised_values["pedx_lidar_R2"] = 1 if (R2 == 1 and ped_xup == 1) else 0
        # unnormalised_values["pedx_lidar_R3"] = 1 if (unnormalised_values["lidar_box_R3"] == 1 and ped_xup == 1) else 0


        # if unnormalised_values["pedx_lidar_L2"]==1:
        #     ic(unnormalised_values["pedx_lidar_L2"])

        # unnormalised_values["ped_crossing"] = 1 if (-1.6 <= opponent_state.orientation <= -1.4) else 0
        # ic(unnormalised_values["ped_crossing"])
        # unnormalised_values["beam_x_crossing"] = 1 if ((-1.6 <= opponent_state.orientation <= -1.4) and
        #        (any(ped_body.bounding_box().intersects(lidar_left) for ped_body in self.pedestrians))) else 0
        # ic(unnormalised_values["beam_x_crossing"])


        # ************** time-to-collision
        ttc=True
        if(ttc):
            #choose the closest pedestrian TODO we only have one so use[0]
            # pb = self.pedestrians[0].constants
            # length = 10.5, width = 14.0, wheelbase = 5.25, track = 14.0, min_velocity = 0, max_velocity = 22.4, min_throttle = -22.4, max_throttle = 22.4, min_steering_angle = -1.2566370614359172, max_steering_angle = 1.2566370614359172
            braking_distance = (self_state.velocity**2)/(2*-self.body.constants.min_throttle)
            reaction_distance = self_state.velocity * 0.675
            stopping_distance = braking_distance + reaction_distance
            ped_width = 0.50

            time_ped_to_road = (opponent_state.position.y - self_state.position.y) / opponent_state.velocity if opponent_state.velocity > 0 else (opponent_state.position.y - self_state.position.y) / 0.1
            time_ego_to_ped =  (opponent_state.position.x
                                - stopping_distance
                                - self.body.constants.length/2 # half length of vehicle
                                - ped_width                           # width of the pedestrian
                                - self_state.position.x) / self_state.velocity if self_state.velocity > 0 \
                else (opponent_state.position.x + stopping_distance - self_state.position.x) / 0.1

            delta = (self.body.constants.width / 2) / opponent_state.velocity if opponent_state.velocity > 0 else (self.body.constants.width / 2) / 0.1

            # ********* Safe to pass IN FRONT
            in_front = (opponent_state.position.x
                        - self.body.constants.length/2
                        - self_state.position.x
                        + ped_width # width of the ped
                        - stopping_distance) > 0
            safety_time = abs(time_ego_to_ped) - delta - abs(time_ped_to_road) #if in_front else -1 #this one makes little difference
            safety_binary = 0 if (safety_time <= 0 and in_front) else 1 #this one makes big difference

            unnormalised_values["safety_binary"] = safety_binary
            # unnormalised_values["safety_time"] = safety_time

            #************* SAFE TO PASS BEHIND
            # time_ped_safe = (py - ey - w / 2) / vp
            safe_y = (opponent_state.position.y
                      - self_state.position.y -
                      self.body.constants.width / 2
                      - ped_width)
            time_ped_safe = safe_y / opponent_state.velocity if opponent_state.velocity > 0 else safe_y / 0.1
            # time_ego_safe = (px - ex + L/2) / ve
            safe_x = (opponent_state.position.x
                      - self_state.position.x
                      + self.body.constants.length/2
                      + ped_width)
            time_ego_safe = safe_x / self_state.velocity if self_state.velocity > 0 else safe_x / 0.1

            safety_pass_time = abs(time_ped_safe) - time_ego_safe
            safe_pass_behind = 1 if (safety_pass_time > 0) else 0

            # safety_binary = safety_binary & safe_pass_behind
            # ic(safe_x)
            # ic(safe_y)
            # ic(time_ped_safe)
            # ic(time_ego_safe)
            # ic(safety_pass_time)
            # ic(safe_pass_behind)
            unnormalised_values["safe_pass_behind"] = safe_pass_behind
            # unnormalised_values["safety_pass_time"] = safety_pass_time


        #********************************
        lane_position = False
        if(lane_position):
            half_lane_width = 29 # also happens to be absolute y-posn of lane center
            lane_rel_pos = abs(self_state.position.y - half_lane_width) / half_lane_width
            unnormalised_values["lane_position"] = 1 - lane_rel_pos
            if lane_rel_pos > 1:
                unnormalised_values["lane_position"] = 1 - lane_rel_pos**2
            # ic(unnormalised_values["lane_position"])


        # ********************************
        if self.feature_config.distance_x:
            unnormalised_values["distance_x"] = self_state.position.distance_x(opponent_state.position)
            # ic(unnormalised_values["distance_x"])
        if self.feature_config.distance_y:
            unnormalised_values["distance_y"] = self_state.position.distance_y(opponent_state.position)
        if self.feature_config.distance:
            unnormalised_values["distance"] = self_state.position.distance(opponent_state.position)
        if self.feature_config.relative_angle:
            unnormalised_values["relative_angle"] = abs(geometry.normalise_angle(geometry.Line(start=self_state.position, end=opponent_state.position).orientation() - self_state.orientation))
            # ic(unnormalised_values["relative_angle"])
        if self.feature_config.heading:
            unnormalised_values["heading"] = abs(geometry.normalise_angle(geometry.Line(start=opponent_state.position, end=self_state.position).orientation() - opponent_state.orientation))
        # GC Added the following
        if self.feature_config.on_road:
            # unnormalised_values["on_road"] = 1 if any(ped_body.bounding_box().intersects(self.road_polgon)) else 0
            unnormalised_values["on_road"] = 1 if any(ped_body.bounding_box().intersects(self.road_polgon) for ped_body in self.pedestrians) else 0
            # ic(unnormalised_values["on_road"])
        if self.feature_config.inverse_distance:
            # x = unnormalised_values["distance"] if "distance" in unnormalised_values else self_state.state.position.distance(opponent_state.position)
            x = self_state.position.distance(opponent_state.position)
            unnormalised_values["inverse_distance"] = 1 - (x / self.x_max) ** self.n  # thanks to Ram Varadarajan
            # ic(unnormalised_values["inverse_distance"])

        # if unnormalised_values["beam_detect"]==1:
            # ic(unnormalised_values["distance_x"])
            # ic(unnormalised_values["distance_y"])
            # ic(unnormalised_values["relative_angle"])
            # ic(unnormalised_values["beam_detect"])

        normalised_values = {feature: normalise(feature_value, *self.feature_bounds[feature]) for feature, feature_value in unnormalised_values.items()}
        return normalised_values
