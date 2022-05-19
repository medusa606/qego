import datetime
import math
import pathlib
import timeit

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import uniform_filter1d

import reporting
from config import Mode, AgentType
from examples.election import Election
from examples.agents.ego import QLearningEgoAgent
from examples.constants import M2PX

from icecream import ic

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
ic(gpu_devices)
ic(tf.test.is_built_with_cuda)
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
ic(tf.config.get_visible_devices())
import random

from keras.callbacks import LambdaCallback


# use this to improve training time for batch learning
tf.compat.v1.disable_eager_execution()

# set random seed
tf.random.set_seed(0)
random.seed(0)

# set baseline mode, ego drives max speed
BASELINE_MODE = False

# For DQN agent
GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 1000000
BATCH_SIZE = 50 #50
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
DENSE_NODES = 12 #24
SOLVER_FREQ = 1 #how often to run dqn solver, 1=every timestep, 20=every 20 steps

# ic.disable()

class DQNSolver:
    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = len(action_space)
        self.ego_throttle_actions = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.store_action = []

        self.model = Sequential()
        self.model.add(Dense(DENSE_NODES, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(DENSE_NODES, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))
        ic(self.model.summary())

    def get_model_weights(self):
        l1 = self.model.layers[0].get_weights()[0]
        l2 = self.model.layers[1].get_weights()[0]
        l3 = self.model.layers[2].get_weights()[0]
        return np.concatenate((l1.flatten(), l2.flatten(), l3.flatten()))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # ic(self.memory)

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        state = np.reshape(state, [1, 8]) #TODO
        # ic( state.shape )
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                a=self.model.predict(state_next)[0]
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            # return the index location of the current action
            # ic(action)
            # ic(self.ego_throttle_actions)
            # ic(self.ego_throttle_actions[action])
            # input("95")
            q_values[0][action] = q_update

            # ic(q_values)
            # ic(q_values[0])
            # ic(q_values.shape)
            # ic("101", action)
            self.model.fit(state, q_values, verbose=0)

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def save_model(self):
        self.model.save('models')

class Simulation:
    def __init__(self, env, agents, config, keyboard_agent):
        assert len(env.bodies) == len(agents), "each body must be assigned an agent and vice versa"
        self.env = env
        self.agents = agents
        self.config = config
        self.keyboard_agent = keyboard_agent
        self.reward_monitor = []
        self.weights_plot = []
        self.graph_save_episode = 5
        self.ego_win_rate = []
        self.ego_win_draw_rate = []
        self.ego = self.agents[0] #handle for ego agent
        self.ego_body = env.bodies[0] #handle for ego body

        self.width = env.constants.viewer_width
        self.height = env.constants.viewer_height
        self.M2PX = M2PX

        self.min_ego_throttle = env.bodies[0].constants.min_velocity
        self.max_ego_throttle = env.bodies[0].constants.max_velocity
        self.min_ped_velocity = env.bodies[1].constants.min_velocity
        self.max_ped_velocity = env.bodies[1].constants.max_velocity

        self.norm_limits = [[0, self.width],[0, self.height],[self.min_ego_throttle, self.max_ego_throttle],[0, 2*math.pi]]
        norm_limits_ped = [[0, self.width],[0, self.height],[self.min_ped_velocity, self.max_ped_velocity],[0, 2 * math.pi]]

        # generate 2D list of low-high norm values for each agent
        for non_ego_agent in range(len(agents[1:])):
            [self.norm_limits.append(limit) for limit in norm_limits_ped]

        ic(self.norm_limits)

        self.ego_body = env.bodies[0]
        self.noop_action = [0.0, 0.0]
        num_throttle_actions = 3
        self.ego_available_actions = [[throttle_action, self.noop_action[1]] for throttle_action in
                                      np.linspace(start=self.ego_body.constants.min_throttle,
                                                  stop=self.ego_body.constants.max_throttle, num=num_throttle_actions,
                                                  endpoint=True)]
        self.DQN_ego_type = False
        if self.DQN_ego_type:
            # The ego discrete velocity actions are defined in config.py line 312
            # Since we only want to control tis with network, we can ignore steering control
            # self.ego_action_space = num_actions # self.env.action_space[0].shape[0] # ego action space
            # # TODO - need to check this - action space == 2 but should be 3??
            # ic(self.ego_action_space)
            # ic(type(self.env.action_space))
            # ic(type(self.env.action_space[0]))
            # # current action space is 'box' for continuous space
            # # can switch to 'discrete' for discrete action space
            # ic(self.env.action_space[0])
            # ic(self.env.action_space[0].low)
            # ic(self.env.action_space[0].high)
            # ic(self.env.action_space[0].shape)
            # ic(self.env.action_space[0].shape[0])
            # input()

            # top-level ego action space (acceleration, steering)
            # self.ego_action_space = self.env.action_space[0].shape[0]
            obs = 0 # observation of all agents, ego + ped(s) in environment
            for index in range(0,len(self.env.observation_space)):
                obs += self.env.observation_space[index].shape[0]
            # ic(obs)
            self.obs = 8 # obs TODO
            # input()

            # each of these has a discrete number of actions ([-144,0,+144],[-pi/2...])
            # ego_steering_actions = 0 # steering disabled!
            # self.ego_action_space = self.ego.num_actions + ego_steering_actions
            # self.ego_throttle_actions = np.linspace(start=self.ego_body.constants.min_throttle, stop=self.ego_body.constants.max_throttle, num=self.ego.num_actions)

            self.Q_ego_type = False


            #available actions are -144,0,+144 if num_action = 3see config.setup.ego_config
            self.ego_throttle_actions = [self.ego_available_actions[i][0] for i in range(len(self.ego_available_actions))]
            self.ego_steering_actions = [self.ego_available_actions[i][1] for i in range(len(self.ego_available_actions))]
            self.dqn_solver = DQNSolver(obs, self.ego_throttle_actions)
            # ic(self.ego_throttle_actions)
            # ic(self.ego_steering_actions)
            # ic(self.ego_available_actions) #available actions are -144,0,+144 if num_action = 3see config.setup.ego_config
            # ic(len(self.ego_available_actions))
            # how often to solve the DQN
            self.solver_freq = SOLVER_FREQ
        else:
            self.Q_ego_type = True

        if self.keyboard_agent is not None:
            assert self.config.mode_config.mode is Mode.RENDER, "keyboard agents only work in render mode"

        if self.config.tester_config.agent is AgentType.ELECTION:
            self.election = Election(env, agents)
        else:
            self.election = None

        self.console = reporting.get_console(self.config.verbosity)

        self.episode_file = None
        if self.config.episode_log is not None:
            self.episode_file = reporting.get_episode_file_logger(self.config.episode_log)

        self.run_file = None
        if self.config.run_log is not None:
            self.run_file = reporting.get_run_file_logger(self.config.run_log)

    def normalise(self, value, min_bound, max_bound):
        if value < min_bound:
            return 0.0
        elif value > max_bound:
            return 1.0
        else:
            return (value - min_bound) / (max_bound - min_bound)

    def should_render(self, episode):
        return self.config.mode_config.mode is Mode.RENDER and episode % self.config.mode_config.episode_condition == 0

    def run(self):
        episode_data = list()
        run_start_time = timeit.default_timer()
        time_plot = []
        count = 0
        ego_wins = 0
        ego_win_ratio = 0
        ego_win_draw_ratio = 0

        for episode in range(1, self.config.episodes+1):
            episode_start_time = timeit.default_timer()

            state = self.env.reset()
            info = self.env.info()
            reward_plot = []

            self.console.debug(f"state={state}")

            for agent in self.agents:
                agent.reset()

            video_file = None
            if self.should_render(episode):
                frame = self.env.render(mode='rgb_array' if self.config.mode_config.video_dir is not None else 'human')
                if self.config.mode_config.video_dir is not None:
                    pathlib.Path(self.config.mode_config.video_dir).mkdir(parents=True, exist_ok=True)
                    import cv2
                    video_file = cv2.VideoWriter(f"{self.config.mode_config.video_dir}/episode{episode:0{len(str(self.config.episodes))}}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), self.env.frequency, (self.env.constants.viewer_width, self.env.constants.viewer_height), True)
                    video_file.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype("uint8"))
                if self.keyboard_agent is not None and self.env.unwrapped.viewer.window.on_key_press is not self.keyboard_agent.key_press:  # must render before key_press can be assigned
                    self.env.unwrapped.viewer.window.on_key_press = self.keyboard_agent.key_press

            final_timestep = self.config.max_timesteps
            for timestep in range(1, self.config.max_timesteps+1):

                # ---------------------------------------------
                # CHOOSE ACTION
                # ---------------------------------------------
                if BASELINE_MODE:
                    ego_action = self.ego_available_actions[-1]
                    opponent_action = [agent.choose_action(state, action_space, info) for agent, action_space in
                                       zip(self.agents[1:], self.env.action_space)]
                    joint_action = opponent_action
                    joint_action.insert(0, ego_action)  # add ego action back to start of the nested list

                if self.DQN_ego_type:
                    # normalise each agent state (position, orientation, velocity)
                    # use normalise() based on each variable min/max

                    # flatten list for network input
                    flat_state = [item for sublist in state for item in sublist]
                    np_flat_state = np.array(flat_state)
                    # ic(np_flat_state)


                    dqn_action = self.dqn_solver.act(np_flat_state)
                    ego_action = self.ego_available_actions[dqn_action]
                    self.dqn_solver.store_action.append(dqn_action)
                    # ic(dqn_action)
                    # ic(ego_action)
                    # input("261")
                    # now add pedestrian action to joint action space
                    opponent_action = [agent.choose_action(state, action_space, info) for agent, action_space in zip(self.agents[1:], self.env.action_space)]
                    # joint_action = list(zip([ego_action, 0.0], opponent_action))
                    joint_action = opponent_action
                    joint_action.insert(0, ego_action) #add ego action back to start of the nested list
                    # ic(joint_action)
                else:
                    joint_action = [agent.choose_action(state, action_space, info) for agent, action_space in zip(self.agents, self.env.action_space)]

                if self.election:
                    joint_action = self.election.result(state, joint_action)

                previous_state = state
                state, joint_reward, done, info, win_ego = self.env.step(joint_action)
                # if done:
                    # ic(done, win_ego, joint_reward)

                # monitor reward development over time
                reward_plot.append(joint_reward[0])


                #monitor the win ratio of the ego, regardless of absolute score
                if win_ego:
                    ego_wins+=1
                    ego_win_draw_ratio+=1
                    # print("Ego wins = %d" % ego_wins)
                if count>1:
                    ego_win_ratio = 100*ego_wins/count
                    self.ego_win_rate.append(ego_win_ratio)
                    # print("Ego win percentage = %5.1f" % ego_win_ratio)

                self.console.debug(f"timestep={timestep}")
                self.console.debug(f"action={joint_action}")
                self.console.debug(f"state={state}")
                self.console.debug(f"reward={joint_reward}")
                self.console.debug(f"done={done}")
                self.console.debug(f"info={info}")

                # for agent, action, reward in zip(self.agents, joint_action, joint_reward):
                #     agent.process_feedback(previous_state, action, state, reward)

                # ---------------------------------------------
                # PROCCESS FEEDBACK
                # ---------------------------------------------
                if BASELINE_MODE:
                    for agent, action, reward in zip(self.agents[1:], joint_action[1:], joint_reward[1:]):
                        agent.process_feedback(previous_state, action, state, reward, done)
                if self.DQN_ego_type:
                    # normalise state data (pos_x, pos_y, velocity, orientation)
                    flat_state = [item for sublist in state for item in sublist]
                    flat_previous_state = [item for sublist in previous_state for item in sublist]

                    np_flat_state = np.array(flat_state)
                    np_flat_previous_state = np.array(flat_previous_state)

                    norm_state = [self.normalise(flat_state[index], *self.norm_limits[index]) for index in range(len(flat_state))]
                    norm_prev_state = [self.normalise(flat_previous_state[index], *self.norm_limits[index]) for index in range(len(flat_previous_state))]

                    # ic(norm_state)
                    # ic(norm_prev_state)

                    # np_flat_state = np.reshape(flat_state, [1, self.obs])
                    # np_flat_previous_state = np.reshape(flat_previous_state, [1, self.obs])
                    np_flat_state = np.reshape(norm_state, [1, self.obs])
                    np_flat_previous_state = np.reshape(norm_prev_state, [1, self.obs])

                    # discrete action transpose
                    # -1 = -144, 0=0, +1=+144
                    ego_action_index  = np.where(self.ego_throttle_actions == joint_action[0][0])
                    ego_action_index  = ego_action_index[0][0] - 1

                    # ic(joint_action[0][0])
                    # ic(self.ego_throttle_actions)
                    # ic(ego_action_index[0][0])
                    # ic(self.ego_throttle_actions[ego_action_index[0][0]])

                    self.dqn_solver.remember(np_flat_previous_state, ego_action_index, joint_reward[0], np_flat_state, done)
                    if count % self.solver_freq ==0:
                        self.dqn_solver.experience_replay()

                    # solve for other agents
                    for agent, action, reward in zip(self.agents[1:], joint_action[1:], joint_reward[1:]):
                        agent.process_feedback(previous_state, action, state, reward, done)
                else:
                    for agent, action, reward in zip(self.agents, joint_action, joint_reward):
                        agent.process_feedback(previous_state, action, state, reward, done)


                # monitor rewards and feature weights
                ego = self.agents[0]
                # print("R ",  joint_reward[0])
                #enabled features
                # print("enabled_features list value ", list(ego.enabled_features[1]))
                # # for feature in ego.feature_weights:
                # print("FW type " , type(ego.feature_weights))
                # print("FW value " , ego.feature_weights[1].values())
                # print("FW type " , type(ego.feature_weights[1].values()))
                # print("FW list value ", list(ego.feature_weights[1].values())) #the [1] refs the value not name
                # #ego q value
                # print("Ego q-values",ego.store_q_values)
                # print("Reward + Weights ", joint_reward[0], list(ego.feature_weights[1].values()))


                if self.should_render(episode):
                    frame = self.env.render(mode='rgb_array' if self.config.mode_config.video_dir is not None else 'human')
                    if self.config.mode_config.video_dir is not None:
                        import cv2
                        video_file.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype("uint8"))

                if done:
                    final_timestep = timestep
                    break

            if video_file is not None:
                video_file.release()

            if self.should_render(episode) and not self.should_render(episode+1):
                self.env.close()  # closes viewer rather than environment

            # save the DQN model
            if self.DQN_ego_type and count%10==0:
                self.dqn_solver.save_model()

            episode_end_time = timeit.default_timer()
            episode_results = reporting.analyse_episode(episode, episode_start_time, episode_end_time, final_timestep, info, self.config, self.env)
            episode_data.append(episode_results)
            self.console.info(episode_results.console_message())
            if self.episode_file:
                self.episode_file.info(episode_results.file_message())

            self.reward_monitor.append(np.sum(reward_plot)) # monitor the rewards over the episodes

            if self.DQN_ego_type:
                self.weights_plot.append(list( self.dqn_solver.get_model_weights() )) # store the weights for plotting
            else:
                self.weights_plot.append(list(ego.feature_weights[1].values())) # store the weights for plotting

            plot_episode_graph = True
            if plot_episode_graph:
                if count > 2:
                    s_avg_1d_10 = uniform_filter1d(np.ndarray.flatten(np.array(self.reward_monitor)), size=10)
                    w_avg_1d_10 = uniform_filter1d(np.array(self.weights_plot), axis=0, size=10)
                    # ic(self.reward_monitor)
                    # ic(avg_1d_10)
                    s_avg_1d_100 = uniform_filter1d(np.ndarray.flatten(np.array(self.reward_monitor)), size=100)
                    w_avg_1d_100 = uniform_filter1d(np.array(self.weights_plot), axis=0, size=100)
                if count==1:
                    fig, ax1 = plt.subplots()
                if count >2:
                    plt.clf()
                    array_width = len(self.weights_plot[0])
                    array_length = len(self.weights_plot)
                    num_plots = array_width
                    # ic(np.shape(np.array(self.reward_monitor)))
                    # ic(array_width,array_length)
                    x = np.arange(array_length)
                    y= self.weights_plot
                    labels = list(ego.enabled_features[1])

                    # Have a look at the colormaps here and decide which one you'd like:
                    # http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html

                    colormap = plt.cm.cool
                    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.cool(np.linspace(0, 1, num_plots))))
                    for i in range(1, num_plots + 1):
                        plt.plot(x, y,'o', mfc='none', markersize=1, markeredgewidth=0.5)
                        if count>10:
                            plt.plot(w_avg_1d_10, '--', linewidth=0.1)
                            # if count>100:
                            #     plt.plot(w_avg_1d_100, 'r:', label='N=100')
                            #     labels.append('N=100')
                    # if count>10:
                    #     labels.append('N=10')
                    plt.xlabel('Episode')
                    plt.ylabel('Feature Weight')
                    plt.title('Feature Weights: Q-Ego win = %5.1f' % (ego_win_ratio))
                    if not self.DQN_ego_type:
                        plt.legend(labels, fontsize=8, loc='lower left')

                    score_subplot=True
                    action_subplot=True
                    if score_subplot:
                        left, bottom, width, height = [0.67, 0.17, 0.2, 0.2]
                        ax2 = fig.add_axes([left, bottom, width, height])
                        ax2.plot(self.ego_win_rate, color='green')
                        ax2.set_xlabel('win ratio')
                        ax2.set_xticks([])
                    if action_subplot:
                        left, bottom, width, height = [0.18, 0.17, 0.2, 0.2]
                        ax3 = fig.add_axes([left, bottom, width, height])
                        if self.DQN_ego_type:
                            ax3.hist(self.dqn_solver.store_action, color='green')
                        else:
                            ax3.hist(ego.store_action, color='green')
                        ax3.set_xlabel('throttle choice')
                        ax3.set_xticks([])
                        ax3.set_yticks([])
                    plt.pause(0.01)

                # save training progress intermittently
                if count % self.graph_save_episode == 0:
                    ts = datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
                    plt.savefig('plots/Weights_%s.png' % ts)
            count += 1


        else:
            run_end_time = timeit.default_timer()
            run_results = reporting.analyse_run(episode_data, run_start_time, run_end_time, self.config, self.env)
            self.console.info(run_results.console_message())
            if self.run_file:
                self.run_file.info(run_results.file_message())

        ts = datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
        plt.savefig('plots/Weights_%s.png' % ts)

        # np.savetxt('episode_rewards_{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()),self.reward_monitor,fmt='%9.3f' )
        # plt.savefig('epi_reward_{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

        # print the features and weights
        # ic(ego.feature_weights)
        ic(ego_win_ratio)
        self.env.close()  # closes viewer rather than environment