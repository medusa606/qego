import datetime
import pathlib
import timeit

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import uniform_filter1d

import reporting
from config import Mode, AgentType
from examples.election import Election

from icecream import ic

class Simulation:
    def __init__(self, env, agents, config, keyboard_agent):
        assert len(env.bodies) == len(agents), "each body must be assigned an agent and vice versa"
        self.env = env
        self.agents = agents
        self.config = config
        self.keyboard_agent = keyboard_agent
        self.reward_monitor = []
        self.weights_plot = []
        self.ego_win_rate = []
        self.ego_win_draw_rate = []

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
                joint_action = [agent.choose_action(state, action_space, info) for agent, action_space in zip(self.agents, self.env.action_space)]

                if self.election:
                    joint_action = self.election.result(state, joint_action)

                previous_state = state
                state, joint_reward, done, info, win_ego = self.env.step(joint_action)

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

            episode_end_time = timeit.default_timer()
            episode_results = reporting.analyse_episode(episode, episode_start_time, episode_end_time, final_timestep, info, self.config, self.env)
            episode_data.append(episode_results)
            self.console.info(episode_results.console_message())
            if self.episode_file:
                self.episode_file.info(episode_results.file_message())

            self.reward_monitor.append(np.sum(reward_plot)) # monitor the rewards over the episodes
            self.weights_plot.append(list(ego.feature_weights[1].values())) # store the weights for plotting

            plot_episode_graph = False
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
                        plt.plot(x, y,'o', mfc='none', markersize=3, markeredgewidth=0.5)
                        if count>10:
                            plt.plot(w_avg_1d_10, '--')
                            # if count>100:
                            #     plt.plot(w_avg_1d_100, 'r:', label='N=100')
                            #     labels.append('N=100')
                    # if count>10:
                    #     labels.append('N=10')
                    plt.xlabel('Episode')
                    plt.ylabel('Feature Weight')
                    plt.title('Feature Weights: Q-Ego win = %5.1f' % (ego_win_ratio))
                    plt.legend(labels, fontsize=8, loc='lower left')

                    left, bottom, width, height = [0.67, 0.17, 0.2, 0.2]
                    score_subplot=True
                    action_subplot=False
                    if score_subplot:
                        ax2 = fig.add_axes([left, bottom, width, height])
                        ax2.plot(self.ego_win_rate, color='green')
                        ax2.set_xlabel('win ratio')
                        ax2.set_xticks([])
                    if action_subplot:
                        ax2 = fig.add_axes([left, bottom, width, height])
                        ax2.hist(ego.store_action, color='green')
                        ax2.set_xlabel('throttle choice')
                        ax2.set_xticks([])
                        ax2.set_yticks([])
                    plt.pause(0.01)
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