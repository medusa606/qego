from argparse import ArgumentParser, ArgumentTypeError
from multiprocessing import Pool

from examples.constants import M2PX
from reporting import Verbosity
from config import Config, PedestriansConfig, HeadlessConfig, QLearningConfig, FeatureConfig, AgentType, RandomConfig, \
    RandomConstrainedConfig, ProximityConfig, ElectionConfig, CollisionType
from simulation import Simulation


def make_tester_config(agent_type):
    if agent_type is AgentType.RANDOM:
        return RandomConfig(epsilon=0.01)
    elif agent_type is AgentType.RANDOM_CONSTRAINED:
        return RandomConstrainedConfig(epsilon=0.5)
    elif agent_type is AgentType.PROXIMITY:
        return ProximityConfig(threshold=float(M2PX * 34))
    elif agent_type is AgentType.ELECTION:
        return ElectionConfig(threshold=float(M2PX * 34))
    elif agent_type is AgentType.Q_LEARNING:
        return QLearningConfig(
            alpha=0.18,
            gamma=0.87,
            epsilon=0.0005,
            features=FeatureConfig(
                distance_x=True,
                distance_y=True,
                distance=True,
                on_road=False,
                facing=False,
                inverse_distance=True
            ),
            log=None
        )
    else:
        raise NotImplementedError


def make_config(tester_type, alpha, gamma, epsilon):
    log_dir = f"logs/tester={tester_type}/alpha={alpha}/gamma={gamma}/epsilon={epsilon}"
    return log_dir, Config(
        verbosity=Verbosity.SILENT,
        episode_log=f"{log_dir}/episode.log",
        run_log=f"{log_dir}/run.log",
        seed=0,
        episodes=10,
        max_timesteps=1000,
        terminate_collisions=CollisionType.EGO,
        terminate_ego_zones=True,
        terminate_ego_offroad=False,
        reward_win=6000.0,
        reward_draw=2000.0,
        cost_step=4.0,
        scenario_config=PedestriansConfig(
            num_pedestrians=1,
            outbound_pavement=1.0,
            inbound_pavement=1.0
        ),
        ego_config=QLearningConfig(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            features=FeatureConfig(
                distance_x=True,
                distance_y=False,
                distance=True,
                relative_angle=True,
                heading=True,
                on_road=False,
                inverse_distance=False
            ),
            log=f"{log_dir}/ego-qlearning.log"
        ),
        tester_config=make_tester_config(tester_type),
        mode_config=HeadlessConfig()
    )


def run(tester_type, alpha, gamma, epsilon):
    # set a label to match the params
    label = f"tester={tester_type}, alpha={alpha}, gamma={gamma}, epsilon={epsilon}"
    print(f"starting: {label}")

    # this generates a config file for each iterable in the param sweep
    log_dir, config = make_config(tester_type, alpha, gamma, epsilon)
    # make sure the results you need are being captured here
    config.write_json(f"{log_dir}/config.json")

    np_seed, env, agents, keyboard_agent = config.setup()

    # this runs the simulation
    simulation = Simulation(env, agents, config=config, keyboard_agent=keyboard_agent)
    simulation.run()

    print(f"finished: {label}")


class PoolParser(ArgumentParser):
    def __init__(self):
        super().__init__()

        def positive_int(value):
            ivalue = int(value)
            if ivalue < 1:
                raise ArgumentTypeError(f"invalid positive int value: {value}")
            return ivalue
        # number of default cores set here for pool
        self.add_argument("-p", "--processes", type=positive_int, default=15, metavar="N", help="set number of processes as %(metavar)s (default: %(default)s)")

    def parse_pool(self):
        args = self.parse_args()
        return Pool(args.processes)


if __name__ == '__main__':
    # set any type of configuration list here
    # tester_types = [AgentType.RANDOM, AgentType.RANDOM_CONSTRAINED, AgentType.PROXIMITY]
    tester_types = [AgentType.RANDOM_CONSTRAINED]
    alphas = [0.1, 0.5, 0.9]
    gammas = [0.1, 0.5, 0.9]
    epsilons = [0.1, 0.5, 0.9]

    parameters = [(tester_type, alpha, gamma, epsilon) for tester_type in tester_types for alpha in alphas for gamma in gammas for epsilon in epsilons]

    parser = PoolParser()
    pool = parser.parse_pool()
    pool.starmap(run, parameters)
