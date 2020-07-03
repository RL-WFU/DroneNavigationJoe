from Mining_network import *
from env import *
from configurationSimple import ConfigSimple as config
import numpy as np
import matplotlib.pyplot as plt
from Mining_utils import *
import collections
import os

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def train(envs, policy, value):
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    episode_lengths = np.zeros(0)
    episode_rewards = np.zeros(0)
    most_common_actions = []
    interval_averages = np.zeros(config.num_episodes // config.plot_interval)
    interval_totals = np.zeros(config.num_episodes // config.plot_interval)
    interval_avg_mining_covered = np.zeros(config.num_episodes // config.plot_interval)
    interval_avg_total_covered = np.zeros(config.num_episodes // config.plot_interval)

    env_interval_averages = np.zeros([len(envs), config.num_episodes // config.plot_interval])
    env_interval_totals = np.zeros([len(envs), config.num_episodes // config.plot_interval])
    env_interval_avg_mining_covered = np.zeros([len(envs), config.num_episodes // config.plot_interval])
    env_interval_avg_total_covered = np.zeros([len(envs), config.num_episodes // config.plot_interval])

    env_interval_counts = np.zeros([len(envs), config.num_episodes // config.plot_interval])


    for i_episode in range(config.num_episodes):
        episode = []

        episode_lengths = np.append(episode_lengths, 0)
        episode_rewards = np.append(episode_rewards, 0)

        episode_env = envs[np.random.randint(0, len(envs))]

        env_interval_counts[episode_env.id, i_episode // config.plot_interval] += 1

        state, flat_local_map = episode_env.reset_environment_mining()

        done = False
        reward = 0
        actions = []

        for t in range(config.max_steps):
            if t < config.seq_length:
                action = np.random.randint(0, 4)

            else:
                states = get_last_t_states(config.seq_length, episode)
                action_probs = policy.predict(states)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_state, flat_local_map, reward, done = episode_env.step_mining(action, t)

            actions.append(action)
            episode.append(Transition(
                state=state, action=action, reward=reward, next_state=next_state, done=done
            ))

            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t


            if t >= config.seq_length:
                states = get_last_t_states(config.seq_length, episode)
                states1 = get_last_t_minus_one_states(config.seq_length, episode)

                value_next = value.predict(states1)
                td_target = reward + config.discount_factor * value_next
                td_error = td_target - value.predict(states)

                value.update(states, td_target, i_episode)
                policy.update(states, td_error, action, i_episode)

            if done:
                break

            state = next_state

        interval_totals[i_episode // config.plot_interval] += episode_rewards[i_episode]
        interval_averages[i_episode // config.plot_interval] = interval_totals[i_episode // config.plot_interval] \
                                                               / ((i_episode % config.plot_interval) + 1)

        percent_mining, total_mining, total_covered = episode_env.localMiningCovered()
        percent_covered = episode_env.calculate_covered('local')

        interval_avg_mining_covered[i_episode // config.plot_interval] = (interval_avg_mining_covered[
                                                                         i_episode // config.plot_interval] * (
                                                                             i_episode % config.plot_interval) + percent_mining) / (
                                                                            i_episode % config.plot_interval + 1)
        interval_avg_total_covered[i_episode // config.plot_interval] = (interval_avg_total_covered[
                                                                              i_episode // config.plot_interval] * (
                                                                                  i_episode % config.plot_interval) + percent_covered) / (
                                                                                 i_episode % config.plot_interval + 1)

        env_interval_totals[episode_env.id, i_episode // config.plot_interval] += episode_rewards[i_episode]
        env_interval_averages[episode_env.id, i_episode // config.plot_interval] = env_interval_totals[episode_env.id, i_episode // config.plot_interval] \
                                                                                    / (env_interval_counts[episode_env.id, i_episode // config.plot_interval])

        env_interval_avg_mining_covered[episode_env.id, i_episode // config.plot_interval] = (
            env_interval_avg_mining_covered[episode_env.id, i_episode // config.plot_interval] * (
            env_interval_counts[episode_env.id, i_episode // config.plot_interval] - 1) + percent_mining) / (
            env_interval_counts[episode_env.id, i_episode // config.plot_interval]
        )

        env_interval_avg_total_covered[episode_env.id, i_episode // config.plot_interval] = (env_interval_avg_total_covered[episode_env.id, i_episode // config.plot_interval] * (
                                                                                            env_interval_counts[episode_env.id, i_episode // config.plot_interval] - 1) + percent_covered) / (
                                                                                            env_interval_counts[episode_env.id, i_episode // config.plot_interval]
                                                                                             )

        if (i_episode + 1) % config.plot_interval == 0:
            """

            intervals = [i * config.plot_interval for i in range((i_episode//config.plot_interval) + 1)]
            plt.plot(intervals, interval_averages[0:(i_episode//config.plot_interval + 1)])
            plt.ylabel('Previous {} Episode Average'.format(config.plot_interval))
            plt.xlabel('Episode')
            plt.savefig(
                os.path.join(config.save_dir, 'Average Reward Episode {}'.format(i_episode))
            )
            plt.clf()

            plt.plot(intervals, interval_avg_mining_covered[0:(i_episode // config.plot_interval + 1)], linestyle='dashed', label='Mining')
            plt.plot(intervals, interval_avg_total_covered[0:(i_episode // config.plot_interval + 1)], label='Total')
            plt.xlabel('Episode')
            plt.ylabel('Previous {} Episode Average Coverage'.format(config.plot_interval))
            plt.legend()
            plt.savefig(os.path.join(config.save_dir, 'Average Coverages Episode {}'.format(i_episode)))
            plt.clf()

            episode_env.plot_local_path(os.path.join(config.save_dir, 'Drone Local Path Episode {}'.format(i_episode)))

            episode_env.save_GT_local_map(i_episode)
            """
            plot_graphs(i_episode, interval_averages, interval_avg_mining_covered, interval_avg_total_covered, episode_env, env_interval_averages, env_interval_avg_mining_covered, env_interval_avg_total_covered)

        data = collections.Counter(actions)
        most_common_actions.append(data.most_common(1))
        print("Episode {} finished. Reward: {}. Steps: {}. Most Common Action: {}. Mining Covered: {}/{}. Total visited: {}. Env: {}.".format(i_episode,
                                                                                                   episode_rewards[
                                                                                                       i_episode],
                                                                                                   episode_lengths[
                                                                                                       i_episode],
                                                                                                   most_common_actions[
                                                                                                       i_episode],
                                                                                                   total_covered, total_mining,
                                                                                                   percent_covered * 25 * 25,
                                                                                                   episode_env.id))

if __name__ == "__main__":
    environments = []

    i = 0
    for filename in os.listdir(config.img_dir):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".TIF") or filename.endswith(".JPG") or filename.endswith(".tiff"):
            environment = Env(config, os.path.join(config.img_dir, filename), i)
            environments.append(environment)
            i += 1
        else:
            continue

    #environment = Env(config, config.image, 0)
    #environments.append(environment)

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    with tf.Session() as sess:
        policy_net = PolicyEstimator_RNN(config)
        value_net = ValueEstimator_RNN(config)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        train(environments, policy_net, value_net)




#TO DO:
#Need to include 2 layers to lower complexity of local map and then attach to state
#Try adding entropy to the loss function
#Keep training to see if you can replicate 70% coverage
#Migrate to A3C to try to improve stability in training