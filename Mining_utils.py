import numpy as np
from configurationSimple import ConfigSimple as config
import matplotlib.pyplot as plt
import os

def get_last_t_states(t, episode):
    states = []
    for i, transition in enumerate(episode[-t:]):
        states.append(transition.state)

    states = np.asarray(states)
    states = np.reshape(states, [1, t, config.vision_size + 4])

    return states

def get_last_t_minus_one_states(t, episode):
    states = []
    for i, transition in enumerate(episode[-t + 1:]):
        states.append(transition.state)

    states.append(episode[-1].next_state)

    states = np.asarray(states)
    states = np.reshape(states, [1, t, config.vision_size + 4])

    return states


def plot_graphs(i_episode, interval_averages, interval_avg_mining_covered, interval_avg_total_covered, episode_env, env_interval_averages, env_interval_avg_mining_covered, env_interval_avg_total_covered):
    intervals = [i * config.plot_interval for i in range((i_episode // config.plot_interval) + 1)]
    plt.plot(intervals, interval_averages[0:(i_episode // config.plot_interval + 1)])
    plt.ylabel('Previous {} Episode Average'.format(config.plot_interval))
    plt.xlabel('Episode')
    plt.savefig(
        os.path.join(config.save_dir, 'Average Reward Episode {}'.format(i_episode))
    )
    plt.clf()

    plt.plot(intervals, interval_avg_mining_covered[0:(i_episode // config.plot_interval + 1)], linestyle='dashed',
             label='Mining')
    plt.plot(intervals, interval_avg_total_covered[0:(i_episode // config.plot_interval + 1)], label='Total')
    plt.xlabel('Episode')
    plt.ylabel('Previous {} Episode Average Coverage'.format(config.plot_interval))
    plt.legend()
    plt.savefig(os.path.join(config.save_dir, 'Average Coverages Episode {}'.format(i_episode)))
    plt.clf()

    episode_env.plot_local_path(os.path.join(config.save_dir, 'Drone Local Path Episode {}'.format(i_episode)))

    episode_env.save_GT_local_map(i_episode)

    rows = 2
    cols = len(env_interval_averages) / rows
    figs, axs = plt.subplots(rows, int(cols), sharex=True, sharey=True)
    figs.suptitle('Interval Averages by Environment')
    idx = 0
    for i in range(len(axs)):
        for j in range(len(axs[0])):
            axs[i][j].plot(intervals, env_interval_averages[idx, 0:(i_episode // config.plot_interval + 1)])
            axs[i][j].set_title('Image{}.jpg'.format(idx + 1))
            idx += 1

    for ax in axs.flat:
        ax.set(xlabel='Episode', ylabel='Reward')
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(os.path.join(config.save_dir, 'Env Average Reward Episode {}'.format(i_episode)))
    plt.clf()

    rows = 2
    cols = len(env_interval_avg_mining_covered) / rows
    figs, axs = plt.subplots(rows, int(cols), sharex=True, sharey=True)
    figs.suptitle('Interval Average Coverages by Environment')
    idx = 0
    for i in range(len(axs)):
        for j in range(len(axs[0])):
            axs[i][j].plot(intervals, env_interval_avg_mining_covered[idx, 0:(i_episode // config.plot_interval + 1)], linestyle='dashed', label='Mining')
            axs[i][j].plot(intervals, env_interval_avg_total_covered[idx, 0:(i_episode // config.plot_interval + 1)], label='Total')
            axs[i][j].set_title('Image{}.jpg'.format(idx + 1))
            idx += 1

    for ax in axs.flat:
        ax.set(xlabel='Episode', ylabel='Percent Covered')
    for ax in axs.flat:
        ax.label_outer()
    plt.legend()

    plt.savefig(os.path.join(config.save_dir, 'Env Average Coverage Episode {}'.format(i_episode)))
    plt.clf()
