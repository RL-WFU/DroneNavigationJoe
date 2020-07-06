from env import *
from configurationSimple import ConfigSimple as config
import matplotlib.pyplot as plt

#env = Env(config, config.image, 0)
"""
actions = []
for l in range(200):
    actions.append(0)
    actions.append(4)
    actions.append(1)

total_reward = 0
for a in actions:
    state, local_map, reward, done = env.step(a, 1)
    total_reward += reward
    env.save_local_map('local_map_test.png')

print('Reward:', total_reward)
env.plot_path('drone_path_test.png')
env.save_map('map_test.png')
"""

#env.sim.showMap()
#env.sim.showLocalMap(low_row=env.local_map_lower_row, up_row=env.local_map_upper_row, low_col=env.local_map_lower_col, up_col=env.local_map_upper_col)

#_, tot, _ = env.localMiningCovered()
#print(tot)

shape = np.random.randint(0, 10, [6, 10])

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)

print(len(axs), len(axs[0]))

fig.suptitle('Shapes')

idx = 0
for i in range(len(axs)):
    for j in range(len(axs[0])):
        axs[i][j].plot(shape[idx])
        axs[i][j].set_title('Env {}'.format(idx))

        idx += 1

for ax in axs.flat:
    ax.set(xlabel='index', ylabel='num')

for ax in axs.flat:
    ax.label_outer()

plt.show()