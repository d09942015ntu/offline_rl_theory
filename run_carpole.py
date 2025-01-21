import gym
import numpy as np
import matplotlib.pyplot as plt

#env = gym.make("CartPole-v1",render_mode="human")
env = gym.make("CartPole-v1")
def env_render():
    pass
    #print("env render:")
    #print(env.render())
env.reset()
def print_observation_space(env):
    print(f"Observation space high: {env.observation_space.high}")
    print(f"Observation space low: {env.observation_space.low}")
    print(f"Number of actions in the action space: {env.action_space.n}")
print_observation_space(env)


def get_max_velocity(env):
    max_velo_cart = 0
    max_velo_pole = 0
    env.reset()
    done = False
    while not done:
        new_state, _, done, _, _ = env.step(1)
        if (abs(new_state[1]) > max_velo_cart):
            max_velo_cart = abs(new_state[1])
        if abs(new_state[3]) > max_velo_pole:
            max_velo_pole = abs(new_state[3])
        env.render()
        #env_render()
    print(f"Max_velo_cart={max_velo_cart}")
    print(f"Max_velo_pole={max_velo_pole}")
get_max_velocity(env)


DISCRETE_OS_SIZE = [25, 25] #our dimensions
real_observation_space = np.array([env.observation_space.high[2], 3.5]) #disregarding cart data
discrete_os_win_size = (real_observation_space * 2 / DISCRETE_OS_SIZE) #step-size

def get_discrete_state(state):
    #print(state)
    trimmed_state = np.array([state[2], state[3]])
    discrete_state = (trimmed_state + real_observation_space) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))

q_table = np.random.uniform(low=0, high=1, size =(DISCRETE_OS_SIZE + [env.action_space.n]))

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 12000
LOG_FREQUENCY = 2000
epsilon = 0.1
START_DECAY = 1
END_DECAY = EPISODES // 2
epsilon_decay_by = epsilon / (END_DECAY - START_DECAY)

rewards = []
for episode in range(EPISODES):
    #Just some logging info
    if episode % LOG_FREQUENCY == 0:
        render = True
        print(f"Episode {episode}, Reward={np.average(rewards)}")
        rewards.clear()
    else:
        render = False    #Resetting the environment as well as getting state 0
    discrete_state = get_discrete_state(env.reset()[0])
    done = False    #One iteration of the environment
    reward_eposodes = []
    while not done:        #Using epsilon to introduce exploration
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,2)
        new_state, reward, done, _, _ = env.step(action)
        reward_eposodes.append(reward)
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()        # Adjusting the values in our Q-table according to the Q-learning formula
            #env_render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
            discrete_state = new_discrete_state    #Decay epsilon
    rewards.append(np.sum(reward_eposodes))
    if END_DECAY >= episode >= START_DECAY:
        epsilon -= epsilon_decay_by