import numpy as np
import gym
import random


class State:
    def __init__(self):
        self.prob_actions = list([0, 0, 0, 0])

    def get_prob_action(self, action):
        return self.prob_actions[action]

    def get_max_value(self):
        return np.max(self.prob_actions)

    def get_maxarg(self):
        return np.argmax(self.prob_actions)

    def state_update(self, action, max_value, Beta_t, reward, gamma):
        self.prob_actions[action] = self.prob_actions[action] + Beta_t*(reward + gamma*max_value - self.prob_actions[action])


class Game:
    def __init__(self, number_of_states):
        self.Q = list()
        for state in range(number_of_states):
            self.Q.append(State())

    def get_prob(self, state):
        return self.Q[state].get_max_value()

    def get_maxarg(self, state):
        return self.Q[state].get_maxarg()

    def update(self, state, action, new_state, Beta_t, reward, gamma):
        max_value = self.Q[new_state].get_max_value()
        self.Q[state].state_update(action, max_value, Beta_t, reward, gamma)


def main():
    env = gym.make("FrozenLake-v1")

    FrozenGame = Game(16)
    all_episodes = 20000
    Beta_t = 0.7
    max_steps = 99
    gamma = 0.95

    epsilon = 1.0
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.005

    for episode in range(all_episodes):
        state = env.reset()
        step = 0
        done = False

        for step in range(max_steps):
            n = random.uniform(0, 1)
            if n > epsilon:
                action = FrozenGame.Q[state].get_maxarg()
            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)
            FrozenGame.update(state, action, new_state, Beta_t, reward, gamma)

            state = new_state

            if done is True:
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

    env.reset()

    percent = 0
    ile = 1000

    for episode in range(ile):
        state = env.reset()
        step = 0
        done = False
        print(f'EPISODE: {episode}')

        for step in range(max_steps):
            action = FrozenGame.get_maxarg(state)
            new_state, reward, done, info = env.step(action)
            if done:
                env.render()
                if new_state == 15:
                    print("We reached our Goal ")
                    percent += 1
                else:
                    print("We fell into a hole ")

                print("Number of steps", step)

                break
            state = new_state
    env.close()

    print(f'Result: {percent*100/ile}%')


if __name__ == "__main__":
    main()
