#reinforcement learner

import ftapp.core.model as model
import numpy as np
import random as rand


from collections import deque


import ftapp.core.constants as constants


class DQLearner(object):
    def __init__(self, \
                 num_states= constants.NUM_STATES,\
                 num_actions=constants.NUM_ACTIONS, \
                 alpha= constants.ALPHA, \
                 gamma=constants.GAMMA, \
                 rar=constants.RAR, \
                 radr= constants.RADR, \
                 rarm= constants.RARM, \
                 dyna=0, \
                 verbose=False):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.rarm = rarm

        self.verbose = verbose

        self.q = np.zeros(shape=(self.num_states, self.num_actions))

        self.s = np.zeros(shape=(1, self.num_states))
        # print("shape", self.s.shape)
        self.a = 0

        # nn version
        self.model = model.nueral_network(self.num_states, self.num_actions)
        self.memory = deque(maxlen=2000)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.rar:
            return rand.randint(0, self.num_actions - 1)
        # s = s.reshape(self.num_states)
        # print("state shape", state.shape, state)
        act_values = self.model.predict(state)
        # print("actions", act_values)
        return np.argmax(act_values[0])  # returns action

    def load(self, name):
        '''
        load saved model weights
        '''
        self.model.load_weights(name)

    def save(self, name):
        '''
        save model weights
        '''
        self.model.save_weights(name)

    def replay(self, batch_size=32):
        """ vectorized implementation; 30x speed up compared with for loop """
        minibatch = rand.sample(self.memory, batch_size)

        states = np.array([tup[0][0] for tup in minibatch])
        actions = np.array([tup[1] for tup in minibatch])
        rewards = np.array([tup[2] for tup in minibatch])
        next_states = np.array([tup[3][0] for tup in minibatch])
        done = np.array([tup[4] for tup in minibatch])

        # Q(s', a)
        target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)
        # end state target is reward itself (no lookahead)
        target[done] = rewards[done]

        # Q(s, a)
        target_f = self.model.predict(states)
        # make the agent to approximately map the current state to future discounted reward
        target_f[range(batch_size), actions] = target

        self.model.fit(states, target_f, epochs=1, verbose=0)
