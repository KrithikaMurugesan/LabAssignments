import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath("../simulator"))  # Where Trainer.py is located

from Trainer import *
from Simulator import *

from BasePilot import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3" # Shut up tensorflow warnings


class QLearningBaselinePilot(BasePilot):
    """

    Combined with the independent learning approach by Tan et al 1992.

    This is the most simple baseline from the actor-critic methods class.

    We learn one actor for one dedicated agent and share these parameters periodically among the other agents.

    """

    def __init__(self):
        # Override default hyperparameters defined in BasePilot
        self.set_hyperparameters({"num_drones": 3,
                                  "grid_size": (8, 8),
                                  "state_definition": ["drone_location", "other_drones_map", "location_seen_map", "obstacles_discovered_map"],
                                  "metric_gather_train": True,
                                  "num_episodes": 10000,
                                  "max_steps_per_episode": 64,
                                  "e_greedy_strategy": "linear_decay",
                                  "e_greedy": 0.9,
                                  "starting_point_strategy": "top_left_corner",
                                  "num_obstacles": 10,
                                  "obstacle_seeds": [1],
                                  "target_starting_location": (6, 6),
                                  "learning_rate": 1e-5,    # Do not set learning rate too high if using critic as baseline and relu
                                  "test_frequency": 100,
                                  "gif_frequency": None,
                                  "copy_frequency": 100,
                                  "reward_decay": 0.8
                                  })

        np.random.seed(self.hp["np_seed"])
        tf.set_random_seed(self.hp["tf_seed"])

        self.train_drone = 99

        self.episodes = None

        self.episode_cnt = 0

        self.sess = tf.Session()

        self.build_network()
        
        self.sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))



    def build_network(self):
        size_x = self.hp["grid_size"][0]
        size_y = self.hp["grid_size"][1]
        size_states = len(self.hp["state_definition"])

        self.X = tf.placeholder(tf.float32, shape=[None, size_x, size_y, size_states], name="states")
        self.R = tf.placeholder(tf.float32, shape=[None], name="rewards")
        self.A = tf.placeholder(tf.int32, shape=[None], name="actions")

        with tf.variable_scope("policy_network_train"):
            self.policy_network = self.policy_nn()

        # Network with delayed parameters for drones other than train drone
        with tf.variable_scope("policy_network_others"):
            self.policy_network_others = self.policy_nn()

        # Copy network parameters from trained drone to other drones
        var_trained = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="policy_network_train")
        var_target = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="policy_network_others")

        self.copy_ops = [var_target[i].assign(var_trained[i].value()) for i in range(len(var_target))]

        action_mask_one_hot = tf.one_hot(self.A,  # column index
                                         self.policy_network.shape[1],
                                         on_value=True,
                                         off_value=False,
                                         dtype=tf.bool)

        policy = tf.boolean_mask(self.policy_network, action_mask_one_hot)
        
        policy_loss = tf.reduce_sum(tf.squared_difference(policy, self.R))
        
        optimizer = tf.train.GradientDescentOptimizer(self.hp["learning_rate"])
        self.train_policy_ops = optimizer.minimize(policy_loss)


    ## Network contruction functions ###
    @staticmethod
    def dense(input, kernel_shape, activation=True):
        # Have to use get_variable to have the scope adjusted correctly
        kernel = tf.get_variable("kernel", kernel_shape, initializer= \
            tf.variance_scaling_initializer(scale=1.0,
                                            mode="fan_avg",
                                            distribution="uniform"))
        biases = tf.get_variable("biases", kernel_shape[1], initializer=tf.constant_initializer(0.1))
        raw_dense = tf.matmul(input, kernel) + biases
        if activation:
            return tf.nn.relu(raw_dense)
        else:
            return raw_dense

    @staticmethod
    def cnn(input, kernel_shape, num_filters):
        # Have to use get_variable to have the scope adjusted correctly
        conv = tf.layers.conv2d(inputs=input,
                                filters=num_filters,
                                kernel_size=kernel_shape,
                                padding="same")
        return conv

    def policy_nn(self):

        #inception = self.cnn(self.X, kernel_shape=[1, 1], num_filters=1)

        shape = self.X.get_shape().as_list()
        dim = np.prod(shape[1:])
        flatten = tf.reshape(self.X, [-1, dim])
        with tf.variable_scope("hidden1"):
            hidden1 = self.dense(flatten, [dim, dim])
        with tf.variable_scope("hidden2"):
            hidden2 = self.dense(hidden1, [dim, dim])
                
        with tf.variable_scope("action_raw"):
            out_policy = 4
            policy_raw = self.dense(hidden2, [dim, out_policy], activation=False)
        return policy_raw


    def get_policy(self, input):
        input = np.expand_dims(np.moveaxis(input, 0, -1), axis=0)   # Reshape cause cnn expects [None, width, height, channels]
        policies = self.sess.run(self.policy_network, feed_dict={self.X: input})
        if np.isnan(policies).any():
            raise ValueError("NaN occured in policy network")
        return policies

    def get_policy_others(self, input):
        input = np.expand_dims(np.moveaxis(input, 0, -1), axis=0)  # Reshape cause cnn expects [None, width, height, channels]
        policies = self.sess.run(self.policy_network_others, feed_dict={self.X: input})
        if np.isnan(policies).any():
            raise ValueError("NaN occured in policy network")
        return policies


    # Override
    def get_action(self, state, is_test=False, id=None):
        if self.hp["e_greedy_strategy"] == "deterministic" or is_test or id is not self.train_drone:
            return np.argmax(self.get_action_values(id, state).ravel())

        elif self.hp["e_greedy_strategy"] == "constant":
            if np.random.uniform(low=0.0,high=1.0) < self.hp["e_greedy"]:
                return np.random.randint(0, 4)
            else:
                return np.argmax(self.get_action_values(id, state).ravel())
        
        elif self.hp["e_greedy_strategy"] == "linear_decay":
                self.hp["e_greedy"] -= self.hp["e_greedy"]*(0.5/self.hp["num_episodes"]) #decays linearly to 50% of original e_greedy
                if np.random.uniform(low=0.0,high=1.0) < self.hp["e_greedy"]:
                    return np.random.randint(0, 4)
                else:
                    return np.argmax(self.get_action_values(id, state).ravel())
                
        else:
            raise ValueError("Specified e-greedy strategy not implemented <%s>" % self.hp["e_greedy_strategy"])


    # Override
    def get_action_values(self, id, state):
        if id == self.train_drone:
            return self.get_policy(state[id])
        else:
            return self.get_policy_others(state[id])

    # Override
    def store_episodes(self, episodes):
        if self.episodes is None:
            self.episodes = episodes
        else:
            self.episodes += episodes

    # Override
    def learn(self):
        id = self.train_drone
        self.episode_cnt += 1
        episode_count = len(self.episodes.rewards[id])

        next_policies = []
        for i in range(episode_count-1):
            next_policy = max(self.get_action_values(self.train_drone, {self.train_drone: self.episodes.states[id][i+1]}).T)
            next_policies.append(next_policy*self.hp["reward_decay"]+self.episodes.rewards[id][i])
        next_policies = np.concatenate(next_policies,axis=0).tolist()
        next_policies.append(self.episodes.rewards[id][episode_count-1])

        rewards = next_policies
        
        states = self.episodes.states[id]
        states = np.moveaxis(states, 1, -1)   # Reshape cause cnn expects [None, width, height, channels]

        feed_dict = {self.X: states,
                     self.A: self.episodes.actions[id],
                     self.R: rewards}

        self.sess.run(self.train_policy_ops, feed_dict=feed_dict)

        if self.episode_cnt % self.hp["copy_frequency"] == 0:
            self.sess.run(self.copy_ops)

        # Reset the episode data, clear memory
        self.episodes = None

        self.train_drone = np.random.choice(list(self.env.drones.keys()))  # randomizes the train drone every episode

    # Override
    def reset(self, hp={}):
        super().reset(hp)
        tf.reset_default_graph()
        self.__init__()
        self.setup()

    @staticmethod
    def reward_fkt(drone, move_direction, discovery_map, step_num):
        """Move the drone and get the reward."""
        try:
            drone.move(move_direction)
            if "T" in drone.observe_surrounding().values():
                return 1, True
            else:
                return -0.01, False     # Small movement cost
        except (PositioningError, IndexError):
            return -0.1, False          # We hit an obstacle, drone or tried to exit the grid

import multiprocessing
if __name__ == "__main__":
    pilot = QLearningBaselinePilot()

    args = pilot.parse_arguments()
    if args.on_cluster:
        pilot.hp["gif_frequency"] = None

    pilot.setup()
    pilot.run()

    def rolling(data, window):
        return np.convolve(data, np.ones((window,)) / window, mode="valid")


    if not args.on_cluster:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(pilot.trainer.test_rewards)
        plt.title("Rolling test rewards")
        plt.xlabel("test runs (every %sth epoch)" % pilot.hp["test_frequency"])
        plt.ylabel("average rewards")
        plt.savefig("../../../results/test_rewards.png")

        plt.figure()
        plt.plot(rolling(pilot.trainer.train_rewards, window=int(pilot.hp["num_episodes"]*0.1)))
        plt.title("Rolling train rewards")
        plt.xlabel("epochs")
        plt.ylabel("average rewards")
        plt.savefig("../../../results/train_rewards.png")
    else:
        np.save('test_rewards.npy', pilot.trainer.test_rewards)
        np.save('train_rewards.npy', pilot.trainer.train_rewards)


