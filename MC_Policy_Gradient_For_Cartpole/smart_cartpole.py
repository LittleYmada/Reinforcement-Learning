import gym
import tensorflow as tf
import numpy as np

INPUT_SIZE = 4
HIDDEN_UNIT_NUM = 4
OUTPUT_SIZE = 1
LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.95

def Cartpole_policy():
    initializer = tf.contrib.layers.variance_scaling_initializer()
    X = tf.placeholder(tf.float32, shape=(None, INPUT_SIZE))
    hidden = tf.layers.dense(X, HIDDEN_UNIT_NUM, activation = tf.nn.relu,
                             kernel_initializer = initializer)
    logit = tf.layers.dense(hidden, OUTPUT_SIZE, activation = tf.nn.relu,
                             kernel_initializer = initializer)
    output = tf.nn.sigmoid(logit)
    p_left_right = tf.concat(axis = 1, values = [output, 1 - output])
    #多项式采样会将梯度断开吗
    action = tf.multinomial(tf.log(p_left_right), 1)
    return X, logit, action

def Get_Gradient(logit, action):
    y = 1. - tf.to_float(action)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, 
                                                        logits = logit)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    grads_and_vars = optimizer.compute_gradients(cross_entropy)
    gradients = [gradient for gradient, variable in grads_and_vars]
    gradient_placeholders = []
    grads_and_vars_feed = []
    for gradient, variable in grads_and_vars:
        gradient_placeholder = tf.placeholder(tf.float32,
                                              shape = gradient.get_shape())
        gradient_placeholders.append(gradient_placeholder)
        grads_and_vars_feed.append((gradient_placeholder, variable))
    training_op = optimizer.apply_gradients(grads_and_vars_feed)
    return gradient_placeholders, gradients, grads_and_vars, training_op

def discounted_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulated_reward = 0
    for step in reversed(range(len(rewards))):
        cumulated_reward = discount_rate * cumulated_reward + rewards[step]
        discounted_rewards[step] = cumulated_reward
    return discounted_rewards

def discounted_and_normalized_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discounted_rewards(rewards, discount_rate)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    mean = np.mean(flat_rewards)
    std = np.std(flat_rewards)
    return [(discounted_rewards - mean)/std for discounted_rewards in all_discounted_rewards]

if __name__ == "__main__":
    n_iterations = 250
    n_max_steps = 1000
    n_games_per_update = 10
    save_iterations = 10

    env = gym.make("CartPole-v0")
    #obs = env.reset()
    
    X, logit, action = Cartpole_policy()
    gradient_placeholders, gradients, grads_and_vars, training_op = Get_Gradient(logit, action)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            all_rewards = []
            all_gradients = []
            for game in range(n_games_per_update):
                obs = env.reset()
                current_rewards = []
                current_gradients = []
                for step in range(n_max_steps):
                    action_val, gradient_val = sess.run(
                                [action, gradients],
                                feed_dict = {X : obs.reshape(1, INPUT_SIZE)}
                            )
                    obs, reward, done, info = env.step(action_val[0][0])
                    current_rewards.append(reward)
                    current_gradients.append(gradient_val)
                    if done:
                        break
                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)

            all_rewards = discounted_and_normalized_rewards(all_rewards, 
                                                            DISCOUNT_RATE)
            feed_dict = {}
            for var_index, gradient_placeholder in enumerate(gradient_placeholders):
                #all_gradients = [all_gradients[game_index][step][var_index] 
                #                 for game_index, rewards in enumerate(all_rewards)
                #                 for step, reward in enumerate(rewards)]
                mean_gradients = np.mean([all_gradients[game_index][step][var_index]
                    for game_index, rewards in enumerate(all_rewards)
                    for step, reward in enumerate(rewards)], axis = 0)
                feed_dict[gradient_placeholder] = mean_gradients
            sess.run(training_op, feed_dict = feed_dict)
            print("Iteration: %d" % (iteration))
            if iteration % save_iterations == 0:
                saver.save(sess, "./my_policy_net_pg.ckpt")



