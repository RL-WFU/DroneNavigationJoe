import tensorflow as tf

class PolicyEstimator_RNN:
    def __init__(self, config, scope='RNN_model_policy'):
        self.config = config
        self.batch_size = self.config.batch_size
        self.train_length = self.config.seq_length
        self.sight_dim = self.config.sight_dim

        with tf.variable_scope(scope):
            self.local_map = tf.placeholder(shape=[None, self.train_length, 625], dtype=tf.float32, name='local_map')

            self.dense1 = tf.contrib.layers.fully_connected(inputs=self.local_map, num_outputs=100)
            self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=16)

            self.state = tf.placeholder(shape=[None, self.train_length, config.vision_size + 4],
                                        dtype=tf.float32, name='state')


            self.final_state = tf.concat([self.state, self.dense2], axis=2)

            self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(64)
            self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.final_state, initial_state=self.initial_state)
            self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * 64])

            self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=4)


            self.action_probs = tf.squeeze(tf.nn.softmax(self.output))
            #End of predict step

            #Start of update step
            self.action = tf.placeholder(tf.int32, name='action')
            self.target = tf.placeholder(tf.float32, name='target')

            self.picked_action_prob = tf.gather(self.action_probs, self.action)
            self.picked_action_prob = tf.cond(self.picked_action_prob < 1e-30, lambda: tf.constant(1e-30), lambda: tf.identity(self.picked_action_prob))

            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
            self.train_op = self.optimizer.minimize(self.loss)


            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope), max_to_keep=10)

            #if self.args.test:
                #self.saver.restore(sess, self.args.weight_dir + policy_weight)




    def predict(self, states, local_map, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: states, self.local_map: local_map})

    def update(self, states, target, action, episode, local_map, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.state: states, self.action: action, self.target: target, self.local_map: local_map}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)

        if episode % (self.config.plot_interval * 4) == 0 and episode != 0:
            self.saver.save(sess, self.config.weight_dir + '/policy_weights', global_step=episode)

        return loss



class ValueEstimator_RNN:
    def __init__(self, config, scope='RNN_model_value'):
        self.config = config
        self.batch_size = self.config.batch_size
        self.train_length = self.config.seq_length
        self.sight_dim = self.config.sight_dim
        with tf.variable_scope(scope):
            self.local_map = tf.placeholder(shape=[None, self.train_length, 625], dtype=tf.float32, name='local_map')

            self.dense1 = tf.contrib.layers.fully_connected(inputs=self.local_map, num_outputs=100)
            self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=16)

            self.state = tf.placeholder(shape=[None, self.train_length, config.vision_size + 4],
                                        dtype=tf.float32, name='state')

            self.final_state = tf.concat([self.state, self.dense2], axis=2)

            self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(64)
            self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.final_state, initial_state=self.initial_state)
            self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * 64])

            self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=1)

            self.value_estimate = tf.squeeze(self.output)
            #End of predict step

            #Start of update step
            self.target = tf.placeholder(tf.float32, name='target')

            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
            self.train_op = self.optimizer.minimize(self.loss)

            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope), max_to_keep=10)
            #if self.args.test:
                #self.saver.restore(sess, self.args.weight_dir + value_weight)

    def predict(self, states, local_maps, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: states, self.local_map: local_maps})

    def update(self, states, target, episode, local_maps, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: states, self.target: target, self.local_map: local_maps}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        if episode % (self.config.plot_interval * 4) == 0 and episode != 0:
            self.saver.save(sess, self.config.weight_dir + '/value_weights', global_step=episode)

        return loss


#Train a model for the good agent to tell if their agent is corrupt
    #Random choice if act corrupt or not, supervised learning to get good agent to tell if it is corrupt

#Have the corrupt agent decide whether to be corrupt or not based on if the good agent seems to think its corrupt
