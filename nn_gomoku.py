## Refer to http://mcts.ai/

import gym
import gym_gomoku
import tensorflow as tf
import time
import numpy as np
import copy
import random
import multiprocessing
from multiprocessing import Process, Queue, Value, Lock
import os
from collections import deque
import _pickle as pickle
import gzip
from time import localtime, strftime

working_directory = './temp/'

def TestStateNodeReceive(in_queue: Queue):
    game_state, node = in_queue.get()
    print('Data Received.')
    print(game_state.board_state)
    print(node.untried_action_list)


class GameState:
    def __init__(self, board_state, just_decided=2):
        self.board_state = copy.deepcopy(board_state)
        self.size = len(board_state)
        self.just_decided = just_decided
        self.is_win = self.CheckWin()
        self.draw = False

    def Clone(self):
        clone = GameState(self.board_state)
        clone.board_state = copy.deepcopy(self.board_state)
        clone.just_decided = self.just_decided
        return clone

    def GetAvailableAction(self):
        untried_action_list = []

        for row in range(self.size):
            for col in range(self.size):
                if self.board_state[row][col] == 0:
                    untried_action_list.append(row * self.size + col)
        return untried_action_list

    def CheckWin(self):
        win = False
        board_full = True
        for row in range(self.size):
            for col in range(self.size):
                if self.board_state[row][col] == 0:
                    board_full = False
                if self.board_state[row][col] == self.just_decided:
                    # Check Horizontal
                    if col <= self.size - 5:
                        if self.board_state[row][col + 1] == self.just_decided and \
                                self.board_state[row][col + 2] == self.just_decided and \
                                self.board_state[row][col + 3] == self.just_decided and \
                                self.board_state[row][col + 4] == self.just_decided:
                            win = True
                            break
                    # Check Vertical
                    if row <= self.size - 5:
                        if self.board_state[row + 1][col] == self.just_decided and \
                                self.board_state[row + 2][col] == self.just_decided and \
                                self.board_state[row + 3][col] == self.just_decided and \
                                self.board_state[row + 4][col] == self.just_decided:
                            win = True
                            break
                    if row <= self.size - 5 and col <= self.size - 5:
                        # Check Rightward Diagonal
                        if self.board_state[row + 1][col + 1] == self.just_decided and \
                                self.board_state[row + 2][col + 2] == self.just_decided and \
                                self.board_state[row + 3][col + 3] == self.just_decided and \
                                self.board_state[row + 4][col + 4] == self.just_decided:
                            win = True
                            break
                    if row <= self.size - 5 and col >= 4:
                        # Check Leftward Diagonal
                        if self.board_state[row + 1][col - 1] == self.just_decided and \
                                self.board_state[row + 2][col - 2] == self.just_decided and \
                                self.board_state[row + 3][col - 3] == self.just_decided and \
                                self.board_state[row + 4][col - 4] == self.just_decided:
                            win = True
                            break
            if win is True:
                break
        if board_full is True and win is False:
            self.draw = True
        return win

    def DoAction(self, action):
        self.just_decided = 3 - self.just_decided
        self.board_state[int(action / self.size)][action % self.size] = self.just_decided

        # Check Win/Lost
        self.is_win = self.CheckWin()
        return self.is_win


class Node:
    def __init__(self, game_state, parent, action, action_predictor=None):
        self.parent = parent
        self.action = action
        self.child_list = []
        self.untried_action_list = game_state.GetAvailableAction()
        self.action_prob = None
        self.value = 0.0
        self.visit = 0.0
        self.just_decided = game_state.just_decided
        self.action_predictor = action_predictor
        if action_predictor is not None:
            action_prob = action_predictor.predict_action_prob([game_state.board_state])[0]
            self.SetActionProb(action_prob, game_state.size)

    def SetActionProb(self, action_prob, board_size):
        self.action_prob = [0 for _ in range(board_size * board_size)]
        prob_sum = 0
        for action_idx in self.untried_action_list:
            self.action_prob[action_idx] = action_prob[action_idx]
            prob_sum += action_prob[action_idx]
        self.action_prob = (np.array(self.action_prob) / prob_sum).tolist()

    def GetScore(self):
        if self.visit > 0:
            # return self.visit
            return self.value / self.visit
        else:
            return 0.0

    def AddChild(self, state, action):
        state.DoAction(action)
        child_node = Node(state, self, action, action_predictor= self.action_predictor)
        self.child_list.append(child_node)
        self.untried_action_list.remove(action)
        return child_node

    def SearchChild(self, action):
        for child in self.child_list:
            if child.action == action:
                return child
        return None

    def Select(self):
        # print('Selection Among %d Action'%len(self.child_list))
        # s = sorted(self.child_list, key=lambda c: c.win / c.visit + np.sqrt(2 * np.log(self.visit) / c.visit))[-1]
        # return s
        score_list = [0] * len(self.child_list)
        for idx in range(len(self.child_list)):
            child = self.child_list[idx]
            if child.visit <= 0:
                return None
            else:
                score_list[idx] = child.value / child.visit + np.sqrt(2 * np.log(self.visit) / child.visit)

        return self.child_list[np.argmax(score_list)]

    def SearchBestChild(self, randomness=0):
        best_child = None
        best_score = 0.0
        action_score_list = []
        if randomness == 0:
            weight_list = []
            action_list = []
            for cur_child in self.child_list:
                weight_list.append(cur_child.GetScore())
                cur_action = cur_child.action
                action_list.append(cur_action)
                if weight_list[-1] > best_score:
                    best_score = weight_list[-1]
                    best_child = cur_child
            weight_list = (np.array(weight_list) / np.sum(weight_list)).tolist()
            for cur_action, cur_weight in zip(action_list, weight_list):
                action_score_list.append((cur_action, cur_weight))
            return best_child, action_score_list
        else:
            weight_list = []
            action_list = []
            for cur_child in self.child_list:
                weight_list.append(cur_child.GetScore() ** (1/randomness))
                cur_action = cur_child.action
                action_list.append(cur_action)
            weight_list = (np.array(weight_list) / np.sum(weight_list)).tolist()
            for cur_action, cur_weight in zip(action_list, weight_list):
                action_score_list.append((cur_action, cur_weight))
            idx = np.random.choice(range(len(weight_list)), 1, replace=False, p=weight_list)[0]
            return self.child_list[idx], action_score_list

def print_action_score(board_state, action_score_list):
    board_size = len(board_state)
    board_state = (np.array(board_state) * (-1)).tolist()
    for action, score in action_score_list:
        board_state[int(action / board_size)][action % board_size] = int(score * 100)
    print(np.array(board_state))


def GetRolloutProbability(state, candidate_action_list=None):
    if candidate_action_list is None:
        candidate_action_list = state.GetAvailableAction(state)

    return np.array([1] * len(candidate_action_list)) / len(candidate_action_list)


def ProcPredictValue(nn_path, device, input_queue, output_queue, proc_state, saver_lock, freeze_network):
    # Init neural network
    value_predictor = GomokuNetwork(nn_path, device=device, saver_lock=saver_lock, owner_name='ValuePredictor')
    print('GomokuNetwork Initialized.')
    while True:
        if freeze_network.value == 0:
            # Always use the latest network, if this is trainee
            print('Restore network in ProcPredictValue')
            value_predictor.restore_network()
            freeze_network.value = 1
        proc_state.value = 0  # Waiting
        state_list = []
        action_list_list = []
        concrete_value_list = []
        state, action_list = input_queue.get()
        if state.CheckWin() is True:
            if state.just_decided == 1:
                concrete_value_list.append(1.0)
            else:
                concrete_value_list.append(0.0)
        else:
            concrete_value_list.append(-1)
        state_list.append(state.board_state)
        action_list_list.append(action_list)
        # Get more samples from Queue, if it exists
        while input_queue.empty() is False:
            state, action_list = input_queue.get()
            state_list.append(state.board_state)
            action_list_list.append(action_list)

        proc_state.value = 1  # Processing
        pred_val_list = value_predictor.predict(state_list)
        for action_list, pred_val, concrete_val in zip(action_list_list, pred_val_list, concrete_value_list):
            if concrete_val < 0:
                output_queue.put((action_list, pred_val))
            else:
                output_queue.put((action_list, concrete_val))

def TransMatToArr(matrix):
    return np.array(matrix).flatten().tolist()

class GomokuNetwork:
    def __init__(self, network_path=None, replay_buffer_size=10000, board_size=9, lr=0.01, device='/gpu:0',
                 saver_lock=None, train_mode=False, owner_name = ''):
        self.owner_name = owner_name
        self.board_size = board_size
        self.train_mode = train_mode
        self.saver_lock = saver_lock
        self.replay_buffer_size = replay_buffer_size
        self.init_lr = lr
        self.device = device
        self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        if network_path is None:
            network_path = working_directory + 'network_weight/'
        if not os.path.exists(network_path):
            os.makedirs(network_path)
        self.network_path = network_path
        self.save_time = -1
        # If exist previous model file, restore it.
        if train_mode is True:
            # Setup replay buffers
            self.replay_buffer = deque(maxlen=replay_buffer_size)
        if os.path.exists(network_path + 'model.ckpt.data-00000-of-00001'):
            print('Load model.')
            self.load_model(network_path, train_mode)
            print('Done')
        else:
            # Build model.
            print('Build model.')
            self.build_model(train_mode)
            print('Done')

        if self.train_mode is True:
            self.load_training_data()

        self.tb_path_name = working_directory + strftime('tb/%Y-%m-%d_%H-%M-%S/', localtime())
        self.summary_writer = None

        self.batch_size = 64
        self.eval_frequency = 10
        self.save_frequency = 10000
        self.iter_idx = 0
        self.cur_win_rate = 0
        self.is_restored = False

    def load_model(self, network_path, train_mode):
        with self.graph.as_default():
            with tf.device(self.device):
                self.saver = tf.train.import_meta_graph(network_path + 'model.ckpt.meta', clear_devices=True)
                # Assign placeholder and operators
                self.board_state = tf.get_collection('board_state')[0]
                self.predicted_value = tf.get_collection('predicted_value')[0]
                self.predicted_action_out = tf.get_collection('predicted_action_out')[0]
                self.predicted_action_prob = tf.get_collection('predicted_action_prob')[0]
                self.save_time = os.path.getmtime(network_path + 'model.ckpt.data-00000-of-00001')
                if train_mode is True:
                    self.create_train_op()
                self.sess.run(tf.global_variables_initializer())
                self.saver.restore(self.sess, network_path + 'model.ckpt')

    def restore_network(self):
        try:
            save_time = os.path.getmtime(self.network_path + 'model.ckpt.data-00000-of-00001')
            if save_time > self.save_time:
                if self.saver_lock is not None:
                    # print('[Read] Lock Saver')
                    self.saver_lock.acquire()
                self.saver.restore(self.sess, self.network_path + 'model.ckpt')
                if self.saver_lock is not None:
                    # print('[Read] UnLock Saver')
                    self.saver_lock.release()
                print('[%s] Model Restored to the latest one.'%self.owner_name)
                self.save_time = save_time
            else:
                print('[%s] No new model.'%self.owner_name)
            self.is_restored = True
        except FileNotFoundError:
            print('No Model file to restore.')

    def build_model(self, train_mode):
        with self.graph.as_default():
            with tf.device(self.device):
                self.board_state = tf.placeholder(tf.float32, shape=[None, self.board_size, self.board_size, 2])
                tf.add_to_collection('board_state', self.board_state)
                filter_num_list = [32, 128, 256]
                hidden_num = [512] #[1024, 512]
                stone_num = tf.reduce_sum(self.board_state, [1,2])
                black_num, white_num = tf.split(stone_num, [1,1], 1)
                turn_info = black_num - white_num
                conv1 = tf.layers.conv2d(inputs=self.board_state, filters=filter_num_list[0], kernel_size=[3, 3],
                                         padding='VALID', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         bias_initializer=tf.constant_initializer(0.01))
                conv2 = tf.layers.conv2d(inputs=conv1, filters=filter_num_list[1], kernel_size=[3, 3], padding='VALID',
                                         kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                         bias_initializer= tf.constant_initializer(0.01))
                conv3 = tf.layers.conv2d(inputs=conv2, filters=filter_num_list[2], kernel_size=[3, 3], padding='VALID',
                                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         bias_initializer= tf.constant_initializer(0.01))
                flat_size = (self.board_size - 6) ** 2 * filter_num_list[-1]
                flat = tf.concat([tf.reshape(conv3, [-1, flat_size]), turn_info], axis=1)  # maybe 3*3*256=2304
                # flat_size = (self.board_size) ** 2 * 2
                # flat = tf.reshape(self.board_state, [-1, flat_size])  # maybe 3*3*256=2304
                hidden1 = tf.layers.dense(inputs=flat, units=hidden_num[0], activation=tf.nn.relu,
                                         kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                         bias_initializer=tf.constant_initializer(0.01))
                # hidden2 = tf.layers.dense(inputs=hidden1, units=hidden_num[1], activation=tf.nn.relu,
                #                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                #                           bias_initializer=tf.constant_initializer(0.01))
                self.predicted_value = tf.layers.dense(inputs=hidden1, units=1, activation=tf.nn.tanh,
                                                       kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                                       bias_initializer = tf.constant_initializer(0.01))
                prob_hidden = tf.layers.dense(inputs=flat, units=hidden_num[0], activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          bias_initializer=tf.constant_initializer(0.01))
                self.predicted_action_out = tf.layers.dense(inputs=prob_hidden, units=self.board_size * self.board_size,
                                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                        bias_initializer=tf.constant_initializer(0.01))
                self.predicted_action_prob = tf.nn.softmax(self.predicted_action_out)
                tf.add_to_collection('predicted_value', self.predicted_value)
                tf.add_to_collection('predicted_action_out', self.predicted_action_out)
                tf.add_to_collection('predicted_action_prob', self.predicted_action_prob)
                tf.train.export_meta_graph(self.network_path + 'model.ckpt.meta')
                self.saver = tf.train.Saver()
            if train_mode is True:
                self.create_train_op()
            self.sess.run(tf.global_variables_initializer())

    def create_train_op(self):
        with self.graph.as_default():
            with tf.device(self.device):
                self.learning_rate = tf.placeholder(tf.float32)
                self.win_rate = tf.placeholder(tf.float32)
                self.target_value = tf.placeholder(tf.float32)
                self.target_action_prob = tf.placeholder(tf.float32, shape=[None, self.board_size * self.board_size])
                self.value_loss = tf.reduce_mean(((self.target_value - self.predicted_value) ** 2) / 2)
                self.action_prob_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.target_action_prob, logits= self.predicted_action_out))
                self.loss = self.value_loss + 0.1 * self.action_prob_loss
                self.prediction_error = tf.reduce_mean(tf.abs(self.target_value - self.predicted_value))
                # self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
                self.train_op = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
                self.corr = tf.reduce_sum(self.target_value * self.predicted_value)
                tf.summary.scalar('Win Rate', self.win_rate)
                tf.summary.scalar('Loss', self.loss)
                tf.summary.scalar('Value Loss', self.value_loss)
                tf.summary.scalar('Action Prob Loss', self.action_prob_loss)
                tf.summary.scalar('Error', self.prediction_error)
                tf.summary.scalar('Learning rate', self.learning_rate)
                pred_mean = tf.reduce_mean(self.predicted_value)
                tf.summary.scalar('Predict Mean', pred_mean)
                pred_std = tf.sqrt(tf.reduce_mean(tf.square(self.predicted_value - pred_mean)))
                tf.summary.scalar('Predict Std', pred_std)
                target_mean = tf.reduce_mean(self.target_value)
                tf.summary.scalar('Target Mean', target_mean)
                target_std = tf.sqrt(tf.reduce_mean(tf.square(self.target_value - target_mean)))
                tf.summary.scalar('Target Std', target_std)
                tf.summary.scalar('Correlation', self.corr)
                self.merged_summary = tf.summary.merge_all()

    def transform_board_state(self, src_board_state_list):
        out_board_state_list = []
        for src_board_state in src_board_state_list:
            # out_board_state = np.zeros([self.board_size, self.board_size, 2])
            out_board_state = np.full([self.board_size, self.board_size, 2], 0)
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if src_board_state[row][col] == 1:
                        out_board_state[row, col, 0] = 1.0
                    elif src_board_state[row][col] == 2:
                        out_board_state[row, col, 1] = 1.0
            out_board_state_list.append(out_board_state)
        return out_board_state_list

    def add_to_replay_buffer(self, board_state_list, win_rate_list, action_prob_list):
        # 게임판을 회전 및 반전
        for board_state, win_rate, action_prob in zip(board_state_list, win_rate_list, action_prob_list):
            self.replay_buffer.append((self.transform_board_state([board_state])[0], win_rate * 2 - 1, TransMatToArr(action_prob)))
            self.replay_buffer.append((self.transform_board_state([np.fliplr(board_state)])[0], win_rate * 2 - 1, TransMatToArr(np.fliplr(action_prob))))
            self.replay_buffer.append((self.transform_board_state([np.flipud(board_state)])[0], win_rate * 2 - 1, TransMatToArr(np.flipud(action_prob))))
            self.replay_buffer.append((self.transform_board_state([np.rot90(board_state).tolist()])[0], win_rate * 2 - 1, TransMatToArr( np.rot90(action_prob))))
            self.replay_buffer.append((self.transform_board_state([np.rot90(board_state, k=2).tolist()])[0],
                                      win_rate * 2 - 1, TransMatToArr(np.rot90(action_prob, k=2))))
            self.replay_buffer.append((self.transform_board_state([np.rot90(board_state, k=3).tolist()])[0],
                                      win_rate * 2 - 1, TransMatToArr(np.rot90(action_prob, k=3))))

    def predict(self, board_state_list):
        # board state를 입력하여 승율을 예측한다.
        # board state를 입력 가능한 형태로 변경
        input_state_list = self.transform_board_state(board_state_list)
        predicted_value = self.sess.run(self.predicted_value, feed_dict={self.board_state: input_state_list})
        # -1~+1을 0~1로 변경. 1이면 흑(X,  1)이 승리 0이면 백(O, 2)가 승리
        win_rate_list = []
        for value in predicted_value:
            win_rate_list.append((value[0] + 1) / 2)

        return win_rate_list

    def predict_action_prob(self, board_state_list):
        input_state_list = self.transform_board_state(board_state_list)
        predicted_action_prob = self.sess.run(self.predicted_action_prob, feed_dict={self.board_state: input_state_list})
        return predicted_action_prob

    def dump_training_data(self):
        # replay buffer를 파일로 저장한다.
        with gzip.GzipFile(self.network_path + 'train_data.gz', 'wb') as f:
            pickle.dump(self.replay_buffer, f)
        print('Replay buffer dumped.')

    def load_training_data(self):
        # replay buffer를 불러온다.
        if os.path.exists(self.network_path + 'train_data.gz'):
            with gzip.GzipFile(self.network_path + 'train_data.gz', 'rb') as f:
                self.replay_buffer = pickle.load(f)
                # print(self.replay_buffer)
            print('Replay buffers loaded: size = %d'%len(self.replay_buffer))
        else:
            print('No Dumped replay buffers')

    def fit(self):
        if self.summary_writer is None:
            if not os.path.exists(self.tb_path_name):
                os.makedirs(self.tb_path_name)
            self.summary_writer = tf.summary.FileWriter(self.tb_path_name)
            self.cur_lr = self.init_lr

        # 학습데이터 준비
        if len(self.replay_buffer) <= 1000:
            return self.iter_idx
        minibatch = random.sample(
            self.replay_buffer, min(len(self.replay_buffer), self.batch_size))

        board_state_list = []
        target_value_list = []
        target_prob_list = []
        for board_state, target_value, target_prob in minibatch:
            board_state_list.append(board_state)
            target_value_list.append(target_value)
            target_prob_list.append(target_prob)

        # if self.iter_idx%10==0:
        #     print('Win Rate in Minibatch=%.4f'%np.mean(target_value_list))

        before_loss = self.sess.run(self.loss,
                                    feed_dict={self.board_state:board_state_list,
                                               self.target_value:target_value_list,
                                               self.target_action_prob: target_prob_list})

        self.sess.run(self.train_op,
                      feed_dict={self.board_state: board_state_list, self.target_value: target_value_list,
                                 self.learning_rate:self.cur_lr, self.target_action_prob: target_prob_list})
        after_loss = self.sess.run(self.loss,
                                    feed_dict={self.board_state:board_state_list,
                                               self.target_value:target_value_list, self.target_action_prob: target_prob_list})

        # print('Loss : %.4f -> %.4f'%(before_loss, after_loss))
        # print(before_loss)
        # print(after_loss)
        if after_loss > before_loss:
            self.cur_lr *= 0.99
            # print('Current LR = %.4f'%self.cur_lr)

        # Update 후 일정 주기로 loss, error를 summary writer로 출력
        if self.iter_idx%self.eval_frequency == 0:
            summary = self.sess.run(self.merged_summary,
                                        feed_dict={self.board_state:board_state_list,
                                                   self.target_value:target_value_list,
                                                   self.learning_rate: self.cur_lr, self.target_action_prob: target_prob_list,
                                                   self.win_rate: self.cur_win_rate})
            self.summary_writer.add_summary(summary, self.iter_idx)

        # 더 드문 주기로 학습된 weight file 및 학습 데이터를 파일로 저장
        if self.iter_idx%self.save_frequency == 0:
            if self.saver_lock is not None:
                # print('[Write] Lock Saver')
                self.saver_lock.acquire()
            print('[%d] Writing model and replay buffers....'%self.iter_idx)
            self.saver.save(self.sess, self.network_path + 'model.ckpt', write_meta_graph=False)
            time.sleep(3)
            if self.saver_lock is not None:
                # print('[Write] UnLock Saver ')
                self.saver_lock.release()
            self.dump_training_data()
            print('Done.')
        self.iter_idx += 1
        return self.iter_idx

class NNMCTS:
    def __init__(self, game_state, my_id, nn_path, nn_device, time_out=60.0, proc_num=-1, saver_lock=None):
        self.game_state = game_state.Clone()
        self.my_id = my_id
        self.action_predictor = GomokuNetwork(nn_path, device=nn_device, saver_lock=saver_lock)
        self.root_node = Node(game_state=self.game_state, parent=None, action=None, action_predictor=self.action_predictor)
        self.time_out = time_out
        self.proc_num = 1
        self.nn_path = nn_path
        self.nn_device = nn_device
        if proc_num < 0:
            self.proc_num = multiprocessing.cpu_count() - 1
        else:
            self.proc_num = proc_num
        self.proc_list = []
        self.proc_state_list = []
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.saver_lock = saver_lock
        self.freeze_network = Value('i',1)
        self.StartProcess()

    def ResetGameState(self, game_state):
        self.game_state = game_state.Clone()
        self.root_node = Node(game_state=self.game_state, parent=None, action=None, action_predictor=self.action_predictor)
        self.ClearQueue(print_info=False)

    def StartProcess(self):
        for _ in range(self.proc_num):
            proc_state = Value('i', 1)
            proc = Process(target=ProcPredictValue, args=(self.nn_path, self.nn_device, self.task_queue, self.result_queue, proc_state, self.saver_lock, self.freeze_network,))
            self.proc_list.append(proc)
            self.proc_state_list.append(proc_state)
            proc.start()
            while proc_state.value == 1:
                print('Waiting for initializing network...', end='\r')
                time.sleep(1)
                print('Waiting for initializing network......  ', end='\r')


    def TerminateProcess(self):
        for proc in self.proc_list:
            proc.terminate()
        self.proc_list = []

    def UpdateCurrentState(self, new_board_state, print_info=True):
        # 나와 상대의 Action에 따라 바뀐 현재 Node를 갱신한다.
        prev_state = self.game_state.board_state
        x_action, o_action = GetAction(prev_state, new_board_state)
        if self.my_id == 1:
            my_action = x_action
            opp_action = o_action
        else:
            my_action = o_action
            opp_action = x_action
        my_child = self.root_node.SearchChild(my_action)

        if my_child is not None:
            self.root_node = my_child.SearchChild(opp_action)
            self.game_state = GameState(new_board_state, just_decided=3 - self.my_id)
            if self.root_node is None:
                self.root_node = Node(self.game_state, None, opp_action, action_predictor= self.action_predictor)
                if print_info is True:
                    print('Not visited tree. Create new tree.')
            else:
                self.root_node.parent = None
                if print_info is True:
                    print('Reuse tree. visit_num = %d (mean value = %.4f)' % (
                        self.root_node.visit, self.root_node.value / self.root_node.visit))
        else:
            self.game_state = GameState(new_board_state, just_decided=3 - self.my_id)
            self.root_node = Node(self.game_state, None, opp_action, action_predictor= self.action_predictor)
            if print_info is True:
                print('Not visited tree. Create new tree.')

    def ClearQueue(self, print_info = True):
        # 혹시 이전 작업의 잔재가 남아 있으면 제거한다.
        if print_info is True:
            print('Clearing Task Queue(%d)...' % self.task_queue.qsize(), end=' ')
        while self.task_queue.empty() is False:
            self.task_queue.get()
        # Process 들이 대기중인지 확인
        if print_info is True:
            print('Cleared')
            print('Check ValuePredict Processes...', end=' ')
        all_ready = False
        while all_ready is False:
            all_ready = True
            for proc_state in self.proc_state_list:
                if proc_state.value == 1:
                    all_ready = False
                    break
        if print_info is True:
            print('Ready.')
            print('Clearing Result Queue(%d)...' % self.result_queue.qsize(), end=' ')
        while self.result_queue.empty() is False:
            self.result_queue.get()
        if print_info is True:
            print('Cleared')

    def SearchBestActionMultiProc(self, print_info = True, select_randomness= 0):

        # action = random.choice(self.root_node.untried_action_list)
        # # print('Action: %d'%action)
        # return action, 0

        start_time = time.time()

        expand_time = 0
        self.ClearQueue(print_info=print_info)

        while True:
            state = self.game_state.Clone()
            cur_time = time.time()
            # 시간이 다 되었거나 최적의 수가 확실해 졌다면, 최적 탐색 결과를 리턴
            cur_depth = 0
            action_list = []
            if (cur_time - start_time >= self.time_out - 1.0):
                # Child 중 Best를 찾는다.
                if print_info is True:
                    print('\nTimeUp. return best result. Visit Num = %d' % self.root_node.visit)
                best_child, action_score_list = self.root_node.SearchBestChild(randomness=select_randomness)
                if print_info is True:
                    print_action_score(state.board_state, action_score_list)
                return best_child.action, best_child.value / best_child.visit, action_score_list
            # Selection: Expand할 노드가 없을 때까지 내려가며 Score와 Visit 비율에 따라 Selection을 한다.(Exploitation + Exploration)
            cur_node = self.root_node

            while cur_node.untried_action_list == [] and cur_node.child_list != []:
                cur_node = cur_node.Select()
                if cur_node is None:  # visit not updated yet
                    break
                state.DoAction(cur_node.action)
                action_list.append(cur_node.action)
                cur_depth += 1

            # print('\r[%.2f,%d] Selection depth = %d, Task Queue=%d, Result Queue=%d' %
            #       (cur_time - start_time, self.root_node.visit, cur_depth, self.task_queue.qsize(),
            #        self.result_queue.qsize()), end='')
            if cur_node is not None:
                # Expand: Expand할 Child가 있고 Terminal이 아니면, Prior probability에 따라 Expand를 한다.
                expand_start = time.time()
                if cur_node.untried_action_list != []:
                    if cur_node.action_prob is None:
                        action_prob = GetRolloutProbability(state, cur_node.untried_action_list)
                    else:
                        action_prob = []
                        for action in cur_node.untried_action_list:
                            action_prob.append(cur_node.action_prob[action])
                        action_prob = (np.array(action_prob) / np.sum(action_prob)).tolist()
                        # print(action_prob)
                    action = np.random.choice(cur_node.untried_action_list, 1, replace=False, p=action_prob)[0]
                    # action = random.choice(cur_node.untried_action_list)   # Prior를 이용할 수 있으면 np.random.choice를 이용한다.
                    cur_node.AddChild(state, action)
                    action_list.append(action)
                expand_time += time.time() - expand_start

                # Process에 Value evaluation을 의뢰한다.
                self.task_queue.put((state, action_list))

            # Simulation 결과를 받아와 해당 노드들을 업데이트 한다.
            while self.result_queue.empty() is False:
                result_action_list, result = self.result_queue.get()
                # node를 찾는다.
                cur_node = self.root_node
                for action in result_action_list:
                    found_child = cur_node.SearchChild(action)
                    if found_child is None:
                        break
                    else:
                        cur_node = found_child

                # Backpropagation: parent node를 거슬러 올라가며 win, visit update
                while cur_node.parent is not None:
                    cur_node.visit += 1.0
                    if cur_node.just_decided == 1:
                        cur_node.value += result
                    else:
                        cur_node.value += 1-result
                    cur_node = cur_node.parent

                cur_node.visit += 1.0

def GetAction(prev_state, cur_state):
    state_diff = np.array(cur_state) - np.array(prev_state)
    x_action_index = np.nonzero(state_diff == 1)
    x_action = x_action_index[0][0] * len(cur_state) + x_action_index[1][0]
    o_action_index = np.nonzero(state_diff == 2)
    o_action = o_action_index[0][0] * len(cur_state) + o_action_index[1][0]
    return x_action, o_action

def TrainNetwork(network_path, device, replay_buffer, saver_lock, cur_win_rate):
    train_network = GomokuNetwork(network_path=network_path, replay_buffer_size=10000, device=device, saver_lock=saver_lock,
                                  train_mode=True, lr=0.01, owner_name='Trainer')

    print('Replay size = %d'%len(train_network.replay_buffer))
    while True:
        train_network.cur_win_rate = cur_win_rate.value
        new_sample_num = 0
        while replay_buffer.empty() is False:
            board_state, win_rate, action_prob = replay_buffer.get()
            train_network.add_to_replay_buffer([board_state], [win_rate], [action_prob])
            new_sample_num += 1
        if new_sample_num > 0:
            print('%d New samples added. Replay buffer size =%d'%(new_sample_num, len(train_network.replay_buffer)))

        # Update network
        train_network.fit()

def GetActionProb(board_size, action_score_list):
    board = [[0 for _ in range(board_size)] for _ in range(board_size)]
    score_sum = 0
    for action, score in action_score_list:
        score_sum += score
    if score_sum > 0:
        for action, score in action_score_list:
            board[int(action / board_size)][action % board_size] = score / score_sum
    return board

def ProcSelfPlay(process_name, network_path, play_devices, time_out, board_size, replay_buffer, saver_lock, self_play_state):
    init_game_board = [[0 for _ in range(board_size)] for _ in range(board_size)]
    init_game_state = GameState(init_game_board)
    game_state = init_game_state.Clone()
    x_player_mcts = NNMCTS(game_state, 1, network_path, play_devices[0], time_out=time_out, proc_num=1,
                           saver_lock=saver_lock)
    o_player_mcts = NNMCTS(game_state, 2, network_path, play_devices[1], time_out=time_out, proc_num=1,
                           saver_lock=saver_lock)
    played_game = 0
    print('[%s]Start SelfPlay.'%process_name)
    result_list = deque(maxlen=100)
    while True:
        # Restore to the recent network
        x_player_mcts.freeze_network.value = 0
        # o_player_mcts.freeze_network.value = 0
        print('[%s]Try to restore the latest model.' % process_name)
        x_player_mcts.action_predictor.restore_network()
        board_state_list = []
        action_score_list = []
        game_state = init_game_state.Clone()
        x_player_mcts.ResetGameState(game_state)
        action, value, action_score = x_player_mcts.SearchBestActionMultiProc(print_info=False, select_randomness=1)
        game_state.DoAction(action)
        board_state_list.append(game_state.board_state)
        action_score_list.append(action_score)
        o_player_mcts.ResetGameState(game_state)
        action, value, action_score = o_player_mcts.SearchBestActionMultiProc(print_info=False, select_randomness=1)
        game_state.DoAction(action)
        board_state_list.append(game_state.board_state)
        action_score_list.append(action_score)
        is_x_turn = True
        turn_num = 1
        while (game_state.CheckWin() is False and game_state.draw is False):
            if turn_num < 10:
                randomness = 1/turn_num
            else:
                randomness = 0
            if is_x_turn is True:
                x_player_mcts.UpdateCurrentState(game_state.board_state, print_info=False)
                action, value, action_score = x_player_mcts.SearchBestActionMultiProc(print_info=False, select_randomness=randomness)
                game_state.DoAction(action)
                is_x_turn = False
            else:
                o_player_mcts.UpdateCurrentState(game_state.board_state, print_info= False)
                action, value, action_score = o_player_mcts.SearchBestActionMultiProc(print_info=False, select_randomness=randomness)
                game_state.DoAction(action)
                is_x_turn = True
                turn_num += 1
            # print('[%d] Max Score = %.4f'%(turn_num, max([score for _, score in action_score])))
            board_state_list.append(game_state.board_state)
            action_score_list.append(action_score)
            self_play_state.value = turn_num

        for board_state, action_score in  zip(board_state_list, action_score_list):
            if game_state.draw is True:
                target_value = 0.5
            else:
                if game_state.just_decided == 1:
                    target_value = 1.0
                else:
                    target_value = 0.0
            # Make action score to prob
            action_prob = GetActionProb(board_size, action_score)
            replay_buffer.put((board_state, target_value, action_prob))
        # if game_state.draw is True:
        #     target_value = 0.5
        # else:
        #     if game_state.just_decided == 1:
        #         target_value = 1.0
        #     else:
        #         target_value = 0.0
        # replay_buffer.put((board_state_list[-1], target_value))
        played_game += 1
        result_list.append(target_value)
        # print(board_state_list[-1])
        # print(target_value)
        # 승률에 따라 상대의 Value network를 갱신해준다.
        print('[%s]Selfplay Num = %d, win Rate= %.4f, replay size = %d'%(process_name, played_game, np.mean(result_list), len(board_state_list)))

def convertPath(path):
    separator = os.path.sep
    if separator != '/':
        path = path.replace('/',os.path.sep)
    return path

def ProcEvaluate(time_out, eval_game, saver_lock, eval_proc_state, cur_win_rate_share):
    env = gym.make('Gomoku9x9-v0')  # default 'beginner' level opponent policy

    eval_result_list = deque(maxlen=eval_game)
    board_size = 9
    nn_path = working_directory + 'network_weight/'
    best_nn_path =  working_directory + 'best_network_weight/'
    init_game_board = [[0 for _ in range(board_size)] for _ in range(board_size)]
    init_game_state = GameState(init_game_board)
    decision_maker = NNMCTS(init_game_state, 1, nn_path, '/cpu:0', time_out=time_out, proc_num=1,
                            saver_lock=saver_lock)
    best_win_rate = 0
    eval_num = 0
    print('Start Evaluation Process.')
    while True:
        env.reset()
        done = False
        game_state = init_game_state.Clone()
        decision_maker.ResetGameState(game_state)
        turn_num = 0
        print('[Evaluator] Try to restore the latest model.')
        decision_maker.action_predictor.restore_network()
        decision_maker.freeze_network.value = 0
        while done is False:
            action, win_rate,_ = decision_maker.SearchBestActionMultiProc(print_info=False)
            observation, reward, done, info = env.step(action)
            if done is False:
                decision_maker.UpdateCurrentState(observation, print_info=False)
            turn_num += 1
            eval_proc_state.value = turn_num
        if reward > 0:
            eval_result_list.append(1.0)
        elif reward < 0:
            eval_result_list.append(0.0)

        eval_num += 1
        cur_win_rate = np.mean(eval_result_list)
        cur_win_rate_share.value = cur_win_rate
        print('Duel Ends in %d turn. Evaluation Num = %d, recent win rate = %.4f'%(turn_num, eval_num, cur_win_rate))

        if len(eval_result_list) >= eval_game and cur_win_rate > best_win_rate:
            print('Best network updated: Win rate %.4f -> %.4f'%(best_win_rate, cur_win_rate))
            best_win_rate = cur_win_rate
            # Copy cur network to best one
            if not os.path.exists(best_nn_path):
                os.makedirs(best_nn_path)
            saver_lock.acquire()
            command = 'copy %s* %s'%(convertPath(nn_path), convertPath(best_nn_path))
            print(command)
            os.system(command)
            saver_lock.release()

    decision_maker.TerminateProcess()

def SelfPlayTrain():
    network_path = working_directory + 'network_weight/'
    train_device = '/cpu:0'
    play_devices = ['/cpu:0','/cpu:0']
    board_size = 9
    train_time_out = 5
    eval_time_out = 10
    eval_game_num = 50
    saver_lock = Lock()
    replay_buffer = Queue()

    cur_win_rate = Value('d', 0)
    # # Start train process, 1 train process
    train_process = Process(target=TrainNetwork, args=(network_path, train_device, replay_buffer, saver_lock, cur_win_rate,))
    train_process.start()
    #
    # # Start Self-play process, at least 4 processes will run
    selfplay_state = [Value('i',0), Value('i',0)]
    selfplay_process1 = Process(target=ProcSelfPlay, args=('Proc1', network_path, play_devices, train_time_out, board_size, replay_buffer, saver_lock, selfplay_state[0],))
    selfplay_process1.start()
    # selfplay_process2 = Process(target=ProcSelfPlay, args=(
    # 'Proc2', network_path, play_devices, train_time_out, board_size, replay_buffer, saver_lock, selfplay_state[1],))
    # selfplay_process2.start()

    # Start Evaluation process, 2 processes will run
    eval_proc_state = Value('i', 0)
    # eval_process = Process(target=ProcEvaluate, args=(eval_time_out, eval_game_num, saver_lock, eval_proc_state, cur_win_rate,))
    # eval_process.run()
    # eval_process.start()
    ProcEvaluate(eval_time_out, eval_game_num, saver_lock, eval_proc_state, cur_win_rate)

    while True:
        # print('Self_Play Turn Num = %d, Eval Process Turn = %d'%(selfplay_state.value, eval_proc_state.value))
        time.sleep(3)

def play_mcts_gomoku():
    env = gym.make('Gomoku9x9-v0')  # default 'beginner' level opponent policy
    env.reset()
    done = False
    board_size = 9
    init_game_board = [[0 for _ in range(board_size)] for _ in range(board_size)]
    game_state = GameState(init_game_board)
    # print('%d CPU available'%multiprocessing.cpu_count())
    nn_path = working_directory + 'network_weight/'
    decision_maker = NNMCTS(game_state, 1, nn_path, '/cpu:0', time_out=60, proc_num=1)
    print('Start')
    while done is False:
        # action, win_rate = decision_maker.SearchBestAction()
        action, win_rate,_ = decision_maker.SearchBestActionMultiProc()
        print('Decision=%d(%.4f)' % (action, win_rate))
        observation, reward, done, info = env.step(action)
        env.render('human')
        if done is False:
            decision_maker.UpdateCurrentState(observation)

    if reward > 0:
        print('Win!!!')
    elif reward < 0:
        print('Lost.')
    else:
        print('Draw')

    decision_maker.TerminateProcess()

    return reward

if __name__ == '__main__':
    SelfPlayTrain()
    # play_mcts_gomoku()