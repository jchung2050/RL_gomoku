## Refer to http://mcts.ai/

import gym
import gym_gomoku
import time
import numpy as np
import copy
import multiprocessing
from multiprocessing import Process, Queue, Value

def TestStateNodeReceive(in_queue:Queue):
    game_state, node = in_queue.get()
    print('Data Received.')
    print(game_state.board_state)
    print(node.untried_action_list)

class GameState:
    def __init__(self, board_state, just_decided=2):
        self.board_state = board_state
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
        self.board_state[int(action/self.size)][action%self.size] = self.just_decided

        # Check Win/Lost
        self.is_win = self.CheckWin()
        return self.is_win

class Node:
    def __init__(self, game_state, parent, action):
        self.parent = parent
        self.action = action
        self.child_list = []
        self.untried_action_list = game_state.GetAvailableAction()
        self.win = 0.0
        self.visit = 0.0
        self.just_decided = game_state.just_decided

    def GetScore(self):
        if self.visit > 0:
            # return self.visit
            return self.win / self.visit
        else:
            return 0.0

    def AddChild(self, state, action):
        state.DoAction(action)
        child_node = Node(state, self, action)
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
                score_list[idx] = child.win / child.visit + np.sqrt(2 * np.log(self.visit) / child.visit)

        return self.child_list[np.argmax(score_list)]

    def SearchBestChild(self):
        best_child = None
        best_score = 0.0
        action_score_list = []
        for cur_child in self.child_list:
            cur_score = cur_child.GetScore()
            cur_action = cur_child.action
            action_score_list.append((cur_action, cur_score))
            if cur_score > best_score:
                best_score = cur_score
                best_child = cur_child
        return best_child, action_score_list

def print_action_score(board_state, action_score_list):
    board_size = len(board_state)
    board_state = (np.array(board_state) * (-1)).tolist()
    for action, score in action_score_list:
        board_state[int(action/board_size)][action%board_size] = int(score * 100)
    print(np.array(board_state))

def GetRolloutProbability(state, candidate_action_list=None):
    if candidate_action_list is None:
        candidate_action_list = state.GetAvailableAction(state)

    return np.array([1]*len(candidate_action_list))/len(candidate_action_list)

def ProcRandomRollOut(input_queue, output_queue, proc_state):
    while True:
        proc_state.value = 0 # Waiting
        state, action_list = input_queue.get()
        proc_state.value = 1 # Processing
        result = 0
        while state.draw is False:
            candidate_action_list = state.GetAvailableAction()
            action_prob = GetRolloutProbability(state, candidate_action_list)
            action = np.random.choice(candidate_action_list, 1, replace=False, p=action_prob)[0]
            win = state.DoAction(action)
            if win is True:
                result = state.just_decided
                break
        output_queue.put((action_list, result))

class MCTS:
    def __init__(self, game_state, my_id, time_out=60.0, proc_num = -1):
        self.game_state = game_state
        self.my_id = my_id
        self.root_node = Node(game_state=self.game_state, parent=None, action=None)
        self.time_out = time_out
        if proc_num < 0:
            self.proc_num = multiprocessing.cpu_count() -1
        else:
            self.proc_num = proc_num
        self.proc_list = []
        self.proc_state_list = []
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.StartProcess()

    def StartProcess(self):
        for _ in range(self.proc_num):
            proc_state = Value('i', 0)
            proc = Process(target= ProcRandomRollOut, args=(self.task_queue, self.result_queue, proc_state,))
            self.proc_list.append(proc)
            self.proc_state_list.append(proc_state)
            proc.start()

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
                self.root_node = Node(self.game_state, None, opp_action)
                if print_info is True:
                    print('Not visited tree. Create new tree.')
            else:
                self.root_node.parent = None
                if print_info is True:
                    print('Reuse tree. visit_num = %d (mean value = %.4f)' % (
                        self.root_node.visit, self.root_node.win / self.root_node.visit))
        else:
            self.game_state = GameState(new_board_state, just_decided=3 - self.my_id)
            self.root_node = Node(self.game_state, None, opp_action)
            if print_info is True:
                print('Not visited tree. Create new tree.')

    def SearchBestActionMultiProc(self):
        start_time = time.time()
        cur_node = self.root_node

        selection_time = 0
        expand_time = 0
        simulation_time = 0
        rollback_time = 0
        # 혹시 이전 작업의 잔재가 남아 있으면 제거한다.
        print('Clearing Task Queue(%d)...'%self.task_queue.qsize(), end=' ')
        while self.task_queue.empty() is False:
            self.task_queue.get()
        # Process 들이 대기중인지 확인
        print('Cleared')
        print('Check Rollout Processes...', end=' ')
        all_ready = False
        while all_ready is False:
            all_ready = True
            for proc_state in self.proc_state_list:
                if proc_state.value == 1:
                    all_ready = False
                    break
        print('Ready.')
        print('Clearing Result Queue(%d)...'%self.result_queue.qsize(), end=' ')
        while self.result_queue.empty() is False:
            self.result_queue.get()
        print('Cleared')

        while True:
            state = self.game_state.Clone()
            cur_time = time.time()
            # 시간이 다 되었거나 최적의 수가 확실해 졌다면, 최적 탐색 결과를 리턴
            cur_depth = 0
            action_list = []
            if (cur_time - start_time >= self.time_out - 1.0):
                # Child 중 Best를 찾는다.
                print('\nTimeUp. return best result. Simulation Num = %d' % self.root_node.visit)
                best_child, action_score_list = self.root_node.SearchBestChild()
                print_action_score(state.board_state, action_score_list)
                return best_child.action, best_child.win / best_child.visit
            # Selection: Expand할 노드가 없을 때까지 내려가며 Score와 Visit 비율에 따라 Selection을 한다.(Exploitation + Exploration)
            cur_node = self.root_node

            while cur_node.untried_action_list == [] and cur_node.child_list != []:
                cur_node = cur_node.Select()
                if cur_node is None:  # visit not updated yet
                    break
                state.DoAction(cur_node.action)
                action_list.append(cur_node.action)
                cur_depth += 1

            print('\r[%.2f,%d] Selection depth = %d, Task Queue=%d, Result Queue=%d' %
                  (cur_time - start_time, self.root_node.visit, cur_depth, self.task_queue.qsize(), self.result_queue.qsize()), end='')
            if cur_node is not None:
                # Expand: Expand할 Child가 있고 Terminal이 아니면, Prior probability에 따라 Expand를 한다.
                expand_start = time.time()
                if cur_node.untried_action_list != []:
                    action_prob = GetRolloutProbability(state, cur_node.untried_action_list)
                    action = np.random.choice(cur_node.untried_action_list, 1, replace=False, p=action_prob)[0]
                    # action = random.choice(cur_node.untried_action_list)   # Prior를 이용할 수 있으면 np.random.choice를 이용한다.
                    cur_node.AddChild(state, action)
                    action_list.append(action)
                expand_time += time.time() - expand_start

                # Process에 시뮬레이션을 의뢰한다.
                self.task_queue.put((state, action_list))

            # Simulation 결과를 받아와 해당 노드들을 업데이트 한다.
            while self.result_queue.empty() is False:
                result_action_list, result = self.result_queue.get()
                # node를 찾는다.
                cur_node = self.root_node
                for action in result_action_list:
                    cur_node = cur_node.SearchChild(action)

                # Backpropagation: parent node를 거슬러 올라가며 win, visit update
                while cur_node.parent is not None:
                    cur_node.visit += 1.0
                    if result == cur_node.just_decided:
                        cur_node.win += 1.0
                    cur_node = cur_node.parent

                cur_node.visit += 1.0

    def SearchBestAction(self):
        start_time = time.time()
        cur_node = self.root_node

        selection_time = 0
        expand_time = 0
        simulation_time = 0
        rollback_time = 0
        while True:
            state = self.game_state.Clone()
            cur_time = time.time()
            # 시간이 다 되었거나 최적의 수가 확실해 졌다면, 최적 탐색 결과를 리턴
            cur_depth = 0
            if (cur_time-start_time >= self.time_out- 1.0):
                # Child 중 Best를 찾는다.
                print('\nTimeUp. return best result. Simulation Num = %d'%self.root_node.visit)
                print('Selection=%.4f, Expand=%.4f, Simulation=%.4f, Rollback=%.4f'%(selection_time, expand_time, simulation_time, rollback_time))
                best_child, action_score_list = self.root_node.SearchBestChild()
                print_action_score(state.board_state, action_score_list)
                return best_child.action, best_child.win / best_child.visit
            # Selection: Expand할 노드가 없을 때까지 내려가며 Score와 Visit 비율에 따라 Selection을 한다.(Exploitation + Exploration)
            select_start = time.time()
            while cur_node.untried_action_list == [] and cur_node.child_list != []:
                cur_node = cur_node.Select()
                state.DoAction(cur_node.action)
                cur_depth += 1
            selection_time += time.time() - select_start
            print('\r[%.2f,%d] Selection depth = %d'%(cur_time-start_time, self.root_node.visit, cur_depth), end='')

            # Expand: Expand할 Child가 있고 Terminal이 아니면, Prior probability에 따라 Expand를 한다.
            expand_start = time.time()
            if cur_node.untried_action_list != []:
                action_prob = GetRolloutProbability(state, cur_node.untried_action_list)
                action = np.random.choice(cur_node.untried_action_list, 1, replace=False, p=action_prob)[0]
                # action = random.choice(cur_node.untried_action_list)   # Prior를 이용할 수 있으면 np.random.choice를 이용한다.
                cur_node = cur_node.AddChild(state, action)
            expand_time += time.time() - expand_start

            # Simulation: Node Expand 없이 번갈아 Decision하면서 결과를 얻는다.(Fast Rollout)
            rollout_num = 0
            result = 0
            simulation_start = time.time()
            while state.draw is False:
                candidate_action_list = state.GetAvailableAction()
                action_prob = GetRolloutProbability(state, candidate_action_list)
                action = np.random.choice(candidate_action_list, 1, replace=False, p=action_prob)[0]
                win = state.DoAction(action)
                rollout_num += 1
                if win is True:
                    result = state.just_decided
                    break
            simulation_time += time.time() - simulation_start

            # print('RolloutNum = %d'%rollout_num)
            # print(np.array(state.board_state))
            # print(state.is_win)
            # Backpropagation: parent node를 거슬러 올라가며 win, visit update
            rollback_start = time.time()
            while cur_node.parent is not None:
                cur_node.visit += 1.0
                if result == cur_node.just_decided:
                    cur_node.win += 1.0
                cur_node = cur_node.parent
            rollback_time += time.time() - rollback_start
            cur_node.visit += 1.0

def GetAction(prev_state, cur_state):
    state_diff = np.array(cur_state) - np.array(prev_state)
    x_action_index = np.nonzero(state_diff==1)
    x_action = x_action_index[0][0] * len(cur_state) + x_action_index[1][0]
    o_action_index = np.nonzero(state_diff==2)
    o_action = o_action_index[0][0] * len(cur_state) + o_action_index[1][0]
    return x_action, o_action

def play_mcts_gomoku():
    env = gym.make('Gomoku9x9-v0')  # default 'beginner' level opponent policy
    env.reset()
    done = False
    board_size = 9
    init_game_board = [[0 for _ in range(board_size)] for _ in range(board_size)]
    game_state = GameState(init_game_board)
    # print('%d CPU available'%multiprocessing.cpu_count())
    decision_maker = MCTS(game_state, 1, time_out=60, proc_num = -1)
    print('Start')
    while done is False:
        # action, win_rate = decision_maker.SearchBestAction()
        action, win_rate = decision_maker.SearchBestActionMultiProc()
        print('Decision=%d(%.4f)'%(action, win_rate))
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

def test_gomoku():
    env = gym.make('Gomoku9x9-v0')  # default 'beginner' level opponent policy
    env.reset()

    for _ in range(100):
        action = env.action_space.sample()  # sample without replacement
        print(action)
        observation, reward, done, info = env.step(action)
        print(observation)
        print(reward)
        print(info)
        # env.render('human')
        if done:
            print("Game is Over")
            break

if __name__ == '__main__':
    # test_gomoku()
    play_mcts_gomoku()