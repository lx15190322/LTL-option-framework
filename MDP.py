import numpy as np
import matplotlib.pyplot as plt
from pandas import *
from copy import deepcopy as dcp

# plotly dependencies
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from plotly import tools
plotly.tools.set_credentials_file(username='SeanLau', api_key='d1fuOtEPoJU7V6GhntKN')

'''
Comment tags:
    Init:       used for initializing the object parameters
    Tool:       inner function called by API, not transparent to the user
    API:        can be called by the user
    VIS:        visualization
    DISCARD:    on developing or discarded functions in old version
'''
# Tool: geometric series function, called by self.transition_matrix()
def GeoSeries(P, it):
    sum = dcp(P)  # + np.identity(len(P))
    for k in range(2, it):
        sum += np.linalg.matrix_power(dcp(P), k)

    return sum

# API: calculating transition matrix for an option
def transition_matrix_(S, goal, unsafe, interruptions, gamma, P, Pi, row, col): # write the correct transition probability according to the policy

    # row, col = 6, 8
    size  = row * col + 1 #l * w + sink
    PP = np.zeros((size, size))
    PP[0, 0] = 1.0
    for state in S:
        s = tuple(state)
        n = (s[0] - 1) * col + s[1] #(1,1) = 1, (2,1)=9

        if s not in goal and s not in unsafe and s not in interruptions:
            PP[n, 0] = 1 - gamma
            for a in Pi[s]:
                for nb in P[s, a]:
                    n_nb = (nb[0] - 1) * col + nb[1]
                    PP[n, n_nb] += P[s, a][nb] * Pi[s][a] * gamma
        else:
            PP[n, n] = 1.0

    sums = GeoSeries(
        dcp(PP),
        50)

    final = {}

    result = sums
    for state in S:
        s = tuple(state)
        n = (s[0]-1) * col + s[1]

        final[s] = {}
        line = []
        for state_ in S:
            s_ = tuple(state_)
            n_ = (s_[0]-1) * col + s_[1]
            line.append(result[n, n_])
        for g in goal:
            ng = (g[0]-1) * col + g[1]
            final[s][g] = result[n, ng]/sum(line)

    return final


class MDP:

    def __init__(self, gamma = 0.9, epsilon = 0.3, gain = 100.0):
        '''
        :param gamma: Discounting factor
        :param epsilon: P[s'=f(s, a)|s, a] = 1 - epsilon
        :param gain: Fixed reward value at goal states

        Data structure:
        self.L: labeling function (L: S -> Q),          e.g. self.L[(2, 3)] = 'g1'
        self.Exp: L^-1: inverse of L (L: Q -> S),       e.g. self.Exp['g1'] = (2, 3)
        self.goal, self.T: goal set list,               e.g. self.goal = [(1, 2), (3, 4)], in product MDP where
                                                        e.g. state is S*Q, self.goal = [((1, 2), 1)] = self.T
                                                        e.g. T only serves for product MDP
        <self.unsafe: globally unsafe states>           e.g. see self.goal
        <self.interruptions: temporal unsafe states>    e.g. see self.goal
        self.S: state set list,                         e.g. see self.goal
        <self.originS>                                  e.g. only serves NON product MDP
        self.P: P(s' | s, a) for stochastic MDP,        e.g. self.P[s, a][s']
        self.Pi: Pi(a | s) policy,                      e.g. self.Pi[s][a]
        self.R: R(s, a) Reward function,                e.g. self.R[s][a]
        self.Q: Q(s, a) Q value function,               e.g. self.Q[s][a]
        self.V: V(s) value function,                    e.g. self.V[s], (self.V_[s] stores V[s] in last iteration)
        self.Po: P(s' | s, o) option transition matrix  e.g. self.Po[s][s']
        self.Opt: MDP object, for composed option       e.g. self.Opt[id], id = (('g1', 'g2'), 'disjunction')
        self.AOpt: MDP object, for atomic option        e.g. self.AOpt[id], id = 'g1'
        self.dfa: DFA object
        self.mdp: non product MDP object

        '''

        self.gamma = gamma # discounting factor
        self.epsilon = epsilon # transition probability, set as constant here
        self.constReward = gain
        self.unsafe = {}

        self.ID = 'root' # MDP id to identify the layer of an MDP, set as 'root' if it's original MDP,

        self.S = [] # state space e.g. [[0,0],[0,1]...]
        self.originS = []
        self.wall_cord = [] # wall state space, e.g. if state space is [[1,1]], wall is [[1,2], [2,1], [1,0], [0,1]]
        self.L = {} # labeling function
        self.Exp = {} # another way to store labeling pairs
        self.goal = [] # goal state set
        self.interruptions = [] # unsafe state set
        self.gridSize = []

        self.A = {"N":(-1, 0), "S":(1, 0), "W":(0, -1), "E":(0, 1)} # actions for grid world

        self.V, self.V_ = {}, {} # value function and V_ denotes memory of the value function in last iteration
        self.init_V, self.init_V_ = {}, {} # store initial value function

        self.Q, self.Q_ = {}, {} # state action value function
        self.T = {} # terminal states
        self.Po = {} # transition matrix

        self.R = {} # reward function
        self.init_R = {} # initial reward function

        self.P = {} # 1 step transition probabilities

        self.Pi, self.Pi_ = {}, {} # policy and memory of last iteration policy
        self.Opt = {} # options library
        self.AOpt = {} # atomic options without production

        self.dfa = None
        self.mdp = None
        self.plotKey = False
        self.svi_plot_key = False

        self.action_special = None
        self.hybrid_special = None
        self.svi_record = 0


    # DISCARD: calculating transition matrix for an option
    def transition_matrix(self): # write the correct transition probability according to the policy
        # PP = {}
        # sink = "sink"
        # PP[sink] = {}
        # PP[sink][sink] = 1.0 #* self.gamma

        # TODO plain matrix work space
        row, col = 6, 8
        size  = row * col + 1 #l * w + sink
        PP = np.zeros((size, size))
        PP[0, 0] = 1.0
        for state in self.S:
            s = tuple(state)
            # translated
            n = (s[0]-1) * col + s[1] #(1,1) = 1, (2,1)=9
            #
            if s not in self.goal and s not in self.unsafe and s not in self.interruptions:
                PP[n, 0] = 1 - self.gamma
                for a in self.Pi[s]:
                    for nb in self.P[s, a]:
                        n_nb = (nb[0] - 1) * col + nb[1]
                        PP[n, n_nb] = self.P[s, a][nb] * self.Pi[s][a] * self.gamma
            else:
                PP[n, n] = 1.0
                # if s in self.goal:
                #     PP[n, n] = 1.0
        #
        '''
        for s in self.S:
            s = tuple(s)
            PP[s] = {}

            if s not in self.goal and s not in self.unsafe and s not in self.interruptions:
                PP[s][sink] = 1 - self.gamma
                for a in self.Pi[s]:
                    for nb in self.P[s, a]:
                        PP[s][nb] = self.P[s, a][nb] * self.Pi[s][a] * self.gamma
                # PP[s][sink] = 0
            else:
                # PP[s][sink] = 0.0
                # PP[s][s] = 1.0
                PP[s][s] = 1.0#self.gamma
        '''

        # reference = DataFrame(PP).T.fillna(0.0)
        # print PP
        sums = GeoSeries(dcp(PP), 50)
        print self.goal, sums

        #test matrix
        # print self.goal
        final = {}
        # line = {}
        # result = sums.transpose()
        result = sums
        for state in self.S:
            s = tuple(state)
            n = (s[0]-1) * col + s[1]

            final[s] = {}
            line = []
            for state_ in self.S:
                s_ = tuple(state_)
                n_ = (s_[0]-1) * col + s_[1]
                # final[s][s_] = result[s][s_]/sum(result[s])
                # if s_ != sink:
                line.append(result[n, n_])
            for g in self.goal:
                ng = (g[0]-1) * col + g[1]

                final[s][g] = result[n, ng]/sum(line)#min(result[s][g], 1.0)
                # print s,g,result[n, ng], sum(line)
            # print s, max(line), sum(line), line
        # print self.goal, final

        self.TransitionMatrix = dcp(final)

    # Init: set labeling function
    def set_L(self, label):
        self.L = label

    # Init: set
    def set_Exp(self, exp):
        self.Exp = exp

    # Tool: a small math tool for the state space cross product
    def crossproduct(self, a, b):
        return [(tuple(y), x) for x in a for y in b]

    # API: Generate composed options
    def option_factory(self):
        # print  self.Exp
        s_index = self.dfa.state_info
        for q in s_index:

            id = s_index[q]['safe']
            print id, len(id)

            # if id == [] or len(id) > 1:
            #     continue
            if id == []:
                continue

            ctype = 'disjunction'
            sample = self.AOpt[id[0]]

            ID = tuple(id), ctype
            self.Opt[ID] = MDP()

            self.Opt[ID].ID = ID
            self.Opt[ID].plotKey = True
            self.Opt[ID].S = self.originS
            self.Opt[ID].R = sample.R
            self.Opt[ID].P = sample.P

            sumG = []
            sumT = []
            sumUnsafe = []

            for ap in id:
                sumG += dcp(self.AOpt[ap].goal)
                sumT += dcp(self.AOpt[ap].T)
                sumUnsafe += dcp(self.AOpt[ap].unsafe)

            self.Opt[ID].goal = list(set(sumG))
            self.Opt[ID].T = list(set(sumT))
            self.Opt[ID].unsafe = list(set(sumUnsafe))
            self.Opt[ID].set_Size(self.gridSize[0], self.gridSize[1])

            self.Opt[ID].V = self.option_composition(ctype, id)
            self.Opt[ID].V_ = dcp(self.Opt[ID].V)

            interruptid = s_index[q]['unsafe']
            interruptions = []

            for ap in interruptid:
                interruptions += self.Exp[ap]

            self.Opt[ID].interruptions = list(set(interruptions))

            for s in self.Opt[ID].interruptions:
                self.Opt[ID].V[tuple(s)] = 0.0
            for s in self.Opt[ID].unsafe:
                self.Opt[ID].V[tuple(s)] = 0.0
            for s in self.Opt[ID].goal:
                self.Opt[ID].V[tuple(s)] = 100.0

            self.Opt[ID].SVI(100000000000)
            # print  "policy", ID, self.Opt[ID].Pi

            self.Opt[ID].TransitionMatrix = transition_matrix_(
                self.Opt[ID].S,
                self.Opt[ID].goal,
                self.Opt[ID].unsafe,
                self.Opt[ID].interruptions,
                self.Opt[ID].gamma,
                self.Opt[ID].P,
                self.Opt[ID].Pi,
                self.gridSize[0],
                self.gridSize[1]
            )

    # Tool: Composing value functions of two options for conjunction or disjunction
    def option_composition(self, ctype, Opt_list):
        result = {}

        if ctype == 'disjunction':
            for state in self.originS:
                s = tuple(state)
                result[s] = sum([self.AOpt[name].V[s] for name in Opt_list])/ len(Opt_list)

        if ctype == 'conjunction':
            for state in self.originS:
                s = tuple(state)
                result[s] = sum([self.AOpt[name].V[s] for name in Opt_list]) / len(Opt_list)

        return dcp(result)

    # VIS: Visulizing policy direction and weight in grid world
    def draw_quiver(self, name):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # for s in self.Pi:
        for state in self.S:
            s = tuple(state)
            ma, mpi = '', 0
            if s not in self.goal and s not in self.unsafe and s not in self.interruptions:
                for a in self.Pi[s]:
                    if self.Pi[s][a] > mpi:
                        ma, mpi = a, self.Pi[s][a]
                # print np.array(s), np.array(self.A[ma])
                # s_ = tuple(np.array(s)+np.array(self.A[ma]))
                ax.quiver(s[0],s[1], self.A[ma][0], self.A[ma][1], angles='xy', scale_units='xy', scale=3/mpi)
        # plt.xticks(range(-5, 6))
        # plt.yticks(range(-5, 6))
        plt.grid()
        plt.draw()
        # plt.show()
        plt.savefig(name+'policy.png')
        return

    # API: Generate atomic options from decomposition algorithm
    def option_generation(self, dfa):

        goals = dcp(self.Exp)
        del goals["phi"]

        delList = []
        for exp in goals:
            if exp not in dfa.effTS:
                delList.append(exp)

        unsafe_map = {}
        for delExp in delList:
            unsafe_map[delExp] = dcp(goals[delExp])
            del goals[delExp]

        filtered_S = []
        # # print  'wall', self.wall_cord
        for s in self.S:  # new_S
            if s not in self.wall_cord:  # new_wall
                filtered_S.append(s)

        AOpt = {}
        # for qs in dfa.state_info.keys():
        opstacles = set([])
        for exp in goals:

            AOpt[exp] = MDP()
            AOpt[exp].ID = exp
            AOpt[exp].plotKey = True
            # AOpt[exp].unsafe = self.crossproduct(dfa.sink_states, filtered_S)

            AOpt[exp].set_S(filtered_S)
            AOpt[exp].goal = dcp(goals[exp])
            AOpt[exp].T = dcp(goals[exp])
            AOpt[exp].T += self.Exp[dfa.g_unsafe]
            AOpt[exp].unsafe = self.Exp[dfa.g_unsafe]
            AOpt[exp].set_Size(self.gridSize[0], self.gridSize[1])

            AOpt[exp].P = {}
            for key in self.P:
                pindex = tuple([key[0],key[1]])
                if pindex not in AOpt[exp].P:
                    AOpt[exp].P[pindex] = {}
                AOpt[exp].P[pindex][key[2]] = self.P[key]

            for s in filtered_S:
                AOpt[exp].V[tuple(s)], AOpt[exp].V_[tuple(s)] = 0.0, 0.0
            for goal in goals[exp]:
                AOpt[exp].V[goal], AOpt[exp].V_[goal] = 100.0, 0.0

            AOpt[exp].R = {}
            for state in filtered_S:
                s = tuple(state)
                for a in self.A:
                    if (s, a) in AOpt[exp].P:
                        AOpt[exp].R[s, a] = 0

            AOpt[exp].SVI(0.000001)
            # print 'atomic policy', exp, AOpt[exp].Pi
            # AOpt[exp].draw_quiver(exp)

        return dcp(AOpt)

    # DISCARD: option segmentation from product state space based on DFA transitions
    def segmentation(self):
        goals = dcp(self.mdp.Exp)
        TS_tree = dcp(self.dfa.transition_tree)
        del goals["phi"]

        delList = []
        for exp in goals:
            if exp not in self.dfa.effTS:
                delList.append(exp)

        unsafe_map = {}
        for delExp in delList:
            unsafe_map[delExp] = dcp(goals[delExp])
            del goals[delExp]

        # # print  delList
        # print  unsafe_map

        # print  'unsafe',self.unsafe
        # print  "!!!!!",self.T
        # print  self.P
        # print  self.dfa.sink_states
        # print  self.dfa.transition_tree

        # for element in self.P:
        #     if "sink" in self.P[element]:
        #         # print  element, "sink"

        q_unsafe = dcp(self.dfa.sink_states)
        # # print  self.mdp.Exp
        # print  'dfa_transitions: ', self.dfa.state_transitions
        # collect transitions into efficient transition, get rid of same state transition and sink transition
        # print  'effTs', self.dfa.effTS
        # print  'goals: ', goals
        for exp in goals:
            # TODO: need to get the transition!!!
            # # print  exp, goals[exp]
            # # print  exp
            for transitions in self.dfa.effTS[exp]:
                # print  "transition", transitions
                self.Opt[exp, tuple(transitions)] = MDP()
                self.Opt[exp, tuple(transitions)].ID = exp, tuple(transitions)
                self.Opt[exp, tuple(transitions)].plotKey = True
                self.Opt[exp, tuple(transitions)].unsafe = dcp(self.unsafe)
                # self.Opt[exp, tuple(transitions)].A = dcp(self.A)

                # TODO special transition need to deal with, if the transition share the same origin as current
                # TODO but different non-fail target, need to get rid of it!!
                interruptions = {}
                for target in TS_tree[transitions[0]]:
                    if target != transitions[1] and target not in q_unsafe:
                        for ts in self.dfa.invEffTS[transitions[0], target]:
                            if ts not in interruptions:
                                interruptions[ts] = target # {:}
                # print  "interruptions: ", interruptions

                # interruptions = {} # ignore the interruptions

                S = []
                g = []
                for state in self.S:
                    # # print  state
                    if state[0] not in goals[exp]:
                        if self.mdp.L[state[0]].v in unsafe_map:
                            # # print  transitions, state
                            if state[1] in q_unsafe and state not in S:
                                # print  "unsafe", state
                                S.append(state)
                        elif self.mdp.L[state[0]].v in interruptions:
                            if state[1] == interruptions[self.mdp.L[state[0]].v] and state not in S:
                                S.append(state)
                                self.Opt[exp, tuple(transitions)].interruptions.append(state)
                        else:
                            if state[1] == transitions[0] and state not in S:
                                S.append(state)
                    else:
                        if state[1] == transitions[1] and state not in S:
                            S.append(state)
                            if state not in g:
                                g.append(state)

                # print  "states", S
                # print  "goals", g


                self.Opt[exp, tuple(transitions)].set_S(S)

                self.Opt[exp, tuple(transitions)].goal = dcp(g)
                self.Opt[exp, tuple(transitions)].T = dcp(g)

                P = {}
                for prior in self.P.keys():
                    if prior[0] in S and prior[0] not in g:
                        if prior not in P:
                            P[prior] = dcp(self.P[prior])

                self.Opt[exp, tuple(transitions)].P = P

                # print  "!!!!!!!!!!",len(self.Opt[exp, tuple(transitions)].P.keys())
                # print  len(self.Opt[exp, tuple(transitions)].S)
                # print  "probability", P

                # value function initiation, if doing this step, no need to assign value to reward function
                for s in S:
                    self.Opt[exp, tuple(transitions)].V[s] = 0
                    self.Opt[exp, tuple(transitions)].V_[s] = 0
                for goal in g:
                    self.Opt[exp, tuple(transitions)].V[goal] = 100.0
                    self.Opt[exp, tuple(transitions)].V_[goal] = 0

                self.Opt[exp, tuple(transitions)].R = self.R

                self.Opt[exp, tuple(transitions)].SVI(0.000001)
                # print  "policy", self.Opt[exp, tuple(transitions)].Pi
                self.Opt[exp, tuple(transitions)].transition_matrix()
                # # print  "++++++++++"
                # for s in self.Opt[exp, tuple(transitions)].S:
                #     # print  self.Opt[exp, tuple(transitions)].TransitionMatrix[s][]

        return 0

    # API: DFA * MDP product
    def product(self, dfa, mdp):
        result = MDP()
        result.Exp = dcp(mdp.Exp)
        result.L = self.L

        new_A = mdp.A

        filtered_S = []
        for s in mdp.S: #new_S
            if s not in mdp.wall_cord: # new_wall
                filtered_S.append(s)

        result.originS = dcp(filtered_S)

        new_sink = self.crossproduct(dfa.final_states, filtered_S)
        new_unsafe = self.crossproduct(dfa.sink_states, filtered_S)

        new_P, new_R = {}, {}
        new_V, new_V_ = {}, {}

        true_new_s = []

        sink = "fail"

        for p in mdp.P.keys():

            for q in dfa.states:
                new_s = (p[0], q)
                new_a = p[1]
                if (new_s, new_a) not in new_P:
                    new_P[new_s, new_a] = {}

                if new_s not in new_sink: #(mdp.L[p[2]].display(), q) in dfa.state_transitions
                    q_ = dfa.state_transitions[mdp.L[p[0]], q]

                    new_s_ = (p[2], q_)
                    if q == q_:
                        new_P[new_s, new_a][new_s_] = mdp.P[p]
                    else:
                        new_s__ = (p[2], q)
                        new_P[new_s, new_a][new_s_] = mdp.P[p]
                        new_P[new_s, new_a][new_s__] = 0.0

                    if tuple(new_s_) not in true_new_s:
                        true_new_s.append(tuple(new_s_))

                else:
                    new_s_ = sink
                    new_P[new_s, new_a][new_s_] = 1

                if q in dfa.final_states and q not in dfa.sink_states:
                    new_R[new_s, new_a] = 0
                    new_V[new_s] = 100.0
                    new_V_[new_s] = 0
                elif q in dfa.sink_states:
                    new_R[new_s, new_a] = 0
                    new_V[new_s] = 0  # -1
                    new_V_[new_s] = 0
                elif q not in dfa.final_states:
                    new_R[new_s, new_a] = 0
                    new_V[new_s] = 0
                    new_V_[new_s] = 0

                if new_s not in true_new_s:
                    true_new_s.append(tuple(new_s))

        result.set_S(true_new_s)

        result.P = dcp(new_P)
        result.A = dcp(new_A)
        result.R = dcp(new_R)
        result.T = dcp(new_sink)
        result.V = dcp(new_V)
        result.V_ = dcp(new_V_)
        result.dfa = dcp(dfa)
        result.mdp = dcp(mdp)
        result.unsafe = dcp(new_unsafe)
        result.set_Size(self.gridSize[0], self.gridSize[1])

        result.init_V, result.init_V_ = dcp(new_V), dcp(new_V_)
        result.init_R = dcp(new_R)

        return result

    # Init: preparation for 1 step transition probability generation, not important
    def add_wall(self, inners):
        wall_cords = []
        for state in inners:
            for action in self.A:
                # # print  state, self.A[action]
                temp = list(np.array(state) + np.array(self.A[action]))
                if temp not in inners:
                    wall_cords.append(temp)
        return wall_cords

    # Init: preparation for 1 step transition probability generation, not important
    def set_WallCord(self, wall_cord):
        for element in wall_cord:
            self.wall_cord.append(element)
            if element not in self.S:
                self.S.append(element)

    # Init: init state space
    def set_S(self, in_s = None):
        if in_s == None:
            for i in range(13):
                for j in range(13):
                    self.S.append([i,j])
        else:
            self.S = dcp(in_s)

    # Init: set grid size
    def set_Size(self, row, col):
        self.gridSize = [row, col]

    # Init: init goal state set
    def set_goal(self, goal=None, in_g=None):
        if goal!=None and in_g == None:
            self.goal.append(goal)
        if goal == None and in_g!=None:
            self.goal = dcp(in_g)

    # Init: init state value function
    def init_V(self):
        for state in self.S:
            s = tuple(state)
            if s not in self.V:
                self.V[s], self.V_[s] = 0.0, 0.0
            if state in self.goal:
                self.V[s], self.V_[s] = 100.0, 0.0

    # Init: generating 1 step transition probabilities for 2d grid world
    def set_P(self):
        self.P = {}
        filtered_S = []
        for s in self.S:  # new_S
            if s not in self.wall_cord:  # new_wall
                filtered_S.append(s)
        self.S = filtered_S

        for state in self.S:
            s = tuple(state)
            explore = []

            for act in self.A.keys():
                temp = tuple(np.array(s) + np.array(self.A[act]))
                explore.append(temp)
            # print s, explore
            for a in self.A.keys():
                # selfP[s, a] = {}
                self.P[s, a, s] = 0.0

                s_ = tuple(np.array(s) + np.array(self.A[a]))
                unit = self.epsilon / 3

                if list(s_) in self.S:

                    self.P[s, a, s_] = 1 - self.epsilon
                    for _s_ in explore:
                        if tuple(_s_) != s_:
                            if list(_s_) in self.S:
                                self.P[s, a, tuple(_s_)] = unit
                            else:
                                self.P[s, a, s] += unit
                else:
                    self.P[s, a, s] = 1 - self.epsilon
                    for _s_ in explore:
                        if _s_ != s_:
                            if list(_s_) in self.S:
                                self.P[s, a, tuple(_s_)] = unit
                            else:
                                self.P[s, a, s] += unit
        return

    # Tool: turning dictionary structure to vector
    def Dict2Vec(self, V, S):
        # # print  S
        v = []
        # # print  "Vkeys",V.keys()
        for s in S:
            v.append(V[tuple(s)])
        return np.array(v)

    # Tool: sum operator in value iteration algorithm, called by self.SVI()
    def Sigma_(self, s, a):
        total = 0.0

        if self.ID != "root":
            if s in self.unsafe:
                return total
            elif s in self.interruptions:
                return total

        for s_ in self.P[s, a].keys():
            total += self.P[s, a][s_] * self.V_[s_]
        return total

    # Tool: sum operator in value iteration algorithm, called by self.SVI_option()
    def Sigma_opt(self, s, opt):

        # g = tuple(self.Opt[opt].goal[0])
        total = 0
        if s[0] in self.Opt[opt].goal:
            return total
        # print s, opt
        for g in self.Opt[opt].goal:
            vlist = []

            for ap in opt[0]:
                if g in self.Exp[ap]:
                    vlist.append(self.V_[g, s[1]])

            v_avg = max(vlist)
            Po = self.Opt[opt].TransitionMatrix[s[0]][g]

            temp = Po * v_avg
            result = temp
            total += result #self.V_[g]
        # if s == ((2, 8), 3):
        #     print total

        return total

    # API: option && action hybrid SVI runner
    def SVI_option(self, threshold):
        tau = 1.0

        self.V, self.V_ = dcp(self.init_V), dcp(self.init_V_)

        self.R = dcp(self.init_R)

        self.Pi, self.Pi_ = {}, {}

        self.Q = {}

        V_current, V_last = self.Dict2Vec(self.V, self.S), self.Dict2Vec(self.V_, self.S)

        it = 1
        diff = []
        val = []
        special = []
        diff.append(np.inner(V_current - V_last, V_current - V_last))

        v_flag = True

        print self.P[((2,8),3),'S']
        print self.P[((2, 8), 3), 'N']
        print self.P[((2, 8), 3), 'W']
        print self.P[((2, 8), 3), 'E']
        while np.inner(V_current - V_last, V_current - V_last) > threshold:
            if it > 20:
                break

            val.append(np.linalg.norm(V_current))
            special.append(self.V[(3, 3),0])

            for s in self.S:
                self.V_[s] = self.V[s]

                if s not in self.Pi: # for softmax
                    self.Pi[s] = {}
                if s not in self.Q:
                    self.Q[s] = {}

            for s in self.S:
                # # print  "state # print :", s, tuple(s)
                if s not in self.T:

                    for a in self.A:
                        if (tuple(s), a) in self.P:

                            v = self.R[tuple(s), a] + self.gamma * self.Sigma_(tuple(s), a)
                            self.Q[tuple(s)][a] = np.exp(v / tau) # softmax solution

                    for opt in self.Opt.keys():
                        os = tuple(s[0])
                        if set(opt[0]).intersection(self.dfa.state_info[s[1]]['safe']) == set([]):
                            continue

                        if tuple([tuple(s), opt]) not in self.R:
                            self.R[tuple(s), opt] = 0.0

                        v = self.R[tuple(s), opt] + self.Sigma_opt(s, opt)

                        self.Q[tuple(s)][opt] = np.exp(v / tau)

                    # softmax solution
                    sumQ = sum(self.Q[tuple(s)].values())
                    for choice in self.Q[tuple(s)]:
                        self.Pi[tuple(s)][choice] = self.Q[tuple(s)][choice]/sumQ
                    if v_flag:
                        self.V[tuple(s)] = tau * np.log(sumQ)

                else:
                    if s not in self.unsafe:
                        self.V[tuple(s)], self.Pi[tuple(s)] = 100.0, None
                    else:
                        self.V[tuple(s)], self.Pi[tuple(s)] = 0.0, None
            print it, self.V[(3, 3), 0], self.V[(2, 8),1], self.V[(2, 8),2], self.V[(2, 8),3], self.V[(2, 8), 4]
            # print

            V_current, V_last = self.Dict2Vec(self.V, self.S), self.Dict2Vec(self.V_, self.S)
            # diff.append(np.inner(V_current - V_last, V_current - V_last))

            it += 1
        self.plot_map(it)
        print  "option special point: ", special

        self.hybrid_diff = diff
        self.hybrid_val = val
        self.hybrid_special = special
        return 0

    # API: action SVI runner
    def SVI(self, threshold):
        # S, A, P, R, sink, V, V_ = inS, inA, inP, inR, in_sink, inV, inV_
        tau = 1.0

        self.Pi, self.Pi_ = {}, {}

        V_current, V_last = self.Dict2Vec(self.V, self.S), self.Dict2Vec(self.V_, self.S)
        it = 1
        diff = []
        val = []
        pi_diff = []
        special = []

        diff.append(np.inner(V_current - V_last, V_current - V_last))

        while np.inner(V_current - V_last, V_current - V_last) > threshold or it < 2: # np.inner(V_current - V_last, V_current - V_last) > threshold

            if it > 1 and threshold > 10000:
                print "break"
                break
            # self.plot_map(it)

            if tuple([(3, 3), 0]) in self.V:
                special.append(self.V[(3, 3), 0])

            for s in self.S:
                self.V_[tuple(s)] = self.V[tuple(s)]

                if tuple(s) not in self.Pi: # for softmax
                    self.Pi[tuple(s)] = {}
                if tuple(s) not in self.Q:
                    self.Q[tuple(s)] = {}

            for s in self.S:

                if tuple(s) not in self.T and tuple(s) not in self.interruptions:
                    # # print  s, self.T
                    max_v, max_a = -0.001, None

                    for a in self.A:
                        if (tuple(s), a) in self.P:

                            v = self.R[tuple(s), a] + self.gamma * self.Sigma_(tuple(s), a)

                            self.Q[tuple(s)][a] = np.exp(v / tau)  # softmax solution
                            # if self.plotKey:
                            #     # print  v, self.R[tuple(s), a], self.Sigma_(tuple(s), a)

                            if v > max_v:
                                max_v, max_a = v, a

                    sumQ = sum(self.Q[tuple(s)].values())
                    for choice in self.Q[tuple(s)]:
                        self.Pi[tuple(s)][choice] = self.Q[tuple(s)][choice] / sumQ
                    self.V[tuple(s)] = tau * np.log(sumQ)

            V_current, V_last = self.Dict2Vec(self.V, self.S), self.Dict2Vec(self.V_, self.S)
            diff.append(np.inner(V_current - V_last, V_current - V_last))
            # if self.plotKey:
            #     # print  self.gamma
            #     # print  self.V
            #     # print  self.P
            # # print  V_current, V_last
            it += 1

        if not self.plotKey:
            self.svi_record = special[len(special) - 1]

            print "svi_record:", self.svi_record
            print "action special point: ", special
            # print len(self.V), self.V
        self.action_diff = diff
        self.action_val = val
        self.action_special = special

        return 0

    # DISCARD
    def testPolicy(self, ss, Policy, S, A, V, P, dfa, mdp):
        cs = ss
        cp = 1
        # print  "-----------------------------------------"
        # print  "start at:", cs, V[cs], "(objective value)"
        while True:
            act = Policy[cs]
            ns = tuple(np.array(cs[0]) + np.array(A[act]))
            nq = dfa.state_transitions[mdp.L[ns].display(), cs[1]]
            cp *= P[cs, act][tuple([ns, nq])]

            cs = tuple([ns, nq])

    # VIS: plotting heatmap using either matplotlib ot plotly to visualize value function for all options
    def layer_plot(self, title, dir=None):
        z = {}
        for key in self.Opt:
            # # print  self.Opt[key].S,  len(self.Opt[key].S)
            q1 = key[1][0]
            q2 = key[1][1]
            g = key[0]

            # # print  self.V
            temp = np.random.random((16, 18))
            for state in self.Opt[key].S:
                    # key = ((i, j), )
                    # if (i, j) not in self.V:
                    #     temp[(i, j)] = -1
                # # print  state[0], state
                i, j = 15-(state[0][0]-1), state[0][1]-1
                temp[i, j] = self.V[state]

            name = "layer-" + str(q1) + "-" + str(g) + "-" + str(q2)
            z[name] = temp

            # plt.figure()
            # plt.imshow(temp, cmap='hot', interpolation='nearest')

            # folder = "" if dir == None else dir

            # plt.savefig(folder + name + ".png")  # "../DFA/comparing test/"

        # title = "test_layers"
        # fname = 'RL/LTL_SVI/' + title
        fname = dir + title

        trace = []
        names = z.keys()
        fig = tools.make_subplots(rows=2, cols=2,
                                  subplot_titles=(names[0], names[1], names[2], names[3])
                                  )
        # specs = [[{'is_3d': True}, {'is_3d': True}], [{'is_3d': True}, {'is_3d': True}]]
        # subplot_titles=(names[0], names[1], names[2], names[3])

        trace.append(go.Heatmap(z=z[names[0]], showscale=True, colorscale='Jet'))
        trace.append(go.Heatmap(z=z[names[1]], showscale=False, colorscale='Jet'))
        trace.append(go.Heatmap(z=z[names[2]], showscale=False, colorscale='Jet'))
        trace.append(go.Heatmap(z=z[names[3]], showscale=False, colorscale='Jet'))

        fig.append_trace(trace[0], 1, 1)
        fig.append_trace(trace[1], 1, 2)
        fig.append_trace(trace[2], 2, 1)
        fig.append_trace(trace[3], 2, 2)

        # py.iplot(
        #     [
        #      dict(z=z[0]+1.0, showscale=False, opacity=0.9, type='surface'),
        #      dict(z=z[1]+2.0, showscale=False, opacity=0.9, type='surface'),
        #      dict(z=z[2]+3.0, showscale=False, opacity=0.9, type='surface'),
        #      dict(z=z[3]+4.0, showscale=False, opacity=0.9, type='surface')],
        #     filename=fname)

        fig['layout'].update(title=title)

        py.iplot(fig, filename=fname)

        return 0

    # VIS: plot whole product state space value functions
    def plot_map(self, iteration):
        Aphi = self.dfa.state_info
        for q in Aphi:
            # if q == 1:
            #     print
            goals, obs = [], []
            for ap in Aphi[q]['safe']:
                goals += self.Exp[ap]
            goals = list(set(goals))

            for ap in Aphi[q]['unsafe']:
                obs += self.Exp[ap]
            obs = list(set(obs))

            temp = np.random.random((6, 8))
            for state in self.originS:
                v = 0
                count = 0
                if tuple(state) in goals:
                    for ap in Aphi[q]['safe']:
                        if tuple(state) in self.Exp[ap]:
                            v += self.V[tuple(state), self.dfa.state_transitions[ap,q]]
                            count += 1
                    temp[state[0] - 1, state[1] - 1] = v/count
                elif tuple(state) in obs:
                    temp[state[0] - 1, state[1] - 1] = 0.0
                else:
                    temp[state[0] - 1, state[1] - 1] = self.V[tuple(state), q]
            folder = '../whole system/option/'
            name = 'value function at automata state:'+ str(q) + " at it:" + str(iteration)
            fig = plt.figure()
            cax = plt.imshow(temp, cmap='hot', interpolation='nearest')
            cbar = fig.colorbar(cax) #ticks=[-1, 0, 1]
            # cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar
            plt.savefig(folder + name + ".png")  # "../DFA/comparing test/"

    # VIS: plotting single heatmap when necessary, used for debugging!!
    def plot_heat(self, key, dir=None):

        g, q1, q2 = None, None, None
        folder = "" if dir == None else dir

        if type(key) == type("string") or type(key) == type(tuple([1,2,3])):
            try:
                name = "reaching option" + key
            except:
                name = "reaching option"
                for element in key[0]:
                    name = name + '-' + element
            plt.figure()

            temp = np.random.random((6, 8))
            for state in self.S:
                if state not in self.wall_cord:
                    temp[state[0] - 1, state[1] - 1] = self.V[tuple(state)]

            plt.imshow(temp, cmap='hot', interpolation='nearest')
            plt.savefig(folder + name + ".png")
            return 0
        else:

            temp = np.random.random((6, 8))
            for state in self.S:
                temp[state[0][0] - 1, state[0][1] - 1] = self.V[state]

            q1 = key[1][0]
            q2 = key[1][1]
            g = key[0]


        name = "layer-" + str(q1) + "-" + str(g) + "-" + str(q2)
        plt.figure()
        plt.imshow(temp, cmap='hot', interpolation='nearest')
        plt.savefig(folder + name + ".png")  # "../DFA/comparing test/"

        ct = go.Contour(
            z=temp,
            type='surface',
            colorscale='Jet',
        )
        fname = 'RL/LTL_SVI/' + name

        py.iplot(
            [   ct,
                dict(z=temp, showscale=False, opacity=0.9, type='surface')],
            filename=fname)

        return 0

    # VIS: comparing the trend of two curves for any function or variable
    def plot_curve(self, trace1, trace2, name):
        plt.figure()

        # # print  x
        # # print  trace1
        # # print  trace2
        l1, = plt.plot(trace1, label="action")
        l2, = plt.plot(trace2, label="hybrid")

        plt.legend(handles=[l1, l2])

        plt.ylabel('Value iteration difference')
        plt.savefig(name + ".png")


