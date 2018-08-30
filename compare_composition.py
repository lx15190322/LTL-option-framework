from MDP import MDP
import time
from copy import deepcopy as dcp
from DFA import Action

if __name__ == '__main__':

    g1 = Action('g1')
    g2 = Action('g2')  # set as an globally unsafe obstacle
    g3 = Action('g3')
    g23 = Action('g2-g3')
    g23_ = Action('g2,g3')
    obs = Action('obs')
    phi = Action('phi')
    whole = Action('whole')

    row, col = 6, 8
    i, j = range(1, row + 1), range(1, col + 1)
    states = [[x, y] for x in i for y in j]
    S = [tuple([x, y]) for x in i for y in j]

    s_q = {}
    q_s = {}
    q_s[g1.v] = [(3, 1), (3, 2)]  # (1, 2), (1, 3)
    q_s[obs.v] = [(2, 4), (3, 4), (4, 4), (5, 1), (5, 2), (6, 1), (6, 2), (1, 6), (2, 6),
                  (6, 6)]  # (3, 4), (4, 4), (5, 4)
    q_s[g2.v] = [(3, 8), (4, 8)]  # (4, 8)
    q_s[g23.v] = [(4, 8)]
    q_s[g3.v] = [(5, 8), (4, 8)]  # (4, 8),
    q_s[phi.v] = list(set(S) - set(q_s[g1.v] + q_s[g2.v] + q_s[g3.v] + q_s[obs.v]))
    q_s[whole.v] = S
    for s in S:
        s_q[s] = []
        if s in q_s[g1.v] and g1.v not in s_q[s]:
            s_q[s].append(g1.v)
        if s in q_s[g2.v] and g2.v not in s_q[s]:
            s_q[s].append(g2.v)
        if s in q_s[g3.v] and g3.v not in s_q[s]:
            s_q[s].append(g3.v)
        if s in q_s[g23.v] and g23.v not in s_q[s]:
            # s_q[s].append(g23.v)
            s_q[s] = [g23.v]
            # temp = g23.v.split('-')
            # for element in temp:
            #     if element not in s_q[s]:
            #         s_q[s].append(element)
        if s in q_s[obs.v] and obs.v not in s_q[s]:
            s_q[s].append(obs.v)
        if s in q_s[phi.v] and phi.v not in s_q[s]:
            s_q[s].append(phi.v)
    # # print (g23_.v)

    threshold = 0.001
    # initialize origin MDP
    mdp1 = MDP()

    # a = ['a', 'a-b', 'c', 'd', 'a-c', 'a-e', 'b-d', 'e']
    # b = mdp.bubble(a)

    mdp1.set_S(states)
    # mdp1.set_WallCord(mdp1.add_wall(states))
    mdp1.set_P()
    mdp1.set_L(s_q)  # L: labeling function (L: S -> Q), e.g. self.L[(2, 3)] = 'g1'
    mdp1.set_Exp(q_s)  # L^-1: inverse of L (L: Q -> S), e.g. self.Exp['g1'] = (2, 3)
    mdp1.set_Size(6, 8)
    mdp1.interruptions = q_s[obs.v]
    mdp1.T = q_s[g2.v]

    mdp2 = MDP()
    mdp2.set_S(states)
    # mdp1.set_WallCord(mdp1.add_wall(states))
    mdp2.set_P()
    mdp2.set_L(s_q)  # L: labeling function (L: S -> Q), e.g. self.L[(2, 3)] = 'g1'
    mdp2.set_Exp(q_s)  # L^-1: inverse of L (L: Q -> S), e.g. self.Exp['g1'] = (2, 3)
    mdp2.set_Size(6, 8)
    mdp2.interruptions = q_s[obs.v]
    mdp2.T = q_s[g3.v]