from DFA import DFA
from DFA import DRA
from DFA import DRA2
from DFA import Action
from MDP import MDP
import time
from copy import deepcopy as dcp

if __name__ == '__main__':
    # DFA generated from LTL:
    # (F (G1 and G3) or F (G3 and G1)) and G !G2
    g1 = Action('g1')
    g2 = Action('g2') # set as an globally unsafe obstacle
    g3 = Action('g3')
    g23 = Action('g23')
    obs = Action('obs')
    phi = Action('phi')
    whole = Action('whole')

    row, col = 6, 8
    i, j = range(1, row + 1), range(1, col + 1)
    states = [[x, y] for x in i for y in j]
    S = [tuple([x, y]) for x in i for y in j]

    # print S

    s_q = {}
    q_s = {}
    q_s[g1.v] = [(3, 1), (3, 2)]  # (1, 2), (1, 3)
    q_s[obs.v] = [(2, 4), (3, 4), (4, 4), (5, 1), (5, 2), (6, 1), (6, 2), (1, 6), (2, 6),
                  (6, 6)]  # (3, 4), (4, 4), (5, 4)
    q_s[g2.v] = [(2, 8), (4, 8)] # (4, 8)
    q_s[g23.v] = [(4, 8)]
    q_s[g3.v] = [(5, 8), (4, 8)]  # (4, 8),
    q_s[phi.v] = list(set(S) - set(q_s[g1.v] + q_s[g2.v] + q_s[g3.v] + q_s[obs.v]))
    q_s[whole.v] = S
    for s in S:
        if s in q_s[g1.v]:
            s_q[s] = g1.v
        elif s in q_s[g2.v]:
            s_q[s] = g2.v
        elif s in q_s[g3.v]:
            s_q[s] = g3.v
        elif s in q_s[g23.v]:
            s_q[s] = g23.v
        elif s in q_s[obs.v]:
            s_q[s] = obs.v
        else:
            s_q[s] = phi.v

    # initialize origin MDP
    mdp = MDP()
    mdp.set_S(states)
    mdp.set_WallCord(mdp.add_wall(states))
    mdp.set_P()
    mdp.set_L(s_q) # L: labeling function (L: S -> Q), e.g. self.L[(2, 3)] = 'g1'
    mdp.set_Exp(q_s) # L^-1: inverse of L (L: Q -> S), e.g. self.Exp['g1'] = (2, 3)
    mdp.set_Size(6, 8)
    # print "probabilities", len(mdp.P)

    dfa = DFA(0, [g1, g2, obs, g3, phi, whole])

    dfa.set_final(4)
    dfa.set_sink(5)

    sink = list(dfa.sink_states)[0]

    for i in range(sink + 1):
        dfa.add_transition(phi.display(), i, i)
        if i < sink:
            dfa.add_transition(obs.display(), i, sink)

    dfa.add_transition(whole.display(), sink, sink)

    dfa.add_transition(g1.display(), 0, 1)
    for i in range(1, sink + 1):
        dfa.add_transition(g1.display(), i, i)

    dfa.add_transition(g2.display(), 1, 2)
    dfa.add_transition(g2.display(), 3, 4)
    dfa.add_transition(g2.display(), 0, 0)
    dfa.add_transition(g2.display(), 2, 2)

    dfa.add_transition(g3.display(), 1, 3)
    dfa.add_transition(g3.display(), 2, 4)
    dfa.add_transition(g3.display(), 0, 0)
    dfa.add_transition(g3.display(), 3, 3)

    dfa.add_transition(g23.display(), 1, 4)
    dfa.add_transition(g23.display(), 0, 0)
    dfa.add_transition(g23.display(), 2, 2)
    dfa.add_transition(g23.display(), 3, 3)

    dfa.toDot("DFA")
    dfa.prune_eff_transition()
    dfa.g_unsafe = 'obs'

    print "transitions:"
    print dfa.state_transitions
    print "safe final:"
    print dfa.final_states
    print "sinks:"
    print dfa.sink_states

    print "========================================"

    t0 = time.time()
    result = mdp.product(dfa, mdp)
    print result.goal
    print result.T
    result.plotKey = False
    result.SVI(1.0)
    t1 = time.time()
    print "action time", t1 - t0

    result.AOpt = mdp.option_generation(dfa)
    # result.segmentation()
    t2 = time.time()
    print "option generating time", t2-t1
    result.option_factory()
    t3 = time.time()
    # print t3-t2
    # print "before option",result.V

    result.SVI_option(1.0)
    t4 = time.time()
    print "option time",t4-t3

'''
# ==================== task2
#     dfa.clear()
#     dfa.reset()
#     dfa.clear()
    dfa2 = DRA(0, [g1, g2, obs, g3, phi, whole])
    # print dfa2.state_transitions
    # print dfa2.alphabet
    # print dfa2.states

    dfa2.set_final(3)
    dfa2.set_sink(4)

    sink = list(dfa2.sink_states)[0]

    for i in range(sink + 1):
        dfa2.add_transition(phi.display(), i, i)
        if i < sink:
            dfa2.add_transition(obs.display(), i, sink)

    dfa2.add_transition(whole.display(), sink, sink)

    dfa2.add_transition(g1.display(), 0, sink)

    for i in range(2, sink + 1):
        dfa2.add_transition(g1.display(), i, i)
    dfa2.add_transition(g1.display(), 1, 3)

    dfa2.add_transition(g2.display(), 0, 2)
    for i in range(1, sink + 1):
        dfa2.add_transition(g2.display(), i, i)

    dfa2.add_transition(g3.display(), 2, 1)
    dfa2.add_transition(g3.display(), 0, 0)
    dfa2.add_transition(g3.display(), 1, 1)
    dfa2.add_transition(g3.display(), 3, 3)

    # dfa2.add_transition(g23.display(), 0, 1)
    # dfa2.add_transition(g23.display(), 1, 1)
    # dfa2.add_transition(g23.display(), 2, 2)
    # dfa2.add_transition(g23.display(), 3, 3)


    dfa2.toDot("DFA2")
    dfa2.prune_eff_transition()
    dfa2.g_unsafe = 'obs'

    mdp.set_P()
    # print "probabilities", len(mdp.P)
    t0 = time.time()
    result2 = mdp.product(dfa2, mdp)
    result2.plotKey = False
    result2.SVI(1.0)
    t1 = time.time()
    print "action time", t1 - t0

    result2.AOpt = dcp(result.AOpt)
    # result.segmentation()
    t2 = time.time()
    # print t2 - t1
    result2.option_factory()
    t3 = time.time()
    # print t3 - t2
    # print "before option", result.V

    result2.SVI_option(1.0)
    t4 = time.time()
    print "option time", t4 - t3
    # print t4
# task 3 ==============================
    dfa3 = DRA2(0, [g1, g2, obs, g3, phi, whole])

    dfa3.set_final(3)
    dfa3.set_sink(4)

    sink = list(dfa3.sink_states)[0]

    for i in range(sink + 1):
        dfa3.add_transition(phi.display(), i, i)
        if i < sink:
            dfa3.add_transition(obs.display(), i, sink)

    dfa3.add_transition(whole.display(), sink, sink)

    dfa3.add_transition(g1.display(), 0, 2)
    dfa3.add_transition(g1.display(), 1, 3)
    for i in range(2, sink + 1):
        dfa3.add_transition(g1.display(), i, i)

    dfa3.add_transition(g2.display(), 0, sink)
    dfa3.add_transition(g2.display(), 1, 1)
    dfa3.add_transition(g2.display(), 2, sink)
    dfa3.add_transition(g2.display(), 3, 3)

    dfa3.add_transition(g3.display(), 0, 1)
    dfa3.add_transition(g3.display(), 1, 1)
    dfa3.add_transition(g3.display(), 2, 3)
    dfa3.add_transition(g3.display(), 3, 3)

    # dfa3.add_transition(g23.display(), 0, 0)
    # dfa3.add_transition(g23.display(), 1, 1)
    # dfa3.add_transition(g23.display(), 2, 2)
    # dfa3.add_transition(g23.display(), 3, 3)

    dfa3.toDot("DFA3")
    dfa3.prune_eff_transition()
    dfa3.g_unsafe = 'obs'

    mdp.set_P()
    # print "probabilities", len(mdp.P)
    t0 = time.time()
    result3 = mdp.product(dfa3, mdp)
    result3.plotKey = False
    result3.SVI(1.0)
    t1 = time.time()
    print "action time",t1 - t0

    result3.AOpt = dcp(result.AOpt)
    # result.segmentation()
    t2 = time.time()
    # print t2 - t1
    result3.option_factory()
    t3 = time.time()
    # print t3 - t2
    # print "before option", result.V

    result3.SVI_option(1.0)
    t4 = time.time()
    print "option time",t4 - t3






'''