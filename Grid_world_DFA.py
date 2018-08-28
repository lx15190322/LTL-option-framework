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
    g23 = Action('g2-g3')
    g23_ = Action('g2,g3')
    obs = Action('obs')
    phi = Action('phi')
    whole = Action('whole')

    row, col = 6, 8
    i, j = range(1, row + 1), range(1, col + 1)
    states = [[x, y] for x in i for y in j]
    S = [tuple([x, y]) for x in i for y in j]

    # # print  S

    s_q = {}
    q_s = {}
    q_s[g1.v] = [(3, 1), (3, 2)]  # (1, 2), (1, 3)
    q_s[obs.v] = [(2, 4), (3, 4), (4, 4), (5, 1), (5, 2), (6, 1), (6, 2), (1, 6), (2, 6),
                  (6, 6)]  # (3, 4), (4, 4), (5, 4)
    q_s[g2.v] = [(3, 8), (4, 8)] # (4, 8)
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

    # initialize origin MDP
    mdp = MDP()

    # a = ['a', 'a-b', 'c', 'd', 'a-c', 'a-e', 'b-d', 'e']
    # b = mdp.bubble(a)

    mdp.set_S(states)
    mdp.set_WallCord(mdp.add_wall(states))
    mdp.set_P()
    mdp.set_L(s_q) # L: labeling function (L: S -> Q), e.g. self.L[(2, 3)] = 'g1'
    mdp.set_Exp(q_s) # L^-1: inverse of L (L: Q -> S), e.g. self.Exp['g1'] = (2, 3)
    mdp.set_Size(6, 8)
    # # print  "probabilities", len(mdp.P)

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
    dfa.add_transition(g23.display(), 2, 4)
    dfa.add_transition(g23.display(), 3, 4)

    dfa.toDot("DFA")
    dfa.prune_eff_transition()
    dfa.g_unsafe = 'obs'

    # print  "transitions:"
    # print  dfa.state_transitions
    # print  "safe final:"
    # print  dfa.final_states
    # print  "sinks:"
    # print  dfa.sink_states

    # print  "========================================"
    curve = {}
    t0 = time.time()
    result = mdp.product(dfa, mdp)
    # print  result.goal
    # print  result.T
    # print  "setup reward is:"
    # print  result.R
    result.plotKey = False
    curve['action'] = result.SVI(0.001)
    V_action = result.goal_probability(result.Pi, result.P, ((3, 3), 0), 0.001)

    # print ('rate for action is:', result.evaluation(result.Pi, result.P, ((3, 7), 0), trial=10000))

    t1 = time.time()
    # print  "action time", t1 - t0

    result.AOpt = mdp.option_generation(dfa)
    # result.segmentation()
    t2 = time.time()
    # print  "option generating time", t2-t1
    result.option_factory()
    result.layer_plot()
    result.option_plot()
    t3 = time.time()
    # # print  t3-t2
    # # print  "before option",result.V

    curve['option'] = result.SVI_option(1.0)
    Policy = result.policy_evaluation(result.V)
    V_option = result.goal_probability(Policy, result.P, ((3, 3), 0), 0.001)
    # print('rate for option is:', result.evaluation(Policy, result.P, ((3, 7), 0), trial=10000))

    t4 = time.time()
    curve['hybrid'] = result.SVI_option(1.0, hybrid=True)
    Policy = result.policy_evaluation(result.V)
    V_hybrid = result.goal_probability(Policy, result.P, ((3, 3), 0), 0.001)
    # print('rate for hybrid is:', result.evaluation(Policy, result.P, ((3, 7), 0), trial=10000))
    # print  "option time",t4-t3
    result.plot_curve(curve, 'compare_result_normalized_reward')

    result.compute_norm(V_action, V_option)
    result.compute_norm(V_action, V_hybrid)
'''
# ==================== task2
#     dfa.clear()
#     dfa.reset()
#     dfa.clear()
    dfa2 = DRA(0, [g1, g2, obs, g3, phi, whole])
    # # print  dfa2.state_transitions
    # # print  dfa2.alphabet
    # # print  dfa2.states

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
    dfa2.add_transi
    
    tion(g3.display(), 3, 3)

    # dfa2.add_transition(g23.display(), 0, 1)
    # dfa2.add_transition(g23.display(), 1, 1)
    # dfa2.add_transition(g23.display(), 2, 2)
    # dfa2.add_transition(g23.display(), 3, 3)








    dfa2.toDot("DFA2")
    dfa2.prune_eff_transition()
    dfa2.g_unsafe = 'obs'

    mdp.set_P()
    # # print  "probabilities", len(mdp.P)
    t0 = time.time()
    result2 = mdp.product(dfa2, mdp)
    result2.plotKey = False
    result2.SVI(1.0)
    t1 = time.time()
    # print  "action time", t1 - t0

    result2.AOpt = dcp(result.AOpt)
    # result.segmentation()
    t2 = time.time()
    # # print  t2 - t1
    result2.option_factory()
    t3 = time.time()
    # # print  t3 - t2
    # # print  "before option", result.V

    result2.SVI_option(1.0)
    t4 = time.time()
    # print  "option time", t4 - t3
    # # print  t4
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
    # # print  "probabilities", len(mdp.P)
    t0 = time.time()
    result3 = mdp.product(dfa3, mdp)
    result3.plotKey = False
    result3.SVI(1.0)
    t1 = time.time()
    # print  "action time",t1 - t0

    result3.AOpt = dcp(result.AOpt)
    # result.segmentation()
    t2 = time.time()
    # # print  t2 - t1
    result3.option_factory()
    t3 = time.time()
    # # print  t3 - t2
    # # print  "before option", result.V

    result3.SVI_option(1.0)
    t4 = time.time()
    # print  "option time",t4 - t3






'''