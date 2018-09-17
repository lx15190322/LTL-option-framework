
# a stacked bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt


N = 4
optionError2 = (0.1857143517654766, 0.1295029466143666, 0.12771737604013453, 0)
hybridError2 = (0.07143359849846807, 0.008156022009548219, 0.007214475297557925, 0)

optionError = (0.06405260894515845, 0.072114763976038, 0.07101221812066082, 0)
hybridError = (0.027776576651151685, 0.00521980474900248, 0.004351941657409625, 0)

ind = np.arange(N)    # the x locations for the groups
width = 0.1       # the width of the bars: can also be len(x) sequence
ind2 = [x+width+0.02 for x in ind]
ind3 = [x+(width+0.02)*2 for x in ind]
ind4 = [x+(width+0.02)*3 for x in ind]
ind_mid = [x+width/2 for x in ind]

p1 = plt.bar(ind, optionError, width) # color='#d62728'
p2 = plt.bar(ind2, hybridError, width) #  bottom=optionError
p3 = plt.bar(ind3, optionError2, width) # color='#d62728'
p4 = plt.bar(ind4, hybridError2, width) # bottom=optionError2

plt.ylabel('2 Norm and $\infty$ Norm Error')
plt.title('Evaluated value function error for option-action and hybrid-action')
plt.xticks(ind_mid, ('$\phi_1$', '$\phi_2$', '$\phi_3$'))
# plt.yticks(np.arange(0, 0.06, 10))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Option $||\cdot||^2$', 'Hybrid $||\cdot||^2$', 'Option $||\cdot||^\infty$', 'Hybrid $||\cdot||^\infty$'))
plt.show()