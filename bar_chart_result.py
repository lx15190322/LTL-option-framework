
# a stacked bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt


N = 4
optionError2 = (0.024972824138408435, 0.4484616895296258, 0.024972824138408435, 0)
hybridError2 = (0.008830631908655587, 0.01118868471725386, 0.008830631908655587, 0)

optionError = (0.07206194696779192, 0.12831071146799922, 0.07206194696779192, 0)
hybridError = (0.020875919901878495, 0.00793809720320826, 0.020875919901878495, 0)

ind = np.arange(N)    # the x locations for the groups
width = 0.25       # the width of the bars: can also be len(x) sequence
ind2 = [x+width+0.02 for x in ind]
ind_mid = [x+width/2 for x in ind]

p1 = plt.bar(ind, optionError, width) # color='#d62728'
p2 = plt.bar(ind, hybridError, width, bottom=optionError)
p3 = plt.bar(ind2, optionError2, width) # color='#d62728'
p4 = plt.bar(ind2, hybridError2, width, bottom=optionError2)

plt.ylabel('2 Norm and $\infty$ Norm Error')
plt.title('Evaluated value function error for option-action and hybrid-action')
plt.xticks(ind_mid, ('$\phi_1$', '$\phi_2$', '$\phi_3$'))
# plt.yticks(np.arange(0, 0.06, 10))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Option $||\cdot||^2$', 'Hybrid $||\cdot||^2$', 'Option $||\cdot||^\infty$', 'Hybrid $||\cdot||^\infty$'))
plt.show()