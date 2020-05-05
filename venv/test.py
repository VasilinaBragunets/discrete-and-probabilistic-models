import numpy as np
import matplotlib.pyplot as plt




u1 =  [9.86510116e-01, 1.33984834e-02, 9.09870898e-05, 4.13314626e-07]
norm_p = np.sqrt(sum(list(map(lambda x: x ** 2, u1))))
print ('norm_p_last = ', norm_func)

'''

x = np.random.randint(low=1, high=11, size=50)
y = x + np.random.randint(1, 5, size=x.size)
data = np.column_stack((x, y))


fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(1, 2, 1)

ax1.scatter(x=x, y=y, marker='o', c='r', edgecolor='b')
ax1.set_title('Scatter: $x$ versus $y$')
ax1.set_xlabel('$x$')

plt.text(0.5, 0.9, 'среднее время между заявками', ha='left', va='top', transform=fig.transFigure
)


plt.show()
'''