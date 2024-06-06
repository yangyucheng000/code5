import matplotlib.pyplot as plt
import numpy as np
import os

plt.rc('font', family='Times New Roman')
fig = plt.figure(figsize=(8,3))
x_labels111 = ['Ranking Loss', 'One Error', 'Coverage', 'MAP', 'Macro F1']

########################################### k=5
k5 = np.array([[0.246, 0.560, 0.347, 0.507, 0.226],
[0.233, 0.552, 0.337, 0.525, 0.244],
[0.232, 0.534, 0.332, 0.531, 0.256],
[0.222, 0.522, 0.315, 0.536, 0.276],
[0.218, 0.478, 0.287, 0.543, 0.301],
[0.217, 0.465, 0.263, 0.557, 0.327],
[0.219, 0.463, 0.267, 0.554, 0.302],
[0.222, 0.467, 0.271, 0.551, 0.295]])

Ranking = k5[:, 0]
One = k5[:, 1]
Covergae = k5[:, 2]
MAP = k5[:, 3]
F1 = k5[:, 4]

ax = fig.add_subplot(1, 2, 1)

xlabel = ['0', r'$\lfloor {n \choose 2} /{50}\rfloor$', '', r'$\lfloor{n \choose 2} /{30}\rfloor$', '', r'$\lfloor {n \choose 2} /{10}\rfloor$', '', r'$ {n \choose 2} $']
xticks = np.arange(len(xlabel))

# ax.plot(xticks, Ranking, linestyle='-', label='Ranking Loss', color=[242/255, 5/255, 5/255], marker='s', linewidth=3, markersize=8)
# ax.plot(xticks, One, label='One Error', color='royalblue', marker='v', linewidth=3, markersize=8)
# ax.plot(xticks, Covergae, label='Coverage', marker='H', linewidth=3, markersize=8)
ax.plot(xticks, MAP, label='MAP', color=[242/255, 5/255, 5/255], marker='D', linewidth=3, markersize=8)
ax.plot(xticks, F1, label='Macro F1', color='royalblue', marker='h', linewidth=3, markersize=8)


font_xlable = {  # 用 dict 单独指定 xlabel 样式
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 15,
    'usetex': True,
}
plt.tick_params(pad=0)
ax.set_xticks(xticks)
ax.set_xticklabels(xlabel, font_xlable)
ax.set_title(r"$K_2=5$", fontsize=18)
# ax.legend(fontsize=15, bbox_to_anchor=(0.5, 0.82))
ax.legend(fontsize=18)
plt.yticks(fontsize=18)



########################################### k=10
k10 = np.array([
    [0.297, 0.661, 0.321, 0.536, 0.269],
    [0.288, 0.641, 0.290, 0.548, 0.312],
    [0.279, 0.620, 0.267, 0.575, 0.337],
    [0.276, 0.556, 0.249, 0.586, 0.361],
    [0.173, 0.481, 0.236, 0.591, 0.370],
    [0.182, 0.491, 0.242, 0.588, 0.359],
    [0.180, 0.484, 0.246, 0.587, 0.368],
    [0.182, 0.493, 0.247, 0.584, 0.354]]
)

Ranking = k10[:, 0]
One = k10[:, 1]
Covergae = k10[:, 2]
MAP = k10[:, 3]
F1 = k10[:, 4]

ax = fig.add_subplot(1, 2, 2)

xlabel = ['0', r'$\lfloor {n \choose 2} /{50}\rfloor$', '', r'$\lfloor{n \choose 2} /{30}\rfloor$', '', r'$\lfloor {n \choose 2} /{10}\rfloor$', '', r'$ {n \choose 2} $']
xticks = np.arange(len(xlabel))

# ax.plot(xticks, Ranking, linestyle='-', label='Ranking Loss', color=[242/255, 5/255, 5/255], marker='s', linewidth=3, markersize=8)
# ax.plot(xticks, One, label='One Error', color='royalblue', marker='v', linewidth=3, markersize=8)
# ax.plot(xticks, Covergae, label='Coverage', marker='H', linewidth=3, markersize=8)
ax.plot(xticks, MAP, label='MAP', color=[242/255, 5/255, 5/255],  marker='D', linewidth=3, markersize=8)
ax.plot(xticks, F1, label='Macro F1', color='royalblue', marker='h', linewidth=3, markersize=8)

font_xlable = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 15,
    'usetex': True,
}
plt.tick_params(pad=0)
ax.set_xticks(xticks)
ax.set_xticklabels(xlabel, font_xlable)
ax.set_title(r"$K_2=10$", fontsize=18)
# ax.legend(fontsize=12)
plt.yticks(fontsize=18)
# plt.ylabel('MAP', fontsize=18)
# plt.xlabel('Value', fontsize=18)
plt.subplots_adjust(left=0.057, bottom=0.098, right=0.99, top=0.905, wspace=0.157, hspace=0.2)

plt.show()