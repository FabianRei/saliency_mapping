import itertools as it

import numpy as np

stations = [0,1,2,3,4,5]

group_arr = [[],[],[],[],[],[],[],[]]
groups = [0,1,2,3,4,5,6,7]
groupings = it.combinations(groups, 2)
groupings = list(groupings)


groups = []
groupings_done = []
teams_i = []
for i in range(6):
    group = []
    teams = []
    for g in groupings:
        if g in group:
            continue
        if g in groupings_done:
            continue
        if g[0] in teams or g[1] in teams:
            continue
        group.append(g)
        teams.append(g[0])
        teams.append(g[1])
        groupings_done.append(g)
    groups.append(group)
print('nice')

