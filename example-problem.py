from mip import Model, maximize, BINARY, xsum

Targets = ['Early Morning', 'Morning', 'Noon', 'Afternoon', 'Night']
T = range(len(Targets))

Omega = ['Commuters', 'Elderly People', 'Office Workers', 'Shoppers']
O = range(len(Omega))

# S = legal schedules as singleton sets of targets
Schedules = [x for x in T] 
S = range(len(Schedules))

# Define the payoff matrix for all targets
U_DEF = [[5, -10], [5, -10], [5, -10], [5, -10], [30, -10]]
U_ATT = [[0, 10], [0, 5], [0, 5], [0, 5], [0, 10]]

# M: S x T 
# It's diagonal because we defined legal schedules to be singleton 
# sets of targets
M = [[1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 1, 0],
     [0, 0, 0, 0, 1]]

# Coverage capabilities: S x Omega 
Ca = [[1, 1, 0, 0],  # {Early Morning}
      [0, 1, 1, 0],  # {Morning}
      [0, 0, 1, 1],  # {Noon}
      [0, 1, 0, 1],  # {Afternoon}
      [1, 0, 0, 1]]  # {Night}

# We have one unit of each resource
R = [0.4, 0.1, 0.25, 0.25]

# Z: large constant, relative to the maximum payoff
Z = 10e10 

def utility(t, c, covered, uncovered):
	return c[t] * covered + (1 - c[t]) * uncovered

m = Model() 
d = m.add_var('d')
k = m.add_var('k')

m.objective = maximize(d) # (19)

a = [m.add_var(var_type=BINARY) for t in T] # (20)
c = [m.add_var() for t in T] # (21)
q = [m.add_var() for s in S] # (22)
h = [[m.add_var() for o in Omega] for s in S] # (23)

m += xsum(a) == 1 # (24)

# (25)
for s in S:
	m += xsum(h[s]) == q[s]

# (26)
for t in T:
	m += xsum([q[s]*M[s][t] for s in S]) == c[t]

# (27)
for w in O:
	m += xsum([h[s][w]*Ca[s][w] for s in S]) <= R[w]

# (28)
for s in S:
	for w in O:
		m += h[s][w] <= Ca[s][w]

# (29)
for t in T:
    m += d - utility(t, c, U_DEF[t][0], U_DEF[t][1]) <= (1 - a[t]) * Z 

# (30)
for t in T:
	delta = k - utility(t, c, U_ATT[t][0], U_ATT[t][1])
	m += 0 <= delta
	m += delta <= (1 - a[t]) * Z

m.optimize(max_seconds=60)

print('maximum payoff: {}'.format(round(d.x, 3)))
print('attack vector: {}'.format([round(x.x, 3) for x in a]))
print('coverage vector: {}'.format([round(x.x, 3) for x in c]))

# pretty print results
results = [['']+Omega] + [[Targets[s]]+[str(round(x.x,3)) for x in h[s]] for s in S]
col_width = max(len(word) for row in results for word in row) + 2 
for row in results:
    print ("".join(word.ljust(col_width) for word in row))

