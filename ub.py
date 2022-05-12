import numpy as np
from itertools import combinations
from gurobipy import *

INIT_CAP = 30 #10,20,30
RETURN_SCALE = 60 #40,60,80
KAPPAS = [0.5, 5e-3, 5e-5]
names = ['high','medium','low'] # for the kappa values

np.random.seed(6360)

num_prod = 5

num_cust = 5

T = 200

inventory = [INIT_CAP]*num_prod

prices = np.array([30,25,20,15,10])

p_salvage = prices/2 #salvaged price is 1/2

#return distribution is discrete uniform over 1,2,...,10 days
p_return = np.zeros((num_prod, T))
for p in range(num_prod):
    for t in range(1,11):
        p_return[p,t] = 1/10

prob_return = prices/RETURN_SCALE

# prob_return = np.ones(5)*1/3

prob_good = 0.85


periods = np.array([180,140,100,60,20])

kappa = 0.5

prob_arriv = np.zeros((T, num_cust))

for t in range(T):
    for c in range(num_cust):
        prob_arriv[t,c] = np.exp(-kappa*np.abs(t - periods[c]))
    prob_arriv[t] /= np.sum(prob_arriv[t])

arrivals = np.zeros(T)
for t in range(T):
    arrivals[t] = np.random.choice(np.arange(num_cust), p = prob_arriv[t])


w = np.zeros((num_cust, num_prod)) #utilities
for c in range(num_cust):
    for p in range(num_prod):
        if p<=c:
            w[c,p] = np.random.uniform()

w = np.load('w.npy')
w_0 = np.zeros(num_cust)

for c in range(num_cust):
    w_0[c] = np.sum(w[c])/9

def phi(S, j, i):
    ''' S is assortment, j is customer, i is product, e.g. S = [0,3,4]
        NOTE: these arguments always denote first product/item with index 0'''
    if len(S) > 0:
        denom = w_0[j]
        for s in S:
            denom += w[j, s]
        num = 0
        if i in S:
            num = w[j,i]
        return num/denom
    else:
        return 0

products = [0,1,2,3,4]

assort = []
for a in range(0,6):
    for i in combinations(products,a):
        assort.append(i)

arrivals = np.zeros(T)
m = Model('lp_ub')
m.Params.LogToConsole = 0
y = m.addVars(len(assort),T,name = 'y')
x = m.addVars(num_prod,T,name = 'x')
cons = {}
cons1 = {}
for i in range(num_prod):
    for k in range(T):
        cons[i,k] = LinExpr()
        cons1[i,k] = x[i,k]
        for t in range(k+1):
            cons[i,k] -= prob_good*x[i,t]
            for S in range(len(assort)):
                temp = phi(assort[S], int(arrivals[t]), i)*y[S,t]
                cons[i,k] += temp
                cons1[i,k] -= temp*p_return[i,k-t]*prob_return[i]

# add capacity constraints  
cap = m.addConstrs((cons[i,k] <= inventory[i] for i in range(num_prod) for k in range(T)),name = 'cap')


# add constraints w.r.t product returns
ret = m.addConstrs((cons1[i,k] == 0 for i in range(num_prod) for k in range(T)),name = 'ret')

# an assortment must be offered at every time period
m.addConstrs((sum(y[S,t] for S in range(len(assort))) == 1 for t in range(T)))

#set objective
obj = LinExpr()
for t in range(T):
    for i in products:
        obj += -prices[i]*x[i,t] + p_salvage[i]*(1-prob_good)*x[i,t]
        for S in range(len(assort)):
            obj += prices[i]*phi(assort[S], int(arrivals[t]), i)*y[S,t]
m.setObjective(obj,GRB.MAXIMIZE)
# m.write('1.lp')
# m.optimize()

def modelUpdate(arr):
    ''' Update optimization model, 
       arr represents a sample path of customer arrivals {z_t}_{t=1}^{T}'''
    for i in range(num_prod):
        for k in range(T):
            for t in range(k):
                for S in range(len(assort)):
                    temp = phi(assort[S], int(arr[t]), i)
                    m.chgCoeff(cap[i,k], y[S,t], temp)
                    m.chgCoeff(ret[i,k], y[S,t], -temp*p_return[i,k-t]*prob_return[i])
    obj = LinExpr()
    for t in range(T):
        for i in products:
            obj += -prices[i]*x[i,t] + p_salvage[i]*(1-prob_good)*x[i,t]
            for S in range(len(assort)):
                obj += prices[i]*phi(assort[S], int(arr[t]), i)*y[S,t]
    m.setObjective(obj,GRB.MAXIMIZE)
    
def getUB(e):
    '''e is index of kappa value'''
    for i in range(10):
        arrivals = np.load('arrivals' + str(i)+'_'+names[e]+'.npy')
        modelUpdate(arrivals)
        m.optimize()
        obj = m.getObjective()
        print(names[e], i, ': ',obj.getValue())

#EXAMPLE USAGE
# e = 1 #e.g. kappa = 5e-3
# getUB(e)
