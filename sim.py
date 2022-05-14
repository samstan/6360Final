import numpy as np
from itertools import combinations

INIT_CAP = 30 #10,20,30
RETURN_SCALE = 60 #40,60,80
RETURN_TIME = 0 #0 is uniform, 1 is decreasing from day 1 to day 10, 2 is peaking at day 5/6
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
        if RETURN_TIME == 0:
            p_return[p,t] = 1/10
        elif RETURN_TIME == 1:
            p_return[p,t] = (11-t)/55
        else:
            p_return[p,t] = np.minimum(t, 11-t)/30

prob_return = prices/RETURN_SCALE

# prob_return = np.ones(5)*1/3

prob_good = 0.85

periods = np.array([180, 140, 100, 60, 20])


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

def genArrivals(kappa):
    prob_arriv = np.zeros((T, num_cust))

    for t in range(T):
        for c in range(num_cust):
            prob_arriv[t,c] = np.exp(-kappa*np.abs(t - periods[c]))
        prob_arriv[t] /= np.sum(prob_arriv[t])

    arrivals = np.zeros(T)
    for t in range(T):
        arrivals[t] = np.random.choice(np.arange(num_cust), p = prob_arriv[t])
    return arrivals


def psi(x, ib):
    if ib == 'myopic':
        return (x>0)
    elif ib == 'linear':
        return x
    elif ib == 'exponential':
        return (np.exp(1)/(np.exp(1)-1))*(1- np.exp(-x))
    else:
        return

def sim(arrivals, ib):
    invs_allT = np.copy(inventory)
    revenue = 0
    return_event = [[] for t in range(T)]
    for t in range(T):
        for i in return_event[t]:
            revenue-=prices[i]
            if np.random.uniform()<prob_good: #if good condition
                invs_allT[i]+=1
            else:
                revenue+=p_salvage[i] #salvage price
        arriv = int(arrivals[t])
        max = -np.inf
        argmax = -1
        for a in assort:
            tmp = 0
            for i in a:
                tmp += prices[i]*phi(a, arriv, i)*psi(invs_allT[i]/inventory[i],ib)
            if tmp>max:
                max = tmp
                argmax = a
        denom = w_0[arriv]
        for i in argmax:
            denom += w[arriv,i]
        num = [w[arriv,i] for i in argmax] + [w_0[arriv]]
        probs = num/denom
        options = list(argmax) + [-1] #-1 is no purchase
        purchase = np.random.choice(options, p = probs)
        if purchase != -1:
            revenue+=prices[purchase]
            invs_allT[purchase]-=1 #remove from future inventory
            if np.random.uniform()<prob_return[purchase]: #if returned
                day = np.random.choice(1+np.arange(10))
                if t+day<T: #returned in time
                    return_event[t+day].append(purchase)
    return revenue


#just for generating arrival sequences
# for i in range(10):
#     for e,k in enumerate(KAPPAS):
#         arrivals = genArrivals(k)
#         np.save('arrivals'+str(i)+'_'+names[e]+'.npy', arrivals)

def getRevenue(e, pot):
    '''e is index of kappa value, pot is potential function'''
    revs = np.zeros((10,100))
    for i in range(10):
        arrivals = np.load('arrivals'+str(i)+'_'+names[e]+'.npy')
        for j in range(100):
            revs[i,j] = sim(arrivals, pot)
    return revs.mean(axis = 1)

#EXAMPLE USAGE
e = 1 #e.g. kappa = 5e-3
print('myopic: ', getRevenue(e,'myopic'))
print('linear: ', getRevenue(e,'linear'))
print('exponential: ', getRevenue(e,'exponential'))

