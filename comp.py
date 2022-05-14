import numpy as np

INIT_CAPS = [30,20,10]
KAPPAS = [0.5, 5e-3, 5e-5]
RETURN_TIMES = ['flat', 'right', 'midpeak']
pots = ['myopic','linear','exponential']
names = ['high','medium','low'] # for the kappa values

for i in range(3):
    for r in range(3):
        for e in range(3):
            for p in pots:
                tmp1 = np.load('revenues'+str(INIT_CAPS[i])+str(r)+names[e]+p+'.npy',getRevenue(e,p))
                tmp2 = np.load('revenues'+str(INIT_CAPS[i])+str(e)+names[e]+'ub'+'.npy',getRevenue(e,p))
                print("Capacity: " + str(INIT_CAPS[i]) + ', Return dist: '+ str(RETURN_TIMES[r]) + ', Kappa: ' + names[e]+ ', Potential: ' + p)
                print(np.mean(np.divide(tmp1, tmp2)))