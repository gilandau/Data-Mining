
# coding: utf-8

# In[118]:


# Import data, lec 21 - 25

import numpy as np
import csv
import random as r
header_mappings = dict([])
x_list = []
with open('yahoo_ad_clicks.csv', encoding = "utf-8") as csvfile:
    yelpreader = csv.reader(csvfile, delimiter=',')
    i = 0
    for r in yelpreader:
        x_list.append(r)
X = np.array(x_list).astype(int)
        
#     for row in yelpreader:
#         print(row['useful'])


# In[119]:


X.shape


# In[120]:


# Part 2 - Implementation of partial-feedback - EXP3

import random as r
import matplotlib.pyplot as plt
import numpy as np

# initial regret matrix
total_reward = np.full(X.shape[0], 0)

my_rewards = np.zeros(X.shape[1])

est_regret = np.zeros(X.shape[1])
best_est_regret = np.zeros(X.shape[1])

aavg_total_regret = np.zeros(X.shape[1])
abest_avg_total_regret = np.zeros(X.shape[1])

best_arm = np.argmax(np.sum(X[:, :], axis = 1))
eta = np.sqrt((2*np.log(X.shape[0]))/(X.shape[1] * X.shape[0]))

# p =
# for each round
for i in range(X.shape[1]):
#     Set distribution
    pd = np.exp(eta * total_reward) / np.sum(np.exp(eta * total_reward))
#     print(pd)

    if(i%1000 == 0):
        print(i)
#     pick random arm from distribution
    arm = np.random.choice(X.shape[0],1, p= pd)[0]
    
    obs_reward = X[arm, i]
    my_rewards[i] = obs_reward
    total_reward = total_reward + 1
    total_reward[arm] = total_reward[arm] -  (((1-obs_reward)/pd[arm]))
    
#     Calc regret
#     if np.count_nonzero(X[:,i]) > 0:
#         optimal_rewards[i] = 1
    opt_reward = max(np.sum(X[:, :i+1], axis = 1))
#     print("OPT")
#     print(opt_reward)
    est_regret[i] = opt_reward - sum(my_rewards[:i+1])
    best_est_regret[i] = sum(X[best_arm, :i+1]) - sum(my_rewards[:i+1])
#     print("EST")
#     print(est_regret[i])
    aavg_total_regret[i] = est_regret[i]/(i+1)
    abest_avg_total_regret[i] = best_est_regret[i]/(i+1)
#     print("AVG")
#     print(avg_total_regret[i])

    


# In[121]:


avg_total_regret


# In[122]:


print(opt_reward)
print(sum(X[best_arm, :]))
print(sum(my_rewards))
print(est_regret)


# In[123]:


# Part 3 - Implementation of full-feedback - EXP3

import random as r
import matplotlib.pyplot as plt
import numpy as np


import random as r
import matplotlib.pyplot as plt
import numpy as np

# initial regret matrix
total_reward = np.full(X.shape[0], 0)

my_rewards = np.zeros(X.shape[1])

est_regret = np.zeros(X.shape[1])
best_est_regret = np.zeros(X.shape[1])

bavg_total_regret = np.zeros(X.shape[1])
bbest_avg_total_regret = np.zeros(X.shape[1])
best_arm = np.argmax(np.sum(X[:, :], axis = 1))
eta = np.sqrt((2*np.log(X.shape[0]))/(X.shape[1] * X.shape[0]))

# p =
# for each round
for i in range(X.shape[1]):
#     Set distribution
    pd = np.exp(eta * total_reward) / np.sum(np.exp(eta * total_reward))
#     print(pd)

    if(i%1000 == 0):
        print(i)
#     pick random arm from distribution
    arm = np.random.choice(X.shape[0],1, p= pd)[0]
    
    obs_reward = X[arm, i]
    my_rewards[i] = obs_reward
    total_reward = total_reward + X[:,i]

#     total_reward[arm] = total_reward[arm] -  (((1-obs_reward)/pd[arm]))
    
#     Calc regret
#     if np.count_nonzero(X[:,i]) > 0:
#         optimal_rewards[i] = 1
    opt_reward = max(np.sum(X[:, :i+1], axis = 1))
#     print("OPT")
#     print(opt_reward)
    est_regret[i] = opt_reward - sum(my_rewards[:i+1])
    best_est_regret[i] = sum(X[best_arm, :i+1]) - sum(my_rewards[:i+1])
#     print("EST")
#     print(est_regret[i])
    bavg_total_regret[i] = est_regret[i]/(i+1)
    bbest_avg_total_regret[i] = best_est_regret[i]/(i+1)
#     print("AVG")
#     print(avg_total_regret[i])


# In[129]:


plt.plot(aavg_total_regret, label = "roaming average regret, partial")
plt.plot(abest_avg_total_regret,  label = "optimal arm average regret, partial")
plt.plot(bavg_total_regret, label = "roaming average regret, full")
plt.plot(bbest_avg_total_regret, label = "optimal arm average regret, full")
plt.legend(loc='upper right')
plt.xlabel('Rounds')
plt.ylabel('Average Regret')
plt.savefig('ese545p4plot.png')


# In[125]:


print(avg_total_regret)
print(best_avg_total_regret)


# In[134]:


aavg_total_regret[10000:]


# In[135]:


abest_avg_total_regret[10000:]


# In[136]:


bavg_total_regret[10000:]


# In[137]:


bbest_avg_total_regret[10000:]

