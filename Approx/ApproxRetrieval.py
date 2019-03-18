
# coding: utf-8

# In[1]:


import numpy as np
import random

movie_index = {}
movie_list =[]
user_index = {}
user_list = []

user_data_count = {}
no_user_list = {}
movie_size = 0


################PART 1#####################
#read in valid movie data and also write it to document

def movie_read(file):
    result = np.zeros((movie_size, len(user_data_count)), dtype=np.uint32)
    with open(file, 'r') as raw_data:
        m_id = 0
        u_id = 0
        movie = "NOTHING"
        for line in raw_data:
            data = line.split(',')
            if len(data) > 1:
                user = data[0]
                if user in user_data_count:
                    if user not in user_index:
                        user_index[user] = u_id
                        user_list.append(user)
                        u_id += 1
                    result[movie_index[movie]][user_index[user]] = int(1)
            else:
                movie = data[0][:-2]
                movie_index[movie] = m_id
                movie_list.append(movie)
                m_id +=1
                result[movie_index[movie]] = [int(0)] * (len(user_data_count))
#     with open('./user_hash', 'w') as user_data, open("./movie_hash", 'w') as movie_data,open("./filter_data", 'w') as all_data:
#         for u in user_index:
#             user_data.write(str(u) + "," + str(user_index[u]) + "\n")
#         for m in movie_index:
#             movie_data.write(str(m) + "," + str(movie_index[m]) + "\n")
#         for row in range(0,movie_size):
#             all_data.write(str(row) + "," + np.array_str(result[row]) + "\n")

    return result



    #     for line in raw_data:
    #         if len(line.split(',')) == 1:
    #
    #         else:
    #             data = line.split(',')

# read in file and get valid users
def user_read(file):
    ms = 0
    with open(file, 'r') as raw_data:
        for line in raw_data:
            data = line.split(',')
            if len(data) > 1:
                user = data[0]
                rating = int(data[1])
                if user not in no_user_list and rating >= 3:
                    if user not in user_data_count:
                        user_data_count[user] = 1
                    else:
                        user_data_count[user] = user_data_count[user] +1
                        if user_data_count[user] > 20:
                            del user_data_count[user]
                            no_user_list[user] = 1
            else:
                ms += 1
    raw_data.close()
    return ms





#Problem 1
movie_size = user_read("./Netflix_data.txt")
movie_data_matrix = movie_read("./Netflix_data.txt")
print(len(user_data_count))
print(movie_size)


# In[56]:


import matplotlib.pyplot as plt
import random
################PART 2#####################
def jaccard_distance(A, B, sig):
    if sig:
        intersect = (A!=B).sum()
        dist= intersect/A.size
    else:
        intersect = 0
    #         for i in range(A.size):
    #             if A[i] == 1 and B[i] == 1:
    #                 intersect += 1
        intersect = np.sum(np.bitwise_and(A,B))
        union = np.sum(A) + np.sum(B) - intersect
        dist = 1 - (intersect / union)
    return dist



#Problem 2
histo_data = []
for i in range(10000):
    random_sample = random.sample(range(len(user_data_count)), 2)
    i1 = random_sample[0]
    i2 = random_sample[1]
    dist = jaccard_distance(movie_data_matrix[:,i1],movie_data_matrix[:,i2], False)
    histo_data.append(dist)


# In[3]:


################PART 2#####################
#Problem 2 Cont
min_val = min(histo_data)
avg_val = sum(histo_data) / float(len(histo_data))

plt.hist(histo_data, bins=100)
# plt.xticks([.3,.35, .4, .45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1])
plt.xlabel("Jaccard Dist Values")
plt.ylabel("Freq")
plt.show
print(min_val)
print(avg_val)
print(len(histo_data))
plt.savefig("abc.png")


# In[4]:


################PART 3#####################
#Problem 3 Create data structure
from random import shuffle
N_HASHES = 700
PRIME = 8017
SIG_HASH = np.full((700,len(user_data_count)), np.inf)
MOVIE_COORDS = np.nonzero(movie_data_matrix)
INIT_HASHES = np.zeros((N_HASHES,2))
for i in range(N_HASHES):
    INIT_HASHES[i,0] = random.randint(0,PRIME)
    INIT_HASHES[i,1] =random.randint(0,PRIME)
def gen_perm(r_size):
    def min_hash(a,b,r,p):
        return (((a*r) + b) % p)
    print(MOVIE_COORDS[0].size)
    for rc in range(MOVIE_COORDS[0].size):
        if rc%100000 == 0:
            print(rc)
        r = MOVIE_COORDS[0][rc]
        c = MOVIE_COORDS[1][rc]
        r_hashes = np.full((1,N_HASHES), r)
        hashed_vector = ((INIT_HASHES[:,0]*r_hashes) + INIT_HASHES[:,1])%PRIME
        SIG_HASH[:, c] = np.minimum( hashed_vector,SIG_HASH[:, c]) 
            
gen_perm(movie_size)


# In[5]:


################PART 4#####################
#Problem 4 Generate Bands and Hashes
NUM_BANDS = 70
NUM_ROWS = 10
T = .65
HASHES =[]
BIG_PRIME = 8017
HASHED_RESULTS = np.zeros((NUM_BANDS*NUM_ROWS,len(user_data_count) ))
for i in range(NUM_ROWS):
    h = (random.randint(1,BIG_PRIME),random.randint(1,BIG_PRIME))
    HASHES.append(h)

for r in range(NUM_BANDS*NUM_ROWS):
    hash_var = HASHES[r%NUM_ROWS]
    HASHED_RESULTS[r] = ((hash_var[0]*SIG_HASH[r, :])+hash_var[1])%BIG_PRIME
FINAL_HASH = np.zeros(((NUM_BANDS), len(user_data_count)))
for i in range(NUM_BANDS):
    start = i*NUM_ROWS
    end = start + NUM_ROWS
    FINAL_HASH[i] = np.sum(HASHED_RESULTS[start:end, :], axis=0)

print(FINAL_HASH)


# In[6]:


################PART 4#####################
# #FORM POWER SETS AND REMOVE DUPLICATES
from itertools import combinations
FINAL_H = set([])
for i in range(NUM_BANDS):
    NEW_H = {}
    print("BAND: " + str(i) + " -- " +str(len(FINAL_H)))
#     Get Buckets
    for j in range(len(user_data_count)):
        k = FINAL_HASH[i,j]
        if k in NEW_H:
            NEW_H[k].append(j)
        else:
            NEW_H[k] = [j]
    
    for bucket in NEW_H:
        combos = combinations(NEW_H[bucket],2)
        for c in combos:
            if jaccard_distance(SIG_HASH[:,c[0]],SIG_HASH[:,c[1]], True) <= .35:
                FINAL_H.add(c)
                


# In[83]:


import csv
with open('similarPairs.csv','w') as writeFile:
    similarWriter = csv.writer(writeFile, delimiter=',')
    for tup in FINAL_H:
        similarWriter.writerow([tup[0], tup[1]])


# In[81]:


################PART 5#####################

def user_query(movie_index_list):
    closest_users = []
    current_min = [[0],np.inf]
    sig_user = np.full((1,N_HASHES), np.inf)
    for r in movie_index_list:
        r_hashes = np.full((1,N_HASHES), r)
        hashed_vector = ((INIT_HASHES[:,0]*r_hashes) + INIT_HASHES[:,1])%PRIME
        sig_user = np.minimum( hashed_vector,sig_user)   

    for u in range(len(user_data_count)):
        current_dist = jaccard_distance(SIG_HASH[:,u], sig_user,  True)
        if current_dist <= .35:
            closest_users.append(u)
        if current_dist < current_min[1]:
            current_min[0] = [u]
            current_min[1] = current_dist
        elif current_dist == current_min[1]:
            current_min[0].append(u)
                
                
    if len(closest_users) == 0:
        return current_min[0]
    return closest_users

