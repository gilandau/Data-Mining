
# coding: utf-8

# In[1]:


# LEC 19 - 22
import numpy as np
import csv
import random as r
header_mappings = dict([])
x_list = []
with open('yelp.csv', encoding = "utf-8") as csvfile:
    yelpreader = csv.reader(csvfile, delimiter=',')
    headers = next(yelpreader)
    i = 0
    for h in headers:
        header_mappings[h] = i
        i+=1
    i = 0
    for r in yelpreader:
        x_list.append(r)
X = np.array(x_list)
        
#     for row in yelpreader:
#         print(row['useful'])


# In[2]:


# Q1 - Formulate problem -- maybe PCA features
print(header_mappings)


# In[3]:


# Part 2


c_locs = []
batch_size = 10
c_num = 10
def mini_k_means(C1, b, X, T):
#     c - locations, b - batch, C - data matrix, T -iterations
# 
#     ITER
    C = C1
    t = 1
    change = True
    
    print(C.shape)
    
    
    assigned_clusters_prev = np.zeros(b)

    while change and t <= T:
        
        
        
        
        #         get minibatch
        batch = np.array(sorted(r.sample(range(0,X.shape[0]),b)))
        X_batch = X[batch, :]
        n  = 1.0/(t)
        
#         n = min(1, ) TODO
#        MAPS a data point x to a cluster index 
        # CALC CLOSEST CLUSER FOR minibatch
        diff_prenorm = X_batch[: , np.newaxis] - C
#         print(diff_prenorm)
#         print("SUBTRACTED")

        
        normed = np.linalg.norm(diff_prenorm,2, axis=2)**2
#         print(normed)
#         print("NORMED")

        assigned_clusters = np.argmin(normed, axis=1)

#         print(np.unique(assigned_clusters))        
#         print("UNIQUE IN LOOP")
#         print(assigned_clusters.shape)
#         print("ASSIGNED")

#         for idx, x in enumerate(X_batch):
#             min_dist_idx = int(np.argmin(np.linalg.norm(C-x,2, axis=1)**2))
#             assigned_clusters[idx]= min_dist_idx
            
            

        # UPDATE
        num_change = (assigned_clusters_prev - assigned_clusters).sum(0)
#         C =  np.array([C[k] + (X_batch[assigned_clusters==k] - C[k]) for k in range(C.shape[0])])
        
        for idx, x in enumerate(X_batch):
            ac = int(assigned_clusters[idx])
            C[ac] = C[ac] + n*(x-C[ac])
            
            
            
        
        
        
        t+=1
        assigned_clusters_prev = assigned_clusters
        if num_change == 0:
            print("TOTAL ITERS: " + str(t))
            change = False
    
    return C

# RANDOM START
# C = np.array(sorted(r.sample(range(0,X.shape[0]),c_num)))

    
    
    


# In[4]:


# Part 3
c = 5
s = 10
import random as r

def kpp_means(s, b, X, i):
#     s = number of clusters
#    b = batch size
# X = data
# i = number of iterations
#     First pick
    C = []
    current_c = X[r.randint(0,X.shape[0])]
    D2_1 = np.zeros((X.shape[0],1))
    D2_2 = np.zeros((X.shape[0],1))
    num_c = 1
    C.append(current_c)
    if (s > 1):
#     second pick
        diff_prenorm = X-current_c
#         print(diff_prenorm.shape)
#         print("PRENORMED")
        D2_1= np.linalg.norm(diff_prenorm,2, axis =1) ** 2
#         print(D2_1.shape)
#         print("NORMED FIRST d2")
    
    
        p_D2 = np.divide(D2_1,(D2_1.sum(0)))
#         print(p_D2.shape)
        ind = np.random.choice(X.shape[0],1, p= p_D2)[0]
#         print(ind)
#         print("INDEX")

        current_c = X[ind]
        num_c += 1
        C.append(current_c)
        D2_2= np.linalg.norm(X-current_c,2, axis =1) ** 2

# Pick more
        while num_c <s:
            D2 = np.minimum(D2_1, D2_2)
            D2_1 = D2_2
            D2_2 = D2
            p_D2 = D2/(D2.sum(0))
#             print(D2_2.shape)
#             print("NORMED FIRST d22")
#             print(p_D2.shape)

            ind = np.random.choice(X.shape[0],1, p= p_D2)[0]
#             print(ind)
#             print("INDEX")
            current_c = X[ind]
            C.append(current_c)
            num_c += 1
            D2_2= np.linalg.norm(X-current_c,2, axis=1) ** 2


    C_arr = np.array(C)
    return mini_k_means(C_arr,b,X,i)
# c_locs = kpp_means(s,b,X,i)
# batch_size = 30
# r = mini_k_means(c_locs, batch_size)


# In[5]:


# Q5
# {'user_id': 0, 'name': 1, 'review_count': 2,
# 'yelping_since': 3, 'useful': 4, 'funny': 5, 'cool': 6, 
# 'fans': 7, 'elite': 8, 'average_stars': 9,
# 'compliment_hot': 10, 'compliment_more': 11, 
# 'compliment_profile': 12, 'compliment_cute': 13,
# 'compliment_list': 14, 'compliment_note': 15,
# 'compliment_plain': 16, 'compliment_cool': 17,
#         'compliment_funny': 18, 'compliment_writer': 19,
#                 'compliment_photos': 20}
# 0       1     2   3    4     5    6     7      8     9     10   11     12    13          
# count, fans, hot, more, prof, cute, list, not, plain, cool, funny, writer, photo 
num_reviews = X[:, 2].astype(float)
X_data = X[:, [7, 10,12,17, 18, 19, 20]].astype(float)


# In[6]:


# Alter data
# compl_vect = sum(X_data[:, 5:], axis = 1)
# print(compl_vect.shape)
# a = X_data[:, 5:]
# b = X_data[:, 0]
# X_data[:, 5:] = np.divide(a,b)


# In[7]:


X_data[:, 0] = X_data[:,0 ].astype(float)/1000


# In[8]:


for i in range(1, 7):
    a = X_data[:, i].astype(float)
    b = num_reviews.astype(float)
    X_data[:, i] = np.divide(a, b, out=np.zeros_like(a), where=b!=0)

    
from sklearn.preprocessing import StandardScaler

new_X = StandardScaler().fit_transform(X_data)


from sklearn.decomposition import PCA
pca = PCA(n_components=4)
X_pca = pca.fit_transform(new_X)
print(X_pca.shape)
# X_pca = X_data


# In[9]:


pca.explained_variance_ratio_


# In[10]:


import random as r
# min, max, mean
k_vals = np.zeros((3,6))
t = 0

for i in [5, 10, 50, 100, 500]:
    C = X_pca[np.array(sorted(r.sample(range(0,X.shape[0]),i))), : ]
    C_kmeans = mini_k_means(C, 10000, X_pca, 200)
#         print(C_kmeans)
    print("K-means done")
        #         n = min(1, ) TODO
#        MAPS a data point x to a cluster index 
        # CALC CLOSEST CLUSER FOR minibatch
    diff_prenorm = X_pca[: , np.newaxis] - C_kmeans
#         print(diff_prenorm.shape)
#         print("SUBTRACTED")

        
    normed = np.linalg.norm(diff_prenorm,2, axis=2)
#         print(normed)
#         print("NORMED")

    assigned_clusters = np.argmin(normed, axis=1)
#         print(assigned_clusters)
#         print("ASSIGNED")
        
        
#         print(np.unique(assigned_clusters))        
#         print("UNIQUE")
    ac_values = C_kmeans[assigned_clusters]
#         print(ac_values)
#         print(np.unique(ac_values))        
#         print("UNIQUE")
    d_normed = ac_values - X_pca
    dist_matrix =  np.linalg.norm(d_normed ,2, axis = 1)

    k_vals[0,t]= np.min(dist_matrix)
    k_vals[1, t] = np.max(dist_matrix)
    k_vals[2,t] = np.mean(dist_matrix)
    print(k_vals[:,t])
    t+=1
        


# In[10]:


import random as r

k2_vals = np.zeros((3,6))
t= 0
for i in [5, 10, 50, 100, 500]:
    if i >= 4:
        C_KPP = kpp_means(i,10000,X_pca,200)
        diff_prenorm = X_pca[: , np.newaxis] - C_KPP
#         print(diff_prenorm.shape)
#         print("SUBTRACTED")

        
        normed = np.linalg.norm(diff_prenorm,2, axis=2)
#         print(normed)
#         print("NORMED")

        assigned_clusters = np.argmin(normed, axis=1)
#         print(assigned_clusters)
#         print("ASSIGNED")
        
        
#         print(np.unique(assigned_clusters))        
#         print("UNIQUE")
        ac_values = C_KPP[assigned_clusters]
#         print(ac_values)
#         print(np.unique(ac_values))        
#         print("UNIQUE")
        d_normed = ac_values - X_pca
        dist_matrix =  np.linalg.norm(d_normed ,2, axis = 1)

        k2_vals[0,t]= np.min(dist_matrix)
        k2_vals[1, t] = np.max(dist_matrix)
        k2_vals[2,t] = np.mean(dist_matrix)
        print(k2_vals[:,t])
        t+=1
        


#         k_min;
#         k_max;
#         k_mean = 0;
#         for x in X_data:
#             if k_min == None:
#                 k_min = min(np.linalg.norm(x-C_KPP,2)**2)
#             else:
#                 k_min = min(k_min, min(np.linalg.norm(x-C_KPP,2)**2))
                
#             if k_max == None:
#                 k_max = min(np.linalg.norm(x-C_KPP,2)**2)
#             else:
#                 k_max = max(k_min, min(np.linalg.norm(x-C_KPP,2)**2))
#             k_mean += min(np.linalg.norm(x-C_KPP,2)**2)
#         k_mean = k_mean/X_data.shape(0)
#         k2_vals[0,i]= k_min
#         k2_vals[1, i] = k_max
#         k2_vals[2,i] = k_mean


# In[10]:


# Part 4
c = 5
s = 10
def my_means(s, b, X, i):
#     s = number of clusters
#    b = batch size
# X = data
# i = number of iterations
#     First pick

# Start with the origin
    C = []
    
    D2_1 = np.zeros((X.shape[0],1))
    D2_2 = np.zeros((X.shape[0],1))
    num_c = 0 
#     second pick
    maxpc1 = max(X[:,0])
    maxpc2 = max(X[:,1])
    maxpc3 = max(X[:,2])
    maxpc4 = max(X[:,3])
    
    minpc1 = min(X[:,0])
    minpc2 = min(X[:,1])
    minpc3 = min(X[:,2])
    minpc4 = min(X[:,3])
    
    OG_C = []
    C.append([maxpc1, minpc2, minpc3, minpc4])
    OG_C.append([maxpc1, minpc2, minpc3, minpc4])
    num_c += 1
    
    C.append([minpc1, maxpc2, minpc3, minpc4])
    OG_C.append([minpc1, maxpc2, minpc3, minpc4])
    num_c += 1
    
    
    C.append([minpc1, minpc2, maxpc3, minpc4])
    OG_C.append([minpc1, minpc2, maxpc3, minpc4])
    num_c += 1
    
    
    C.append([minpc1, minpc2, minpc3, maxpc4])
    OG_C.append([minpc1, minpc2, minpc3, maxpc4])
    num_c += 1

        
#         6th choice

    current_c = X[r.randint(0,X.shape[0])]
    
    C.append(current_c)
    num_c += 1
    

# Pick more
    while num_c <s:
        current_og = 0
        current_max = 0
        og_count = 0
        for og in OG_C:
            d_to_og= np.linalg.norm(og-current_c,2) ** 2
            if d_to_og > current_max:
                current_max = d_to_og
                current_og = og_count
            og_count += 1
        og_endpoint = OG_C[current_og]
        current_c = abs(og_endpoint + current_c)/2
        OG_C.append(current_c)
        C.append(current_c)

        num_c += 1

        

    C_arr = np.array(C)
    return mini_k_means(C_arr,b,X,i)
# c_locs = kpp_means(s,b,X,i)
# batch_size = 30
# r = mini_k_means(c_locs, batch_size)


# In[11]:


k3_vals = np.zeros((3,6))
t= 0
for i in [5, 10, 50, 100, 500]:
    if i >= 4:
        C_KPP = my_means(i,10000,X_pca,200)
        diff_prenorm = X_pca[: , np.newaxis] - C_KPP
#         print(diff_prenorm.shape)
#         print("SUBTRACTED")

        
        normed = np.linalg.norm(diff_prenorm,2, axis=2)
#         print(normed)
#         print("NORMED")

        assigned_clusters = np.argmin(normed, axis=1)
#         print(assigned_clusters)
#         print("ASSIGNED")
        
        
#         print(np.unique(assigned_clusters))        
#         print("UNIQUE")
        ac_values = C_KPP[assigned_clusters]
#         print(ac_values)
#         print(np.unique(ac_values))        
#         print("UNIQUE")
        d_normed = ac_values - X_pca
        dist_matrix =  np.linalg.norm(d_normed ,2, axis = 1)

        k3_vals[0,t]= np.min(dist_matrix)
        k3_vals[1, t] = np.max(dist_matrix)
        k3_vals[2,t] = np.mean(dist_matrix)
        print(k3_vals[:,t])
        t+=1
        


#         k_min;
#         k_max;
#         k_mean = 0;
#         for x in X_data:
#             if k_min == None:
#                 k_min = min(np.linalg.norm(x-C_KPP,2)**2)
#             else:
#                 k_min = min(k_min, min(np.linalg.norm(x-C_KPP,2)**2))
                
#             if k_max == None:
#                 k_max = min(np.linalg.norm(x-C_KPP,2)**2)
#             else:
#                 k_max = max(k_min, min(np.linalg.norm(x-C_KPP,2)**2))
#             k_mean += min(np.linalg.norm(x-C_KPP,2)**2)
#         k_mean = k_mean/X_data.shape(0)
#         k2_vals[0,i]= k_min
#         k2_vals[1, i] = k_max
#         k2_vals[2,i] = k_mean

