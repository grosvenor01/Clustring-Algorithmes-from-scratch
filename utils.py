import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns 
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd

def hamming(arr1, arr2):
    return np.sum(arr1 != arr2)

def manhattan(atr1, atr2):
    return np.abs(atr1 - atr2)

def distance(row, medoid):
    dist = 0
    for col in row.index:
        if col not in ["cluster", "distance"]:
            if row[col].dtype == "object":
                dist += (row[col] != medoid[col])
            else:
                dist += np.abs(row[col] - medoid[col])
    return dist

def affect_clusters(df, medoids):
    distances = np.zeros((len(df), len(medoids)))
    for j in range(len(medoids)):
        distances[:, j] = df.apply(lambda row: distance(row, medoids.iloc[j]), axis=1)
    clusters = np.argmin(distances, axis=1)
    return clusters

def global_distances(df, medoids):
    df['distance'] = np.zeros(len(df))
    for i in range(len(df)):
        cluster_index = int(df.iloc[i]["cluster"])
        if cluster_index < len(medoids):
            df.at[i, "distance"] = distance(df.iloc[i], medoids.iloc[cluster_index])
        else:
            print(f"Warning: cluster index {cluster_index} is out of bounds for medoids.")
    return df

def medoids_change_test(df, max_neighbor, nbr_clusters, medoids):
    for _ in range(max_neighbor): 
        for j in range(nbr_clusters):
            df_test = df[df["cluster"] == j]
            current_distance = df_test["distance"].sum()
            #print(f"Cluster {j} ---> Current distance: {current_distance}")

            # randomly select another medoid
            if not df_test.empty: 
                med = df_test.sample(n=1).reset_index(drop=True)
                new_meds = medoids.copy()
                new_meds.loc[j] = med.iloc[0]

                # calculate distances with new medoid
                df_test_updated = global_distances(df_test, new_meds)
                second_distance = df_test_updated["distance"].sum()

                #print(f"New medoid for cluster {j}: {new_meds.iloc[j]}")
                print(f"Second distance ---> {second_distance}")

                if second_distance < current_distance: 
                    medoids.loc[j] = new_meds.loc[j]
                    #print(f"Medoid for cluster {j} updated.")

    return medoids

def Clarans(v1, nbr_clusters, max_neighbor, numlocal):
    loop = 0
    min_distance = float("inf")
    global_meds = None
    df = v1.copy()

    while loop < numlocal: 
        # Randomly select medoids
        medoids = df.sample(n=nbr_clusters).reset_index(drop=True)
        print("Medoids are set")

        # Calculate distances and affect cluster number
        df["cluster"] = affect_clusters(df, medoids)
        print("Clusters are set")
        
        df = global_distances(df, medoids)
        print("Global distances added")

        # Update medoids based on the distance change test
        medoids = medoids_change_test(df, max_neighbor, nbr_clusters, medoids)

        current_distance = df["distance"].sum()
        if current_distance < min_distance: 
            print("Meds changed")
            global_meds = medoids
            min_distance = current_distance  # Update minimum distance

        print("Done")
        loop += 1

    return df, global_meds

def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def define_core_points(v1, max_ngbr, circle_size):
    result=v1.copy()
    df = v1.values
    all_neighbors = {}
    
    for i in range(len(df)):
        neighbors = []
        for j in range(len(df)):
            if i != j:
                dist = calculate_distance(df[i], df[j])
                if dist <= circle_size:
                    neighbors.append(j)
        
        if max_ngbr is not None:
            neighbors = neighbors[:]

        all_neighbors[i] = neighbors
        if len(all_neighbors[i]) > max_ngbr : 
            result.loc[i , "is_core"] = True
        else : 
            result.loc[i , "is_core"] = False
    return all_neighbors , result

def clustering(df , neighbor_matrix):
    nbr_cluster = 0
    clusters = df.copy()
    not_core=[]
    while True:
        try :
            test = clusters[pd.isna(clusters["cluster"])]
            if len(test[test["is_core"]==False])==len(test):
                break
        except Exception as e:
            test = clusters.copy()
        # randomly select a core point
        while True:
            point = test.sample(n=1)
            point = point[point["is_core"]==True]
            if len(point) == 1:
                break
        index = df.index[point.iloc[0].name]
        stack = [index]
        # set of core points najoutiw el false point w nkhrjou les voisin ta3 other core points
        while len(stack) > 0:
            print(len(stack))
            index = stack.pop()
            clusters.loc[index, "cluster"] = nbr_cluster
            
            for i in neighbor_matrix[index]:
                if clusters.iloc[i]["is_core"] == True and pd.isna(clusters.iloc[i]["cluster"]):
                    stack.append(i)

                elif clusters.iloc[i]["is_core"] == False and i not in not_core:
                    not_core.append(i)
        for j in not_core:
            clusters.loc[j, "cluster"] = nbr_cluster
        
        nbr_cluster+=1
    return clusters

def dbScan(v1,circle_size , max_ngbr):
    # define core points (points li andhom niehgbors kter man max ngbr) , neighbore m3ntha distance < l9otr ta3 da2ira
    # nselectioniw core point wahed w ndkhlouh f cluster apres nseclctioniw neighbore ta3ou w njoutiwhom l cluster (lazem ykounou core points )
    # ki nkmlou ga3 li f cluster wahed nsotiw l non core point li jazna 3lihom w ndkhlohom f hadak el cluster b chart ykounou neihbore ta3 core points man hadak el cluster 
    # le faite li nkmlou n3wdou nkhyrou random core point whdokher w najoutiwah l cluster wahdokher ... etc 
    # repeter juesqu'a ykhlassou ga3 
    neighbor_matrix , df= define_core_points(v1 , max_ngbr , circle_size)
    df["cluster"] = pd.NA
    df = clustering(df , neighbor_matrix) 
    return df





