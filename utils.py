import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns 
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd


# Clustering part

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


# Regression part

# Decision tree

import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def calculate_mse(data, target):
    return ((data[target] - data[target].mean()) ** 2).mean()

def features_2_dict(df):
    return {feature: df.columns.get_loc(feature) for feature in df.columns}

def end_criterion(df, depth, max_depth):
    return len(df) < 2 or depth >= max_depth or len(df.columns) <= 1

def find_best_feature_and_threshold(df, target):
    best_feature = None
    best_threshold = None
    best_score = float("inf")


    features = [col for col in df.columns if col not in ["Qair_Winter", "Qair_Spring", "Qair_Summer", "Qair_Fall"]]
    print(f"looking for best feature among these: {features}")

    for feature in features:
        thresholds = df[feature].unique()
        thresholds.sort()

        for i in range(0, len(thresholds)-1):
            #threshold = (thresholds[i] + thresholds[i - 1]) / 2
            threshold = thresholds[i]

            left_data = df[df[feature] <= threshold]
            right_data = df[df[feature] > threshold]
            #print(f"\n left : {len(left_data)} \n write : {len(right_data)}")

            if len(left_data) == 0 or len(right_data) == 0:
                print(f" skip this feature: {feature}")
                continue

            mse_left = calculate_mse(left_data, target)
            mse_right = calculate_mse(right_data, target)
            score = (len(left_data) * mse_left + len(right_data) * mse_right) / len(df)

            if score < best_score:
                best_feature = feature
                best_threshold = threshold
                best_score = score
                #print(" Best So far updated!")
    
    print(f"\n\n Result: best feature: {best_feature} , best threshold: {best_threshold}\n Results comin from {len(df)} instances, and score of {best_score}\nResearch is done for this node\n")
    return best_feature, best_threshold, best_score

def split(df, feature, threshold):
    left_df = df[df[feature] <= threshold]
    right_df = df[df[feature] > threshold]
    return left_df, right_df

def leaf_value(df, target):
    print("created leaf")
    return df[target].mean()

def build_tree(df, target='Qair_Winter', depth=0, max_depth=200, min_impurity=2):
    if end_criterion(df, depth, max_depth):
        return Node(value=leaf_value(df, target))

    impurity = calculate_mse(df, target)
    best_feature, best_threshold, weighted_impurity = find_best_feature_and_threshold(df, target)

    if (weighted_impurity / impurity) * 100 < min_impurity:
        return Node(value=leaf_value(df, target))
    
    print(f"splitting on feature: {best_feature} with threshold = {best_threshold}")
    left_son, right_son = split(df, best_feature, best_threshold)
    print("affecting sons ...")
    return Node(
        feature=best_feature,
        threshold=best_threshold,
        left=build_tree(left_son, target, depth + 1, max_depth, min_impurity),
        right=build_tree(right_son, target, depth + 1, max_depth, min_impurity),
    )

def predict(root, line, features_indexes_dict):
    if root.left is None and root.right is None:
        return root.value

    feature_index = features_indexes_dict[root.feature]
    if line[feature_index] < root.threshold:
        return predict(root.left, line, features_indexes_dict)
    else:
        return predict(root.right, line, features_indexes_dict)

def fit(df, test_data_percent=20, target='Qair_Winter', max_depth=5, min_impurity=2):
    len_test = int(len(df) * test_data_percent / 100)
    train_data = df.iloc[:-len_test]
    test_data = df.iloc[-len_test:]
    print("creating root")
    root = build_tree(train_data, target, max_depth=max_depth, min_impurity=min_impurity)

    features_indexes = features_2_dict(df)
    predicted = [predict(root, line, features_indexes) for _, line in test_data.iterrows()]

    predicted = pd.Series(predicted, index=test_data.index)
    mse = ((test_data[target] - predicted) ** 2).mean()
    r_square = r2_score(test_data[target], predicted)
    mae = mean_absolute_error(test_data[target], predicted)
    rmse = root_mean_squared_error(test_data[target], predicted)

    return root, mse, r_square, mae, rmse

def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - (ss_res / ss_tot)

def mean_absolute_error(y_true, y_pred):
    return abs(y_true - y_pred).mean()

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())

# Main code:
#root, mse, r_square, mae, rmse = fit(v1, 20, 'Qair_Winter', max_depth=5)
#print(f"Evaluation Metrics:\n MSE: {mse}\n R^2: {r_square}\n MAE: {mae}\n RMSE: {rmse}")


# Random Forest

import numpy as np
import pandas as pd

def create_sample(df, max_features, target = 'Qair_Winter', bootstrap_percent = 60):
    """
    Create a random sample of the dataframe with specified features and bootstrap percentage.
    """
    sampled_features = np.random.choice(df.columns.difference([target]), max_features, replace=False)  # Exclude target
    sampled_df = df[list(sampled_features) + [target]].sample(
        frac=bootstrap_percent / 100, replace=True, random_state=None
    )
    return sampled_df

def Random_forest(df, nbr_trees=10, max_features= 20, bootstrap_percent=60,
                  test_data_percent=20, target='Qair_Winter', max_depth=5, min_impurity=2):
    """
    Build a random forest by training multiple decision trees on bootstrapped samples of the data.
    """
    forest = []
    errors = []

    for i in range(nbr_trees):
        # Create random sampling
        df_sample = create_sample(df, max_features, target, bootstrap_percent)
        print(f"Sample created with features: {len(df_sample.columns) - 1} // rows: {len(df_sample)}")

        # Train a decision tree
        tree, mse, _, _, _ = fit(df, test_data_percent, target, max_depth=max_depth, min_impurity=min_impurity)
        print(f"\n\nTree {i+1} successfully created, mse = {mse:.4f}\n\n")

        # Store the tree and its error
        forest.append(tree)
        errors.append(mse)

    print("Forest complete!")
    return forest, errors

def RF_predict(forest, line, features_indexes):
    """
    Predict the output for a single line using the Random Forest ensemble.
    """
    predictions = np.array([predict(tree, line, features_indexes) for tree in forest])
    return predictions.mean()

def train_test_sets(df, test_data_percent):
    """
    Split the data into train and test sets based on the specified percentage.
    """
    len_test = int(len(df) * test_data_percent / 100)
    train_data = df[:-len_test]
    test_data = df[-len_test:]
    return train_data, test_data

def evaluate_rf(forest, df, target, test_data_percent=20):
    """
    Evaluate the Random Forest model on the test data.
    """
    # Split data
    _, test_data = train_test_sets(df, test_data_percent)

    # Generate feature indexes
    features_indexes = features_2_dict(df)

    # Predict values for test data
    predicted = np.array([RF_predict(forest, line, features_indexes) for _, line in test_data.iterrows()])

    # Actual target values
    y_true = test_data[target].values

    # Calculate metrics
    mse = np.mean((y_true - predicted) ** 2)
    r2 = 1 - mse / np.var(y_true)
    # Calculate MAE
    mae = np.abs(y_true - predicted)
    
    # Calculate RMSE
    rmse = np.sqrt(mse)
    return mse, r2, mae, rmse

# Main code
#forest, mse_list = Random_forest(v1, nbr_trees=5, max_features=20, bootstrap_percent=80,
#                                 test_data_percent=30, target='Qair_Winter', max_depth=5, min_impurity=2)

#print("\n~~~~~~~~~~ Evaluation Part ~~~~~~~~~~\n")
#print(f"Mean of MSE values of all trees: {np.mean(mse_list):.4f}\n")

#mse, r2, mae, rmse = evaluate_rf(forest, v1, target='Qair_Winter', test_data_percent=20)
#print(f"Overall Random Forest Evaluation:\nMSE: {mse:.4f}\n R^2: {r2:.4f}\n MAE: {mae:.4f}\n RMSE: {rmse:.4f}")

# Single Prediction Test
#print("\n~~~~~~~~~~ Prediction Test ~~~~~~~~~~\n")
#line_as_df = v1.sample(1)
#print(line_as_df.columns)
#line = line_as_df.iloc[0]
#my_dict = features_2_dict(v1)
#prediction = RF_predict(forest, line, my_dict)

#print(f"Prediction: {prediction:.4f}\nActual: {line_as_df['Qair_Winter'].values[0]:.4f}\n")
