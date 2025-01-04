import streamlit as st
import pandas as pd
from utils import *

def main():
    st.title("Algorithm Selection Interface")

    # Option to choose dataset
    dataset_option = st.radio("Choose Dataset:", ("Use Current Dataset", "Upload New Dataset"))

    if dataset_option == "Upload New Dataset":
        uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    # Algorithm selection
    algorithm = st.selectbox("Select Algorithm:", ("DBSCAN", "CLARANS", "Decision Tree", "Random Forest"))

    # Parameters for CLARANS
    if algorithm == "CLARANS":
        max_neighbors = st.number_input("Max Neighbors:", min_value=1, value=5)
        num_clusters = st.number_input("Number of Clusters:", min_value=1, value=3)
        num_local = st.number_input("Number of iterations:", min_value=1, value=1)

    # Parameters for DBSCAN
    elif algorithm == "DBSCAN":
        max_distance = st.number_input("Max Distance:", min_value=0.0, value=0.5, format="%.2f")
        min_samples = st.number_input("Min Samples:", min_value=1, value=5)
    
    # Parameters for Decision Tree
    elif algorithm == "Decision Tree":
        target = st.selectbox("Select feature to predict:",
                               ("Qair_Winter", "Qair_Spring", "Qair_Summer", "Qair_Fall"))
        max_depth = st.number_input("Max Depth Of the tree:", min_value=1, value=5)
        test_data_percent = st.number_input("percentage of test data from initial dataset", min_value=1, value=20)

    # Parameters for Random Forest
    elif algorithm == "Random Forest":
        target = st.selectbox("Select feature to predict:",
                              ("Qair_Winter", "Qair_Spring", "Qair_Summer", "Qair_Fall"))
        # Tree data
        max_depth = st.number_input("Max Depth Of the tree:", min_value=1, value=5)
        test_data_percent = st.number_input("percentage of test data from initial dataset", min_value=1, value=2)
        # Forest data
        nbr_trees = st.number_input("Number of Trees:", min_value=1, value=5)
        max_features = st.number_input("nombre de colonnes par sample:", min_value=1, value=20)
        bootstrap_percent = st.number_input("nombre de lignes par sample:", min_value=1, value=80)
    # Button to process the selection
    if st.button("Run Algorithm"):
        st.success("Algorithm has been run with the selected parameters!")
        if dataset_option == "Use Current Dataset":
            v1 = pd.read_csv("final.csv")
        else : 
            v1 = pd.read_csv(uploaded_file)


        if algorithm == "CLARANS":
            st.write("Max Neighbors:", max_neighbors)
            st.write("Number of Clusters:", num_clusters)
            result , meds = Clarans(v1 , num_clusters , max_neighbors , num_local)
            st.dataframe(result)
            st.pyplot(plot_pca_clusters(result))
            
        elif algorithm == "DBSCAN":
            st.write("Max Distance:", max_distance)
            st.write("Min Samples:", min_samples)
            print(f"{max_distance} , {min_samples}")
            result = pd.read_csv(f"predictions/dbScan{max_distance}_{min_samples}.csv")
            #result = dbScan(v1 , max_distance , min_samples) 
            st.dataframe(result)
            st.pyplot(plot_pca_clusters(result))
        
        elif algorithm == "Decision Tree":
            root, mse, r_square, mae, rmse = fit(v1, test_data_percent, target, max_depth)
            st.write(f"Evaluation Metrics:\n MSE: {mse}\n R^2: {r_square}\n MAE: {mae}\n RMSE: {rmse}")

        elif algorithm == "Random Forest":
            forest, mse_list = Random_forest(v1, nbr_trees, max_features, bootstrap_percent,
                                  test_data_percent=30, target=target, max_depth=5, min_impurity=2)

            st.write("\n~~~~~~~~~~ Evaluation Part ~~~~~~~~~~\n")
            st.write(f"Mean of MSE values of all trees: {np.mean(mse_list):.4f}\n")
            mse, r2, mae, rmse = evaluate_rf(forest, v1, target='Qair_Winter', test_data_percent=20)
            print(f"Overall Random Forest Evaluation:\nMSE: {mse:.4f}\n R^2: {r2:.4f}\n MAE: {mae:.4f}\n RMSE: {rmse:.4f}")
            st.write(f"Evaluation Metrics:\n MSE: {mse}\n R^2: {r2}\n MAE: {mae}\n RMSE: {rmse}")

            st.write("\n~~~~~~~~~~ Prediction Test ~~~~~~~~~~\n")


        st.write("Selected Dataset Option:", dataset_option)
        st.write("Selected Algorithm:", algorithm)

if __name__ == "__main__":
    main()