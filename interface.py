import streamlit as st
import pandas as pd
from utils import *



if 'root' not in st.session_state:
    dataset = pd.read_csv("final1.csv")
    st.session_state.root , st.session_state.mse ,st.session_state.r_square , st.session_state.mae , st.session_state.rmse =  fit(dataset, 20, "Qair_Winter", 5)
    st.session_state.forest, st.session_state.mse_list = Random_forest(dataset,8, 20, 80,test_data_percent=20, target="Qair_Winter", max_depth=5, min_impurity=2)
    st.session_state.Fmse, st.session_state.Fr2, st.session_state.Fmae, st.session_state.Frmse = evaluate_rf(st.session_state.forest, dataset, target='Qair_Winter', test_data_percent=20)

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
        index_instance = st.number_input("Instance to test", min_value=1,max_value=589 ,value=5)

    # Parameters for Random Forest
    elif algorithm == "Random Forest":
        index_instance = st.number_input("Instance to test", min_value=1,max_value=589 ,value=5)
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
            dataset = pd.read_csv("final1.csv")
            st.dataframe(dataset)
            #root, mse, r_square, mae, rmse = fit(dataset, test_data_percent, target, max_depth)
            mse = st.session_state.mse
            r_square = st.session_state.r_square
            mae = st.session_state.mae
            rmse = st.session_state.rmse
            st.markdown(
                f"### Evaluation Metrics\n"
                f"- **Mean Squared Error (MSE)**: {mse}\n"
                f"- **R²**: {r_square}\n"
                f"- **Mean Absolute Error (MAE)**: {mae}\n"
                f"- **Root Mean Squared Error (RMSE)**: {rmse}"
            )

            line = dataset.iloc[index_instance]
            line = list(line)
            dictionary ={}
            for index , i in enumerate(dataset):
                dictionary[i]= index
            value =  predict(st.session_state.root ,line , dictionary)
            st.write(f"the prediction result for the sample : {index_instance} is : {value}")

        elif algorithm == "Random Forest":
            st.write("\n~~~~~~~~~~ Prediction Test ~~~~~~~~~~\n")
            dataset = pd.read_csv("final1.csv")
            st.dataframe(dataset)

            mse = st.session_state.Fmse
            r_square = st.session_state.Fr2
            mae = st.session_state.Fmae
            rmse = st.session_state.Frmse
            st.markdown(
                f"### Evaluation Metrics\n"
                f"- **Mean Squared Error (MSE)**: {mse}\n"
                f"- **R²**: {r_square}\n"
                f"- **Mean Absolute Error (MAE)**: {mae}\n"
                f"- **Root Mean Squared Error (RMSE)**: {rmse}"
            )

            line = dataset.iloc[index_instance]
            line = list(line)
            dictionary ={}
            for index , i in enumerate(dataset):
                dictionary[i]= index
            value =  RF_predict(st.session_state.forest ,line , dictionary)
            st.write(f"the prediction result for the sample : {index_instance} is : {value}")
        st.write("Selected Dataset Option:", dataset_option)
        st.write("Selected Algorithm:", algorithm)

if __name__ == "__main__":
    main()