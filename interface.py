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
    algorithm = st.selectbox("Select Algorithm:", 
                              ("DBSCAN", "CLARANS", "Decision Tree", "Random Forest"))

    # Parameters for CLARANS
    if algorithm == "CLARANS":
        max_neighbors = st.number_input("Max Neighbors:", min_value=1, value=5)
        num_clusters = st.number_input("Number of Clusters:", min_value=1, value=3)

    # Parameters for DBSCAN
    elif algorithm == "DBSCAN":
        max_distance = st.number_input("Max Distance:", min_value=0.0, value=0.5, format="%.2f")
        min_samples = st.number_input("Min Samples:", min_value=1, value=5)

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
            """result , meds = Clarans(v1 , num_clusters , max_neighbors)
            st.dataframe(result)"""
            
        elif algorithm == "DBSCAN":
            st.write("Max Distance:", max_distance)
            st.write("Min Samples:", min_samples)
            """print(f"{max_distance} , {min_samples}")
            result = dbScan(v1 , max_distance , min_samples) 
            st.dataframe(result)"""

        st.write("Selected Dataset Option:", dataset_option)
        st.write("Selected Algorithm:", algorithm)

if __name__ == "__main__":
    main()