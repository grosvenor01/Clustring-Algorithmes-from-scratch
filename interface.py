import streamlit as st

# Streamlit Interface
st.title("Decision Tree & Random Forest Interface")

# Dropdown menu for algorithm selection
algorithm = st.selectbox(
    "Select an Algorithm:",
    ("Decision Tree", "Random Forest")
)

# Display the selected option
st.write(f"You selected: {algorithm}")

# Decision Tree Input Fields
if algorithm == "Decision Tree":
    st.write("Enter parameters for the Decision Tree:")
    
    test_data_percent = st.number_input(
        "Test Data Percentage:",
        min_value=1,
        max_value=99,
        value=20,
        step=1
    )
    
    target = st.text_input(
        "Target Column:",
        value="Rainf1"
    )
    
    max_depth = st.number_input(
        "Maximum Depth:",
        min_value=1,
        value=5,
        step=1
    )
    
    min_impurity = st.number_input(
        "Minimum Impurity Reduction Percentage:",
        min_value=0,
        value=2,
        step=1
    )
    
    # Display inputs
    st.write("### Parameters Summary:")
    st.write(f"- Test Data Percentage: {test_data_percent}%")
    st.write(f"- Target Column: {target}")
    st.write(f"- Maximum Depth: {max_depth}")
    st.write(f"- Minimum Impurity: {min_impurity}")
    
    # Placeholder for next action
    if st.button("Run Decision Tree"):
        st.write("Running Decision Tree with the above parameters...")
        # Call your decision tree function here
        # e.g., result = run_decision_tree(df, test_data_percent, target, max_depth, min_impurity)

# Random Forest placeholder (you can implement similar fields later)
else:
    st.write("Random Forest setup is under construction!")
