import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from PIL import Image
from google.oauth2 import service_account
from gsheetsdb import connect

####################
#USER AUTHENTICATION
####################


#adding a login page
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["login"]["LOGIN_PASSWORD"]: #error point to this line!
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.header("Employee Retention Dashboard")
        image = Image.open('invokeanalytics_logo.png')
        st.image(image)
        st.text_input(
        "Please enter your password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Please enter your password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False

    else:
        # Password correct.
        return True


#################
#Load model
###############

if check_password():

    #Load the saved model (V5)
    model=pkl.load(open("modelERv5.pkl","rb"))

    st.set_page_config(
        page_title="Employe Retention Prediction App",
        page_icon="invoke_logo.jpg"

    )

    st.set_option('deprecation.showPyplotGlobalUse', False)


    ######################
    #main page layout
    ######################
    st.image("invoke_logo.jpg")

    st.title("Employee Retention Prediction")

    ######################
    #sidebar layout
    ######################


    st.sidebar.title("Employee Info")
    st.sidebar.image("er.webp")
    st.sidebar.write("Please choose department")

   
    ###########################
    #Importing data from gsheet
    ###########################

    from gsheetsdb import connect
    from google.oauth2 import service_account
    import streamlit as st
    import pandas as pd

    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
            ],
        )

    conn = connect(credentials=credentials)
        
    def run_query(query):
        rows = conn.execute(query, headers=1)
        rows = rows.fetchall()
        return rows

    sheet_url = st.secrets["private_gsheets_url"]
    rows = run_query(f'SELECT * FROM "{sheet_url}"')
    df_input = pd.DataFrame(rows)

    #########
    # Output
    ##########

    # st.dataframe(df_input)
    
    # Filter to only display employee who are still working in the company
    df_input_filtered = df_input[df_input['Month_of_resignation'].isnull()]
    
    #Filter based on department selection
    sorted_unique_dept = sorted(df_input_filtered.dept.unique())
    selected_dept = st.sidebar.multiselect('Department', sorted_unique_dept, sorted_unique_dept)
    df_selected_dept = df_input_filtered[(df_input_filtered.dept.isin(selected_dept))]

    #Display only code name, dept and predicted probability
    df_predictproba = df_selected_dept[['code_name', 'dept', 'Probability_of_Resigning']]

    # Color the rows red for predict_proba >.50 (employee high likely to resign)
    styled_df = df_predictproba.style.applymap(lambda x: 'background-color: red' if x > 0.50 else 'background-color: green', subset=['Probability_of_Resigning'])
    st.write("""
             **Quick Insights: Red and Green Highlights**

            In this table, you'll see numbers highlighted in red and green. 

            - **Green**: High chance the employee will stay.
            - **Red**: Higher chance the employee might leave.

            **Numbers**: They represent the estimated chance of leaving or staying. For example, 0.30 means 30% chance of leaving.


             """)
    st.dataframe(styled_df, hide_index= True)




    