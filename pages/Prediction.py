import streamlit as st
import pandas as pd
import base64
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle as pkl
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit.components.v1 as components
from PIL import Image
from google.oauth2 import service_account
from gsheetsdb import connect
import gspread

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
    st.sidebar.write("Please choose parameters that descibe the employee")

    #input features
    def get_user_input():

        dept = st.sidebar.selectbox("Department: ", ("ANALYTICS", "CALL CENTRE", "COMMERCIAL", "CORPORATE SERVICES", "CREATIVE", "DIGITAL MARKETING", "FINANCE", "PRODUCTS", "SYSTEM", "TALENT MANAGEMENT"))
        age_group =st.sidebar.selectbox("Age Group:",("20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59") )
        postdic =st.sidebar.selectbox("PostDic Score:",("1.1 - 2.9 [Req Imp]", "3.0 - 3.9 [Good Work Attitude]", "Above 4.0 [Positive Influence]", "NOT APPLICABLE"), index = 1)
        promoted =st.sidebar.radio("Promoted within the same year? :", ("Yes", "No"), index = 1)
        travel =st.sidebar.radio("Business Travel? :", ("Yes", "No"))
        distance=st.sidebar.slider("Distance from Home (km):",min_value=1, max_value=100,step=1, value = 15)
        marital =st.sidebar.selectbox("Marital Status :",("Single", "Married", "Divorce", "Widow") )
        edu =st.sidebar.selectbox("Education Level :",("High School", "STP", "Certificate", "Sijil", "Diploma", "Degree", "Masters", "PhD", "Professional Certificate"), index = 5)
        grad =st.sidebar.radio("Local or Overseas Grad? :",("Local", "Overseas") )
        age_joined = st.sidebar.number_input("Age Joined:", min_value=20, max_value=70, step=1, value=23, format="%d")
        age_left = st.sidebar.number_input("Current Age:", min_value=20, max_value=70, step=1, value=24, format="%d")

        ###logic for age
        if age_left >= age_joined:
                st.success("Success: Age accepted.")
        else:
                st.error("Error: Current age must be more than or equal to the age of joining.")

        duration=st.sidebar.number_input("Duration of Employment (months):",min_value=0, max_value=100,step=1, value = 12, format = "%d")
        salary=st.sidebar.slider("Monthly Salary (RM):",min_value=0, max_value=10000,step=50, value = 3000)
        experience=st.sidebar.number_input("Years of Experience:",min_value=0, max_value=30,step=1, value = 1, format = "%d")

        user_input_dict={"dept" : [dept], "age_group":[age_group],"postdic": [postdic],"promoted": [promoted],"travel": [travel],
                        "distance": [distance],"marital": [marital],"edu": [edu],"grad": [grad], 
                        "age_joined":[age_joined],"age_left": [age_left],"duration": [duration],"salary": [salary],"experience": [experience]}
        
        return user_input_dict
    

    #########################
    #Data feature engineering
    #########################


    def preprocess(user_input:pd.DataFrame):

        cleaner_type = {"dept" : {"ANALYTICS" : 0, "CALL CENTRE" : 1, "COMMERCIAL" : 2,
                                "CORPORATE SERVICES" : 3, "CREATIVE" : 4, "DIGITAL MARKETING" : 5,
                                "FINANCE" : 6, "PRODUCTS" : 7, "SYSTEM" : 8, "TALENT MANAGEMENT" : 9},
                        "age_group": {'20-24' : 0, '25-29' : 1, '30-34' : 2, '35-39' : 3, '40-44' : 4,  '45-49' : 5, '50-54': 6, '55-59' : 7},
                        "postdic": {'NOT APPLICABLE': 0, '1.1 - 2.9 [Req Imp]': 1, '3.0 - 3.9 [Good Work Attitude]':2, 'Above 4.0 [Positive Influence]':3},
                        "promoted" : { 'Yes': 1, 'No': 0},
                        "travel" : {'Yes': 1, 'No': 0},
                        "marital" : {'Single': 0, 'Married': 1, 'Divorce':2, 'Widow': 3},
                        "edu" : {'High School': 0, 'STP': 1, 'Certificate': 2, 'Sijil' : 3, 'Diploma': 4, 'Degree': 5, 'Masters': 6, 'PhD': 7, 'Professional Certificate': 8},
                        "grad" : {'Local': 0, 'Overseas':1}
                        }

        for col in user_input.columns:
            if col in cleaner_type:
                user_input[col] = user_input[col].map(cleaner_type[col])

        return user_input

    
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

    ####################
    #SHAP explainability
    ####################
    
    def generate_shap_waterfall_plot(df, user_input):

        #encoding categorical varible
        department_map = {'FINANCE': 0, 'DIGITAL MARKETING': 1, 'CREATIVE': 2, 'CALL CENTRE': 3,
                        'CORPORATE SERVICES': 4, 'COMMERCIAL': 5, 'SYSTEM': 6, 'ANALYTICS': 7,
                        'PRODUCTS': 8, 'TALENT MANAGEMENT': 9}

        postdic_map = {'NOT APPLICABLE': 0, '1.1 - 2.9 [Req Imp]': 1, '3.0 - 3.9 [Good Work Attitude]':2, 'Above 4.0 [Positive Influence]':3}

        business_travel_map = {'YES': 1, 'NO': 0}

        married_map = {'Married': 0, 'Widow': 1, 'Single':2, 'Divorce':3}

        target_map = {'stay': 0, 'left':1}

        promotion_map = {'NO': 0, 'YES':1}

        age_map = {'30-34' : 2, '35-39' : 3, '55-59' : 6, '25-29' : 1, '45-49' : 5, '40-44' : 4,
            '20-24' : 0}

        edu_map = {'HS': 0, 'STP': 1, 'Cert': 2, 'Sijil' : 3, 'Dip': 4, 'Dg': 5, 'M': 6, 'P': 7, 'PC': 8}

        localoverseas_map = {'L': 0, 'O':1}

        # Map values to columns using the map() function
        df['dept'] = df['dept'].map(department_map)
        df['POSTDIC_SCORE_2021'] = df['POSTDIC_SCORE_2021'].map(postdic_map)
        df['postdic'] = df['postdic'].map(postdic_map)
        df['travel'] = df['travel'].map(business_travel_map)
        df['marital'] = df['marital'].map(married_map)
        df['Status'] =df['Status'].map(target_map)
        df['promoted'] =df['promoted'].map(promotion_map)
        df['age_group'] =df['age_group'].map(age_map)
        df['edu'] =df['edu'].map(edu_map)
        df['grad'] =df['grad'].map(localoverseas_map)

        df['distance'] = df['distance'].astype(int)
        df['age_left'] = df['age_left'].astype(int)
        df['salary'] = df['salary'].astype(int)
        df['experience'] = df['experience'].astype(int)

        X = df.drop(columns = ['code_name', 'Month_of_resignation', 'POSTDIC_SCORE_2021' ,'Status_', 'Status', 'Probability_of_Resigning', 'Predicted_Class'], axis = 1)
        y = df[['Status']] 
        # y_ravel = y.values.ravel()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        shap.initjs()
        vectorizer = TfidfVectorizer(min_df=1)
        X_train_v = vectorizer.fit_transform(X_train)
        X_test_v = vectorizer.transform(user_input)

        explainer = shap.KernelExplainer(model.predict, X_train)
        shap_values = explainer(user_input)
        fig = shap.plots.waterfall(shap_values[0])

        return fig

    ############
    #Call def
    ############

    #get the data from get_user_input def
    data = get_user_input()
    user_input = pd.DataFrame(data)

    retention_predict = st.sidebar.button("Predict")

    #display chosen input
    st.subheader(""" Input chosen:
    """)
    st.dataframe(user_input, height = 70, hide_index = True)
    
    # #feature engineering of the user input
    user_input=preprocess(user_input=user_input)

    #Prediction code
    st.write("Press predict to see result")


    if retention_predict:
        pred = model.predict_proba(user_input)[:, 1]

        if pred[0] >= 0.50:
            st.subheader("Result:")
            st.error('Warning! The applicant has a high risk of resigning from Invoke!')
            st.subheader('Percentage of resigning:')
            st.header(f"{round((pred[0]*100), 2).astype(str)}%")
            st.subheader('Result Interpretability:')
            chart = generate_shap_waterfall_plot(df_input, user_input)
            # Display the SHAP waterfall plot
            st.pyplot(chart)
            st.markdown("""
                Understanding the graph:
                1. The graph explains how the model decides if someone will resign for Invoke.
                2. Each person is represented by a top bar, showing the model's final decision for them (e.g., 1 = "will leave" or 0 = "will stay"). 
                3. Colored bars below the top bar represent different aspects about that person, like Post Dic Score, Age Group, and Salary. 
                4. If a bar is on the right side, it means that aspect increased the chances of resigning.
                5. If a bar is on the left side, it means that aspect decreased the chances of resigning.
                6. The graph shows how all these aspects combine to make the model's decision.
                7. By adding up the lengths of the bars, we can see what factors mattered the most in the decision.
                8. It helps us understand why the model said "resign" or "stay" to a person based on individual aspects.
                9. It provides insights into how the machine works and why it made a particular prediction.
                """)
            
        else:
            st.subheader("Result:")
            st.success('It is green! The applicant has a low risk to resigning from Invoke!')
            st.subheader('Percentage of resigning:')
            st.header(f"{round((pred[0]*100), 2).astype(str)}%")
            st.subheader('Result Interpretability:')
            # st.dataframe(df_input)
            chart = generate_shap_waterfall_plot(df_input, user_input)
            st.pyplot(chart)
            st.markdown("""
                Understanding the graph:
                1. The graph explains how the model decides if someone will resign for Invoke.
                2. Each person is represented by a top bar, showing the model's final decision for them (e.g., 1 = "will leave" or 0 = "will stay"). 
                3. Colored bars below the top bar represent different aspects about that person, like Post Dic Score, Age Group, and Salary. 
                4. If a bar is on the right side, it means that aspect increased the chances of resigning.
                5. If a bar is on the left side, it means that aspect decreased the chances of resigning.
                6. The graph shows how all these aspects combine to make the model's decision.
                7. By adding up the lengths of the bars, we can see what factors mattered the most in the decision.
                8. It helps us understand why the model said "resign" or "stay" to a person based on individual aspects.
                9. It provides insights into how the machine works and why it made a particular prediction.
                """)

