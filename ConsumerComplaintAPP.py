import numpy as np
import pandas as pd
import streamlit as st
import joblib
import random
import altair as alt

st.set_page_config(
    page_icon="/Users/Lenovo/Desktop/ISTDSA/DSAG22/proje3/streamlit/CS_logo.png",
    menu_items={
        "Get help": "mailto:juneight79@gmail.com",
        
    }
)


pipe_lr=joblib.load(open("complaints_classifier_LR_TFIDF.pkl", "rb"))



def predict_product(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results




def main():
    st.title("Consumer Complaints Classifier App")
    
    menu= ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.image("https://t3.ftcdn.net/jpg/00/47/31/46/240_F_47314660_FrEaRx6KeHjYJMajIGzXiPxhcQR61Qw7.jpg", width=300)
    with col3:
        st.write(' ')
    
    df = pd.read_csv('complaint30000.csv')
    df.rename(columns= {"Consumer complaint narrative":"consumer_complaint"},inplace=True)
    df.drop(['Unnamed: 0'], axis='columns', inplace=True)

    st.table(df.sample(3, random_state=45))


    if choice == "Home":
        st.subheader("What is The Complaint In Text About ?")

                
        with st.form(key='complaint text'):
            raw_text=st.selectbox("Select", df["consumer_complaint"])
            submit_text = st.form_submit_button(label='Submit')            
                          
                
        if submit_text:
            col1,col2 = st.columns(2)

            prediction = predict_product(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                st.write(prediction)
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                #st.write(probability)
                proba_df=pd.DataFrame(probability, columns=pipe_lr.classes_)
                #st.write(proba_df.T)
                proba_df_clean= proba_df.T.reset_index()
                proba_df_clean.columns = ["Product", "probability"]

                fig=alt.Chart(proba_df_clean).mark_bar().encode(x="Product", y="probability", color="Product")
                st.altair_chart(fig, use_container_width=True)

            
   
    else:
        st.subheader("About")


if __name__=='__main__':
    main()


