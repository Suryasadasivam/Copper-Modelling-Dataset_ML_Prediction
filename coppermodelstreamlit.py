import streamlit as st
from datetime import date
import numpy as np
import sklearn
import pickle

st.set_page_config(page_title= "Copper Modelling Prediction",
                   layout= "wide",
                   initial_sidebar_state='expanded')   

# user input options
class options:

    country_values = [25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0, 
                    78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]

    status_values = ['Won', 'Lost']
    status_dict = {'Lost':0, 'Won':1}

    item_type_values = ['W', 'WI', 'S', 'PL', 'IPL', 'SLAWR', 'Others']
    item_type_dict = {'W':5.0, 'WI':6.0, 'S':3.0, 'Others':1.0, 'PL':2.0, 'IPL':0.0, 'SLAWR':4.0}

    application_values = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 
                        27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0, 
                        59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]

    product_ref_values = [611728, 611733, 611993, 628112, 628117, 628377, 640400, 
                        640405, 640665, 164141591, 164336407, 164337175, 929423819, 
                        1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 
                        1665584642, 1665584662, 1668701376, 1668701698, 1668701718, 
                        1668701725, 1670798778, 1671863738, 1671876026, 1690738206, 
                        1690738219, 1693867550, 1693867563, 1721130331, 1722207579]
    
# Streamlit page custom design
page = st.sidebar.selectbox("Select", ["About","Selling Price Prediction", "Status Prediction"], index=0)
if page=="About":
  st.title("Welcome to the Copper Modelling Selling Price/Status Prediction")
  if st.button("Overview"):
    st.write(''' The Copper Modelling Price/Status Prediction tool is a user-friendly
             web application built to predict the price or status from the given  data. 
             Leveraging the power of Streamlit and Machine Learning.''')
    st.header("Key Features")
    st.markdown(''' 
                - Data Understanding
                - Data Preprocessing
                - EDA
                - Feature Engineering
                - Model Building & Evalution''')
    st.header("Technology List ")
    st.markdown(''' 
                - Python
                - Pandas
                - Machine Learning
                - Streamlit''')
    
# Get input data from users both regression  methods
if page=="Selling Price Prediction":
     st.title("Welcome to Copper Modelling Selling Price Prediction")
     st.header("Please fill the below form")
     col=st.columns((3,3),gap='medium')
     with col[1]:
                quantity_tons_log = st.text_input(label='Quantity Tons (Min: 0.00001 & Max: 1000000000)')
                
                thickness_log = st.number_input(label='Thickness', min_value=0.1, max_value=2500000.0, value=1.0)
                
                item_date = st.date_input(label='Item Date', min_value=date(2020,7,1), 
                                        max_value=date(2021,5,31), value=date(2020,7,1))
                
                
                delivery_date = st.date_input(label='Delivery Date', min_value=date(2020,8,1), 
                                            max_value=date(2022,2,28), value=date(2020,8,1))

                product_ref = st.selectbox(label='Product Ref', options=options.product_ref_values)


            
     with col[0]:
                customer = st.text_input(label='Customer ID (Min: 12458000 & Max: 2147484000)')
                
                country = st.selectbox(label='Country', options=options.country_values)
                
                status = st.selectbox(label='Status', options=options.status_values)
                
                item_type = st.selectbox(label='Item Type', options=options.item_type_values)
                
                application = st.selectbox(label='Application', options=options.application_values)
                
                width = st.number_input(label='Width', min_value=1.0, max_value=2990000.0, value=1.0)
# user entered the all input values and click the button          
     if st.button('Submit'):
#input transforming using scalar method
        pickel_in_reg=open("x_scaled_data_fit_reg","rb")
        model_data_reg=pickle.load(pickel_in_reg)
        user_data_reg = np.array([[customer,
                                   country, 
                               options.status_dict[status],
                               options.item_type_dict[item_type],
                               application, 
                               width, 
                               product_ref, 
                               np.log(float(quantity_tons_log)), 
                               np.log(float(thickness_log)),
                               item_date.day, item_date.month, item_date.year,
                               delivery_date.day, delivery_date.month, delivery_date.year]])
        user_data_scaled_reg=model_data_reg.transform(user_data_reg)
        # Input give to model and predict the price 
    
        pickel_in_reg_pred=open("model_XGB_regression.pkl","rb")
        reg_model=pickle.load(pickel_in_reg_pred)
        y_p = reg_model.predict(user_data_scaled_reg)
        selling_price=np.exp(y_p[0])          
        st.success(f"The predicted selling price for the given data is: {selling_price}")        
 # Get input data from users both classification  methods               
if page=="Status Prediction":
     st.title("Welcome to Copper Modelling status prediction")
     st.header("Please fill the below form")
    
     col=st.columns((3,3),gap='medium')
     with col[1]:
                quantity_tons_log = st.text_input(label='Quantity Tons (Min: 0.00001 & Max: 1000000000)')
                
                thickness_log = st.number_input(label='Thickness', min_value=0.1, max_value=2500000.0, value=1.0)
                
                item_date = st.date_input(label='Item Date', min_value=date(2020,7,1), 
                                        max_value=date(2021,5,31), value=date(2020,7,1))
                
                
                delivery_date = st.date_input(label='Delivery Date', min_value=date(2020,8,1), 
                                            max_value=date(2022,2,28), value=date(2020,8,1))

                product_ref = st.selectbox(label='Product Ref', options=options.product_ref_values)


            
     with col[0]:
                customer = st.text_input(label='Customer ID (Min: 12458000 & Max: 2147484000)')
                
                country = st.selectbox(label='Country', options=options.country_values)
                
                selling_price_log = st.text_input(label='Selling Price (Min: 0.1 & Max: 100001000)')
                
                item_type = st.selectbox(label='Item Type', options=options.item_type_values)
                
                application = st.selectbox(label='Application', options=options.application_values)
                
                width = st.number_input(label='Width', min_value=1.0, max_value=2990000.0, value=1.0)
      # user entered the all input values and click the button            
     if st.button('Submit'):
         #input transforming using scalar method
         pickel_in=open("x_scaled_data_fit","rb")
         model_data_class=pickle.load(pickel_in)
         user_data = np.array([[customer,
                                country, 
                               options.item_type_dict[item_type],
                               application, 
                               width, 
                               product_ref, 
                               np.log(float(quantity_tons_log)), 
                               np.log(float(thickness_log)),
                               np.log(float(selling_price_log)),
                               item_date.day, item_date.month, item_date.year,
                               delivery_date.day, delivery_date.month, delivery_date.year]])
         user_data_scaled=model_data_class.transform(user_data)
          # Input give to model and predict the price
         pickel_in_class_pred=open("model_XGB_class.pkl","rb")
         class_model=pickle.load(pickel_in_class_pred)
         y_p = class_model.predict(user_data_scaled)
         if y_p[0] == 1:
            st.success('The predicted status for the given data is Won')
         else:
            st.error('The predicted status for the given data is Lost')        

                

                