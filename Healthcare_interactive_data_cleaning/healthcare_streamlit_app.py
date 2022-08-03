import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn import preprocessing
from healthcare_streamlit import data_import
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_tags import st_tags

st.set_page_config(layout="wide")
scatter_column, settings_colunm = st.columns((4, 1))
scatter_column.title("Interactive way to Clean Data and Understand Using Streamlit and Stream_tags")
settings_colunm.title("Settings")
uploaded_file = settings_colunm.file_uploader('choose_file')

if uploaded_file is not None:
	data_import = pd.read_csv(uploaded_file)

	uploaded_file = st.text('> View top 5 Rows of datasets.')
	st.dataframe(data_import.head())

	st.markdown(st.write("The Description of dataframe."))

	st.write(data_import.describe())

	# modifying the date and time into standard form
	# base_data['ScheduledDay'] = pd.to_datetime(base_data['ScheduledDay']).dt.date.astype('datetime64[ns]')
	# base_data['AppointmentDay'] = pd.to_datetime(base_data['AppointmentDay']).dt.date.astype('datetime64[ns]')

	options = st.multiselect("Change DataType", options=data_import.columns)

	if len(options) > 1:
		# st.write(data_import[options[0]])
		data_import[options[0]] = pd.to_datetime(data_import[options[0]])
		data_import[options[1]] = pd.to_datetime(data_import[options[1]])
		st.markdown('Columns type is converted of **{0}** & **{1}** as **{1}**'.format(options[0],options[1],data_import['ScheduledDay'].dtype))
	else:
		st.error("Please Select attribute to change DataType.")

	st.markdown("For the schedule day and appointment day storing the weekdays only into a variable")
	# weekday = st.multiselect("Change DataType", options=data_import.columns)

	data_import['sch_weekday'] = data_import[options[0]].dt.dayofweek
	data_import['app_weekday'] = data_import[options[1]].dt.dayofweek
	st.markdown('Columns weekday are set for **{0}** & **{1}**'.format(options[0],options[1]))	

	st.markdown('#### Renaming Columns name')
	# with st.form(key="form"):
	col_to_change = st.multiselect("Column to change", data_import.columns)
		# new_col_name = st.text_input("New name", value=)
	
	keywords = st_tags(
    label='Enter Keywords:',
    text='Press enter to add more',
    value=['PatientId', 'AppointmentID', 'Gender', 
                 'ScheduledDay', 'AppointmentDay', 'Age', 
                 'Neighbourhood', 'Scholarship', 'Hyptertension', 
                 'Diabetes', 'Alcoholism', 'Handicap', 'SMSReceived', 'NoShow'],
    suggestions=['PatientId', 'AppointmentID', 'Gender', 
                 'ScheduledDay', 'AppointmentDay', 'Age', 
                 'Neighbourhood', 'Scholarship', 'Hyptertension', 
                 'Diabetes', 'Alcoholism', 'Handicap', 'SMSReceived', 'NoShow'],
    maxtags = len(data_import.columns),
    key='1')

	submit_button = st.button(label='Submit')

	st.write(keywords[0])
	if st.button:

		for i in range(len(keywords)):
				data_import = data_import.rename(columns={col_to_change[i]: keywords[i]})

	st.dataframe(data_import.head())

	st.markdown('### - calculating the % of appointments or not')

	st.markdown(st.write(100*data_import['NoShow'].value_counts()/len(data_import['NoShow'])))

	st.markdown('#### Count Value of Number of appointments.')

	st.markdown(st.write(data_import['NoShow'].value_counts()))

	st.markdown('#### Treating missing Values.')

	#check for null values
	if st.checkbox('Missing Values'):
		st.markdown("Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")
		st.write("Number of rows:", len(data_import))
		dfnull = data_import.isnull().sum()/len(data_import)*100
		totalmiss = dfnull.sum().round(2)
		st.write("Percentage of total missing values:",totalmiss)
		st.write(dfnull)
		if totalmiss <= 30:
				st.success("Looks good! as we have less then 30 percent of missing values.")       
	else:
		st.success("Poor data quality due to greater than 30 percent of missing value.")
	st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there’s no hard and fast rule to decide this threshold. It can vary from problem to problem.")

	st.markdown("#### Generate Dummies Variable")
	labels = ["{0} - {1}".format(i, i + 20) for i in range(1, 118, 20)]

	data_import['Age_group'] = pd.cut(data_import.Age, range(1, 130, 20), right=False, labels=labels)

	data_import.drop(['Age'], axis=1, inplace=True)

#C:\Users\kd\Saved Games\Projects\Projects\EDAforHealthcare-main>

	def get_dummies(data_import):

		data_import['NoShow'] = np.where(data_import.NoShow == 'Yes',1,0)

		data_count = data_import.NoShow.value_counts()

		# Convert all the categorical variables into dummy variables
		base_data_dummies = pd.get_dummies(data_import)
		
		return base_data_dummies, data_count

	base_data_dummies, data_count = get_dummies(data_import)

	st.write(base_data_dummies.head())
	# Build a corelation of all predictors with 'NoShow'
	plt.figure(figsize=(20,8))
	st.write(base_data_dummies.corr()['NoShow'].sort_values(ascending = False).plot(kind='bar'))

	plt.figure(figsize=(12,12))
	sns.heatmap(base_data_dummies.corr(), cmap="Paired")

	# Bivariate Analysis

	new_df1_target0=data_import.loc[data_import["NoShow"]==0]
	new_df1_target1=data_import.loc[data_import["NoShow"]==1] 

	st.markdown(st.write("The number of NoShow is below where NoShow attribute value is **zero** with Age category"))

	st.write(new_df1_target0)

	st.markdown(st.write("The number of NoShow is below where NoShow attribute value is **one** with Age category"))

	st.write(new_df1_target1)