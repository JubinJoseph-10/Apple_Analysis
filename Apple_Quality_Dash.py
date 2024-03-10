#buildin a streamlit dashboard in order to share live insights
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from scipy.stats import shapiro
from scipy.stats import ttest_ind

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout='wide',initial_sidebar_state='collapsed')

#reading the dataset
apple_data = pd.read_csv("Data/Dataset (Apple Fruit Quantity).csv")
#dropping the last row which had the author source
apple_data.dropna(inplace=True)
apple_data['Quality'].replace({'good':1,'bad':0},inplace=True)
#converting the acidity column to float
apple_data['Acidity'] = apple_data['Acidity'].astype('float64')
apple_data = apple_data.drop('A_id',axis=1)

#the starting of the page
st.title('Apple Quality Analysis')
st.write('\n')


###################### SECTION ON PAIRPLOT ######################
pairplot_space = st.container(border=True)
pairplot_space_1,pairplot_space_2_ = pairplot_space.columns([.8,.2])
pairplot_space_1_ = pairplot_space_1.container(border=True)
pairplot_space_2 = pairplot_space_2_.container(border=True)
# Create pairplot
fig = px.scatter_matrix(apple_data, dimensions=apple_data.columns.to_list(), color='Quality',height=800,color_continuous_scale='tropic')
fig.update_layout(yaxis=dict(tickangle=45)) 
pairplot_space_1_.plotly_chart(fig,use_container_width=True)
pairplot_space_2.write('Pairplot to see mutltiple relationships at the same time')
pairplot_space_2.write('\n')
pairplot_space_2.markdown('<div style="text-align: justify; font-size: 12px">When examining a pairplot illustrating the relationships between seven independent features and the quality of apples as the dependent variable, one can easily discern various insights. The diagonal plots provide individual feature distributions, offering a glimpse into their respective ranges and distributions. Meanwhile, the off-diagonal plots reveal associations between pairs of features, with each scatter plot showcasing the correlation between two variables. By observing color-coded points, representing different levels of apple quality, one can discern how quality varies concerning the features being compared. Identifying trends, patterns, outliers, and anomalies within the scatter plots provides further depth to the analysis. Strong positive or negative correlations between features and apple quality are indicative of impactful relationships. Ultimately, the pairplot serves as a comprehensive visual aid, enabling the observer to extract valuable insights regarding the interplay between independent features and the quality of apples.</div>',unsafe_allow_html=True)
pairplot_space_2.write('\n')


###################### SECTION ON NORMALITY ######################

#Testing the normality of the columns
exploring_normality = st.container(border=True)
exploring_normality.markdown('<div style="text-align: center; font-size: 16px">Exploring the normality of the independent variables</div>',unsafe_allow_html=True)
exploring_normality.write('\n')
des_,plot_space = exploring_normality.columns([.2,.8])

des = des_.container(border=True)
des.write('\n')
select_var = des.selectbox("Select a variable:", ['Size','Weight',
                                                    'Sweetness','Crunchiness','Juiciness','Ripeness','Acidity'],key='1')

# Assuming 'data' is your variable for which you want to test normality
statistic, p_value = shapiro(apple_data[select_var])
# Check significance
alpha = 0.05
if p_value < alpha:
    test_res = (f"{select_var} does not follow a normal distribution (reject H0), with p-value as {round(p_value,3)}.")
else:
    test_res = (f"{select_var} follows a normal distribution (fail to reject H0), with p-value as {round(p_value,3)}.")

des.write('\n')
description = f'The following graphs help us understand the normality of the {select_var} visually. Using the shapiro wilk test we understand that {test_res}'

des.markdown('<div style="text-align: justify; font-size: 14px">{}</div>'.format(description), unsafe_allow_html=True)
des.write('\n')

plots = plot_space.container(border=True)
plots_den,plot_vio,plot_box = plots.columns(3)
density_plot =  px.histogram(apple_data, y=select_var, title=f"{select_var} Density Plot",histnorm='probability density')
violin = px.violin(apple_data,y=select_var,title=f'ViolinPlot_{select_var}') 
box = px.box(apple_data,y=select_var,title=f'Boxplot_{select_var}')
plots_den.plotly_chart(density_plot,use_container_width=True)
plot_vio.plotly_chart(violin,use_container_width=True)
plot_box.plotly_chart(box,use_container_width=True)


###################### SECTION ON CORRELATION BETWEEN VARIABLES ######################
corr_section = st.container(border=True)
#splitting space to columns
corr_heatmap,corr_des = corr_section.columns([.78,.22])
#creatign individual containers
corr_heatmap_ = corr_heatmap.container(border=True)
corr_des_ = corr_des.container(border=True)
#creating a heatmap
heatmap = px.imshow(apple_data.corr(method='spearman').round(3),labels=dict(x="Features", y="Features", color="Correlation"),aspect='auto',text_auto=True,color_continuous_scale ='jet',title='Correlation Heatmap for all the Variables')
corr_heatmap_.plotly_chart(heatmap,use_container_width=True)
#writting description
corr_des_.markdown('<div style="text-align: center; font-size: 16px">Lets Break it down!</div>',unsafe_allow_html=True)
corr_des_.write('\n')
corr_des_.markdown('<div style="text-align: justify; font-size: 13px">Interpreting a correlation heatmap involves scrutinizing the colors and values displayed. Positive correlations, denoted by warmer colors, suggest variables moving in tandem, while negative correlations, depicted by cooler hues, signify opposing trends. The strength of correlation lies between -1 and 1, where values closer to 1 or -1 imply stronger associations, whereas those near 0 indicate weaker connections. Spearman correlation, unlike Pearson, evaluates monotonic relationships, rendering it advantageous for datasets with ordinal or skewed distributions.</div>',unsafe_allow_html=True)
corr_des_.write('\n')
corr_des_.write('\n')

###################### SECTION ON DIFFERENCE IN MEANS ######################
#creating a structure for the difference in mean analysis
statistical_difference = st.container(border=True)
statistical_difference.markdown('<div style="text-align: center; font-size: 16px">Lets analyse the difference in means!</div>',unsafe_allow_html=True)
statistical_difference.write('\n')
stat_des_,stat_diff_plot = statistical_difference.columns([.35,.65])
stat_des = stat_des_.container(border=True)

#section for des and chart
stat_des.markdown('<div style="text-align: center; font-size: 16px">Lets Break it down!</div>',unsafe_allow_html=True)
stat_des.write('\n')
#writting the description
stat_des.markdown('<div style="text-align: justify; font-size: 14px">When we compare the average height of boys and girls in a class, differences suggest variations between the groups. Statistical significance helps determine if these differences are likely due to real distinctions or random fluctuations. This assurance enables informed decisions, ensuring fair treatment and tailored interventions to promote equality and fairness for all students.</div>',unsafe_allow_html=True)
stat_des.write('\n')
#creating a dropdown for columns
sel_col = stat_des.selectbox("Select a variable:", ['Size','Weight',
                                                    'Sweetness','Crunchiness','Juiciness','Ripeness','Acidity'],key='2')
# Perform t-test test
statistic, p_value = ttest_ind(apple_data[apple_data['Quality']==1][sel_col], apple_data[apple_data['Quality']==0][sel_col])
# Check significance
alpha = 0.05
if p_value < alpha:
    sig = f"There is a significant difference between the average of {sel_col} across good quality and bad quality apples."
else:
    sig = f"There is no significant difference between the average of {sel_col} across good quality and bad quality apples."

#writting the statistical significane
stat_des.markdown('<div style="text-align: justify; font-size: 14px">{}</div>'.format(sig),unsafe_allow_html=True)
stat_des.write('\n')
#creating barplot
stat_diff_plot_ = stat_diff_plot.container(border=True)
y = [round(apple_data[apple_data['Quality']==0][sel_col].mean(),3),round(apple_data[apple_data['Quality']==1][sel_col].mean(),3)]
x = ['Bad','Good']
diff_plot = px.bar(x=x,y=y,color=x,text=y,labels=dict(x='Quality',y=f'Average Value {sel_col}'),title='Difference in mean')
stat_diff_plot_.plotly_chart(diff_plot,use_container_width=True)



####################### Logistic Regression model Space #######################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# Calculate the accuracy score
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import accuracy_score

features = apple_data.drop('Quality',axis=1)
target = apple_data['Quality']

LR_Space = st.container(border=True)
model_des, model_accuracy, model_auc_roc = LR_Space.columns([.22,.33,.45])
model_des_ = model_des.container(border=True)
model_des_.write('Model Building')
model_des_.markdown('<div style="text-align: justify; font-size: 12px">A confusion matrix summarizes a model\'s classification performance by tabulating correct and incorrect predictions. It includes true positives (correctly predicted positives), true negatives (correctly predicted negatives), false positives (incorrectly predicted positives), and false negatives (incorrectly predicted negatives). Meanwhile, a feature importance plot in logistic regression illustrates the significance of predictors in predicting outcomes. Positive coefficients indicate a positive impact on the outcome, while negative coefficients suggest the opposite. These plots aid in identifying crucial predictors and understanding their influence on model predictions.</div>',unsafe_allow_html=True)
model_des_.write('\n')

model_sel, feat_import= LR_Space.columns([.22,.78])
model_sel = model_sel.container(border=True)
LR_Space_ = LR_Space.container(border=True)
sel_features = model_sel.multiselect('Select Features for the Model:',features.columns.to_list(),default=features.columns.to_list())
# Check if at least one value is selected
if not sel_features:
    st.warning('Please select at least one value.')
    
model_features = features[sel_features]

X_train,X_test,y_train,y_test = train_test_split(model_features,target,test_size=.25,random_state=42) 

model_lr = LogisticRegression() 
model_lr.fit(X_train,y_train)
pred_lr = model_lr.predict(X_test)



#visualisation of accracy
cm = confusion_matrix(y_test, pred_lr)
# Create the heatmap plot using Plotly Express
con_mat = px.imshow(cm, labels=dict(x="Predicted", y="True"), x=['Bad', 'Good'], y=['Bad', 'Good'], color_continuous_scale='Blues',text_auto=True,title='Confusion Matrix Logistic Regression')
# Update the color axis to hide the scale
con_mat.update_coloraxes(showscale=False)
#creating a container for model_accuracy
model_accuracy_ = model_accuracy.container(border=True)
# Show the plot
model_accuracy_.plotly_chart(con_mat,use_container_width=True)


#aucroc curve
model_auc_roc_ = model_auc_roc.container(border=True)
from sklearn.metrics import roc_curve, auc
# Predict probabilities
y_prob = model_lr.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Create DataFrame for ROC curve data
roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})

# Plot ROC curve
plt.figure(figsize=(6, 6.2))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve Logistic Regression',y=1.03)
plt.legend(loc='lower right')
model_auc_roc_.pyplot(plt.show(),use_container_width=True)


#Faeture importance plot
# Extract coefficients and feature names
coefficients = model_lr.coef_[0]
feature_names = [f'{i}' for i in model_features.columns.to_list()]
# Create a DataFrame with feature names and coefficients
df_coefficients = pd.DataFrame({'Feature': feature_names, 'Coefficient Value': coefficients.round(2)})

# Sort the DataFrame by coefficient values
df_coefficients_sorted = df_coefficients.sort_values(by='Coefficient Value',ascending=False)

# Create the bar plot using Plotly Express
feat_importance = px.bar(df_coefficients_sorted, 
             y='Coefficient Value', 
             x='Feature', 
             orientation='v', 
             title='Feature Importance in Logistic Regression',
             labels={'Coefficient Value': 'Coefficient Value', 'Feature': 'Feature'},color='Feature',text='Coefficient Value')

#crating container for feature importance plot
model_feat_imp_ = feat_import.container(border=True)
# Show the plot
model_feat_imp_.plotly_chart(feat_importance,use_container_width=True)


class__ = st.container(border=True) 
class__.write(f'<h3 style="text-align: center;">Classification Report & Log Odds</h3>', unsafe_allow_html=True)
class_1,class_2 = class__.columns(2)
class_ = class_1.container(border=True)
class_2_ = class_2.container(border=True)
# Example classification report
report = classification_report(y_test, pred_lr)
# Display classification report with HTML formatting
accuracy = accuracy_score(y_test, pred_lr)
class_.write(f"Accuracy: {accuracy}")
# Calculate additional metrics
class_.text(report)
class_.write('\n')

# Interpret coefficients
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coefficients_df['Odds Ratio'] = np.exp(coefficients_df['Coefficient'])
class_2_.text(coefficients_df)






