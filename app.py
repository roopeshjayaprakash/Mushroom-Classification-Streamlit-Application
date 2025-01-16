from enum import auto
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import pair_confusion_matrix, confusion_matrix
from sklearn.metrics import pair_confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay 
from sklearn.metrics import precision_score, recall_score 

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your Mushrooms poisonous 🍄")
    st.sidebar.markdown("Are your Mushrooms poisonous 🍄")

    #Function for loading the dataset and since All the columns in the data set are categorical we perform label encoding'''
    @st.cache_data(persist=True) #Using this decorator to cache the output which we run 
    def load_data():
        data=pd.read_csv('/home/roopesh-j/notebooks/ML Project/mushrooms.csv') 
        label=LabelEncoder()

        for col in data.columns:
            data[col]=label.fit_transform(data[col])
        return data
    
    '''Function for Train Test Split'''
    @st.cache_data(persist=True)
    def split(df):
        y=df['class']
        x=df.drop(columns=['class'])
        x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    
    '''Function for Evaluation Metrics'''
    def plot_metrics(metrics_list): #Argument is one or more user selected options
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list: #More caution to False Negetives i.e Type 2 errors 
            st.subheader("Plot ROC Curve")
            roc_display = RocCurveDisplay.from_estimator(model, x_test, y_test)
            st.pyplot(roc_display.figure_)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            pr_display = PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
            st.pyplot(pr_display.figure_)

    #Checkbox widget showing raw dataframe on mainpage
    @st.cache_data(persist=True) #Using this decorator to cache the output which we run 
    def raw_data():
        df=pd.read_csv("/home/roopesh-j/notebooks/ML Project/mushrooms.csv")
        return df
    
    if st.sidebar.checkbox("Show Raw Data", False):
        st.subheader("Mushroom Dataset (Classification)")
        st.write(raw_data())
        st.subheader("Attributes Description")
        descriptions = """
        **class**: Indicates whether the mushroom is edible (`e`) or poisonous (`p`).  
        **cap-shape**: Shape of the mushroom cap (e.g., `x` for convex, `b` for bell, etc.).  
        **cap-surface**: Texture of the mushroom cap (e.g., `s` for smooth, `y` for scaly, etc.).  
        **cap-color**: Color of the mushroom cap (e.g., `n` for brown, `w` for white, etc.).  
        **bruises**: Whether the mushroom shows bruising (`t` for true, `f` for false).  
        **odor**: Odor of the mushroom (e.g., `a` for almond, `p` for pungent, etc.).  
        **gill-attachment**: How the gills are attached to the stalk (e.g., `f` for free, `a` for attached).  
        **gill-spacing**: Spacing between the gills (e.g., `c` for close, `w` for wide).  
        **gill-size**: Size of the gills (e.g., `b` for broad, `n` for narrow).  
        **gill-color**: Color of the gills (e.g., `k` for black, `n` for brown, etc.).  
        **stalk-shape**: Shape of the stalk (`e` for enlarging, `t` for tapering).  
        **stalk-root**: Type of root at the base of the stalk (e.g., `b` for bulbous, `c` for club, `e` for equal, etc.).  
        **stalk-surface-above-ring**: Texture of the stalk surface above the ring (e.g., `s` for smooth, `f` for fibrous).  
        **stalk-surface-below-ring**: Texture of the stalk surface below the ring.  
        **stalk-color-above-ring**: Color of the stalk above the ring.  
        **stalk-color-below-ring**: Color of the stalk below the ring.  
        **veil-type**: Type of veil covering the mushroom (usually `p` for partial).  
        **veil-color**: Color of the veil (e.g., `w` for white, `o` for orange, etc.).  
        **ring-number**: Number of rings on the mushroom stalk (e.g., `o` for one, `t` for two).  
        **ring-type**: Type of ring (e.g., `p` for pendant, `e` for evanescent).  
        **spore-print-color**: Color of the spore print (e.g., `k` for black, `n` for brown, etc.).  
        **population**: Mushroom population size (e.g., `s` for scattered, `n` for numerous, etc.).  
        **habitat**: Habitat where the mushroom grows (e.g., `g` for grasses, `m` for meadows).  
        """
        st.markdown(descriptions)

    df=load_data()
    x_train, x_test, y_train, y_test=split(df)
    class_names=['Edible', 'Poisonous']
    st.sidebar.subheader("Choose Classifier") #User can select the type of classifier they want to use
    Classifier=st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if Classifier=='Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularisation Parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel=st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma=st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics=st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",'ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='Classify'):
            st.subheader("Support Vector Machine(SVM) Results")
            model=SVC(C=C, kernel=kernel, gamma=gamma)#Values taken from user
            model.fit(x_train, y_train)
            accuracy=model.score(x_test, y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred, labels=class_names), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred, labels=class_names),2))
            plot_metrics(metrics)#User selected metrics from above
    
    if Classifier=='Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularisation Parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter=st.sidebar.slider("Maximum number of Iterations", 100, 500, key="max_iter")

        metrics=st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",'ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='Classify'):
            st.subheader("Logistic Regression Results")
            model=LogisticRegression(C=C, max_iter=max_iter)#Values taken from user
            model.fit(x_train, y_train)
            accuracy=model.score(x_test, y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred, labels=class_names), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred, labels=class_names),2))
            plot_metrics(metrics)#User selected metrics from above
    
    if Classifier=='Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimator = st.sidebar.slider(
    "The number of trees in the forest", 
    min_value=100, 
    max_value=5000, 
    step=10, 
    key="n_estimator"
)
        max_depth=st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = bool(st.sidebar.radio("Bootstrap samples when building trees",('True','False'), key='bootstrap'))
        metrics=st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",'ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='Classify'):
            st.subheader("Random Forest Results")
            model=RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth, bootstrap=bootstrap) #telling scikit learn to use all the cores of our CPU for the computation to be more faster
            model.fit(x_train, y_train)
            accuracy=model.score(x_test, y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred, labels=class_names), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred, labels=class_names),2))
            plot_metrics(metrics)#User selected metrics from above

    #Future scope - K Fold Cross Validation, using .predict() function, 
if __name__ == '__main__':
    main()