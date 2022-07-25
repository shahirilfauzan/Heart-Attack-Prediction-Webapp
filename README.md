# Heart Attack Prediction By Using Machine Learning With Web Application

## Project Description
According to World Health Organisation (WHO), every year around 17.9 million deaths are due to cardiovascular diseases (CVDs) predisposing CVD becoming the leading cause of death globally. CVDs are a group of disorders of the heart and blood vessels, if left untreated it may cause heart attack. Heart attack occurs due to the presence of obstruction of blood flow into the heart. The presence of blockage may be due to the accumulation of fat, cholesterol, and other substances. Despite treatment has improved over the years and most CVD’s pathophysiology have been elucidated, heart attack can still be fatal.

Thus, clinicians believe that prevention of heart attack is always better than curing it. After many years of research, scientists and clinicians discovered that, the probability of one’s getting heart attack can be determined by analysing the patient’s age, gender, exercise induced angina, number of major vessels, chest pain indication, resting blood pressure, cholesterol level, fasting blood sugar, resting electrocardiographic results, and maximum heart rate achieved.

The purpose of this project is to predict the chance of someone having a heart attack by using Machine Learning and create application using streamlit by **Analysing Below Features**:

- Age

- Chest Pain Type

- Resting blood pressure (in mm Hg)

- Maximum heart rate achieved

- Exercise induced angina

- ST depression induced by exercise relative to rest

- Number of major vessels

- Thalassemia

# How to Install and Run the Project
To run this model on your pc, you may need to download all things inside the repository. All the folder inside are already arranged according to the coding, so it's not recommend to change the folder path such as "model" folder to another path. By using the spyder and upload the heart_attack_train.py and run it to get the result. By upload and run the app.py you may see the result of model deployment by using 10 sets of data from the dataset to get a better accuracy of the model and also can access the webapp by using streamlit.

Software required: Spyder, Python(preferably the latest version)

Additional modules needed: Tensorflow, Sklearn, matplotlib, streamlit

# Model Development
The Machine Learning algorithms used for this project :

1) MinMaxScaler + Logistic Regression (Scikit-learn)

2) StandardScaler + Logistic Regression (Scikit-learn)

3) MinMaxScaler + DecisionTreeClassifier(Scikit-learn)

4) StandardScaler + DecisionTreeClassifier(Scikit-learn)

5) MinMaxScaler + RandomForestClassifier(Scikit-learn)

6) StandardScaler + RandomForestClassifier(Scikit-learn)

7) MinMaxScaler + KNeighborsClassifier(Scikit-learn)

8) StandardScaler + KNeighborsClassifier(Scikit-learn)

9) MinMaxScaler + GradientBoostingClassifier(Scikit-learn)

10) StandardScaler + GradientBoostingClassifier(Scikit-learn)

11) MinMaxScaler + SVC(Scikit-learn)

12) StandardScaler + SVC(Scikit-learn)

By using the pipeline method to get which is the best method for the model. Thus the result is **MinMaxScaler + Logistic Regression**

# Result

The images below show the result of the model. The model accuracy is 0.84 after using pipeline to compare the best Machine Learning algorithms and further tuning the parameters by using gridsearchCV to increase the accuracy and this accuracy still can be increased with more data collected, better features selection and model parameter tuning.

![Result Accuracy](https://github.com/shahirilfauzan/Heart-Attack-Prediction-Webapp/blob/1821dd6c814ebc9c1720481ec2ea961982a816f4/static/result.PNG)

![Result](https://github.com/shahirilfauzan/Heart-Attack-Prediction-Webapp/blob/a338b05234e5fc0248063efa8a2fdba270be9e23/static/matrix_result1.png)

# Web Application
By using the web application you can fill the data needed such as patient age, chest pain type, resting blood pressure (in mm Hg), maximum heart rate achieved, exercise induced angina, ST depression induced by exercise relative to rest, number of major vessels, and thalassemia.

![streamlit](https://github.com/shahirilfauzan/Heart-Attack-Prediction-Webapp/blob/23133fed24cbee8c2fa0862dea8882f0966591d9/static/result_final_.gif)

Below is the result of the web application by using 10 dataset from the data.csv :

![streamlit](https://github.com/shahirilfauzan/Heart-Attack-Prediction-Webapp/blob/23133fed24cbee8c2fa0862dea8882f0966591d9/static/data_result.PNG)

The result came out with 9 out of 10 dataset able to predict the same outcome, this show that this application have 90% accuracy

# Credits
This dataset is provided by [RASHIK RAHMAN](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/discussion/234843?sort=votes)

# These Codes powered by
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
 ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
 ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
