# Supervised Machine Learning for Predicting Sustainable Energy Needs

In this project, I developed machine learning models to predict future energy consumption for 176 countries using the Global Data on Sustainable Energy. The dataset spans from 2000 to 2020 and includes 21 measures such as population count, access to clean fuels, and electricity generation sources. I applied data pre-processing, model selection, training, and evaluation techniques, and visualized energy consumption and carbon emission trends for the next five years for three selected countries. This work aims to support the United Nations' Sustainable Development Goal 7 by providing insights into global energy needs and environmental impact.

# The Dataset
The dataset contains key metrics for 176 countries where each country has 21 records spanning the years 2000 – 2020. There are 21 variables in total including Energy Consumption per capita, Renewable electricity generation, Carbon (CO2 Emissions), Access to clean cooking fuels and financial flows to developing countries (from developed countries). Table 1 summarizes all variables along with the descriptions

![image](https://github.com/Mabrar92/Supervised-ML-Based-Energy-Consumption-Predictions/assets/18236632/e05853a6-9c41-437d-aaa7-91e5a84c7d77)


Table 1 Names and description of all features from the Global Data on Sustainable Energy

#  Data Preprocessing
Data preprocessing is an important step in the machine learning pipeline that involves cleaning, organizing, transforming, and optimizing raw data that can be utilized by machine learning models to produce optimized outputs. The tasks involved include handling missing values through imputation algorithms, scaling features through standardization or normalization and converting categorical features to numerical data through encoding.

The columns such as Year, Density and Land Area have incorrect data types. Moreover, the name of Density has irregular characters in it, so we rename it.

![image](https://github.com/Mabrar92/Supervised-ML-Based-Energy-Consumption-Predictions/assets/18236632/366d3ae8-558c-4506-9d9e-73c69303dadf)


# Feature Transformation

Feature transformation is part of data preprocessing that involves transforming feature data to make it suitable for machine learning models. Feature transformation for this dataset involves Missing value imputations, Encoding categorical data and data scaling and standardization.

**Missing Values Imputation**

The missing data is imputed because they introduce bias in the dataset leading to inaccurate predictions and secondly missing values degrades the performance of ML models. Researchers have divided imputation methods into statistical methods such as replacing values with mean or mode and ML based methods estimate values using predictive model trained on existing data and then fill missing values (Pereira et al., 2020).

![image](https://github.com/Mabrar92/Supervised-ML-Based-Energy-Consumption-Predictions/assets/18236632/c3d2b3e5-9443-4501-9c8b-3c231090df59)

  Figure 1 Missing values Imputation Techniques (Pereira et al., 2020)



The missing data analysis shows the presence of two columns with more than 50 percent of missing values. Thus, handling missing values through imputation is crucial for the analysis.

![image](https://github.com/Mabrar92/Supervised-ML-Based-Energy-Consumption-Predictions/assets/18236632/6fdf819a-18ac-4e4f-89d6-17b4958b5013)

The goal of an imputation method is to recover the correct distribution, rather than the exact value of each missing feature (Shadbahr et al., 2023). Thus, missing values are imputed through KNN Imputer and Iterative Imputers (regression-based) and the effectiveness of imputation is analyzed through the change in distribution.


**KNN Imputer vs Iterative Imputer**

The missing data imputation for this analysis has been performed using KNN imputer and Iterative imputers. KNN imputer is a machine learning based imputer that utilizes KNN algorithm to predict missing values based on other values in the dataset. Iterative imputer performs predictions over multiple iterations using regression until the error is reduced to a threshold value.
The KNN imputer is used to predict missing vales for different values of K and based on the density distribution of features we find the optimal value of k. The closer the variance (distribution) to the original density of the column the better the imputation. There is no significant difference between the distributions of imputed data for both imputers, However, the KNN imputer for k=9 shows slightly better results in terms of the closeness to the original distribution, Thus KNN imputer is selected.


![image](https://github.com/Mabrar92/Supervised-ML-Based-Energy-Consumption-Predictions/assets/18236632/1c4b6c77-e6c7-4ddb-903f-f1836392ff59)


Figure 2 The Kernel Density Estimate (KDE) distribution plot for original and imputed data (left: KNN; Right: Iterative)


**Feature Scaling**

The features in the dataset often exhibit large differences in ranges of data. This can introduce biases in the predictions, as algorithms give more weight to the features with data in a high range. Thus, the process of balancing features and bringing the data ranges to similar ranges is known as Standardization. The process of transforming features into a common scale (unit) is known as Normalization. The feature scaling is comprised of Standardization or normalization of data.
As our dataset has huge differences in features in terms of ranges, such as financial flows to developing countries (US $) has a high range while Electricity from fossil fuels (TWh) has values in extremely lower ranges, thus it can negatively affect the accuracy of our ML Model. Thus, we apply Standard Scalar to standardize the values across all features. the resultant standardized dataset will have a mean equal to zero and a standard deviation equal to 1.

![image](https://github.com/Mabrar92/Supervised-ML-Based-Energy-Consumption-Predictions/assets/18236632/d81559c1-286c-4a8f-8157-bd9e73d05482)

**Encoding Categorical Data**

Categorical data can be either nominal or ordinal. The underlying dataset contains a categorical column Entity containing names of the countries. Thus, to transform nominal values into numerical values, One Hot Encoder from the scikit learn library is used. To obtain optimal results, the data is split into test and train parts. We see that the shape transforms from 11 columns into 185 columns when encoded. This was the last step in Feature Transformation. The next part of preprocessing is Feature Selection which is explained in the next section.

# Feature Selection

In machine learning the number of features has a direct impact on the prediction accuracy of ML models. There is a need for an optimal number of features to obtain better predictions. Increasing the number of features beyond a threshold in the ML model causes overfitting. This problem is known as the Curse of Dimensionality. On the contrary, selecting too few features can introduce underfitting. Thus, to avoid this the role of feature selection and feature extraction come into play.

Feature extraction algorithm involves Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA). However, in the assessment, we find the optimal number of features using feature selection algorithms such as Recursive Feature Elimination (RFE) to find the most optimal feature space. As the requirement in this study is to build two predictive models, one for Energy Consumption predictions and the other for Carbon Emission Predictions, so the feature selection for both models is discussed in this section to find optimal feature sets.


Recursive Feature Elimination (RFE) is an embedded feature selection method that leverages a Machine learning algorithm to rank features based on the chosen evaluation metric.  It iterates multiple times that’s why recursive until it finds the most optimal feature subset. The RFECV object from the SKlearn library has been used to find the optimal feature subset. The prediction algorithm for RFECV is a Random Forest Regressor in our case.


**Energy Consumption Prediction Model Features**
The plot reveals a plateau effect in prediction accuracy scores when selecting 9 to 11 features, showing similar accuracy against the optimal number of features (9-11) for our dataset. The Recursive Feature Elimination (RFE) determines the final feature subset based on this optimal number of features. As we increase the number of features beyond 10, the accuracy reduces showing the adverse effect of overfitting or introduction of correlated features. This graph also highlights the significance of balancing number of features to avoid underfitting or overfitting.

![image](https://github.com/Mabrar92/Supervised-ML-Based-Energy-Consumption-Predictions/assets/18236632/bba68a7b-4875-4742-be35-2e8740a05bab)

Figure Mean Test Accuracy for Energy Consumption Prediction Model

 Thus, the final features for energy consumption prediction model are shown below.
![image](https://github.com/Mabrar92/Supervised-ML-Based-Energy-Consumption-Predictions/assets/18236632/241b2d5a-c2c8-4f0c-aef7-67053feb2a1a)

Now we have our final features and pre-processed data ready to be fed into machine-learning models for prediction.


# Model selection and Training

The model selection and training phase involves selecting a range of machine learning algorithms to train on the preprocessed dataset and based on the prediction results, the best-performing model is chosen. The selected prediction algorithms depend on the type of input features and output label. If the feature to be predicted is numerical in nature, the regression algorithm is utilized, on other hand, if the feature to be predicted is categorical, classification algorithm would be the appropriate model. Since the underlying dataset has numerical input features and output label, we choose Regression Algorithms for our analysis. To train our model, a Cross Validation technique is employed, so that our model is evaluated on different subsets of data thus avoiding underfitting or overfitting problems.

  
**Model Training with cross-validation:**

In cross-validation, the ML models are tested on varied subsets of training data. Cross-validation ensures the selection of the optimum model that shows consistent and accurate results on different subsets of training data across multiple folds or iterations. We select 5 regression models for evaluation using the cross-validation technique. First, we store input and output features in separate variables. And then the dataset is split into Testing and Training data with a ratio of 1/3. The models are then evaluated based on evaluation parameters such as R2.

The cross-validation object accepts multiple parameters to refine the process, in our case, the cross-validation is performed with 5 subsets of training data also known as the number of folds. The results show that the best-performing model is the Random Forest Regressor with an R2 of 0.984 for the Energy Consumption Prediction Model.

![image](https://github.com/Mabrar92/Supervised-ML-Based-Energy-Consumption-Predictions/assets/18236632/e98acd07-4b00-4ea1-9fd3-07b955f6ce13)


# Model Selection on Test Data

The output of the above code reveals that the Random Forest Regressor has the highest r2 score of 0.989 and an adjusted r2 score of 0.986. The second high-performing model is Gradient boosting with a 0.972 r2 score. While SVM was the worst-performing model with an R2 score less than zero.  Thus, the chosen model for the Prediction of Energy Consumption data would be a **Random Forest Regressor**.

![image](https://github.com/Mabrar92/Supervised-ML-Based-Energy-Consumption-Predictions/assets/18236632/f674888d-25ef-4d5d-837d-64fcee2a12fd)

The prediction error of the Random Forest Regressor model is shown in the plot.  The plot presents the accuracy of the predictions and explains how well the predicted values from the model align with the actual values from the dataset, where blue points represent the actual (y) vs predicted (y’) value distribution. The visualization shows that the points are closely aligned to the ideal fit thus suggesting high accuracy.

![image](https://github.com/Mabrar92/Supervised-ML-Based-Energy-Consumption-Predictions/assets/18236632/37100964-d03c-4233-9307-7d74b2a360de)

The residual plot along with the distribution of residuals for the Random Forest Regressor model is shown in the figure below. The residual plot is an effective visualization that shows the difference between the predicted value and the actual value against the predicted value.  The closer a data point is to the horizontal line the lesser the error and thus the more accurate the model is. The plot shows that random forest regressors have a normal distribution of residuals along the origin, which is zero, indicating unbiased predictions.

![image](https://github.com/Mabrar92/Supervised-ML-Based-Energy-Consumption-Predictions/assets/18236632/fe867e4f-fa74-4f87-ac67-18945c43d0e2)


# Forecasting
The BRICS nations are key players in the global energy dynamics because of their large population and economic growth. Thus, predicting future energy for the next 5 years of some of these countries would prove to understand their influence on the global stage and in shaping the global strategies of energy. For this purpose, ARIMA (Autoregressive Integrated Moving Average) was employed.

The countries chosen for energy consumption forecasting are India, China, and Turkey. Using ARIMA model, the energy consumption for the years 2021 to 2025 is forecasted. The results of the forecast are visualized with historical data, to better understand the context. The Figure 4 forecast suggests a moderate decrease in energy consumption per capita which aligns perfectly with the shift to more energy-efficient practices of the country but could also suggest less accessibility of the energy in the rural areas. For China and Turkey, a profound growth in energy consumption per capita is predicted which is indicative of continued industrial growth and urbanization in both countries. 


![image](https://github.com/Mabrar92/Supervised-ML-Based-Energy-Consumption-Predictions/assets/18236632/eeb1c2a9-f8a7-4295-94da-60c9686340d3)


