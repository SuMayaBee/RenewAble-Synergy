# Technical Documentation

<p align="center">
  <img src="https://images.ncsl.org/image/upload/c_fill,g_auto,w_1100/f_auto,q_auto/v1676057138/website/NU-solar-wind-turbine-clean-energy-498769592_1x.jpg" alt="Renewable Energy" width="800px" style="border-radius: 15px;">
</p>

<p align="justify">
This repository is dedicated to the development of AI-driven models for <strong>RenewAble Synergy</strong>, focusing on integrating AI in the renewable energy sector. The project aims to enhance the efficiency and management of renewable resources using machine learning techniques.
</p>

## 📊 $\textcolor{darkblue}{\textbf{Renewable\ Energy\ Generation\ Forecast}}$

---

## $\textcolor{orange}{\textbf{🌞\ Solar\ Energy\ Generation\ Forecast}}$

### 1. Introduction
$\textcolor{gray}{\text{We focus on forecasting solar energy generation using historical data. We preprocess the data and apply machine learning models to predict energy output.}}$

- **The Notebook and Dataset:**
  - [Solar Energy Generation Forecast](https://github.com/SuMayaBee/RenewAble-Synergy/blob/main/Renewable%20Energy%20Generation%20Forecast/Solar%20Energy%20Generation%20Forecast/Solar%20energy%20generation%20forecast.ipynb)
  - [Dataset](https://github.com/SuMayaBee/RenewAble-Synergy/tree/main/Renewable%20Energy%20Generation%20Forecast/Solar%20Energy%20Generation%20Forecast/Dataset)

### 2. Data Loading and Preprocessing
We load time-series data from CSV files and perform:
- **$\textcolor{darkgreen}{\textbf{Missing\ Value\ Handling:}}$** Interpolation and forward-fill.
- **$\textcolor{darkgreen}{\textbf{Feature\ Engineering:}}$** Adding time-based features like hour and day.
- **$\textcolor{darkgreen}{\textbf{Scaling:}}$** Using MinMaxScaler or StandardScaler.
- **$\textcolor{darkgreen}{\textbf{Data\ Splitting:}}$** Training, validation, and test sets.

### 3. Exploratory Data Analysis (EDA)
We analyze trends, distributions, and relationships using:
- **$\textcolor{orange}{\textbf{Time-Series\ Plots:}}$** For visualizing temporal trends.
- **$\textcolor{orange}{\textbf{Correlation\ Matrices:}}$** To explore feature relationships.
- **$\textcolor{orange}{\textbf{Histograms\ and\ Boxplots:}}$** For understanding data distribution.

### 4. Model Selection and Implementation
We experiment with:
- **$\textcolor{teal}{\textbf{Linear\ Regression}}$**: As a baseline model.
- **$\textcolor{teal}{\textbf{Decision\ Tree\ \&\ Random\ Forest\ Regressors}}$**: For capturing non-linear relationships.
- **$\textcolor{teal}{\textbf{Gradient\ Boosting\ Models\ (XGBoost,\ LightGBM)}}$**: For advanced, iterative improvements.
- **$\textcolor{teal}{\textbf{Neural\ Networks\ (if\ applicable)}}$**: For modeling complex patterns.

We tune hyperparameters using GridSearchCV or RandomizedSearchCV.

### 5. Evaluation Metrics
We use:
- **$\textcolor{blue}{\textbf{Mean\ Absolute\ Error\ (MAE):}}$** To measure average prediction errors.
- **$\textcolor{blue}{\textbf{Root\ Mean\ Squared\ Error\ (RMSE):}}$** To penalize larger errors.
- **$\textcolor{blue}{\textbf{R²\ (R-Squared):}}$** To explain variance captured by the model.

### 6. Model Training and Validation
We apply cross-validation and plot learning/validation curves to monitor performance and diagnose overfitting.

### 7. Forecasting and Results Visualization
We predict solar energy generation using the best model and compare predictions against actual data using time-series plots and residual plots.

---

## $\textcolor{deepskyblue}{\textbf{💨\ Wind\ Energy\ Generation\ Forecast}}$

### 1. Introduction
$\textcolor{gray}{\text{We focus on predicting wind energy generation using historical data. Our approach involves preprocessing the data and applying time-series forecasting models like ARIMA to predict wind energy output for the next 15 days.}}$

- **The Notebook and Dataset:**
  - [Wind Energy Generation Forecast](https://github.com/SuMayaBee/RenewAble-Synergy/blob/main/Renewable%20Energy%20Generation%20Forecast/Wind%20Energy%20Generation%20Forecast/Wind%20energy%20generation%20forecast.ipynb)
  - [Dataset](https://github.com/SuMayaBee/RenewAble-Synergy/tree/main/Renewable%20Energy%20Generation%20Forecast/Wind%20Energy%20Generation%20Forecast/Dataset)

### 2. Data Loading and Preprocessing
We load time-series data from CSV files containing information on wind turbine power generation. The preprocessing steps include:
- **$\textcolor{darkgreen}{\textbf{Handling\ Missing\ Data:}}$** Dealing with missing values and outliers.
- **$\textcolor{darkgreen}{\textbf{Feature\ Engineering:}}$** Generating time-lagged features to improve model performance.
- **$\textcolor{darkgreen}{\textbf{Data\ Visualization:}}$** Using scatter plots and histograms to explore data distribution.

### 3. Exploratory Data Analysis (EDA)
We visualize trends and correlations within the dataset:
- **$\textcolor{orange}{\textbf{Time-Series\ Visualization:}}$** Plotting active power generation over time.
- **$\textcolor{orange}{\textbf{Correlation\ Analysis:}}$** Checking relationships between lagged values and current power output.

### 4. Model Selection and Implementation
We implement and evaluate ARIMA for time-series forecasting:
- **$\textcolor{teal}{\textbf{ARIMA\ (AutoRegressive\ Integrated\ Moving\ Average):}}$** Configured with optimal parameters `(p=2, d=0, q=3)` after tuning.
- **$\textcolor{teal}{\textbf{Model\ Diagnostics:}}$** Analyzing residuals for normality and autocorrelation.
- **$\textcolor{teal}{\textbf{Train-Test\ Split:}}$** Splitting the data to validate model performance over a 15-day forecast.

### 5. Evaluation Metrics
We use key metrics to evaluate forecast accuracy:
- **$\textcolor{blue}{\textbf{Mean\ Absolute\ Percentage\ Error\ (MAPE):}}$** ~2.6% indicating 97.4% accuracy.
- **$\textcolor{blue}{\textbf{Root\ Mean\ Squared\ Error\ (RMSE):}}$** For evaluating model precision.

### 6. Model Training and Validation
We train the ARIMA model on historical data and validate its performance using test data. We plot the actual vs. predicted values and confidence intervals to evaluate the accuracy visually.

### 7. Forecasting and Results Visualization
The final model is used to predict wind energy generation for the next 15 days. We visualize the forecast against actual values, displaying prediction intervals and residual errors.


# Maintenance Fault Detection

<h2 align="center">Transmission Line Fault Detection</h2>

### 1. Introduction
We focus on detecting faults in transmission lines using a multi-class classification approach. The objective is to identify fault types across lines A, B, C, and Ground (G) based on voltage and current measurements.

- **The Notebook and Dataset:**
  - [Transmission Line Fault Detection](https://github.com/SuMayaBee/RenewAble-Synergy/blob/main/Maintenance%20Fault%20Detection/Transmission%20Line%20Fault%20Detection/Transmission%20line%20fault%20detection.ipynb)
  - [Dataset](https://github.com/SuMayaBee/RenewAble-Synergy/tree/main/Maintenance%20Fault%20Detection/Transmission%20Line%20Fault%20Detection/Dataset)

### 2. Data Loading and Preprocessing
We load time-series data containing voltage and current measurements for fault detection. Preprocessing steps include:
- **Feature Engineering:** Creating fault labels for each line (A, B, C, and Ground).
- **Data Balancing:** Ensuring an even distribution of fault classes across the dataset.

### 3. Exploratory Data Analysis (EDA)
We analyze fault patterns using:
- **Voltage and Current Fluctuations:** Identifying trends where faults occur.
- **Correlation Analysis:** Investigating relationships between fault types across different lines.

### 4. Model Selection and Implementation
We explore multiple classification models:
- **Random Forest:** As the base classifier for multi-class detection.
- **MultiOutput Classifiers:** Extending Random Forest, XGBoost, and other models to handle multiple fault types simultaneously.
- **Hyperparameter Tuning:** Using Optuna for optimizing model performance.

### 5. Evaluation Metrics
We use key metrics for multi-class classification:
- **Accuracy Score:** Evaluating overall classification performance.
- **Classification Report:** Detailing precision, recall, and F1-score for each fault type (Ground, Line A, Line B, Line C).

### 6. Model Training and Validation
We train and validate the models using an 80/20 train-test split. We assess performance through classification reports and confusion matrices for each fault category.

### 7. Results Visualization and Conclusion
The final model is used to classify transmission line faults, and we visualize results with confusion matrices and fault classification accuracy across different line faults.


<h2 align="center">Solar Panel Fault Detection</h2>

### 1. Introduction
We focus on detecting faults in solar panels using image data and deep learning techniques. The objective is to classify faults that can reduce solar power generation, such as dirt accumulation, shading, and physical damage.

- **The Notebook and Dataset:**
  - [Solar Panel Fault Detection](https://github.com/SuMayaBee/RenewAble-Synergy/blob/main/Maintenance%20Fault%20Detection/Solar%20Panel%20Fault%20Detection/Solar%20Panel%20Fault%20Detection.ipynb)
  - [Dataset](https://github.com/SuMayaBee/RenewAble-Synergy/tree/main/Maintenance%20Fault%20Detection/Solar%20Panel%20Fault%20Detection/Dataset/Faulty_solar_panel)

### 2. Data Loading and Preprocessing
We use image data of solar panels, preprocessing it with:
- **Image Augmentation:** Techniques such as rotation, flipping, and scaling to increase model robustness.
- **Data Normalization:** Scaling pixel values for consistent input to the neural network.

### 3. Model Selection and Implementation
We implement a Convolutional Neural Network (CNN) for image classification:
- **Model Architecture:** Consists of multiple convolutional layers, ReLU activations, and pooling layers.
- **Output Layer:** Softmax activation for multi-class classification of fault types.

### 4. Model Training and Evaluation
We train the model using:
- **Loss Function:** Categorical cross-entropy for multi-class classification.
- **Optimizer:** Adam optimizer with learning rate scheduling.
- **Early Stopping:** To prevent overfitting based on validation loss.

### 5. Evaluation Metrics
We assess model performance using:
- **Accuracy:** To evaluate classification success.
- **Confusion Matrix:** Visualizing true positives, false positives, and misclassifications.

### 6. Results and Visualization
We visualize the model’s predictions on validation images, highlighting correctly and incorrectly classified samples with confidence scores.

<h2 align="center">Wind Turbine Failure Detection</h2>

### 1. Introduction
We focus on detecting faults in wind turbines using sensor data and various machine learning models. The aim is to classify different failure modes such as bearing faults, blade faults, and motor failures to enhance predictive maintenance.

- **The Notebook and Dataset:**
  - [Wind Turbine Failure Detection](https://github.com/SuMayaBee/RenewAble-Synergy/blob/main/Maintenance%20Fault%20Detection/Wind%20Turbine%20Fault%20Detection/Wind%20turbine%20failure%20detection.ipynb)
  - [Dataset](https://github.com/SuMayaBee/RenewAble-Synergy/tree/main/Maintenance%20Fault%20Detection/Wind%20Turbine%20Fault%20Detection/Dataset)

### 2. Data Loading and Preprocessing
We load sensor data containing operational metrics (e.g., temperature, vibration) and preprocess it by:
- **Handling Imbalanced Data:** Using techniques like SMOTE to address class imbalance.
- **Feature Scaling:** Normalizing features to improve model convergence.

### 3. Model Selection and Implementation
We explore multiple classification models:
- **Logistic Regression & SVM:** Initial baselines for comparison.
- **Decision Tree & Random Forest:** Tree-based models to handle non-linearity in the data.
- **XGBoost:** Gradient boosting for enhanced classification performance.

### 4. Model Training and Evaluation
We train and evaluate models using:
- **Cross-Validation:** Ensuring model generalization across multiple folds.
- **Performance Metrics:** Precision, recall, and F1-score, focusing on critical failure classes.

### 5. Results and Model Tuning
The Random Forest model showed the best performance with a 93.5% accuracy. However, addressing rare failure modes (e.g., motor failures) required oversampling techniques and fine-tuning of hyperparameters.

# Smart Grid Stability Prediction

### 1. Introduction
We focus on predicting the stability of smart grids in response to varying load conditions and renewable energy inputs. The goal is to classify grid stability as "stable" or "unstable" using machine learning techniques, allowing for proactive grid management.

- **The Notebook and Dataset:**
  - [Smart Grid Stability Prediction](https://github.com/SuMayaBee/RenewAble-Synergy/blob/main/Smart%20Grid%20Stability%20Prediction/Smart%20grid%20stability%20prediction.ipynb)
  - [Dataset](https://github.com/SuMayaBee/RenewAble-Synergy/tree/main/Smart%20Grid%20Stability%20Prediction/Dataset)

### 2. Data Loading and Preprocessing
We work with data that captures smart grid metrics such as voltage, current, and power factors. Preprocessing includes:
- **Feature Engineering:** Creating interaction terms between power metrics.
- **Data Normalization:** Standardizing input features for better model performance.

### 3. Model Selection and Implementation
We implement and compare various classification models:
- **Logistic Regression:** Baseline model for binary classification.
- **Random Forest & XGBoost:** Tree-based models that capture complex interactions between grid features.
- **Neural Networks:** A deep learning approach to capture non-linear relationships.

### 4. Model Training and Evaluation
We evaluate models using:
- **Accuracy, Precision, Recall, and F1-Score:** Standard metrics to assess classification performance.
- **Confusion Matrix:** To visualize false positives and false negatives.

### 5. Hyperparameter Tuning
We fine-tune models using RandomizedSearchCV for selecting the best parameters for XGBoost, optimizing factors like `max_depth`, `gamma`, and `reg_alpha`.

### 6. Results and Conclusion
XGBoost and Neural Networks deliver the best performance, achieving high accuracy in predicting grid stability. Future improvements involve integrating more real-time data and exploring ensemble techniques.

