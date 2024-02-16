# Singapore Resale Flat Price Prediction

## Overview

The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model is based on historical data of resale flat transactions, aiming to assist both potential buyers and sellers in estimating the resale value of a flat.

## Motivation

The resale flat market in Singapore is highly competitive, making it challenging to accurately estimate resale values. Various factors, such as location, flat type, floor area, and lease duration, influence resale prices. This project addresses these challenges by providing users with an estimated resale price based on these factors.

## Project Scope

### Tasks

1. **Data Collection and Preprocessing:**
   - Collect a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB) for the years 1990 to Till Date.
   - Preprocess the data to clean and structure it for machine learning.

2. **Feature Engineering:**
   - Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date.
   - Create any additional features that may enhance prediction accuracy.

3. **Model Selection and Training:**
   - Choose an appropriate machine learning model for regression (e.g., linear regression, decision trees, or random forests).
   - Train the model on historical data, using a portion of the dataset for training.

4. **Model Evaluation:**
   - Evaluate the model's predictive performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) and R2 Score.

5. **Streamlit Web Application:**
   - Develop a user-friendly web application using Streamlit that allows users to input details of a flat (town, flat type, storey range, etc.).
   - Utilize the trained machine learning model to predict the resale price based on user inputs.

6. **Deployment on Render:**
   - Deploy the Streamlit application on the Render platform to make it accessible to users over the internet.

7. **Testing and Validation:**
   - Thoroughly test the deployed application to ensure it functions correctly and provides accurate predictions.

### Deliverables

- A well-trained machine learning model for resale price prediction.
- A user-friendly web application (built with Streamlit/Flask/Django) deployed on the Render platform or any cloud platform.
- Documentation and instructions for using the application.
- A project report summarizing the data analysis, model development, and deployment process.

## Data Source

[Data Source: Singapore Government Data](https://beta.data.gov.sg/collections/189/view)

## Results

The project will benefit both potential buyers and sellers in the Singapore housing market. Buyers can use the application to estimate resale prices and make informed decisions, while sellers can get an idea of their flat's potential market value. Additionally, the project demonstrates the practical application of machine learning in real estate and web development.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/singapore-resale-flat-prediction.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

   Access the application in your web browser at `http://localhost:8501`.

## Contribution

Contributions are welcome! Please create a new branch for any proposed changes and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
