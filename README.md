# Disaster Response Pipeline Project
(Part of Udacity's Data Scientist Nanodegree using data from Figure Eight)
In this project, disaster data is analyzed to build a model for an API that classifies disaster messages.

### Installing the project
Clone the project into your repository

git clone https://github.com/michka127/udacity-data_scientist-disaster_response_pipelines.git

### Running te project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Important files
data/process_data.py: ETL pipeline used for data import, data cleaning, and loading data in a SQLite database

models/train_classifier.py: A machine learning pipeline used for loading a a SQLite database, creation and training  a classifier, and stores the classifier into a pickle file

run.py: File used for running a Flask web app containing three visualizations using data from the SQLite database. When a user inputs a message into the app, the app returns classification results for all categories of the classifier.
