# Prediction of Store Sales!
## Built Regression models on data from Rossman store chain 
In this project we predict future sales of the supermarket chain [Rossman](https://www.kaggle.com/competitions/rossmann-store-sales)
using two different models. A regression 
forest which uses many regression trees, and a deep neural network.
The project involves:
* **Data preprocessing**, handling missing values and data format.
* **Exploratory data analysis** (EDA) using statistical methods. Goal is to identify (ir)relevant 
features.
* Construct, train and save a **deep neural network** built based on processed data.
* Construct, train and save a **regression forest** that leverages many regression trees that are trained sequentially to
emphasize difficult and under-represented samples.
* Create an **interactive website**, allowing user to input data and receive predictions from chosen model.
* **Containerize** a portion of the code using [Docker](https://www.docker.com/get-started/).
* Provide the user with an interactive version of the project using **jupyter notebooks**.


# Performance
Using the metric: Root mean square percentage error (RMSPE), the results are:
* Deep neural network rmspe of about 20% after 20 epochs
* Regression forest rmspe of about 17.5%.

# Files Description
- **Datasets**: Training and validation data
- **Misc_Files**: .json files to specify input order in streamlit made website, see below.
- **Trained Models**: Trained regression forest and deep neural network ready to be loaded and used.
- **export2MySQL.py, connect.py, config.py, app.ini**: Used on personal machine to upload csv 
data to MySQL server and then directly using the data from the server in the project. 
- **import_csv_from_mysql.py, Full_Project.ipynb** modified version of the code that uses csv files and shows the user
used code along with resulting plots. Does not save trained models!
- **Dockerfile**: When run in a container, see below, displays plots yielded in exploratory daya analysis.
- **DNN.py, regression_forest.py**: Deep neural network and regression forest models respectively.
- **eda.py**: File containing both visual and statistical exploratory data analysis along with data preprocessing.
- **main.py**: File that calls other files and goes through the project from start to finish.


# Download and use the code
## Clone Repo
To get and run the files on your machine, the easiest way is to clone this repo and dive into **main.py** file!
## Dockerfile
The dockerfile could be built into a docker image which could then be run in a container, this is 
very useful since it only requires the user to have [Docker](https://www.docker.com/get-started/) installed.

To run the dockerfile:
1. Install [docker](https://www.docker.com/get-started/).
2. Open docker application.
3. Open the command line in a directory / folder and type:
   * ```docker build -t rossman_image .```
   * ```docker run --name rosssman_container rossman_image```

Don't miss that . (dot) at the end of the first listed line!

After the code is done running, type in the command line:
* ```docker cp rossman_container:/app/Figures .```
This will return a copy of the folder containing the figures from the exploratory data analysis
and place it in the current directory.


# To Fix
* The .ipynb file requires **Datasets** folder to work and also the package jupyter notebook which is not included in
pipfile!
* The **web_hosted_models.py** is supposed to allow any user to open a website
and interact with the models by giving data and receiving predictions, but the ***website*** is ***not deployed yet***. 
