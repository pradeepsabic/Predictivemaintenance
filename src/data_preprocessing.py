import pandas as pd

#Load data from maintenance logs and sensor readings
def load_data(log_file, sensor_file):
    maintenance_logs = pd.read_csv(log_file)
    sensor_data = pd.read_csv(sensor_file)
    return maintenance_logs, sensor_data

#Clean and preprocess the data
def clean_data(maintenance_logs, sensor_data):
    # Drop duplicates and handle missing values
    maintenance_logs.drop_duplicates(inplace=True)
    sensor_data.dropna(inplace=True)
    return maintenance_logs, sensor_data
