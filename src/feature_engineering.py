
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# Extract features from the logs and sensor data
def extract_features(maintenance_logs, sensor_data):
    # Example: extract time features, categorical encoding, etc.
    maintenance_logs['date'] = pd.to_datetime(maintenance_logs['date'])
    maintenance_logs['day_of_week'] = maintenance_logs['date'].dt.dayofweek
    return maintenance_logs

#Exploratory Data Analysis (EDA)


def plot_data(maintenance_logs):
    plt.figure(figsize=(10, 5))
    sns.countplot(data=maintenance_logs, x='failure_type')
    plt.title('Failure Types Distribution')
    plt.show()
    
    #Split the data into training and testing sets
   
def split_data(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


