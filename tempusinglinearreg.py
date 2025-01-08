import warnings
import pandas as pd
import numpy as np
import datetime
import pymysql as MySQLdb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import time

# Database connection
db = MySQLdb.connect(
    host="localhost",
    user="root",
    passwd="root",
    db="predictionlr",
    port=3307
)
cur = db.cursor()

# Function to suppress warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# Read and preprocess the dataset
df = pd.read_csv("niteditedfinal.csv")
df = df.fillna(0)

# Ensure non-zero mean for non-zero elements (if needed)
nonzero_mean = df[df != 0].mean()

print(df.head())

# Prepare data for training
cols = [0, 1, 2, 3, 4]  # Features
X = df[df.columns[cols]].values

cols = [5]  # Target: Temperature
Y_temp = df[df.columns[cols]].values

cols = [6]  # Target: GHI
Y_ghi = df[df.columns[cols]].values

# Split data into training and testing sets
x_train, x_test, y_temp_train, y_temp_test = train_test_split(X, Y_temp, random_state=42)
x_train, x_test, y_ghi_train, y_ghi_test = train_test_split(X, Y_ghi, random_state=42)

# Train models
lr1 = LinearRegression()
lr2 = LinearRegression()

lr1.fit(x_train, y_temp_train.ravel())
lr2.fit(x_train, y_ghi_train.ravel())

# Main loop for predictions
try:
    for _ in range(10):  # Run for 10 iterations (or replace with `while True` for infinite loop)
        # Predict for the next time interval
        nextTime = datetime.datetime.now() + datetime.timedelta(minutes=15)
        now = nextTime.strftime("%Y,%m,%d,%H,%M")
        now = list(map(float, now.split(",")))  # Convert to float
        now = np.array(now).reshape(1, -1)     # Ensure 2D array for prediction

        temp = lr1.predict(now)[0]  # Predict temperature
        ghi = lr2.predict(now)[0]   # Predict GHI

        print("Predicted Temperature:", temp)
        print("Predicted GHI:", ghi)

        # Calculate power
        f = 0.18 * 7.4322 * ghi
        insi = temp - 25
        midd = 0.95 * insi
        power = f * midd

        print("Calculated Power:", power)

        # Insert into database
        time_str = nextTime.strftime("%Y-%m-%d %H:%M:%S")  # Ensure proper DATETIME format
        sql = """INSERT INTO power_predictionlr (time_updated, Temperature, GHI, power)
                 VALUES (%s, %s, %s, %s)"""
        values = (time_str, float(temp), float(ghi), float(power))

        try:
            print("Writing to the database...")
            cur.execute(sql, values)
            db.commit()
            print("Write complete")
        except MySQLdb.Error as e:
            print(f"Error while writing to database: {e}")

        # Wait before next iteration
        time.sleep(10)

except KeyboardInterrupt:
    print("Process interrupted by user.")

finally:
    # Close the database connection
    db.close()
