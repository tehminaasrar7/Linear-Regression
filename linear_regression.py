# Step 1: Importing Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml  # Import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 2: Loading the Dataset
boston = fetch_openml(data_id=531)  # Fetch the Boston Housing Prices dataset
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# Step 3: Preparing the Data
X = df[['RM']]  # Input feature: Average number of rooms
y = df['PRICE']  # Target variable: House price

# Step 4: Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Building and Training the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Making Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluating the Model
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R-squared:', r2_score(y_test, y_pred))

# Step 8: Visualizing the Results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('House Price')
plt.title('Linear Regression Model')
plt.show()
