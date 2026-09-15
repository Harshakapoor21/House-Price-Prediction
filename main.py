import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_excel("House Price Prediction Dataset.csv.xlsx")

print("Shape of dataset:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nDuplicate values:")
print(df.duplicated().sum())

le = LabelEncoder()

df["Location"] = le.fit_transform(df["Location"])
df["Condition"] = le.fit_transform(df["Condition"])
df["Garage"] = le.fit_transform(df["Garage"])

# Display information after encoding
print("\nInformation after encoding:")
df.info()

# Correlation
print("\nCorrelation:")
corr = df.corr()
print(corr)

# Select input features
x = df[
    [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Floors",
        "YearBuilt",
        "Location",
        "Condition",
        "Garage"
    ]
]

# Select target
y = df["Price"]

# Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:")
print(x_train)

print("\nTesting data:")
print(x_test)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(x_train, y_train)

print("\nModel trained successfully!")

# Make predictions
y_pred = model.predict(x_test)

print("\nPredicted prices:")
print(y_pred)

# Evaluate model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nR2 Score:", r2)
print("Mean Squared Error:", mse)