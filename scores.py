import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import keras
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import ipywidgets as widgets
from IPython.display import display, clear_output

# Load dataset from Google Drive
file_url = 'https://drive.google.com/uc?id=1qlm4OnaAM3bdgti9xyUqK3eSFokjYdA-'
ipl = pd.read_csv(file_url)

# Drop unnecessary columns
columns_to_drop = ['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'mid', 'striker', 'non-striker']
df = ipl.drop(columns_to_drop, axis=1)

# Encode categorical features
label_encoders = {}
for column in ['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Prepare X and y
X = df.drop(['total'], axis=1)
y = df['total']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Define the neural network model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer
    keras.layers.Dense(512, activation='relu'),  # Hidden layer with 512 units and ReLU activation
    keras.layers.Dense(216, activation='relu'),  # Hidden layer with 216 units and ReLU activation
    keras.layers.Dense(1, activation='linear')   # Output layer with linear activation for regression
])

# Compile the model with Huber loss
huber_loss = tf.keras.losses.Huber(delta=1.0)  # Adjust the 'delta' parameter as needed
model.compile(optimizer='adam', loss=huber_loss)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error on Test Set: {mae}")

# Define widgets for prediction interface
venue_widget = widgets.Dropdown(options=df['venue'].unique(), description='Select Venue:')
batting_team_widget = widgets.Dropdown(options=df['bat_team'].unique(), description='Select Batting Team:')
bowling_team_widget = widgets.Dropdown(options=df['bowl_team'].unique(), description='Select Bowling Team:')
striker_widget = widgets.Dropdown(options=df['batsman'].unique(), description='Select Striker:')
bowler_widget = widgets.Dropdown(options=df['bowler'].unique(), description='Select Bowler:')
predict_button = widgets.Button(description="Predict Score")
output_widget = widgets.Output()

# Prediction function
def predict_score(button):
    with output_widget:
        clear_output()
        # Decode selected values from widgets
        decoded_values = {
            'venue': label_encoders['venue'].transform([venue_widget.value])[0],
            'bat_team': label_encoders['bat_team'].transform([batting_team_widget.value])[0],
            'bowl_team': label_encoders['bowl_team'].transform([bowling_team_widget.value])[0],
            'batsman': label_encoders['batsman'].transform([striker_widget.value])[0],
            'bowler': label_encoders['bowler'].transform([bowler_widget.value])[0]
        }
        
        # Prepare input for prediction
        input_data = np.array([list(decoded_values.values())])
        input_data_scaled = scaler.transform(input_data)
        
        # Predict score
        predicted_score = model.predict(input_data_scaled)
        predicted_score = int(predicted_score[0, 0])
        
        # Display prediction
        print(f"Predicted Score: {predicted_score}")

# Link prediction function to button click
predict_button.on_click(predict_score)

# Display widgets and output
display(venue_widget, batting_team_widget, bowling_team_widget, striker_widget, bowler_widget, predict_button, output_widget)
