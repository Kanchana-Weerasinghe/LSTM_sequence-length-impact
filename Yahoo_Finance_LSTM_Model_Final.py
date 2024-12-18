#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from datetime import datetime,timedelta
from sklearn.model_selection import TimeSeriesSplit
import sweetviz
import random

# Define training parameters
training_start_date = '1970-02-02'
training_end_date = '2024-06-01'
prediction_start_date = '2024-06-02'
prediction_end_date = '2024-06-12'
n_future = 1
batch_size = 16
hidden_size = 64
output_size = 1

SEED = 300
random.seed(SEED)               # Python random module
np.random.seed(SEED)            # Numpy
torch.manual_seed(SEED)         # PyTorch
torch.cuda.manual_seed(SEED)    # PyTorch for GPU (if using CUDA)
torch.cuda.manual_seed_all(SEED) # All GPU devices
torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior in PyTorch
torch.backends.cudnn.benchmark = False 

df_full = pd.read_csv('data/Yahoo/yahoo_finance.csv')
df_full['Date'] = pd.to_datetime(df_full['Date'])
df_full.head()
#%%
# yahoo_finance_eda= sweetviz.analyze(df_full)
# yahoo_finance_eda.show_html("Yahoo_Finance.html")
#%%

cols = list(df_full)[1:2]  # Adjust based on your DataFrame structure
df_for_training = df_full[(df_full['Date'] >= training_start_date) & (df_full['Date'] <= training_end_date)][cols].astype(float)

# Convert 'Date' column to index for time series analysis
df_for_training.set_index(df_full[(df_full['Date'] >= training_start_date) & (df_full['Date'] <= training_end_date)]['Date'], inplace=True)

# Plot the time series data for a visual check
plt.figure(figsize=(12, 6))
plt.plot(df_for_training, label="Operning Price")
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Operning Price Overview')
plt.legend()
plt.show()

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x[:, -1, :])  # Use only the last output
        x = self.fc(x)
        return x

# Prepare the training data
def prepare_data(df, n_past):

    cols = list(df)[1:6]  # Adjust based on your DataFrame structure
    df_for_training = df[(df['Date'] >= training_start_date) & (df['Date'] <= training_end_date)]
    df_for_training = df_for_training.sort_values(by='Date', ascending=True).reset_index(drop=True)
    df_for_training =df_for_training[cols].astype(float)

    # Normalize training data 
    scaler = StandardScaler()
    df_for_training_scaled = scaler.fit_transform(df_for_training)

    # Prepare data for LSTM
    trainX, trainY = [], []
    for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
        trainX.append(df_for_training_scaled[i - n_past:i, :])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])  # Predicting the first column

    trainX, trainY = np.array(trainX), np.array(trainY)
    return trainX, trainY, scaler, df_for_training, cols

def train_model_single_split(n_past,prediction_period ,patience=5, min_delta=0.001):
    trainX, trainY, scaler, df_for_training, cols = prepare_data(df_full, n_past)

    # Convert numpy arrays to torch tensors
    trainX = torch.from_numpy(trainX).float()
    trainY = torch.from_numpy(trainY).float().view(-1, 1)

    # Split the data into training and validation sets
    train_size = int(len(trainX) * 0.8)
    trainX_tensor, valX_tensor = trainX[:train_size], trainX[train_size:]
    trainY_tensor, valY_tensor = trainY[:train_size], trainY[train_size:]

    # Create data loaders for training and validation
    train_dataset = torch.utils.data.TensorDataset(trainX_tensor, trainY_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, criterion, and optimizer
    model = LSTMModel(input_size=trainX.shape[2], hidden_size=hidden_size, output_size=output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early stopping parameters
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1000):  # Use a large number of epochs; training will stop early if needed
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(valX_tensor)
            val_loss = criterion(val_outputs, valY_tensor).item()

        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0  # Reset counter if validation loss improves
            best_model_weights = model.state_dict()  # Save the best model weights
        else:
            patience_counter += 1

        # Stop if patience is reached
        if patience_counter >= patience:
            print("Early stopping triggered.")
            model.load_state_dict(best_model_weights)  # Restore best model weights
            break

    # Final validation metrics
    val_outputs = val_outputs.numpy()
    valY_tensor = valY_tensor.numpy()

    mse = mean_squared_error(valY_tensor, val_outputs)
    mae = mean_absolute_error(valY_tensor, val_outputs)
    r2 = r2_score(valY_tensor, val_outputs)

    print(f"Window Size -{n_past} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    without_immediate_past_actual_df = predict_future_without_immediate_past_actual(model, scaler, n_past, prediction_period, cols)
    with_immediate_past_actual_df = predict_future_with_immediate_past_actual(model, scaler, n_past, prediction_period, cols)

    return  without_immediate_past_actual_df, with_immediate_past_actual_df

# Define function to train model with a given window size
def train_model_rolling_window(prediction_period,n_past, n_future, window_size_ratio, step_size_ratio, patience, min_delta):
    trainX, trainY, scaler, df_for_training, cols = prepare_data(df_full, n_past)

    # Convert numpy arrays to torch tensors
    trainX = torch.from_numpy(trainX).float()
    trainY = torch.from_numpy(trainY).float().view(-1, 1)

    # Compute integer window size and step size
    total_samples = len(trainX)
    window_size = int(window_size_ratio * total_samples)  # Convert ratio to absolute size
    step_size = int(step_size_ratio * total_samples)  # Convert ratio to absolute step

    # Rolling window cross-validation
    mse_scores, mae_scores, r2_scores = [], [], []

    # Loop over the rolling windows
    for start in range(0, total_samples - window_size, step_size):
        end = start + window_size
        if end + n_future > total_samples:  # Ensure we have enough data for validation
            break

        trainX_window = trainX[start:end]
        trainY_window = trainY[start:end]
        valX_window = trainX[end:end + n_future]
        valY_window = trainY[end:end + n_future]

        # Create data loaders for the current rolling window
        train_dataset = torch.utils.data.TensorDataset(trainX_window, trainY_window)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model, criterion, and optimizer
        model = LSTMModel(input_size=trainX.shape[2], hidden_size=hidden_size, output_size=output_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Early stopping parameters
        best_val_loss = float("inf")
        patience_counter = 0

        # Train the model for the current window
        for epoch in range(1000):  # Use a large number of epochs; training will stop early if needed
            model.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(valX_window)
                val_loss = criterion(val_outputs, valY_window).item()

            print(f"Window [{start}:{end}], Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0  # Reset counter if validation loss improves
                best_model_weights = model.state_dict()  # Save the best model weights
            else:
                patience_counter += 1

            # Stop if patience is reached
            if patience_counter >= patience:
                print("Early stopping triggered.")
                model.load_state_dict(best_model_weights)  # Restore best model weights
                break

        # Evaluate the model on the validation set
        val_outputs = val_outputs.numpy()
        valY_window = valY_window.numpy()

        mse = mean_squared_error(valY_window, val_outputs)
        mae = mean_absolute_error(valY_window, val_outputs)
        r2 = r2_score(valY_window, val_outputs)

        mse_scores.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)

        print(f"Window [{start}:{end}] - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # Aggregate results from all windows
    avg_mse = np.mean(mse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)

    # Make final predictions for the entire dataset
    without_immediate_past_actual_df = predict_future_without_immediate_past_actual(model, scaler, n_past, prediction_period, cols)
    with_immediate_past_actual_df = predict_future_with_immediate_past_actual(model, scaler, n_past, prediction_period, cols)

    return  without_immediate_past_actual_df, with_immediate_past_actual_df

# Define function to make predictions and visualize results

def predict_future_without_immediate_past_actual(model, scaler, n_past, n_future, cols):
    output_folder = "Prediction"
    start_date = pd.to_datetime(prediction_start_date)
    sequance_start_date = start_date - timedelta(days=n_past)
    
    # Get the latest available data from df_full
    df_for_prediction = df_full[(df_full['Date'] >= sequance_start_date) & (df_full['Date'] < prediction_start_date)]
    df_for_prediction = df_for_prediction.sort_values(by='Date', ascending=True).reset_index(drop=True)
    df_for_prediction = df_for_prediction[cols].astype(float)
    
    prediction_scaled = scaler.transform(df_for_prediction)  # Normalize
    
    # Convert the data to the right shape for the model (samples, time steps, features)
    model_input = torch.from_numpy(prediction_scaled[-n_past:]).float().unsqueeze(0)  # Use only the last `n_past` rows
    
    future_predictions = []
    future_dates = pd.date_range(prediction_start_date, periods=n_future, freq='B')  # Generate future business dates

    for i in range(n_future):
        # Make the prediction for the next day
        model.eval()
        with torch.no_grad():
            predicted_scaled = model(model_input)  # Predict the next day's scaled value
        
        # Extract the prediction and store it
        predicted_scaled = predicted_scaled.item()  # Convert tensor to scalar
        future_predictions.append(predicted_scaled)
        
        # Update the model input
        # Create a placeholder array matching the scaler's input shape
        next_input_scaled = np.zeros((1, prediction_scaled.shape[1]))
        next_input_scaled[0, 0] = predicted_scaled  # Insert the predicted "Open" price
        
        # Append the new input and slide the window
        prediction_scaled = np.vstack((prediction_scaled[1:], next_input_scaled))
        model_input = torch.from_numpy(prediction_scaled[-n_past:]).float().unsqueeze(0)

    # Inverse scale the predicted values
    predictions_array = np.zeros((len(future_predictions), prediction_scaled.shape[1]))  # Create a dummy array
    predictions_array[:, 0] = future_predictions  # Fill in the predicted "Open" prices
    future_predictions_unscaled = scaler.inverse_transform(predictions_array)[:, 0]  # Only extract the "Open" prices

    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Future_Predicted_Value': future_predictions_unscaled
    })
    output_csv_path = os.path.join(output_folder, "filtered_predictions.csv")
    predictions_df.to_csv(output_csv_path, index=False)

    return predictions_df

def predict_future_with_immediate_past_actual(model, scaler, n_past, n_future, cols):

    output_folder = "Prediction"
    start_date = pd.to_datetime(prediction_start_date)
    sequence_start_date = start_date - timedelta(days=n_past)
    
    # Get the latest available data from df_full
    df_for_prediction = df_full[(df_full['Date'] >= sequence_start_date) & (df_full['Date'] < prediction_start_date)]
    df_for_prediction = df_for_prediction.sort_values(by='Date', ascending=True).reset_index(drop=True)
    df_for_prediction = df_for_prediction[cols].astype(float)
    
    prediction_scaled = scaler.transform(df_for_prediction)  # Normalize
    
    # Convert the data to the right shape for the model (samples, time steps, features)
    model_input = torch.from_numpy(prediction_scaled[-n_past:]).float().unsqueeze(0)  # Use only the last `n_past` rows
    
    future_predictions = []
    future_dates = pd.date_range(prediction_start_date, periods=n_future, freq='B')  # Generate future business dates

    for i in range(n_future):
        # Make the prediction for the next day
        model.eval()
        with torch.no_grad():
            predicted_scaled = model(model_input)  # Predict the next day's scaled value
        
        # Extract the prediction and store it
        predicted_scaled = predicted_scaled.item()  # Convert tensor to scalar
        future_predictions.append(predicted_scaled)
        
        # Update the model input with the actual past value (from prediction_scaled)
        if i < len(prediction_scaled) - 1:
            # Use the actual value from `prediction_scaled`
            next_input_scaled = prediction_scaled[i + 1]
        else:
            # If beyond the length of `prediction_scaled`, use the last predicted value
            next_input_scaled = np.zeros((prediction_scaled.shape[1],))
            next_input_scaled[0] = predicted_scaled  # Only insert the predicted value
        
        # Append the new input and slide the window
        updated_input_scaled = np.vstack((model_input.numpy().squeeze(0)[1:], next_input_scaled))  # Slide the window
        model_input = torch.from_numpy(updated_input_scaled).float().unsqueeze(0)

    # Inverse scale the predicted values
    predictions_array = np.zeros((len(future_predictions), prediction_scaled.shape[1]))  # Create a dummy array
    predictions_array[:, 0] = future_predictions  # Fill in the predicted "Open" prices
    future_predictions_unscaled = scaler.inverse_transform(predictions_array)[:, 0]  # Only extract the "Open" prices

    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Future_Predicted_Value': future_predictions_unscaled
    })
    output_csv_path = os.path.join(output_folder, "filtered_predictions.csv")
    predictions_df.to_csv(output_csv_path, index=False)

    return predictions_df

def combine_and_plot(df_prediction_actual, predictions_df,n_past,trainig_type):
    # Ensure the `Date` column is a datetime type for both DataFrames
    df_prediction_actual['Date'] = pd.to_datetime(df_prediction_actual['Date'])
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])

    # Merge the DataFrames on the `Date` column
    combined_df = pd.merge(
        df_prediction_actual[['Date', 'Open']],  # Select only Date and Actual Open value
        predictions_df.rename(columns={'Future_Predicted_Value': 'Predicted_Open'}),  # Rename for clarity
        on='Date',
        how='inner'  # Ensure alignment by common dates
    )

    y_actual = combined_df['Open']  # Actual Open values
    y_predicted = combined_df['Predicted_Open']  # Predicted Open values

    # Calculate metrics
    mse = mean_squared_error(y_actual, y_predicted)
    mae = mean_absolute_error(y_actual, y_predicted)
    r2 = r2_score(y_actual, y_predicted)

    metrics_df = pd.DataFrame({
    "Metric": ["Mean Squared Error", "Mean Absolute Error", "R-Squared"],
    "Value": [mse, mae, r2]
    })

    output_folder = "plots"
    file_path = os.path.join(output_folder, f"metrics_{n_past}_{trainig_type}.txt")
    metrics_df.to_csv(file_path, sep='\t', index=False)

    print(f"MSE (Mean Squared Error): {mse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"R² (Coefficient of Determination): {r2:.4f}")
    # Plot the actual and predicted open values
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    file_path = os.path.join(output_folder, f"Actual_vs_Predicted_{n_past}_{trainig_type}.png")

    plt.figure(figsize=(14, 7))
    plt.title(f'Actual vs Predicted Open Prices (Window Size: {n_past}) - {trainig_type}')
    plt.plot(combined_df['Date'], combined_df['Open'], label='Actual Open Price', color='blue')
    plt.plot(combined_df['Date'], combined_df['Predicted_Open'], label='Predicted Open Price', color='orange', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.ylim(0, 180)  # Set y-axis limits
    plt.legend()  # Add a legend for better interpretation
    plt.savefig(file_path, dpi=300)
    plt.close()

window_sizes = [5,10,20,60,100]  # Example window sizes
results = []

batch_size = 32  # Example batch size for training
hidden_size = 64  # Hidden size for LSTM
output_size = 1  # Output size, typically 1 for regression

start_date = datetime.strptime(prediction_start_date, '%Y-%m-%d')
end_date = datetime.strptime(prediction_end_date, '%Y-%m-%d')
date_difference = end_date - start_date
prediction_period=int(date_difference.days)

df_prediction_actual = df_full[(df_full['Date'] >= prediction_start_date) & (df_full['Date'] <= prediction_end_date)]

#%%

for window_size in window_sizes:
    n_past=window_size
    without_immediate_past_actual_df,with_immediate_past_actual_df =train_model_rolling_window(prediction_period,n_past, 
    n_future=1, 
    window_size_ratio=0.8, 
    step_size_ratio=0.1, 
    patience=5, 
    min_delta=0.001) 
    combine_and_plot(df_prediction_actual,without_immediate_past_actual_df,n_past,"Rolling window Traning - Without Immidiate Pass")
    combine_and_plot(df_prediction_actual,with_immediate_past_actual_df,n_past,"Rolling window Traning - With Immidiate Pass")
  

# # %%
for window_size in window_sizes:
    n_past=window_size
    without_immediate_past_actual_df, with_immediate_past_actual_df = train_model_single_split(n_past, prediction_period,patience=5, min_delta=0.001) 
    combine_and_plot(df_prediction_actual,without_immediate_past_actual_df,n_past,"Single Spit Traning - Without Immidiate Pass")
    combine_and_plot(df_prediction_actual,with_immediate_past_actual_df,n_past,"Single Spit Traning - With Immidiate Pass")
  



# %%
