import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
import librosa

# Load the saved/trained model
model_path = "C:/Age Gender Detection/Voice Recognition/voice_recognition_model.h5"
model = load_model(model_path)

if model is None:
    print("Error: Failed to load the model")
    exit(1)

def extract_feature(file, mel=True, n_mfcc=20):
    try:
        # Load audio file
        y, sr = librosa.load(file, sr=None)

        # Extract features
        if mel:
            feature = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        else:
            feature = librosa.feature.melspectrogram(y=y, sr=sr)

        # Pad or truncate the features to ensure consistent length along the time axis
        target_length = 128
        if feature.shape[1] < target_length:
            feature = np.pad(feature, ((0, 0), (0, target_length - feature.shape[1])), mode='constant')
        elif feature.shape[1] > target_length:
            feature = feature[:, :target_length]

        # Add channel dimension
        feature = np.expand_dims(feature, axis=-1)

        return feature
    except Exception as e:
        print(f"Error extracting features from the file: {e}")
        return None

def predict_gender():
    # Get the path to the WAV file
    file = file_entry.get()

    if not os.path.isfile(file):
        messagebox.showerror("Error", f"File '{file}' does not exist")
        return

    # Extract features from the WAV file
    features = extract_feature(file)

    if features is None:
        messagebox.showerror("Error", "Feature extraction failed. Please check the WAV file.")
        return

    # Perform inference (prediction)
    predictions = model.predict(features)

    # Interpret prediction
    prediction = predictions[0][0]  # Assuming it's a single value representing probability for one class

    # Determine gender based on prediction probability
    gender = "male" if prediction >= 0.5 else "female"

    # Display the result
    result_label.config(text=f"Result: {gender}")

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

# Create the GUI window
root = tk.Tk()
root.title("Voice Gender Predictor")

# Create widgets
file_label = tk.Label(root, text="Select WAV file:")
file_entry = tk.Entry(root, width=50)
browse_button = tk.Button(root, text="Browse", command=browse_file)
predict_button = tk.Button(root, text="Predict Gender", command=predict_gender)
result_label = tk.Label(root, text="")

# Arrange widgets using grid layout
file_label.grid(row=0, column=0, padx=5, pady=5)
file_entry.grid(row=0, column=1, padx=5, pady=5)
browse_button.grid(row=0, column=2, padx=5, pady=5)
predict_button.grid(row=1, column=1, padx=5, pady=5)
result_label.grid(row=2, column=1, padx=5, pady=5)

# Start the GUI event loop
root.mainloop()
