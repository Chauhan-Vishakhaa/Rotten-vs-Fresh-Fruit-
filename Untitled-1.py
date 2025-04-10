# %%
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# machine learning model here
model = load_model('rottenvsfresh.h5')

# Function to preprocess the image for model prediction
def preprocess_image(image):
    # Resize the image to match the input size of your model
    resized_image = cv2.resize(image, (100, 100))
    # Convert the resized image to RGB (assuming the model expects RGB input)
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to range between 0 and 1
    normalized_image = rgb_image / 255.0
    # Add an additional dimension to represent batch size (1 in this case)
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    return preprocessed_image

#Function to classify fruit using the loaded model
def classify_fruit(frame):
    # Preprocess the frame
    processed_frame = preprocess_image(frame)
    # Make prediction using model
    prediction = model.predict(processed_frame)
    # Determine the label based on the prediction
    label = "Fresh" if prediction[0][0] < 0.711 else "Rotten"  # Adjust threshold if needed
    print("Prediction probabilities:", prediction)
    return label





# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Unable to open webcam.")
    exit()

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    
    # Check if the frame is empty
    if not ret:
        print("Error: Unable to read frame from webcam.")
        break
    
    # Classify the fruit
    fruit_label = classify_fruit(frame)
    
    # Display the classification label on the frame
    cv2.putText(frame, fruit_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Fruit Classifier', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()


# %%



