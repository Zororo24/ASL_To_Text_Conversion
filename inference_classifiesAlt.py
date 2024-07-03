import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import Counter

# Load your model
model_dict = pickle.load(open('./model_ensemble_newdata.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define dictionary to map label indices to characters
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
               12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X',
               23: 'Y', 24: 'J', 25: 'Z'}

# Number of predictions to collect for majority vote
num_predictions = 5

# Accumulate predicted characters
predicted_sentence = ""

# Main loop
while True:
    # Read frame from camera
    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    try:
        if results.multi_hand_landmarks:
            predictions = []
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract hand landmarks and preprocess data
                data_aux = []
                x_ = []
                y_ = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Make prediction
                prediction = model.predict([np.asarray(data_aux)])
                predictions.append(int(prediction[0]))

            # Perform majority vote
            most_common_prediction = Counter(predictions).most_common(1)[0][0]
            predicted_character = labels_dict[most_common_prediction]

            # Output sentence
            a = predicted_character

            # Draw bounding box and predicted character
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
    except Exception as e:
        # print("Error:",e)
        cv2.putText(frame, "Error:Unexpected input", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Show the frame
    cv2.imshow('frame', frame)

    # Check for user input
    key = cv2.waitKey(1)
    if key == ord('q'):  # Quit when 'q' is pressed
        predicted_sentence += a
        print("Predicted sentence:", predicted_sentence)
    if key == ord('a'):
        predicted_sentence = predicted_sentence[:-1]
        print("Predicted sentence:", predicted_sentence)
    if key == ord('z'):
        predicted_sentence += ' '
        print("Predicted sentence:", predicted_sentence)
    if key == ord('x'):
        break
# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()