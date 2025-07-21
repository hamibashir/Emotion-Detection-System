import pickle
import cv2
from utils import get_face_landmarks, close_resources

EMOTIONS = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISED']
MODEL_FILE = "model.pkl"

def main():
    # Load model
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting camera. Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        landmarks = get_face_landmarks(frame, draw=True)

        if len(landmarks) == 1404:
            prediction = model.predict([landmarks])
            emotion_idx = int(prediction[0])
            if 0 <= emotion_idx < len(EMOTIONS):
                label = EMOTIONS[emotion_idx]
            else:
                label = "Unknown"
            cv2.putText(
                frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    close_resources()

if __name__ == "__main__":
    main()
