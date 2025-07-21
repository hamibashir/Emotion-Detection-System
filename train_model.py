import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_FILE = "data.txt"
MODEL_FILE = "model.pkl"

def train():
    data = np.loadtxt(DATA_FILE)
    X, y = data[:, :-1], data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Use tuned hyperparameters for better performance (can be adjusted further)
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,  # Use all cores
        class_weight='balanced'  # Handle class imbalance if present
    )

    print("Training Random Forest classifier...")
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(rf, f)

    print(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train()