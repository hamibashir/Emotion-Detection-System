import cv2
import mediapipe as mp

# Initialize MediaPipe FaceMesh once globally to avoid re-initializing on every call
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def get_face_landmarks(image, draw=False):
    """
    Extract normalized face landmarks (x,y,z) from an image.
    Returns a flat list of normalized coordinates or empty list if no face.
    """
    if image is None:
        return []

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image once for landmarks
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return []

    face_landmarks = results.multi_face_landmarks[0]

    if draw:
        mp_drawing.draw_landmarks(
            image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec, drawing_spec
        )

    xs = [lm.x for lm in face_landmarks.landmark]
    ys = [lm.y for lm in face_landmarks.landmark]
    zs = [lm.z for lm in face_landmarks.landmark]

    # Normalize by subtracting min to keep scale invariant
    min_x, min_y, min_z = min(xs), min(ys), min(zs)
    normalized_landmarks = []
    for x, y, z in zip(xs, ys, zs):
        normalized_landmarks.extend([x - min_x, y - min_y, z - min_z])

    return normalized_landmarks


def close_resources():
    face_mesh.close()