import cv2
import numpy as np
import base64

from bson import ObjectId
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from tensorflow.keras.models import load_model
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
from datetime import datetime, timedelta
import time



app = Flask(__name__)

app.config["MONGO_URI"] = "mongodb+srv://pabasaramjayamanne:xN1nXbjCCUbD7mrr@goodnight.11wmo.mongodb.net/good_night?retryWrites=true&w=majority&appName=GoodNight"
mongo = PyMongo(app)

try:
    mongo = PyMongo(app)
    print("✅ Connected to MongoDB successfully!")
except Exception as e:
    print(f"❌ Failed to connect to MongoDB: {e}")

# Get reference to a collection
db = mongo.db.emotions

@app.route('/emotions', methods=['POST'])
def add_emotion():
    try:
        data = request.json
        print(data)
        # Validate required fields
        if not all(key in data for key in ["mood", "note", "imageUrl"]):
            return jsonify({"error": "Missing required fields"}), 400

        # Create document
        emotion_doc = {
            "mood": data["mood"],
            "note": data["note"],
            "level": data["level"],
            "imageUrl": data["imageUrl"],
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        }

        # Insert into MongoDB
        inserted_id = db.insert_one(emotion_doc).inserted_id

        return jsonify({"message": "Emotion added successfully", "id": str(inserted_id)}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# GET: Fetch All Emotions
@app.route('/emotions', methods=['GET'])
def get_emotions():
    try:
        emotions = mongo.db.emotions.find()
        emotion_list = []

        for emotion in emotions:
            emotion_list.append({
                "_id": str(emotion["_id"]),
                "mood": emotion["mood"],
                "note": emotion["note"],
                "level": emotion.get("level"),
                "image_url": emotion["imageUrl"],
                "createdAt": emotion["createdAt"],
                "updatedAt": emotion["updatedAt"]
            })

        print("Fetched Emotions:", emotion_list)  # <--- Added print statement

        return jsonify(emotion_list), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# GET: Fetch Emotions by Month
@app.route('/emotions/month/<int:year>/<int:month>', methods=['GET'])
def get_emotions_by_month(year, month):
    try:
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)

        emotions = mongo.db.emotions.find({
            "createdAt": {"$gte": start_date, "$lt": end_date}
        })

        emotion_list = []
        for emotion in emotions:
            emotion_list.append({
                "_id": str(emotion["_id"]),
                "mood": emotion["mood"],
                "note": emotion["note"],
                "level": emotion.get("level"),
                "image_url": emotion["imageUrl"],
                "createdAt": emotion["createdAt"],
                "updatedAt": emotion["updatedAt"]
            })

        return jsonify(emotion_list), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# DELETE: Remove an Emotion and Image
@app.route('/emotions/<emotion_id>', methods=['DELETE'])
def delete_emotion(emotion_id):
    try:
        # Find the emotion
        emotion = mongo.db.emotions.find_one({"_id": ObjectId(emotion_id)})

        if not emotion:
            return jsonify({"error": "Emotion not found"}), 404

        # If an image exists, delete it from GridFS
        if "image_id" in emotion:
            fs.delete(emotion["image_id"])

        # Delete the emotion document from MongoDB
        mongo.db.emotions.delete_one({"_id": ObjectId(emotion_id)})

        return jsonify({"message": "Emotion deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

CORS(app, resources={r"/": {"origins": ""}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Load the pre-trained model
print("Loading model...")
new_model = load_model("model_epoch20_Final.h5")
print("Model loaded successfully")

# Load face detection model
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_dict = {0: "Stress", 1: "Angry", 2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprise"}

# Frame processing lock to prevent parallel processing of frames from same client
processing_locks = {}


# Keep the HTTP endpoint for compatibility
@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    try:
        print('22222222222222')
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = base64.b64decode(data["image"])

        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        result = process_frame(frame)
        return jsonify(result)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


def process_frame(frame):
    """Process a single frame and return emotion detection results"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces - optimize parameters for speed
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,  # Increased from 1.1 for faster processing
        minNeighbors=4,
        minSize=(30, 30),  # Minimum face size to detect
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    emotion = "Unknown"
    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (224, 224))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / 255.0

        predictions = new_model.predict(face_roi)
        emotion_index = np.argmax(predictions)
        emotion = emotion_dict.get(emotion_index, "Neutral")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert back to base64
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(img_encoded).decode("utf-8")

    return {"emotion": emotion, "processed_image": img_base64, "message": "Emotion detected."}


# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Initialize processing lock for this client
    processing_locks[request.sid] = threading.Lock()


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    # Remove processing lock for this client
    if request.sid in processing_locks:
        del processing_locks[request.sid]


@socketio.on('frame')
def handle_frame(data):

    print('1111111111111111')
    client_sid = request.sid

    # Check if we're already processing a frame for this client
    if client_sid not in processing_locks or processing_locks[client_sid].locked():
        # Skip this frame if we're already processing one
        return

    # Acquire lock for this client
    with processing_locks[client_sid]:
        try:
            # Decode the base64 image
            image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
            image_bytes = base64.b64decode(image_data)

            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                emit('emotion_result', {'status': 'error', 'message': 'Invalid frame'})
                return

            # Resize frame to reduce processing time
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Optimize face detection parameters
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=4,
                minSize=(30, 30)
            )

            results = []

            for (x, y, w, h) in faces:
                # Scale coordinates back to original size
                x, y, w, h = x * 2, y * 2, w * 2, h * 2

                # Extract face region from original frame
                roi_color = frame[y // 2:(y + h) // 2, x // 2:(x + w) // 2]

                if roi_color.size == 0:
                    continue

                # Preprocess the face region for emotion detection
                try:
                    face_roi = cv2.resize(roi_color, (224, 224))
                    face_roi = np.expand_dims(face_roi, axis=0)
                    face_roi = face_roi / 255.0

                    # Predict the emotion
                    predictions = new_model.predict(face_roi)
                    emotion_index = np.argmax(predictions)
                    status = emotion_dict.get(emotion_index, "Neutral")

                    results.append({
                        'face': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                        'emotion': status,
                        'confidence': float(predictions[0][emotion_index])
                    })
                except Exception as e:
                    print(f"Error processing face: {str(e)}")
                    continue

            # Send results back to the client
            emit('emotion_result', {'status': 'success', 'results': results})

        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            emit('emotion_result', {'status': 'error', 'message': str(e)})


if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)