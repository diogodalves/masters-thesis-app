import time
from fastapi import APIRouter, WebSocket
import cv2
import numpy as np
import base64
from collections import deque, Counter

from domain.sentiment_analyzer import InferencePipeline
from PIL import Image

router = APIRouter()

haar_cascade = cv2.CascadeClassifier('models/face_detection/haarcascade_frontalface_default.xml')

best_model_path = "models/sentiment_analysis/distilled_lottery_ticket_590k.pt"
label_encoder_path = "models/sentiment_analysis/label_encoder.pkl"

pipeline = InferencePipeline(model_path=best_model_path,
                             label_encoder_path=label_encoder_path)

buffer_size = 25
face_history = deque(maxlen=buffer_size)
prediction_history = deque(maxlen=buffer_size)
inference_times = deque(maxlen=buffer_size)

last_detected_face = None
interpolated_face = None
no_face_frame_count = 0
MAX_NO_FACE_FRAMES = 10
interpolation_alpha = 0.5

@router.websocket("/camera/ws")
async def websocket_endpoint(websocket: WebSocket):
    global last_detected_face, interpolated_face, no_face_frame_count

    await websocket.accept()
    try:
        while True:
            start_time = time.time()

            data = await websocket.receive_text()
            image_data = data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            scaling_factor = 0.5
            small_img = cv2.resize(gray, (0, 0), fx=scaling_factor, fy=scaling_factor)
            faces = haar_cascade.detectMultiScale(
                small_img, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                no_face_frame_count = 0
                
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = [int(coord / scaling_factor) for coord in largest_face]

                if interpolated_face is None:
                    interpolated_face = (x, y, w, h)
                else:
                    ix, iy, iw, ih = interpolated_face
                    interpolated_face = (
                        int(interpolation_alpha * x + (1 - interpolation_alpha) * ix),
                        int(interpolation_alpha * y + (1 - interpolation_alpha) * iy),
                        int(interpolation_alpha * w + (1 - interpolation_alpha) * iw),
                        int(interpolation_alpha * h + (1 - interpolation_alpha) * ih)
                    )

                last_detected_face = interpolated_face

                face_img = img[y:y + h, x:x + w]

                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

                predicted_label, confidence = pipeline.predict_pil_image(face_pil)

                prediction_history.append((predicted_label, confidence))
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

            else:
                no_face_frame_count += 1
                if no_face_frame_count >= MAX_NO_FACE_FRAMES:
                    interpolated_face = None

            if len(prediction_history) > 0:
                labels, confidences = zip(*prediction_history)
                current_label, label_count = Counter(labels).most_common(1)[0]

                average_confidence = sum(confidences) / len(confidences)
                average_inference_time = sum(inference_times) / len(inference_times)

            if last_detected_face is not None and no_face_frame_count < MAX_NO_FACE_FRAMES:
                (x, y, w, h) = last_detected_face

                alpha = max(1 - no_face_frame_count / MAX_NO_FACE_FRAMES, 0)
                box_color = (0, int(255 * alpha), 0)

                cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)

                label_text = f"Prediction: {current_label}"
                confidence_text = f"Confidence: {average_confidence:.2f}"
                inference_time_text = f"Avg Inference Time: {average_inference_time:.2f}s"

                font_scale = 0.5
                thickness = 2

                cv2.putText(img, label_text, (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, thickness)
                cv2.putText(img, confidence_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, thickness)
                cv2.putText(img, inference_time_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, thickness)

            # cv2.imshow('Debug Window', img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            _, buffer = cv2.imencode('.jpg', img)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_text(f"data:image/jpeg;base64,{jpg_as_text}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()
        # cv2.destroyAllWindows() 
