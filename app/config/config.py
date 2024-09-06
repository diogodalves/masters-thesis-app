from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    best_model_path: str = "app/models/sentiment_analysis/distilled_lottery_ticket_590k.pt"
    label_encoder_path: str = "app/models/sentiment_analysis/label_encoder.pkl"

    haar_cascade_path: str = 'app/models/face_detection/haarcascade_frontalface_default.xml'