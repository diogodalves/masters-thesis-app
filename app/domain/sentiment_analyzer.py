import pickle

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn
import torch.nn.functional as F

class InferencePipeline:
    def __init__(self, model_path, label_encoder_path, image_size=(224, 224)):
        self.model = self.load_model(model_path)
        self.model.eval()
        self.label_encoder = self.load_label_encoder(label_encoder_path)

        self.transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def load_model(self, model_path):
        model = MobileNetV3(num_labels=7)
        model.load_state_dict(torch.load(model_path))
        return model

    def load_label_encoder(self, label_encoder_path):
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        return label_encoder

    def preprocess_pil_image(self, pil_image):
        if self.transforms:
            image = self.transforms(pil_image)
        return image.unsqueeze(0)

    def predict_pil_image(self, pil_image):
        image = self.preprocess_pil_image(pil_image)

        with torch.no_grad():
            output = self.model(image)

            probabilities = F.softmax(output, dim=1)

            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_label = self.label_encoder.inverse_transform([predicted_idx.item()])
        return predicted_label[0], confidence.item()





def MobileNetV3(num_labels=7):
    mobilenet_v3_small = models.mobilenet_v3_small(pretrained=False)
    mobilenet_v3_small.classifier[3] = torch.nn.Linear(mobilenet_v3_small.classifier[3].in_features,
                                                       num_labels)
    mobilenet_v3_small.train()
    return mobilenet_v3_small