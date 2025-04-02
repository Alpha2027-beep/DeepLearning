import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import torchvision.models as models
import numpy as np
import cv2

class DeepfakeEnsembleDetector:
    def __init__(self, model_paths):
        """
        Initialize ensemble deepfake detection models
        
        Args:
            model_paths (dict): Dictionary of model paths
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models with specific architectures
        self.models = [
            self._load_efficientnet_b0(model_paths['deepfake_model1.pth']),
            self._load_efficientnet_b0(model_paths['deepfake_model2.pth']),
            self._load_efficientnet_b0(model_paths['deepfake_detector.pth']),
            self._load_resnet50(model_paths['best_realfake_model.pth']),
            self._load_ensemble_model(model_paths['best_ensemble_model.pth'])
        ]
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_efficientnet_b0(self, model_path):
        """Load EfficientNet-B0 model"""
        try:
            # Create EfficientNet-B0 model
            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Handle different state dict formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Clean state dict
            state_dict = {k.replace('module.', '').replace('classifier.', ''): v 
                          for k, v in state_dict.items()}
            
            # Load state dict
            model.load_state_dict(state_dict, strict=False)
            
            model.eval()
            return model.to(self.device)
        
        except Exception as e:
            print(f"Error loading EfficientNet model {model_path}: {e}")
            raise

    def _load_resnet50(self, model_path):
        """Load ResNet-50 model"""
        try:
            # Create ResNet-50 model
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Handle different state dict formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Clean state dict
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load state dict
            model.load_state_dict(state_dict, strict=False)
            
            model.eval()
            return model.to(self.device)
        
        except Exception as e:
            print(f"Error loading ResNet model {model_path}: {e}")
            raise

    def _load_ensemble_model(self, model_path):
        """Load Ensemble model"""
        try:
            # This might be a custom ensemble model
            # For now, we'll use a standard ResNet50 with binary classification
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Handle different state dict formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Clean state dict
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load state dict
            model.load_state_dict(state_dict, strict=False)
            
            model.eval()
            return model.to(self.device)
        
        except Exception as e:
            print(f"Error loading Ensemble model {model_path}: {e}")
            raise

    def predict(self, image):
        """
        Ensemble prediction across multiple models with majority voting
        
        Args:
            image (PIL.Image or numpy.ndarray): Input image
        
        Returns:
            dict: Prediction results
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Prepare image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Collect predictions from different models
        model_predictions = []
        model_confidences = []
        
        with torch.no_grad():
            for model in self.models:
                output = torch.softmax(model(input_tensor), dim=1)
                pred_prob = output[:, 1].cpu().numpy()[0]
                model_predictions.append(pred_prob > 0.5)
                model_confidences.append(pred_prob)
        
        # Majority voting
        fake_votes = sum(model_predictions)
        total_models = len(self.models)
        
        # Determine final prediction
        is_fake = fake_votes > total_models / 2
        
        return {
            'model_predictions': model_predictions,
            'model_confidences': model_confidences,
            'fake_votes': fake_votes,
            'total_models': total_models,
            'ensemble_prediction': sum(model_confidences) / total_models,
            'is_fake': is_fake
        }