import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm

class AMNet(nn.Module):
    def __init__(self):
        super(AMNet, self).__init__()
        # Load pre-trained ResNet
        resnet = models.resnet50(pretrained=True)
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Add new layers
        self.fc1 = nn.Linear(2048, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class MemorabilityPredictor:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model = AMNet().to(self.device)
        self.model.eval()
        
        # Standard ImageNet transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict_image(self, image_path):
        """Predict memorability score for a single image."""
        try:
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                score = self.model(image_tensor).item()
            
            return score
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def predict_batch(self, image_folder, batch_size=32):
        """Predict memorability scores for all images in a folder."""
        results = {}
        image_paths = []
        
        # Get all image files
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image_paths.append(os.path.join(image_folder, filename))
        
        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            # Prepare batch
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    tensor = self.transform(image)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"Error loading {img_path}: {str(e)}")
                    continue
            
            if batch_tensors:
                # Stack tensors and predict
                batch = torch.stack(batch_tensors).to(self.device)
                with torch.no_grad():
                    scores = self.model(batch).cpu().numpy().flatten()
                
                # Store results
                for path, score in zip(batch_paths, scores):
                    results[os.path.basename(path)] = score
        
        return results

# Only if you want to test the code directly:
if __name__ == "__main__":
    predictor = MemorabilityPredictor()