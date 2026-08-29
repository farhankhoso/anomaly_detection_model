import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from pathlib import Path
import os

# Set page config
st.set_page_config(page_title="Anomaly Detection", layout="wide")

# Define the ResNet Feature Extractor
class resnet_feature_extractor(torch.nn.Module):
    def __init__(self):
        super(resnet_feature_extractor, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        def hook(module, input, output):
            self.features.append(output)
        
        self.model.layer2[-1].register_forward_hook(hook)            
        self.model.layer3[-1].register_forward_hook(hook) 

    def forward(self, input):
        self.features = []
        with torch.no_grad():
            _ = self.model(input)
        
        self.avg = torch.nn.AvgPool2d(3, stride=1)
        fmap_size = self.features[0].shape[-2]
        self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)
        
        resized_maps = [self.resize(self.avg(fmap)) for fmap in self.features]
        patch = torch.cat(resized_maps, 1)
        
        return patch

# Define the Autoencoder
class FeatCAE(nn.Module):
    def __init__(self, in_channels=1536, latent_dim=100, is_bn=True):
        super(FeatCAE, self).__init__()
        
        layers = []
        layers += [nn.Conv2d(in_channels, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)]
        
        self.encoder = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.Conv2d(latent_dim, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, in_channels, kernel_size=1, stride=1, padding=0)]
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Decision function - takes top 10 values
def decision_function(segm_map):  
    mean_top_10_values = []
    
    for map in segm_map:
        # Flatten the tensor
        flattened_tensor = map.reshape(-1)
        
        # Sort the flattened tensor (descending order)
        sorted_tensor, _ = torch.sort(flattened_tensor, descending=True)
        
        # Take the top 10 values
        mean_top_10_value = sorted_tensor[:10].mean()
        
        mean_top_10_values.append(mean_top_10_value)
    
    return torch.stack(mean_top_10_values)

# Get list of available model folders
def get_model_folders(base_path="models"):
    """Scan for folders containing the required model files"""
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return []
    
    model_folders = []
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            # Check if folder contains required files
            has_checkpoint = os.path.exists(os.path.join(folder_path, 'anomaly_model_complete.pth'))
            has_threshold = os.path.exists(os.path.join(folder_path, 'threshold_config.json'))
            
            if has_checkpoint and has_threshold:
                model_folders.append(folder)
    
    return sorted(model_folders)

# Load models from selected folder
@st.cache_resource
def load_models(model_folder_path):
    """Load model, backbone, and configs from specified folder"""
    try:
        # Construct file paths
        checkpoint_path = os.path.join(model_folder_path, 'anomaly_model_complete.pth')
        threshold_path = os.path.join(model_folder_path, 'threshold_config.json')
        
        # Verify files exist
        if not os.path.exists(checkpoint_path):
            st.error(f"❌ Model file not found: {checkpoint_path}")
            return None, None, None, None
        
        if not os.path.exists(threshold_path):
            st.error(f"❌ Threshold config not found: {threshold_path}")
            return None, None, None, None
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load threshold config
        with open(threshold_path, 'r') as f:
            threshold_config = json.load(f)
        
        # Initialize backbone
        backbone = resnet_feature_extractor()
        backbone.eval()
        
        # Initialize autoencoder
        model_config = checkpoint['model_config']
        model = FeatCAE(
            in_channels=model_config['in_channels'],
            latent_dim=model_config['latent_dim'],
            is_bn=model_config['is_bn']
        )
        
        # Load weights
        model.load_state_dict(checkpoint['autoencoder_state_dict'])
        model.eval()
        
        # Setup transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        return backbone, model, transform, threshold_config
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, None

# Function to detect anomalies
def detect_anomaly(image, backbone, model, transform, best_threshold, heat_map_min, heat_map_max):
    # Preprocess image
    test_image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        # Extract features
        features = backbone(test_image)
        
        # Reconstruct
        recon = model(features)
        
        # Calculate reconstruction error per pixel
        segm_map = ((features - recon) ** 2).mean(axis=1)
        
        # Get anomaly score using decision function (top 10 values)
        y_score_image = decision_function(segm_map=segm_map)
        
        # Binary prediction
        y_pred_image = 1 * (y_score_image >= best_threshold)
        
        # Prepare heatmap (resize to 128x128 like notebook)
        heat_map = segm_map.squeeze().cpu().numpy()
        heat_map_resized = cv2.resize(heat_map, (128, 128))
        
        # Segmentation mask (threshold * 10 for visualization)
        segmentation_mask = (heat_map_resized > heat_map_max * 9).astype(np.uint8)
    
    return {
        'original_image': test_image.squeeze().permute(1, 2, 0).cpu().numpy(),
        'heat_map': heat_map_resized,
        'segmentation_mask': segmentation_mask,
        'anomaly_score': y_score_image[0].cpu().item(),
        'is_anomaly': bool(y_pred_image.item()),
        'normalized_score': y_score_image[0].cpu().item() / best_threshold
    }

# Main app
def main():
    st.title("🔍 Anomaly Detection - Multi-Model System")
    st.markdown("Select a model folder and upload images for defect detection.")
    
    # Model selection in sidebar
    st.sidebar.header("🗂️ Model Selection")
    
    # Base path for models
    base_models_path = st.sidebar.text_input(
        "Models Base Path", 
        value="models",
        help="Folder containing your model subfolders"
    )
    
    # Get available model folders
    model_folders = get_model_folders(base_models_path)
    
    if not model_folders:
        st.warning(f"""
        ⚠️ No model folders found in `{base_models_path}/`
        
        ### 📁 Expected Structure:
        ```
        {base_models_path}/
        ├── carpet_model_v1/
        │   ├── anomaly_model_complete.pth
        │   ├── threshold_config.json
        │   └── training_history.pth (optional)
        ├── carpet_model_v2/
        │   ├── anomaly_model_complete.pth
        │   └── threshold_config.json
        └── ...
        ```
        
        ### 🔧 To get started:
        1. Create a folder named `{base_models_path}`
        2. Inside, create subfolders for each model
        3. Place your model files in each subfolder
        4. Refresh this page
        """)
        return
    
    # Model selector
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=model_folders,
        help="Choose which trained model to use"
    )
    
    model_folder_path = os.path.join(base_models_path, selected_model)
    
    # Display model info
    st.sidebar.success(f"✅ Selected: **{selected_model}**")
    
    # Load models
    with st.spinner(f"Loading model from {selected_model}..."):
        backbone, model, transform, threshold_config = load_models(model_folder_path)
    
    if backbone is None or model is None:
        st.error("Failed to load model. Please check the files.")
        return
    
    st.sidebar.success("✅ Model loaded successfully!")
    
    # Settings
    st.sidebar.header("⚙️ Settings")
    
    # Threshold configuration
    st.sidebar.subheader("Threshold Configuration")
    
    default_threshold = threshold_config.get('threshold', threshold_config['mean_error'] + 3 * threshold_config['std_error'])
    
    use_custom_threshold = st.sidebar.checkbox("Use Custom Threshold", value=False)
    
    if use_custom_threshold:
        best_threshold = st.sidebar.slider(
            "Anomaly Threshold",
            min_value=0.0,
            max_value=default_threshold * 2,
            value=default_threshold,
            format="%.6f"
        )
    else:
        best_threshold = default_threshold
        st.sidebar.info(f"Using auto threshold: {best_threshold:.6f}")
    
    # Heatmap range configuration
    st.sidebar.subheader("Heatmap Display Range")
    heat_map_min = threshold_config.get('min_error', 0.0)
    heat_map_max = threshold_config.get('max_error', 1.0)
    
    use_custom_range = st.sidebar.checkbox("Customize Heatmap Range", value=False)
    
    if use_custom_range:
        heat_map_min = st.sidebar.number_input("Min Value", value=float(heat_map_min), format="%.6f")
        heat_map_max = st.sidebar.number_input("Max Value", value=float(heat_map_max), format="%.6f")
    
    heatmap_multiplier = st.sidebar.slider("Heatmap Intensity Multiplier", 1, 20, 10, 1)
    
    # Display statistics
    with st.sidebar.expander("📊 Model Statistics"):
        st.write(f"**Model:** {selected_model}")
        st.write(f"**Mean Error:** {threshold_config['mean_error']:.6f}")
        st.write(f"**Std Error:** {threshold_config['std_error']:.6f}")
        st.write(f"**Min Error:** {threshold_config['min_error']:.6f}")
        st.write(f"**Max Error:** {threshold_config['max_error']:.6f}")
        st.write(f"**Current Threshold:** {best_threshold:.6f}")
        
        # Show training info if available
        if 'training_info' in threshold_config:
            st.write("---")
            st.write("**Training Info:**")
            for key, value in threshold_config['training_info'].items():
                st.write(f"- {key}: {value}")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Choose an image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to check for defects"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Detect anomaly
        with st.spinner("🔍 Analyzing image..."):
            result = detect_anomaly(
                image, backbone, model, transform, 
                best_threshold, heat_map_min, heat_map_max
            )
        
        # Display results
        st.markdown("---")
        
        # Status banner
        if result['is_anomaly']:
            st.error(f"⚠️ **DEFECT DETECTED** | Anomaly Score: {result['normalized_score']:.4f}")
        else:
            st.success(f"✅ **PRODUCT OK** | Anomaly Score: {result['normalized_score']:.4f}")
        
        # Three column layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image)
        
        with col2:
            st.subheader("🔥 Anomaly Heatmap")
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(
                result['heat_map'], 
                cmap='jet', 
                vmin=heat_map_min, 
                vmax=heat_map_max * heatmap_multiplier
            )
            ax.set_title(f"Score: {result['normalized_score']:.4f} | {'NOK' if result['is_anomaly'] else 'OK'}", 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close()
        
        with col3:
            st.subheader("🎯 Segmentation Map")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(result['segmentation_mask'], cmap='gray')
            ax.set_title('Binary Defect Mask', fontsize=12, fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
        
        # Detailed metrics
        st.markdown("---")
        st.subheader("📈 Detailed Analysis")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Anomaly Score", f"{result['anomaly_score']:.6f}")
        
        with metric_col2:
            st.metric("Threshold", f"{best_threshold:.6f}")
        
        with metric_col3:
            st.metric("Normalized Score", f"{result['normalized_score']:.4f}")
        
        with metric_col4:
            status = "DEFECT" if result['is_anomaly'] else "OK"
            st.metric("Classification", status)
        
        # Overlay visualization
        st.markdown("---")
        st.subheader("🖼️ Overlay Visualization")
        
        overlay_alpha = st.slider("Heatmap Opacity", 0.0, 1.0, 0.5, 0.05)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Resize original image to match heatmap
        original_resized = cv2.resize(np.array(image), (128, 128))
        
        ax.imshow(original_resized)
        ax.imshow(
            result['heat_map'], 
            cmap='jet', 
            alpha=overlay_alpha,
            vmin=heat_map_min, 
            vmax=heat_map_max * heatmap_multiplier
        )
        ax.axis('off')
        ax.set_title(f"Overlay - {'DEFECT DETECTED' if result['is_anomaly'] else 'PRODUCT OK'}", 
                    fontsize=14, fontweight='bold')
        
        st.pyplot(fig)
        plt.close()
        
    else:
        st.info("👆 Upload an image to get started!")
        
        # Display instructions
        st.markdown(f"""
        ### 📋 How to use:
        1. **Select a model** from the sidebar dropdown
        2. **Upload** an image (PNG, JPG, JPEG)
        3. The model will **analyze** the image using ResNet50 + Autoencoder
        4. **View results** - heatmap, segmentation, and metrics
        5. **Adjust settings** in the sidebar if needed
        
        ### 🎯 Current Model: **{selected_model}**
        
        ### 📁 Add New Models:
        1. Train a new model using your notebook
        2. Run the save script to generate the 3 files
        3. Create a new folder in `{base_models_path}/`
        4. Copy the files into that folder
        5. Refresh this page and select the new model!
        """)

if __name__ == "__main__":
    main()