import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import base64
from io import BytesIO
from llmres import LLMAdvisor

class ModelManager:
    """Handles model loading, preprocessing, and predictions"""
    
    MODEL_CONFIG = {
        "GeneralModel": {
            "class_names": [
                'anthracnose-cashew', 'bacterial blight-cassava', 'brown spot-cassava',
                'cashew-mosaic-cassava', 'fall armyworm-maize', 'grasshoper-maize',
                'green mite-cassava', 'gumosis-cashew', 'healthy-cashew',
                'healthy-cassava', 'healthy-maize', 'healthy-tomato',
                'leaf beetle-maize', 'leaf blight-maize', 'leaf blight-tomato',
                'leaf curl-tomato', 'leaf miner-cashew', 'leaf spot-maize',
                'red rust-cashew', 'septoria leaf spot-tomato', 'streak virus-maize',
                'verticillium wilt-tomato'
            ],
            "model_path": "models/best_combine_model.pth",
            "input_size": (380, 380),
            "architecture": "efficientnet"
        },
        "Cashew": {
        "class_names": [
            'anthracnose-cashew', 'gumosis-cashew', 'healthy-cashew', 'leaf miner-cashew',
            'red rust-cashew'
        ],
        "model_path": "models/cashew_model.pth",
        "input_size": (380, 380),
        "architecture": "efficientnet"
    },
    "Cassava": {
        "class_names": [
            'bacterial blight-cassava', 'brown spot-cassava',
            'cashew-mosaic-cassava','green mite-cassava','healthy-cassava'
        ],
        "model_path": "models/cassava_model.pth",
        "input_size": (380, 380),
        "architecture": "efficientnet"
    },
    "Maize": {
        "class_names": [
            'fall armyworm-maize', 'grasshoper-maize', 'healthy-maize',
            'leaf beetle-maize', 'leaf blight-maize', 'leaf spot-maize','streak virus-maize'
        ],
        "model_path": "models/maize_model.pth",
        "input_size": (380, 380),
        "architecture": "efficientnet"
    },
    "Tomato": {
        "class_names": [
            'healthy-tomato','leaf blight-tomato','leaf curl-tomato', 'septoria leaf spot-tomato','verticillium wilt-tomato'
        ],
        "model_path": "models/tomato_model.pth",
        "input_size": (380, 380),
        "architecture": "efficientnet"
    }
    }

    @staticmethod
    @st.cache_resource
    def load_model(model_name):
        """Load and cache model based on configuration"""
        config = ModelManager.MODEL_CONFIG[model_name]
        
        try:
            if config["architecture"] == "efficientnet":
                model = models.efficientnet_b4(pretrained=False)
                model.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(1792, len(config["class_names"])))
            
            model.load_state_dict(
                torch.load(config["model_path"], map_location='cpu'))
            model.eval()
            return model
        except Exception as e:
            st.error(f"Model loading error: {str(e)}")
            return None

    @staticmethod
    def preprocess_image(image, target_size):
        """Preprocess image for model input"""
        try:
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            return transform(image).unsqueeze(0)
        except Exception as e:
            st.error(f"Image processing error: {str(e)}")
            return None

    @staticmethod
    def predict(_model, img_tensor):
        """Run model prediction on processed image tensor"""
        with torch.no_grad():
            outputs = _model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_class = torch.max(probabilities, 1)
            return pred_class.item(), conf.item() * 100

class StreamlitApp:
    """Main application class handling Streamlit UI and logic"""
    
    def __init__(self):
        self.setup_page()
        
    def setup_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="CropDoc - Multi-Model Disease Detection",
            layout="centered",
            initial_sidebar_state="expanded"
        )
        self.add_custom_css()
    
    def add_custom_css(self):
        """Inject custom CSS styling"""
        st.markdown("""
        <style>
            .reportview-container {
                background: #f0f2f6;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 10px 24px;
                width: 100%;
            }
            .model-card {
                padding: 20px;
                border-radius: 10px;
                background: #ffffff;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                margin: 20px 0;
            }
        </style>
        """, unsafe_allow_html=True)

    def display_centered_image(self, image):
        """Display uploaded image with styling"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        image_html = f"""
        <div style="display: flex; justify-content: center; margin: 20px 0;">
            <img src="data:image/png;base64,{img_str}" 
                 style="max-width: 300px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        </div>
        """
        st.markdown(image_html, unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666;'>Uploaded Image</p>", 
                    unsafe_allow_html=True)

    def run(self):
        """Main application loop"""
        st.title("🌱 CropDoc - Crop Disease Detection")
        st.markdown("---")

        # File uploader
        uploaded_file = st.file_uploader(
            "📤 Upload Crop Image",
            type=["png", "jpg", "jpeg"],
            help="Select an image of a crop leaf for analysis"
        )

        # Model selection
        selected_model = st.selectbox(
            "🧠 Select Analysis Model",
            options=list(ModelManager.MODEL_CONFIG.keys()),
            index=0,
            help="Choose the machine learning model for diagnosis"
        )

        # Display uploaded image
        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                self.display_centered_image(image)
            except Exception as e:
                st.error(f"Image Error: {str(e)}")
                return

        # Prediction handler
        if st.button("🔍 Analyze Image"):
            if not uploaded_file:
                st.warning("⚠️ Please upload an image first!")
                return

            try:
                config = ModelManager.MODEL_CONFIG[selected_model]
                image = Image.open(uploaded_file).convert("RGB")
                
                # Preprocess image
                img_tensor = ModelManager.preprocess_image(image, config["input_size"])
                if img_tensor is None:
                    return

                # Load model
                model = ModelManager.load_model(selected_model)
                if model is None:
                    return

                # Make prediction
                with st.spinner("🔬 Analyzing image..."):
                    pred_idx, confidence = ModelManager.predict(model, img_tensor)
                    class_name = config["class_names"][pred_idx]

                # Display results
                st.success("📋 Analysis Results")
                st.markdown(f"""
                <div class="model-card">
                    <h3 style="color: #2c3e50; margin-bottom: 15px;">{class_name}</h3>
                    <p style="font-size: 16px; color: #666;">
                        Confidence: <strong>{confidence:.2f}%</strong><br>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                with st.spinner("🌱 Generating expert advice..."):
                    advisor = LLMAdvisor()
                    advice = advisor.generate_advice(class_name, selected_model)

                st.markdown("### 🌱 Expert Advice")
                st.markdown(
                    f"""
                    <div style="
                        background-color: #f9f9f9; 
                        padding: 15px; 
                        border-radius: 10px; 
                        border: 1px solid #ddd; 
                        font-family: Arial, sans-serif; 
                        font-size: 16px; 
                        line-height: 1.5; 
                        color: #333;
                    ">
                        {advice}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()