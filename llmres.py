from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st

class LLMAdvisor:
    """
    Handles local LLM processing for agricultural advice using TinyLlama
    """
    
    def __init__(self):
        self.model, self.tokenizer = self._load_model()
    
    @staticmethod
    @st.cache_resource
    def _load_model():
        """Load and cache the TinyLlama model and tokenizer"""
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return model, tokenizer

    def generate_advice(_self, class_name, model_name):
        """
        Generate agricultural advice using TinyLlama
        Args:
            class_name (str): Detected disease class
            model_name (str): Name of model used for detection
        Returns:
            str: Generated advice text
        """
        try:
            # Format the prompt using TinyLlama's chat template
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an agricultural expert. Provide detailed and practical advice using this structure:\n"
                        "1. **Brief Disease Overview** – A concise introduction to the disease.\n"
                        "2. **Key Identification Symptoms** – Use bold headers for each symptom category.\n"
                        "3. **Prevention Strategies** – List actionable measures to prevent disease spread.\n"
                        "4. **Organic Treatment Protocols** – Step-by-step organic control measures.\n\n"
                        "Formatting Guidelines:\n"
                        "- Use **bold** for all section headers and important points.\n"
                        "- Use bullet points for lists.\n"
                        "- Include tables for better readability where applicable.\n"
                        "- Provide actionable, farmer-friendly advice throughout."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Please analyze {class_name} in {model_name} crops by including the following:\n"
                        "- A detailed symptom identification checklist\n"
                        "- An analysis of climate conditions that favor disease spread\n"
                        "- Stage-wise organic control measures\n"
                        "- Suggestions for companion planting\n"
                        "Also, include cost-effective solutions suitable for small farms."
                    )
                }
            ]
            
            prompt = _self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Generate response
            inputs = _self.tokenizer(prompt, return_tensors="pt").to(_self.model.device)
            outputs = _self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True
            )

            # Decode and clean the response
            response = _self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            return _self._clean_response(response)
        
        except Exception as e:
            st.error(f"LLM Error: {str(e)}")
            return "Could not generate advice at this time."

    @staticmethod
    def _clean_response(text):
        """Clean response while preserving Markdown formatting"""
        # Remove special tokens but keep newlines and Markdown symbols
        cleaned = text.replace("</s>", "").replace("<s>", "").strip()
        # Collapse multiple newlines to two (for paragraph separation)
        cleaned = "\n\n".join([line.strip() for line in cleaned.split("\n") if line.strip()])
        return cleaned