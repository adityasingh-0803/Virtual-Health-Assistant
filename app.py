import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
from typing import List, Dict
import gc
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables for memory optimization
os.environ['TRANSFORMERS_CACHE'] = '/home/user/.cache/huggingface/hub'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class HealthAssistant:
    def __init__(self):
        self.model_id = "microsoft/Phi-2"  # Using smaller Phi-2 model
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.metrics = []
        self.medications = []
        self.device = "cpu"
        self.is_model_loaded = False
        self.max_history_length = 2

    def initialize_model(self):
        try:
            if self.is_model_loaded:
                return True

            logger.info(f"Loading model: {self.model_id}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                model_max_length=256,
                padding_side="left"
            )
            logger.info("Tokenizer loaded")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                device_map=None,
                low_cpu_mem_usage=True
            ).to(self.device)

            gc.collect()
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                model_kwargs={"low_cpu_mem_usage": True}
            )
            
            self.is_model_loaded = True
            logger.info("Model initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in model initialization: {str(e)}")
            raise

    def unload_model(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'pipe') and self.pipe is not None:
            del self.pipe
            self.pipe = None
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.is_model_loaded = False
        gc.collect()
        logger.info("Model unloaded successfully")

    def generate_response(self, message: str, history: List = None) -> str:
        try:
            if not self.is_model_loaded:
                self.initialize_model()
            
            message = message[:200]  # Truncate long messages
            
            prompt = self._prepare_prompt(message, history[-self.max_history_length:] if history else None)

            generation_args = {
                "max_new_tokens": 200,
                "return_full_text": False,
                "temperature": 0.7,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "num_return_sequences": 1,
                "batch_size": 1
            }

            output = self.pipe(prompt, **generation_args)
            response = output[0]['generated_text']

            gc.collect()
            
            return response.strip()

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."

    def _prepare_prompt(self, message: str, history: List = None) -> str:
        prompt_parts = [
            "Medical AI assistant. Be professional, include disclaimers.",
            self._get_health_context()
        ]
        
        if history:
            for h in history:
                if isinstance(h, dict):  # New message format
                    if h['role'] == 'user':
                        prompt_parts.append(f"Human: {h['content'][:100]}")
                    else:
                        prompt_parts.append(f"Assistant: {h['content'][:100]}")
                else:  # Old format (tuple)
                    prompt_parts.extend([
                        f"Human: {h[0][:100]}",
                        f"Assistant: {h[1][:100]}"
                    ])
        
        prompt_parts.extend([
            f"Human: {message}",
            "Assistant:"
        ])
        
        return "\n".join(prompt_parts)

    def _get_health_context(self) -> str:
        if not self.metrics and not self.medications:
            return "No health data"
            
        context = []
        if self.metrics:
            latest = self.metrics[-1]
            context.append(f"Metrics: W:{latest['Weight']}kg S:{latest['Steps']} Sl:{latest['Sleep']}h")
        
        if self.medications:
            meds = [f"{m['Medication']}({m['Dosage']}@{m['Time']})" for m in self.medications[-2:]]
            context.append("Meds: " + ", ".join(meds))
            
        return " | ".join(context)

    def add_metrics(self, weight: float, steps: int, sleep: float) -> bool:
        try:
            if len(self.metrics) >= 5:
                self.metrics.pop(0)
                
            self.metrics.append({
                'Weight': weight,
                'Steps': steps,
                'Sleep': sleep
            })
            return True
        except Exception as e:
            logger.error(f"Error adding metrics: {e}")
            return False

    def add_medication(self, name: str, dosage: str, time: str, notes: str = "") -> bool:
        try:
            if len(self.medications) >= 5:
                self.medications.pop(0)
                
            self.medications.append({
                'Medication': name,
                'Dosage': dosage,
                'Time': time,
                'Notes': notes
            })
            return True
        except Exception as e:
            logger.error(f"Error adding medication: {e}")
            return False

class GradioInterface:
    def __init__(self):
        try:
            logger.info("Initializing Health Assistant...")
            self.assistant = HealthAssistant()
            logger.info("Health Assistant initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Health Assistant: {e}")
            raise

    def chat_response(self, message: str, history: List) -> tuple:
        if not message.strip():
            return "", history
        
        try:
            response = self.assistant.generate_response(message, history)
            # Convert to new message format
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            
            if len(history) % 3 == 0:
                self.assistant.unload_model()
                
            return "", history
        except Exception as e:
            logger.error(f"Error in chat response: {e}")
            return "", history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "I apologize, but I encountered an error. Please try again."}
            ]

    def add_health_metrics(self, weight: float, steps: int, sleep: float) -> str:
        if not all([weight is not None, steps is not None, sleep is not None]):
            return "⚠️ Please fill in all metrics."
        
        if weight <= 0 or steps < 0 or sleep < 0:
            return "⚠️ Please enter valid positive numbers."
        
        if self.assistant.add_metrics(weight, steps, sleep):
            return f"""✅ Health metrics saved successfully!
• Weight: {weight} kg
• Steps: {steps}
• Sleep: {sleep} hours"""
        return "❌ Error saving metrics."

    def add_medication_info(self, name: str, dosage: str, time: str, notes: str) -> str:
        if not all([name, dosage, time]):
            return "⚠️ Please fill in all required fields."
        
        if self.assistant.add_medication(name, dosage, time, notes):
            return f"""✅ Medication added successfully!
• Medication: {name}
• Dosage: {dosage}
• Time: {time}
• Notes: {notes if notes else 'None'}"""
        return "❌ Error adding medication."

    def create_interface(self):
        with gr.Blocks(title="Medical Health Assistant") as demo:
            gr.Markdown("""
            # 🏥 Medical Health Assistant
            This AI assistant provides general health information and guidance.
            """)
            
            with gr.Tabs():
                with gr.Tab("💬 Medical Consultation"):
                    chatbot = gr.Chatbot(
                        value=[],
                        height=400,
                        label=False,
                        type="messages"  # Using new message format
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask your health question...",
                            lines=1,
                            label=False,
                            scale=9
                        )
                        send_btn = gr.Button("Send", scale=1)
                    clear_btn = gr.Button("Clear Chat")

                with gr.Tab("📊 Health Metrics"):
                    gr.Markdown("### Track Your Health Metrics")
                    with gr.Row():
                        weight_input = gr.Number(
                            label="Weight (kg)",
                            minimum=0,
                            maximum=500
                        )
                        steps_input = gr.Number(
                            label="Steps",
                            minimum=0,
                            maximum=100000
                        )
                        sleep_input = gr.Number(
                            label="Hours Slept",
                            minimum=0,
                            maximum=24
                        )
                    metrics_btn = gr.Button("Save Metrics")
                    metrics_status = gr.Markdown()

                with gr.Tab("💊 Medication Manager"):
                    gr.Markdown("### Track Your Medications")
                    med_name = gr.Textbox(
                        label="Medication Name",
                        placeholder="Enter medication name"
                    )
                    with gr.Row():
                        med_dosage = gr.Textbox(
                            label="Dosage",
                            placeholder="e.g., 500mg"
                        )
                        med_time = gr.Textbox(
                            label="Time",
                            placeholder="e.g., 9:00 AM"
                        )
                    med_notes = gr.Textbox(
                        label="Notes (optional)",
                        placeholder="Additional instructions or notes"
                    )
                    med_btn = gr.Button("Add Medication")
                    med_status = gr.Markdown()

            msg.submit(self.chat_response, [msg, chatbot], [msg, chatbot])
            send_btn.click(self.chat_response, [msg, chatbot], [msg, chatbot])
            clear_btn.click(lambda: [], None, chatbot)
            
            metrics_btn.click(
                self.add_health_metrics,
                inputs=[weight_input, steps_input, sleep_input],
                outputs=[metrics_status]
            )
            
            med_btn.click(
                self.add_medication_info,
                inputs=[med_name, med_dosage, med_time, med_notes],
                outputs=[med_status]
            )

            gr.Markdown("""
            ### ⚠️ Medical Disclaimer
            This AI assistant provides general health information only. Not a replacement for professional medical advice.
            Always consult healthcare professionals for medical decisions.
            """)

            demo.queue(max_size=5)
            
        return demo

def main():
    try:
        interface = GradioInterface()
        demo = interface.create_interface()
        demo.launch(
            server_name="0.0.0.0",
            show_error=True,
            share=True
        )
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        raise

if __name__ == "__main__":
    main()