import os
# pyrefly: ignore [missing-import]
import gradio as gr

# Import the inference engine we just built
# Note: We can do this because we will run the app from the root directory
from src.inference import separate_audio_file

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "mamba_dexformer_epoch_115.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "demo_audio", "separated_outputs")

def process_audio(audio_path, max_spk):
    """
    Gradio wrapper function to handle the UI inputs and outputs.
    """
    if not audio_path:
        # If the user clicks submit without uploading, return None for all 5 audio slots
        return [None] * 5
        
    if not os.path.exists(WEIGHTS_PATH):
        raise gr.Error(f"Weights file not found at {WEIGHTS_PATH}. Please ensure it is downloaded.")

    # Call your production inference engine
    extracted_files = separate_audio_file(
        mixture_path=audio_path, 
        checkpoint_path=WEIGHTS_PATH, 
        out_dir=OUTPUT_DIR, 
        max_speakers=int(max_spk)
    )
    
    # Gradio UI expects exactly 5 outputs since we hardcoded 5 audio players.
    # If the user only asked for 3 speakers, we pad the rest with None to hide them.
    while len(extracted_files) < 5:
        extracted_files.append(None)
        
    return extracted_files

# Build the User Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🗣️ Mamba-Dexformer Speech Separator")
    gr.Markdown("Upload a multi-speaker audio mixture. The deflationary Mamba model will iteratively extract individual speakers.")
    
    with gr.Row():
        # Left Column: Inputs
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Mixture (WAV/MP3)")
            spk_slider = gr.Slider(minimum=1, maximum=5, step=1, value=5, label="Max Speakers to Extract")
            submit_btn = gr.Button("Separate Audio", variant="primary")
            
        # Right Column: Outputs
        with gr.Column():
            gr.Markdown("### Extracted Speakers")
            out1 = gr.Audio(label="Speaker 1")
            out2 = gr.Audio(label="Speaker 2")
            out3 = gr.Audio(label="Speaker 3")
            out4 = gr.Audio(label="Speaker 4")
            out5 = gr.Audio(label="Speaker 5")
            
    # Group the output components so we can map them to our function return list
    outputs = [out1, out2, out3, out4, out5]
    
    # Wire the button to the function
    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, spk_slider],
        outputs=outputs
    )

if __name__ == "__main__":
    demo.launch(share=True)