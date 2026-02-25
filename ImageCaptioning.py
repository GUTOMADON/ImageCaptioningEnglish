import torch  
import gradio as gr  #import Gradio to build the web interface
from PIL import Image  # import PIL to handle image conversion
from transformers import (  # import Hugging Face Transformers components
    BlipProcessor,  # processor that handles input preprocessing for BLIP models
    BlipForConditionalGeneration,  # BLIP model variant used for image captioning
    BlipForQuestionAnswering,  # BLIP model variant used for visual question answering
)

# Models!
# Captioning
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")  
caption_model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base") 

# VQA  (model specifically trained to answer questions about images)
vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base") 
vqa_model     = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")  

# Main logic!
def analyze_image(img, open_question, binary_question):  
    if img is None:  
        return "No image provided.", "", ""  

    img_pil = Image.fromarray(img).convert("RGB")  # convert the NumPy array from Gradio into a PIL RGB image

    # 1. Automatic caption
    cap_inputs = caption_processor(img_pil, return_tensors="pt")  # preprocess the image for the captioning model and return PyTorch tensors
    with torch.no_grad():  # disable gradient computation to save memory during inference
        cap_ids = caption_model.generate(**cap_inputs, max_new_tokens=50)  
    caption = caption_processor.decode(cap_ids[0], skip_special_tokens=True)  

    # 2. Open question — uses VQA model
    open_answer = ""  
    if open_question.strip():  # run VQA if the user actually typed a question (ignoring whitespace)
        vqa_inputs = vqa_processor(img_pil, open_question, return_tensors="pt") 
        with torch.no_grad():  # Disable gradient computation during inference
            vqa_ids = vqa_model.generate(**vqa_inputs, max_new_tokens=50)  
        open_answer = vqa_processor.decode(vqa_ids[0], skip_special_tokens=True)  

    # 3. Binary question — uses VQA model and converts to 0/1
    binary_result = ""  # initialize the binary result as an empty string
    if binary_question.strip():  
        bin_inputs = vqa_processor(img_pil, binary_question, return_tensors="pt")  # preprocess the image and binary question for the VQA model
        with torch.no_grad():  # disable gradient computation during inference
            bin_ids = vqa_model.generate(**bin_inputs, max_new_tokens=10)  # enough for a yes/no answer
        binary_answer = vqa_processor.decode(bin_ids[0], skip_special_tokens=True)  # decode the generated token IDs into a string
        binary_result = "1  (Yes)" if "yes" in binary_answer.lower() else "0  (No)"  # Map the answer to 1 if it contains "yes", otherwise map to 0

    return caption, open_answer, binary_result  #return all three results to be displayed in the Gradio interface


#Gradio Interface!
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
# Gradio Interface (CSS moved to end of file to avoid interfering with code)
with gr.Blocks(title="BLIP Image Analyzer") as demo: 
}

/* Ensure Markdown tables (e.g., 'How it works') use dark background and readable text */
.gradio-container .markdown table, .gradio-container .markdown th, .gradio-container .markdown td {
    background: transparent !important;
    color: var(--text) !important;
    border-color: rgba(255,255,255,0.04) !important;
}
"""  # define the custom CSS

with gr.Blocks(css=CSS, title="BLIP Image Analyzer") as demo: 

    gr.Markdown(  
        """
        # BLIP Image Analyzer
        **Captioning · VQA · Binary Classification** — no model training required
        """
    )

    with gr.Row():  
        with gr.Column(scale=1):  #
            img_input = gr.Image(type="numpy", label="Input Image")  
            open_q    = gr.Textbox(  
                label="Open Question",  
                placeholder="ex: What objects are in the image?", 
                lines=2  
            )
            bin_q     = gr.Textbox(  
                label="Binary Question (Yes / No)",  
                placeholder="ex: Is there a person in the image?",  
                lines=2  
            )
            btn = gr.Button("Analyze", variant="primary") 

        with gr.Column(scale=1):  #(scale=1 means equal width) for all model outputs
            out_caption = gr.Textbox(label="Automatic Caption (captioning)")  
            out_open    = gr.Textbox(label="Answer to open question")
            out_binary  = gr.Textbox(label="Binary Result  [ 1 = Yes  |  0 = No ]")  # display the binary classification result

            gr.Markdown(  # Render a Markdown block explaining how each step of the pipeline works
                """
                ---
                ### How it works

                | Step | Model | Description |
                |------|-------|-------------|
                | Captioning | blip-image-captioning-base | Free description of the image without a prompt |
                | Open VQA | blip-vqa-base | Contextual answer to the given question |
                | Binary | blip-vqa-base | VQA answer: yes becomes 1, everything else becomes 0 |
                """
            )

    btn.click(  
        fn=analyze_image,  
        inputs=[img_input, open_q, bin_q],  
        outputs=[out_caption, out_open, out_binary] 
    )

    gr.Examples(  #create a set of pre filled example inputs the user can click to auto-populate the fields
        examples=[  
            [None, "What is happening in this image?",        "Is there a person in the image?"], 
            [None, "What type of building is shown?",         "Is the image outdoors?"], 
            [None, "What is the dominant color in the image?","Is there a car in the image?"],  
        ],
        inputs=[img_input, open_q, bin_q],  
        label="Question examples"  
    )

    # placeholder HTML element; will be filled with CSS later to keep CSS at end of file
    html_css = gr.HTML(value="", elem_id="__custom_css_placeholder")

    # Apply CSS (loaded from the bottom of the file)
    try:
        # Will be updated below after CSS string is defined
        pass
    except Exception:
        pass

    if __name__ == "__main__":
        demo.launch()

    #all CSS here to avoid interfering with code above
    CSS = """
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --bg:      #0d0d10;
        --surface: #13131a;
        --border:  #23232f;
        --accent:  #4f8ef7;
        --text:    #dcdce8;
        --label:   #8888aa;
    }

    body, .gradio-container {
        background: var(--bg) !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        color: var(--text) !important;
    }

    h1 {
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: -0.3px;
        color: var(--text) !important;
    }

    .block, .form {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }

    label span {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.68rem !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--label) !important;
    }

    textarea, input[type=text] {
        background: #0d0d14 !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.875rem !important;
        border-radius: 6px !important;
    }

    button.primary {
        background: var(--accent) !important;
        border: none !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
        font-weight: 600;
        letter-spacing: 1px;
        border-radius: 6px !important;
        padding: 10px 26px !important;
        color: #fff !important;
    }
    button.primary:hover { filter: brightness(1.12); }

    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th, td { padding: 8px 12px; border: 1px solid var(--border); text-align: left; }
    th { background: #1a1a24; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
         text-transform: uppercase; letter-spacing: 1px; color: var(--label); }

    /* Ensure Gradio Examples table uses dark background and readable text */
    .gradio-examples { background: transparent !important; }
    .gradio-examples table, .gradio-examples thead, .gradio-examples tbody, .gradio-examples th, .gradio-examples td {
        background: transparent !important;
        color: var(--text) !important;
        border-color: rgba(255,255,255,0.04) !important;
    }
    .gradio-examples pre, .gradio-examples code { background: transparent !important; color: var(--text) !important; }

    /* Force white/light text for main UI and Markdown (ensures 'How it works' is readable) */
    .gradio-container .markdown, .gradio-container .markdown *,
    .gradio-container h1, .gradio-container h2, .gradio-container h3,
    .gradio-container p, .gradio-container li, .gradio-container strong, .gradio-container em {
        color: var(--text) !important;
    }

    /* Ensure Markdown tables (e.g., 'How it works') use dark background and readable text */
    .gradio-container .markdown table, .gradio-container .markdown th, .gradio-container .markdown td {
        background: transparent !important;
        color: var(--text) !important;
        border-color: rgba(255,255,255,0.04) !important;
    }
    """

    # inject CSS into the placeholder HTML element we created inside the Blocks
    try:
        html_css.update(value=f"<style>{CSS}</style>")
    except Exception:
        # If something goes wrong (e.g., html_css not defined), ignore — UI will still work without custom CSS
        pass