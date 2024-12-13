import gradio as gr
# import numpy as np
# from PIL import Image, ImageDraw
# from typing import Optional, Any, Callable, List, Tuple

from transformers import SamProcessor, SamModel
from diffusers import AutoPipelineForInpainting

from sam_app_utils import ImageSegmentationInpainting



def generate_app(processor: SamProcessor, model: SamModel, pipeline: AutoPipelineForInpainting, device: str = 'cpu'):
    """
    Generates a Gradio app for interactive image segmentation and inpainting using SAM (Segment Anything Model).
    Args:
    - get_processed_inputs: Function that processes the input image and points to generate a segmentation mask.
    - inpaint: Function that performs inpainting on the input image based on the segmentation mask.
    """

    # global input_points
    # global input_image

    image_processor = ImageSegmentationInpainting(processor, model, pipeline, device)
    img_size = {'height': image_processor.IMG_SIZE, 'width': image_processor.IMG_SIZE}

    with gr.Blocks() as demo:

        # Application description
        gr.Markdown(
        """
        # Image Inpainting

        Steps:
        1. Upload an image and click on the subject you want to keep. A SAM mask will be generated.
        2. Refine the mask by adding more points if needed.
        3. Provide prompts for infilling and click "Run inpaint."
        4. Adjust parameters like CFG scale, seed, or toggle mask inversion as needed.

        # Examples
        Try clicking on one of the example images below to get started.
        """)

        # Input and output sections
        with gr.Row():
            display_img = gr.Image(label="Input", interactive=True, type='pil', **img_size)
            sam_mask = gr.AnnotatedImage(label="SAM result", interactive=False, **img_size)
            result = gr.Image(label="Output", interactive=False, type='pil', **img_size)

        # Events
        display_img.select(image_processor.get_points, inputs=[display_img], outputs=[sam_mask, display_img])
        display_img.clear(image_processor.reset_points)
        display_img.change(image_processor.preprocess, inputs=[display_img], outputs=[display_img])

        # Inpainting controls
        with gr.Row():
            cfg = gr.Slider(label="CFG Scale", minimum=0.0, maximum=20.0, value=7.0, step=0.05)
            random_seed = gr.Number(label="Seed", value=74294536, precision=0)
            checkbox = gr.Checkbox(label="Infill subject instead of background")

        with gr.Row():
            prompt = gr.Textbox(label="Prompt for infill")
            neg_prompt = gr.Textbox(label="Negative prompt")
            reset_button = gr.ClearButton(value="Reset", components=[display_img, sam_mask, result, prompt, neg_prompt, checkbox])
            reset_button.click(image_processor.reset_points)
            submit_inpaint = gr.Button(value="Run inpaint")

        # Example section
        with gr.Row():
            examples = gr.Examples(
                [
                    ["images/car.png", "a car driving on planet Mars", "artifacts", 74294536],
                    ["images/dragon.jpeg", "a dragon in a medieval village", "artifacts", 97],
                    ["images/monalisa.png", "a fantasy landscape with flying dragons", "artifacts", 97]
                ],
                inputs=[display_img, prompt, neg_prompt, random_seed]
            )

        # Run inpaint button
        submit_inpaint.click(image_processor.run, inputs=[prompt, neg_prompt, cfg, checkbox, random_seed], outputs=[result])

    demo.queue(max_size=1).launch(share=True, debug=True)
    return demo