import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from typing import Optional, Any, Callable, List, Tuple


# Global variables for managing input points and the original image
input_points: List[List[int]] = []
IMG_SIZE: int = 512
input_image: Image.Image = None


def generate_app(get_processed_inputs: Callable[[Image.Image, List[List[int]]], np.ndarray],
                 inpaint: Callable[[Image.Image, np.ndarray, str, str, int, float], Image.Image]):
    """
    Generates a Gradio app for interactive image segmentation and inpainting using SAM (Segment Anything Model).
    Args:
    - get_processed_inputs: Function that processes the input image and points to generate a segmentation mask.
    - inpaint: Function that performs inpainting on the input image based on the segmentation mask.
    """

    global input_points
    global input_image


    def get_points(img: Image.Image, evt: gr.SelectData) -> Tuple[Tuple[Image.Image, List[Tuple[np.ndarray, str]]], Image.Image]:
        """
        Handles the selection of points on the image, updates the segmentation, and marks points on the image.
        Args:
        - img: The current image displayed in the Gradio app.
        - evt: Event object containing the coordinates of the selected point.
        Returns:
        - sam_output: Updated SAM output.
        - img: The image with marked points.
        """

        global input_image

        # Save the original input image on the first point selection
        if len(input_points) == 0:
            input_image = img.copy()

        # Append the selected point coordinates
        x, y = evt.index[0], evt.index[1]
        input_points.append([x, y])

        # Run the segmentation model (SAM)
        sam_output = run_sam()

        # Mark selected points with a green cross on the image
        draw = ImageDraw.Draw(img)
        size = 10  # Size of the cross mark
        for point in input_points:
            px, py = point
            draw.line((px - size, py, px + size, py), fill="green", width=5)
            draw.line((px, py - size, px, py + size), fill="green", width=5)

        return sam_output, img


    def run_sam() -> Tuple[Image.Image, List[Tuple[np.ndarray, str]]]:
        """
        Runs the SAM (Segment Anything Model) to generate a segmentation mask based on selected points.
        Returns:
        - img: The resized input image.
        - res_masks: A list of tuples containing the mask and its label (background/subject).
        Raises:
        - gr.Error: If no points have been provided.
        """

        if input_image is None:
            raise gr.Error("No points provided. Click on the image to select the object to segment with SAM")

        try:
            # Generate the segmentation mask from the input image and points
            mask = get_processed_inputs(input_image, [input_points])

            # Resize the mask to match the desired output size
            res_mask = np.array(Image.fromarray(mask).resize((IMG_SIZE, IMG_SIZE)))

            # Resized input image
            img = input_image.resize((IMG_SIZE, IMG_SIZE))
            res_masks = [(res_mask, "background"), (~res_mask, "subject")]

            return img, res_masks
        except Exception as e:
            raise gr.Error(str(e))
        

    def run(prompt: str, negative_prompt: str, cfg: float, seed: int, invert: bool) -> Image.Image:
        """
        Runs the inpainting process using the SAM output and user-defined prompts.
        Args:
        - prompt (str): Text prompt for inpainting.
        - negative_prompt (str): Negative text prompt for inpainting.
        - cfg: Configuration parameter for the inpainting model.
        - seed: Random seed for reproducibility.
        - invert: Whether to invert the mask for inpainting.
        Returns:
        - res_inpainted: The inpainted image resized to the desired output size.
        Raises:
        - gr.Error: If no points have been provided.
        """

        if input_image is None:
            raise gr.Error("No points provided. Click on the image to select the object to segment with SAM")

        # Get the segmentation mask
        amask = run_sam()[1][0][0]

        # Invert the mask if needed
        what = 'background' if invert else 'subject'
        amask = ~amask if invert else amask

        gr.Info(f"Inpainting {what}... (this will take up to a few minutes)")
        try:
            # Perform inpainting
            inpainted = inpaint(input_image, amask, prompt, negative_prompt, seed, cfg)
            res_inpainted = inpainted.resize((IMG_SIZE, IMG_SIZE))
        except Exception as e:
            raise gr.Error(str(e))

        return res_inpainted


    def reset_points(*args: Any) -> None:
        """Resets the list of input points to start a new segmentation."""

        input_points.clear()


    def preprocess(input_img: Optional[Image.Image]) -> Optional[Image.Image]:
        """
        Preprocess the input image to ensure it is square and resize it to the specified dimensions.
        Args:
        - input_img: Input image to preprocess.
        Returns:
            Optional[Image.Image]: Preprocessed image or None if input is None.
        """
        if input_img is None:
            return None

        width, height = input_img.size

        # If the image is not square, add white padding to make it square
        if width != height:
            gr.Warning("Image is not square, adding white padding")

            new_size = max(width, height)
            new_image = Image.new("RGB", (new_size, new_size), 'white')

            left = (new_size - width) // 2
            top = (new_size - height) // 2

            new_image.paste(input_img, (left, top))
            input_img = new_image

        res_input_img = input_img.resize((IMG_SIZE, IMG_SIZE))
        return res_input_img




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
            display_img = gr.Image(label="Input", interactive=True, type='pil', height=IMG_SIZE, width=IMG_SIZE)
            sam_mask = gr.AnnotatedImage(label="SAM result", interactive=False, height=IMG_SIZE, width=IMG_SIZE)
            result = gr.Image(label="Output", interactive=False, type='pil', height=IMG_SIZE, width=IMG_SIZE)

        # Events
        display_img.select(get_points, inputs=[display_img], outputs=[sam_mask, display_img])
        display_img.clear(reset_points)
        display_img.change(preprocess, inputs=[display_img], outputs=[display_img])

        # Inpainting controls
        with gr.Row():
            cfg = gr.Slider(label="CFG Scale", minimum=0.0, maximum=20.0, value=7.0, step=0.05)
            random_seed = gr.Number(label="Seed", value=74294536, precision=0)
            checkbox = gr.Checkbox(label="Infill subject instead of background")

        with gr.Row():
            prompt = gr.Textbox(label="Prompt for infill")
            neg_prompt = gr.Textbox(label="Negative prompt")
            reset_button = gr.ClearButton(value="Reset", components=[display_img, sam_mask, result, prompt, neg_prompt, checkbox])
            reset_button.click(reset_points)
            submit_inpaint = gr.Button(value="Run inpaint")

        # Example section
        with gr.Row():
            examples = gr.Examples(
                [
                    ["car.png", "a car driving on planet Mars", "artifacts", 74294536],
                    ["dragon.jpeg", "a dragon in a medieval village", "artifacts", 97],
                    ["monalisa.png", "a fantasy landscape with flying dragons", "artifacts", 97]
                ],
                inputs=[display_img, prompt, neg_prompt, random_seed]
            )

        # Run inpaint button
        submit_inpaint.click(run, inputs=[prompt, neg_prompt, cfg, random_seed, checkbox], outputs=[result])

    demo.queue(max_size=1).launch(share=True, debug=True)
    return demo