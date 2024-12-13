import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from typing import Optional, Any, List, Tuple

from transformers import SamProcessor, SamModel
from diffusers import AutoPipelineForInpainting

from sam_inpainting_utils import get_processed_inputs, inpaint



class ImageSegmentationInpainting:
    """Class to handle image segmentation and inpainting using SAM and user-defined prompts."""

    IMG_SIZE: int = 512
    
    def __init__(self, processor: SamProcessor, model: SamModel, pipeline: AutoPipelineForInpainting, device: str = 'cpu'):

        self.processor = processor
        self.model = model
        self.pipeline = pipeline
        self.device = device

        self.input_image: Image.Image = None
        self.input_points: List[List[int]] = []


    def get_points(self, img: Image.Image, evt: gr.SelectData
                ) -> Tuple[Tuple[Image.Image, List[Tuple[np.ndarray, str]]], Image.Image]:
        """
        Handles the selection of points on the image, updates the segmentation, and marks points on the image.
        Args:
        - img: The current image displayed in the Gradio app.
        - evt: Event object containing the coordinates of the selected point.
        Returns:
        - sam_output: Updated SAM output.
        - img: The image with marked points.
        """

        # Save the original input image on the first point selection
        if len(self.input_points) == 0:
            self.input_image = img.copy()

        # Append the selected point coordinates
        x, y = evt.index[0], evt.index[1]
        self.input_points.append([x, y])

        # Run the segmentation model (SAM)
        sam_output = self.run_sam()

        # Mark selected points with a green cross on the image
        draw = ImageDraw.Draw(img)
        size = 10  # Size of the cross mark
        for point in self.input_points:
            px, py = point
            draw.line((px - size, py, px + size, py), fill="green", width=5)
            draw.line((px, py - size, px, py + size), fill="green", width=5)

        return sam_output, img


    def run_sam(self) -> Tuple[Image.Image, List[Tuple[np.ndarray, str]]]:
        """
        Runs the SAM (Segment Anything Model) to generate a segmentation mask based on selected points.
        Returns:
        - img: The resized input image.
        - res_masks: A list of tuples containing the mask and its label (background/subject).
        Raises:
        - gr.Error: If no points have been provided.
        """

        if self.input_image is None:
            raise gr.Error("No points provided. Click on the image to select the object to segment with SAM")

        try:
            # Generate the segmentation mask from the input image and points
            mask = get_processed_inputs(self.processor, self.model, self.input_image, [self.input_points], self.device)

            # Resize the mask to match the desired output size
            res_mask = np.array(Image.fromarray(mask).resize((self.IMG_SIZE, self.IMG_SIZE)))

            # Resized input image
            img = self.input_image.resize((self.IMG_SIZE, self.IMG_SIZE))
            res_masks = [(res_mask, "background"), (~res_mask, "subject")]

            return img, res_masks
        except Exception as e:
            raise gr.Error(str(e))
        

    def run(self, prompt: str, negative_prompt: str, cfg: float, invert: bool, seed: int) -> Image.Image:
        """
        Runs the inpainting process using the SAM output and user-defined prompts.
        Args:
        - prompt (str): Text prompt for inpainting.
        - negative_prompt (str): Negative text prompt for inpainting.
        - cfg: Configuration parameter for the inpainting model.
        - invert: Whether to invert the mask for inpainting.
        - seed: Random seed for reproducibility.
        Returns:
        - res_inpainted: The inpainted image resized to the desired output size.
        Raises:
        - gr.Error: If no points have been provided.
        """

        if self.input_image is None:
            raise gr.Error("No points provided. Click on the image to select the object to segment with SAM")

        # Get the segmentation mask
        amask = self.run_sam()[1][0][0]

        # Invert the mask if needed
        what = 'background' if invert else 'subject'
        amask = ~amask if invert else amask

        gr.Info(f"Inpainting {what}... (this will take up to a few minutes)")
        try:
            # Perform inpainting
            inpainted = inpaint(self.pipeline, self.input_image, amask, prompt, negative_prompt, cfg, seed)
            res_inpainted = inpainted.resize((self.IMG_SIZE, self.IMG_SIZE))
        except Exception as e:
            raise gr.Error(str(e))

        return res_inpainted


    def reset_points(self, *args: Any) -> None:
        """Resets the list of input points to start a new segmentation."""

        self.input_points.clear()


    def preprocess(self, img: Optional[Image.Image]) -> Optional[Image.Image]:
        """
        Preprocess the input image to ensure it is square and resize it to the specified dimensions.
        Args:
        - img: Input image to preprocess.
        Returns:
            Optional[Image.Image]: Preprocessed image or None if input is None.
        """

        if img is None:
            return None

        width, height = img.size

        # If the image is not square, add white padding to make it square
        if width != height:
            gr.Warning("Image is not square, adding white padding")

            new_size = max(width, height)
            new_image = Image.new("RGB", (new_size, new_size), 'white')

            left = (new_size - width) // 2
            top = (new_size - height) // 2

            new_image.paste(img, (left, top))
            img = new_image

        res_img = img.resize((self.IMG_SIZE, self.IMG_SIZE))
        return res_img