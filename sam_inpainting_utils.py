import numpy as np
import torch
from PIL import Image
from typing import Optional
from transformers import SamProcessor, SamModel
from diffusers import AutoPipelineForInpainting



def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """
    - Transforms a binary mask into an RGBA image for visualization.
    Args:
    - mask: A binary mask where 1 indicates the region of interest.
    Returns:
    - bg_transparent: An RGBA image where the region of interest is highlighted in green.
    """

    # Initialize an RGBA image with all transparent pixels
    bg_transparent = np.zeros(mask.shape + (4,), dtype=np.uint8)

    # Highlight the masked area in green with 50% transparency
    # Color format: [Red, Green, Blue, Alpha]
    bg_transparent[mask == 1] = [0, 255, 0, 127]
    return bg_transparent



def get_processed_inputs(processor: SamProcessor, model: SamModel, image: Image.Image, 
                         input_points: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    Processes an image and input points to generate masks using the SAM model.
    Args:
    - processor: SAM processor to preprocess the inputs.
    - model: SAM model used for generating predictions.
    - image: Input image for segmentation.
    - input_points: Array of points to guide segmentation (e.g., [[x1, y1], [x2, y2]]).
    - device: Device to run the inference on ('cpu' or 'cuda').
    Returns:
    - inv_best_mask: A binary mask of the selected region (1 for background, 0 for the subject).
    """

    # Prepare inputs for the SAM model
    inputs = processor(image, input_points=input_points, return_tensors="pt").to(device)

    # Perform inference to generate segmentation outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract predictions and metadata
    pred_masks = outputs.pred_masks.cpu()
    original_sizes = inputs["original_sizes"].cpu()
    reshaped_input_sizes = inputs["reshaped_input_sizes"].cpu()

    print(f"Prediction masks shape: {pred_masks.shape}")
    print(f"Original sizes shape: {original_sizes.shape}")
    print(f"Reshaped input sizes shape: {reshaped_input_sizes.shape}")

    # Post-process masks to match the original image size
    masks = processor.image_processor.post_process_masks(
        pred_masks, original_sizes, reshaped_input_sizes
    )

    # Select the mask with the highest Intersection over Union (IoU) score
    best_mask = masks[0][0][outputs.iou_scores.argmax()]

    # Invert the mask: 0 represents the subject, 1 represents the background
    inv_best_mask = ~best_mask.cpu().numpy()
    return inv_best_mask



def inpaint(pipeline: AutoPipelineForInpainting, raw_image: Image.Image, input_mask: np.ndarray, 
            prompt: str, negative_prompt: Optional[str] = None, cfgs: float = 7.0, seed: int = 74294536
            ) -> Image.Image:
    """
    Performs inpainting on an image using Stable Diffusion.
    Args:
    - pipeline: The inpainting pipeline from Stable Diffusion.
    - raw_image: The input image to be inpainted.
    - input_mask: Binary mask where 1 indicates the area to be inpainted.
    - prompt: Text prompt describing the desired infill.
    - negative_prompt: Text prompt describing undesired elements in the infill.
    - cfgs (float): Classifier-free guidance scale to balance creativity and fidelity.
    - seed: Random seed for reproducibility of results.
    Returns:
    - image: The resulting inpainted image.
    """

    # Convert the binary mask to a PIL image
    mask_image = Image.fromarray(input_mask)

    # Set the random seed for reproducibility
    rand_gen = torch.manual_seed(seed)

    # Perform inpainting using the Stable Diffusion pipeline
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=raw_image,
        mask_image=mask_image,
        generator=rand_gen,
        guidance_scale=cfgs,
    ).images[0]

    return image