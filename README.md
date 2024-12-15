# Subject-Background Inpainting App (subject-background-inpainting-app)

An **interactive web application** for advanced image segmentation and creative inpainting. 
This app leverages the power of **Segment Anything Model (SAM)** for segmentation and 
**Stable Diffusion Inpainting** for generating subject or background infills based on user-defined prompts.


## Features

### Core Capabilities

- **Interactive Segmentation:** Select objects in an image using SAM by simply clicking on them.
- **Flexible Inpainting:**
  - Infill either the **selected subject** or the **background**.
  - Customize inpainting with prompts and negative prompts.
- **Customizable Parameters:**
  - Adjust **CFG Scale** for controlling the creativity of the diffusion model.
  - Use a specific **random seed** for reproducibility.
  - **Mask inversion** option to switch between subject and background inpainting.

### User-Friendly Interface
  - Drag-and-drop image upload.
  - Real-time segmentation visualization.
  - Example images and prompts to get started quickly.

### Technology Stack
  - **Segment Anything Model (SAM):** For pixel-perfect segmentation.
  - **Stable Diffusion XL Inpainting:** For realistic and creative image generation.
  - **Gradio:** For an intuitive and responsive web-based interface.


## Installation and Setup

### Prerequisites
- Python 3.8 or higher.
- A GPU-enabled environment (CUDA recommended for optimal performance).

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/kkuwaran/subject-background-inpainting-app.git
   cd subject-background-inpainting-app
   ```
2. Install dependencies:
   - Make sure to use Gradio version **3.50.2** for optimal compatibility.
   - Other dependencies are flexible and can be installed via pip as needed.
   ```bash
   pip install gradio==3.50.2
   ```
4. Download pre-trained models:
   - SAM model: `facebook/sam-vit-base`
   - Stable Diffusion Inpainting: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
  
### Running the App
Run the following command to launch the app locally:
```bash
python main.ipynb
```


## Usage Instructions

### How to Use the App

1. **Upload an Image:** Drag and drop or select an image file.
2. **Select Subject:** Click on the subject in the image you want to keep or modify. The SAM model will generate a mask.
3. **Refine Mask:** Add more points if needed to refine the selection.
4. **Set Inpainting Parameters:**
   - Provide a text prompt to describe what you want to generate.
   - Adjust optional parameters such as CFG Scale, seed, or mask inversion.
5. **Run Inpainting:** Click the "Run inpaint" button to generate the output image.
6. **Reset as Needed:** Use the reset button to clear selections and start over.


## Project Structure

```plaintext
📦subject-background-inpainting-app/
 ├── main.ipynb                        # Main script to launch the app
 ├── app_new.py                        # Gradio app user interface
 ├── sam_app_utils.py                  # Helper class for segmentation and inpainting
 ├── sam_inpainting_utils.py           # Core functions for SAM and inpainting logic
 ├── test_sam_inpainting_utils.ipynb   # Test cases for utility functions
 └── images/                           # Example images for testing
```

