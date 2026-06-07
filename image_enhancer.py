#!/usr/bin/env python3
"""
Image Enhancement Script using Local AI (Real-ESRGAN)
This script enhances image quality locally without API or internet.
Uses Real-ESRGAN model for super-resolution and enhancement.

Requirements:
    pip install opencv-python-headless pillow numpy
    
For GPU support (optional):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
"""

import cv2
import numpy as np
from PIL import Image
import os
import sys
import argparse
from pathlib import Path


class ImageEnhancer:
    """Local image enhancer using traditional CV techniques and optional AI models."""
    
    def __init__(self, scale=2, device='cpu'):
        """
        Initialize the image enhancer.
        
        Args:
            scale: Upscaling factor (2 or 4)
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        self.scale = scale
        self.device = device
        self.model = None
        
        # Try to load Real-ESRGAN if available
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            print("✓ Real-ESRGAN found - Using AI enhancement")
            self._load_realesrgan(scale, device)
        except ImportError:
            print("⚠ Real-ESRGAN not installed - Using traditional enhancement")
            print("  For better results, install with: pip install realesrgan basicsr")
    
    def _load_realesrgan(self, scale, device):
        """Load Real-ESRGAN model for AI-based enhancement."""
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            if scale == 4:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                               num_block=23, num_grow_ch=32, scale=4)
            else:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                               num_block=23, num_grow_ch=32, scale=2)
            
            self.model = RealESRGANer(
                scale=scale,
                model_path=f'realesrgan-x{scale}foolplus.pth',
                model=model,
                tile=256,
                tile_pad=10,
                pre_pad=10,
                half=True if device == 'cuda' else False,
                device=device
            )
            print(f"✓ Real-ESRGAN x{scale} loaded successfully")
        except Exception as e:
            print(f"⚠ Could not load Real-ESRGAN: {e}")
            self.model = None
    
    def enhance_traditional(self, image):
        """
        Enhance image using traditional computer vision techniques.
        This works without any external AI models.
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image
        
        # Upscale using Lanczos interpolation (high quality)
        height, width = rgb.shape[:2]
        new_width = width * self.scale
        new_height = height * self.scale
        
        upscaled = cv2.resize(rgb, (new_width, new_height), 
                             interpolation=cv2.INTER_LANCZOS4)
        
        # Apply bilateral filter for edge-preserving smoothing
        denoised = cv2.bilateralFilter(upscaled, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Enhance sharpness using unsharp masking
        gaussian = cv2.GaussianBlur(denoised, (9, 9), 10.0)
        sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
        
        # Adjust contrast and brightness using CLAHE
        if len(sharpened.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge back
            lab_enhanced = cv2.merge((l_enhanced, a, b))
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(sharpened)
        
        return enhanced
    
    def enhance_ai(self, image):
        """Enhance image using Real-ESRGAN AI model."""
        if self.model is None:
            return self.enhance_traditional(image)
        
        try:
            # Real-ESRGAN expects BGR format
            output, _ = self.model.enhance(image, outscale=self.scale)
            # Convert back to RGB
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            return output_rgb
        except Exception as e:
            print(f"⚠ AI enhancement failed: {e}, falling back to traditional")
            return self.enhance_traditional(image)
    
    def enhance(self, image_path, output_path=None, use_ai=True):
        """
        Main enhancement method.
        
        Args:
            image_path: Path to input image
            output_path: Path to save enhanced image (optional)
            use_ai: Whether to try AI enhancement first
            
        Returns:
            Enhanced image as numpy array
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        print(f"Input size: {image.shape[1]}x{image.shape[0]}")
        
        # Enhance
        if use_ai and self.model is not None:
            print("Using AI enhancement (Real-ESRGAN)...")
            enhanced = self.enhance_ai(image)
        else:
            print("Using traditional enhancement...")
            enhanced = self.enhance_traditional(image)
        
        print(f"Output size: {enhanced.shape[1]}x{enhanced.shape[0]}")
        
        # Save if output path provided
        if output_path:
            # Convert BGR to RGB for saving
            if len(enhanced.shape) == 3 and enhanced.shape[2] == 3:
                save_image = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            else:
                save_image = enhanced
            
            cv2.imwrite(str(output_path), save_image)
            print(f"✓ Saved enhanced image to: {output_path}")
        
        return enhanced


def download_model_weights():
    """Download Real-ESRGAN model weights (one-time operation)."""
    import urllib.request
    
    models = {
        'realesrgan-x2foolplus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRGAN_x2plus.pth',
        'realesrgan-x4foolplus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRGAN_x4plus.pth'
    }
    
    for model_name, url in models.items():
        if not os.path.exists(model_name):
            print(f"Downloading {model_name}...")
            urllib.request.urlretrieve(url, model_name)
            print(f"✓ Downloaded {model_name}")
        else:
            print(f"✓ {model_name} already exists")


def main():
    parser = argparse.ArgumentParser(
        description='Enhance image quality locally using AI or traditional methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python image_enhancer.py input.jpg
    python image_enhancer.py input.jpg -o output.jpg -s 4
    python image_enhancer.py input.jpg --ai --device cuda
    
For best results with AI:
    pip install realesrgan basicsr
    python image_enhancer.py --download-models
        """
    )
    
    parser.add_argument('input', type=str, help='Input image path')
    parser.add_argument('-o', '--output', type=str, help='Output image path')
    parser.add_argument('-s', '--scale', type=int, default=2, choices=[2, 4],
                       help='Upscaling factor (default: 2)')
    parser.add_argument('--ai', action='store_true', 
                       help='Use AI enhancement (requires realesrgan)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use (default: cpu)')
    parser.add_argument('--download-models', action='store_true',
                       help='Download Real-ESRGAN model weights')
    
    args = parser.parse_args()
    
    # Download models if requested
    if args.download_models:
        download_model_weights()
        return
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    # Generate output path if not provided
    if not args.output:
        input_path = Path(args.input)
        args.output = f"{input_path.stem}_enhanced_{args.scale}x{input_path.suffix}"
    
    # Create enhancer and process
    print("=" * 60)
    print("Image Enhancement Tool (Local - No API/Internet)")
    print("=" * 60)
    
    enhancer = ImageEnhancer(scale=args.scale, device=args.device)
    
    try:
        enhancer.enhance(args.input, args.output, use_ai=args.ai)
        print("=" * 60)
        print("✓ Enhancement complete!")
        print("=" * 60)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
