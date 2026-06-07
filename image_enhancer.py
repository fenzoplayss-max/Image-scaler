#!/usr/bin/env python3
"""
Image Enhancement Tool - Local Processing (No AI/API/Internet)
Enhances image quality and resolution using traditional algorithms only.
All processing is done locally on your machine.
"""

import cv2
import numpy as np
import argparse
import os
import sys


def enhance_image(image_path, output_path=None, scale=2):
    """
    Enhance image quality and increase resolution using traditional methods.
    
    Args:
        image_path: Path to input image
        output_path: Path for output image (default: input_enhanced.jpg)
        scale: Upscaling factor (default: 2x)
    
    Returns:
        Path to enhanced image or None if failed
    """
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found!")
        return None
    
    # Generate output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_enhanced{ext}"
    
    try:
        # Load image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image '{image_path}'")
            return None
        
        print(f"Loaded image: {img.shape[1]}x{img.shape[0]}")
        
        # Step 1: Denoise using Non-Local Means
        print("Applying denoising...")
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        # Step 2: Upscale using Lanczos interpolation (highest quality)
        print(f"Upscaling by {scale}x...")
        new_width = int(img.shape[1] * scale)
        new_height = int(img.shape[0] * scale)
        upscaled = cv2.resize(denoised, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Step 3: Sharpen the image
        print("Applying sharpening...")
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(upscaled, -1, kernel)
        
        # Blend sharpened with original upscaled for natural look
        blended = cv2.addWeighted(upscaled, 0.7, sharpened, 0.3, 0)
        
        # Step 4: Apply CLAHE for better contrast
        print("Enhancing contrast...")
        lab = cv2.cvtColor(blended, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        enhanced_lab = lab.copy()
        enhanced_lab[:, :, 0] = enhanced_l
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Step 5: Final bilateral filter for edge preservation
        print("Final smoothing...")
        final = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Save the result
        cv2.imwrite(output_path, final)
        print(f"Enhanced image saved to: {output_path}")
        print(f"New resolution: {final.shape[1]}x{final.shape[0]}")
        
        return output_path
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Enhance image quality locally without AI/API/Internet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python image_enhancer.py photo.jpg
    python image_enhancer.py photo.jpg -o output.jpg -s 4
    python image_enhancer.py photo.jpg --scale 2

Features:
    - Denoising
    - High-quality upscaling (Lanczos)
    - Sharpening
    - Contrast enhancement (CLAHE)
    - Edge-preserving smoothing
    
All processing is done locally on your machine!
        """
    )
    
    parser.add_argument('input', type=str, help='Input image path')
    parser.add_argument('-o', '--output', type=str, help='Output image path (optional)')
    parser.add_argument('-s', '--scale', type=int, default=2, 
                       help='Upscaling factor (default: 2)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    print("=" * 60)
    print("Image Enhancement Tool (Local - No AI/API/Internet)")
    print("=" * 60)
    
    # Process image
    result = enhance_image(args.input, args.output, args.scale)
    
    if result:
        print("=" * 60)
        print("Enhancement complete!")
        print("=" * 60)
    else:
        print("Enhancement failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
