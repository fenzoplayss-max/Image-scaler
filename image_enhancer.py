#!/usr/bin/env python3
"""
Image Enhancement Tool - Local Processing Only
Uses advanced algorithms for upscaling, denoising, and color preservation
No AI models, no internet, no external APIs
"""

import cv2
import numpy as np
import argparse


def adaptive_denoise(img, strength=10):
    """
    Advanced denoising with edge preservation using Non-Local Means
    Preserves colors and details while removing noise
    """
    if len(img.shape) == 3:
        # Color image - process in LAB color space to preserve colors
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply denoising only to luminance channel to preserve color
        l_denoised = cv2.fastNlMeansDenoising(l, None, h=strength, templateWindowSize=7, searchWindowSize=21)
        
        # Merge back
        lab_denoised = cv2.merge([l_denoised, a, b])
        result = cv2.cvtColor(lab_denoised, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale image
        result = cv2.fastNlMeansDenoising(img, None, h=strength, templateWindowSize=7, searchWindowSize=21)
    
    return result


def enhance_details(img):
    """
    Enhance details using adaptive histogram equalization and sharpening
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return img_enhanced


def smart_sharpen(img, strength=1.5):
    """
    Smart sharpening using unsharp mask with edge detection
    """
    # Create Gaussian blur
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    
    # Calculate unsharp mask
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    
    # Preserve edges by blending based on edge strength
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_dilated = cv2.dilate(edges, None, iterations=2)
    edges_blurred = cv2.GaussianBlur(edges_dilated.astype(float), (5, 5), 0)
    edge_mask = edges_blurred / 255.0
    
    # Blend original and sharpened based on edge strength
    if len(img.shape) == 3:
        edge_mask_3ch = np.stack([edge_mask] * 3, axis=2)
        result = img * (1 - edge_mask_3ch * 0.3) + sharpened * (edge_mask_3ch * 0.3 + 0.7)
    else:
        result = img * (1 - edge_mask * 0.3) + sharpened * (edge_mask * 0.3 + 0.7)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def pixel_prediction_upscale(img, scale_factor):
    """
    Advanced pixel prediction using multiple interpolation methods and blending
    Simulates intelligent pixel generation without AI models
    """
    height, width = img.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    
    # Method 1: Lanczos (good for general quality)
    lanczos = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Method 2: Cubic (smoother)
    cubic = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Method 3: Area (preserves patterns better)
    area = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Edge detection to guide blending
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_resized_lanczos = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    gray_resized_cubic = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Calculate local variance to detect edges and textures
    kernel_size = 3
    mean = cv2.blur(gray_resized_lanczos.astype(float), (kernel_size, kernel_size))
    mean_sq = cv2.blur((gray_resized_lanczos ** 2).astype(float), (kernel_size, kernel_size))
    variance = mean_sq - mean ** 2
    variance = np.clip(variance, 0, None)
    
    # Normalize variance to create weight map
    weight_map = variance / (variance.max() + 1e-6)
    weight_map = cv2.GaussianBlur(weight_map, (5, 5), 0)
    
    # Blend based on variance: high variance (edges) -> Lanczos, low variance (smooth) -> Cubic
    if len(img.shape) == 3:
        weight_map_3ch = np.stack([weight_map] * 3, axis=2)
        blended = lanczos * weight_map_3ch + cubic * (1 - weight_map_3ch)
        
        # Add some area interpolation for very smooth regions
        smooth_mask = (weight_map_3ch < 0.2).astype(float)
        blended = blended * (1 - smooth_mask * 0.3) + area * (smooth_mask * 0.3)
    else:
        blended = lanczos * weight_map + cubic * (1 - weight_map)
        smooth_mask = (weight_map < 0.2).astype(float)
        blended = blended * (1 - smooth_mask * 0.3) + area * (smooth_mask * 0.3)
    
    return np.clip(blended, 0, 255).astype(np.uint8)


def color_correction(img, original_img):
    """
    Preserve original color characteristics after processing
    Uses 75% original color to maintain natural look
    """
    # Convert both to LAB
    lab_processed = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_original = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
    
    l_proc, a_proc, b_proc = cv2.split(lab_processed)
    l_orig, a_orig, b_orig = cv2.split(lab_original)
    
    # Resize original channels to match processed size
    a_orig_resized = cv2.resize(a_orig, (a_proc.shape[1], a_proc.shape[0]), interpolation=cv2.INTER_CUBIC)
    b_orig_resized = cv2.resize(b_orig, (b_proc.shape[1], b_proc.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Blend color channels: 75% original color, 25% processed color (preserves natural look)
    a_final = (a_orig_resized * 0.75 + a_proc * 0.25).astype(np.uint8)
    b_final = (b_orig_resized * 0.75 + b_proc * 0.25).astype(np.uint8)
    
    # Keep enhanced luminance
    lab_final = cv2.merge([l_proc, a_final, b_final])
    result = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
    
    return result


def bilateral_filter_preserve(img, d=9, sigmaColor=75, sigmaSpace=75):
    """
    Apply bilateral filter for edge-preserving smoothing
    """
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)


def enhance_image(input_path, output_path=None, scale=2, denoise_strength=10):
    """
    Main enhancement pipeline
    """
    # Read image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
    
    print(f"Input image shape: {img.shape}")
    
    # Store original for color reference
    original_img = img.copy()
    
    # Step 1: Advanced pixel prediction upscaling
    print("Step 1: Pixel prediction upscaling...")
    upscaled = pixel_prediction_upscale(img, scale)
    
    # Step 2: Denoise with edge preservation
    print("Step 2: Adaptive denoising...")
    denoised = adaptive_denoise(upscaled, strength=denoise_strength)
    
    # Step 3: Enhance details
    print("Step 3: Detail enhancement...")
    enhanced = enhance_details(denoised)
    
    # Step 4: Smart sharpening
    print("Step 4: Smart sharpening...")
    sharpened = smart_sharpen(enhanced, strength=1.2)
    
    # Step 5: Bilateral filter for final smoothing
    print("Step 5: Edge-preserving smoothing...")
    smoothed = bilateral_filter_preserve(sharpened, d=7, sigmaColor=50, sigmaSpace=50)
    
    # Step 6: Color correction to preserve original look (75% original color)
    print("Step 6: Color correction (preserving original colors)...")
    final = color_correction(smoothed, cv2.resize(original_img, 
                                                   (smoothed.shape[1], smoothed.shape[0]), 
                                                   interpolation=cv2.INTER_CUBIC))
    
    # Save result
    if output_path is None:
        base_name = input_path.rsplit('.', 1)[0]
        ext = input_path.rsplit('.', 1)[1] if '.' in input_path else 'png'
        output_path = f"{base_name}_enhanced.{ext}"
    
    cv2.imwrite(output_path, final)
    print(f"Enhanced image saved to: {output_path}")
    print(f"Output image shape: {final.shape}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Local Image Enhancement Tool')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output image path (optional)')
    parser.add_argument('-s', '--scale', type=float, default=2.0, 
                        help='Scale factor (default: 2.0)')
    parser.add_argument('-d', '--denoise', type=int, default=12, 
                        help='Denoising strength 0-30 (default: 12)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Advanced Image Enhancement Tool (Local Processing)")
    print("=" * 60)
    
    try:
        enhance_image(args.input, args.output, args.scale, args.denoise)
        print("\nEnhancement completed successfully!")
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
