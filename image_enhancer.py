#!/usr/bin/env python3
"""
Advanced Image Enhancement Tool (Local Processing)
Uses traditional algorithms for upscaling, denoising and enhancement.
No AI models, no internet required.
"""

import cv2
import numpy as np
import argparse
import os
import sys


def chunked_upscale(img, scale_factor, chunk_size=800):
    """
    Upscale large images in chunks to avoid memory errors.
    """
    h, w = img.shape[:2]
    new_h, new_w = h * scale_factor, w * scale_factor
    
    # Create output image
    if len(img.shape) == 3:
        enhanced = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    else:
        enhanced = np.zeros((new_h, new_w), dtype=np.uint8)
    
    print(f"Processing in chunks of {chunk_size}x{chunk_size}...")
    
    for y in range(0, h, chunk_size):
        for x in range(0, w, chunk_size):
            # Define chunk boundaries
            y_end = min(y + chunk_size, h)
            x_end = min(x + chunk_size, w)
            
            # Extract chunk with overlap for better edge handling
            overlap = 20
            y_start_chunk = max(0, y - overlap)
            x_start_chunk = max(0, x - overlap)
            y_end_chunk = min(h, y_end + overlap)
            x_end_chunk = min(w, x_end + overlap)
            
            chunk = img[y_start_chunk:y_end_chunk, x_start_chunk:x_end_chunk]
            
            # Upscale chunk
            chunk_upscaled = cv2.resize(chunk, None, fx=scale_factor, fy=scale_factor, 
                                       interpolation=cv2.INTER_LANCZOS4)
            
            # Calculate output coordinates (accounting for overlap)
            out_y_start = y_start_chunk * scale_factor
            out_x_start = x_start_chunk * scale_factor
            
            # Calculate the region to copy (excluding overlap areas)
            actual_y_start = (y - y_start_chunk) * scale_factor
            actual_x_start = (x - x_start_chunk) * scale_factor
            actual_h = (y_end - y) * scale_factor
            actual_w = (x_end - x) * scale_factor
            
            # Copy to output
            enhanced[out_y_start+int(actual_y_start):out_y_start+int(actual_y_start)+int(actual_h),
                    out_x_start+int(actual_x_start):out_x_start+int(actual_x_start)+int(actual_w)] = \
                chunk_upscaled[int(actual_y_start):int(actual_y_start)+int(actual_h),
                              int(actual_x_start):int(actual_x_start)+int(actual_w)]
            
            print(f"Processed chunk at ({x}, {y})")
    
    return enhanced


def smart_pixel_prediction(img, scale_factor):
    """
    Advanced pixel prediction using multiple interpolation methods blending.
    """
    h, w = img.shape[:2]
    
    # Method 1: Lanczos (sharp details)
    lanczos = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, 
                        interpolation=cv2.INTER_LANCZOS4)
    
    # Method 2: Cubic (smooth gradients)
    cubic = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, 
                      interpolation=cv2.INTER_CUBIC)
    
    # Method 3: Area (preserves textures)
    area = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, 
                     interpolation=cv2.INTER_AREA)
    
    # Blend methods based on local characteristics
    # Use Laplacian to detect edges and textures
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edge_mask = np.abs(laplacian) > np.mean(np.abs(laplacian))
    edge_mask = cv2.resize(edge_mask.astype(np.uint8), None, fx=scale_factor, fy=scale_factor,
                          interpolation=cv2.INTER_NEAREST)
    
    # Blend: edges -> lanczos, smooth -> cubic, textures -> area
    if len(img.shape) == 3:
        result = np.where(edge_mask[..., np.newaxis],
                         lanczos * 0.7 + cubic * 0.3,
                         cubic * 0.6 + area * 0.4).astype(np.uint8)
    else:
        result = np.where(edge_mask,
                         lanczos * 0.7 + cubic * 0.3,
                         cubic * 0.6 + area * 0.4).astype(np.uint8)
    
    return result


def advanced_denoise(img, strength=10):
    """
    Advanced non-local means denoising with adaptive strength.
    """
    if len(img.shape) == 3:
        # Color image - use fastNlMeansDenoisingColored
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 
                                                   h=strength, 
                                                   hForColorComponents=strength,
                                                   templateWindowSize=7,
                                                   searchWindowSize=21)
    else:
        # Grayscale
        denoised = cv2.fastNlMeansDenoising(img, None, 
                                           h=strength,
                                           templateWindowSize=7,
                                           searchWindowSize=21)
    return denoised


def color_preservation_blend(original, enhanced, preserve_ratio=0.7):
    """
    Preserve original colors while keeping enhanced details.
    """
    # Convert to LAB color space
    original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    enhanced_lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Keep L (lightness) from enhanced, blend A and B from original
    result_lab = enhanced_lab.copy()
    result_lab[:, :, 1] = original_lab[:, :, 1] * preserve_ratio + enhanced_lab[:, :, 1] * (1 - preserve_ratio)
    result_lab[:, :, 2] = original_lab[:, :, 2] * preserve_ratio + enhanced_lab[:, :, 2] * (1 - preserve_ratio)
    
    result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result


def adaptive_sharpen(img, amount=1.5, radius=1.5, threshold=0):
    """
    Adaptive unsharp masking with edge detection.
    """
    # Create blurred version
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    
    # Apply unsharp mask
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    
    # Edge-aware blending: only sharpen edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.GaussianBlur(edges.astype(float), (5, 5), 2) / 255.0
    
    if len(img.shape) == 3:
        edges = edges[..., np.newaxis]
    
    # Blend sharpened and original based on edge strength
    result = (sharpened * edges + img * (1 - edges)).astype(np.uint8)
    return result


def enhance_local_contrast(img, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Enhance local contrast using CLAHE in LAB color space.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    
    # Merge back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return result


def process_image(input_path, output_path=None, scale=2, denoise_strength=10, 
                 sharpen_amount=1.2, contrast_clip=3.0, color_preserve=0.75):
    """
    Main processing pipeline.
    """
    # Load image
    print(f"Loading image: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not load image: {input_path}")
    
    original = img.copy()
    h, w = img.shape[:2]
    print(f"Input image shape: {img.shape}")
    
    # Step 1: Smart pixel prediction upscaling
    print("Step 1: Smart pixel prediction upscaling...")
    if h * scale > 8000 or w * scale > 8000:
        # Use chunked processing for very large images
        upscaled = chunked_upscale(img, scale, chunk_size=800)
    else:
        upscaled = smart_pixel_prediction(img, scale)
    print(f"Upscaled to: {upscaled.shape}")
    
    # Resize original to match for color preservation
    original_resized = cv2.resize(original, (upscaled.shape[1], upscaled.shape[0]), 
                                 interpolation=cv2.INTER_LANCZOS4)
    
    # Step 2: Denoise
    print("Step 2: Advanced denoising...")
    denoised = advanced_denoise(upscaled, strength=denoise_strength)
    
    # Step 3: Color preservation
    print("Step 3: Preserving original colors...")
    color_preserved = color_preservation_blend(original_resized, denoised, 
                                              preserve_ratio=color_preserve)
    
    # Step 4: Local contrast enhancement
    print("Step 4: Enhancing local contrast...")
    contrast_enhanced = enhance_local_contrast(color_preserved, 
                                              clip_limit=contrast_clip)
    
    # Step 5: Adaptive sharpening
    print("Step 5: Adaptive sharpening...")
    final = adaptive_sharpen(contrast_enhanced, amount=sharpen_amount)
    
    # Save result
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_enhanced{ext}"
    
    print(f"Saving to: {output_path}")
    cv2.imwrite(output_path, final, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    print("\nEnhancement complete!")
    print(f"Output: {output_path}")
    print(f"Resolution: {final.shape[1]}x{final.shape[0]}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Advanced Image Enhancement (Local)')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output image path')
    parser.add_argument('-s', '--scale', type=float, default=2, 
                       help='Scale factor (default: 2)')
    parser.add_argument('-d', '--denoise', type=int, default=10,
                       help='Denoise strength 0-20 (default: 10)')
    parser.add_argument('--sharpen', type=float, default=1.2,
                       help='Sharpen amount 0-3 (default: 1.2)')
    parser.add_argument('--contrast', type=float, default=3.0,
                       help='Contrast clip limit 1-10 (default: 3.0)')
    parser.add_argument('--color-preserve', type=float, default=0.75,
                       help='Color preservation ratio 0-1 (default: 0.75)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Advanced Image Enhancement Tool (Local Processing)")
    print("=" * 60)
    
    try:
        process_image(
            args.input,
            args.output,
            scale=args.scale,
            denoise_strength=args.denoise,
            sharpen_amount=args.sharpen,
            contrast_clip=args.contrast,
            color_preserve=args.color_preserve
        )
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
