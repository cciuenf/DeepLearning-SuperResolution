"""
Image Quality Metrics Module
Implements PSNR and SSIM calculations
"""

import math
import numpy as np
import cv2 as cv
from sewar.full_ref import mse, psnr


def covariance(orig, upsc, mu1, mu2):
    """Calculate covariance between two images"""
    diff1 = orig - mu1
    diff2 = upsc - mu2
    prod = np.multiply(diff1, diff2)
    return np.sum(prod) / (diff1.size - 1)


def calculate_ssim(orig, upsc, lightness=1.0, contrast=1.0, structure=1.0, K1=0.01, K2=0.03, L=255):
    """
    Calculate Structural Similarity Index (SSIM) between two images
    
    Args:
        orig: Original image
        upsc: Processed/upscaled image
        lightness: Weight for lightness comparison (default: 1.0)
        contrast: Weight for contrast comparison (default: 1.0)
        structure: Weight for structure comparison (default: 1.0)
        K1: Stability constant (default: 0.01)
        K2: Stability constant (default: 0.03)
        L: Dynamic range of pixel values (default: 255)
    
    Returns:
        numpy.ndarray: SSIM value(s)
    """
    C1 = pow(K1 * L, 2)
    C2 = pow(K2 * L, 2)
    C3 = C2 / 2.0

    (mu1a, sigma1a) = cv.meanStdDev(orig)
    (mu2a, sigma2a) = cv.meanStdDev(upsc)
    mu1 = mu1a[0]
    mu2 = mu2a[0]
    sigma1 = sigma1a[0]
    sigma2 = sigma2a[0]

    cov = covariance(orig, upsc, mu1, mu2)
    l = (2.0 * mu1 * mu2 + C1) / (pow(mu1, 2) + pow(mu2, 2) + C1)
    c = (2.0 * sigma1 * sigma2 + C2) / (pow(sigma1, 2) + pow(sigma2, 2) + C2)
    s = (cov + C3) / (sigma1 * sigma2 + C3)

    return pow(l, lightness) * pow(c, contrast) * pow(s, structure)


def calculate_psnr(orig, upsc):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images
    
    Args:
        orig: Original image
        upsc: Processed/upscaled image
    
    Returns:
        float: PSNR value in dB
    """
    return psnr(orig, upsc)


def calculate_mse(orig, upsc):
    """
    Calculate Mean Squared Error (MSE) between two images
    
    Args:
        orig: Original image
        upsc: Processed/upscaled image
    
    Returns:
        float: MSE value
    """
    return mse(orig, upsc)


def calculate_all_psnr(originals, upscaled_images):
    """
    Calculate average PSNR across multiple image pairs
    
    Args:
        originals: List of original image dictionaries
        upscaled_images: List of upscaled image dictionaries
    
    Returns:
        float: Average PSNR in dB
    """
    if len(originals) == 0 or len(upscaled_images) == 0:
        return 0.0
    
    psnr_total = 0.0
    count = min(len(originals), len(upscaled_images))
    
    for i in range(count):
        orig = originals[i]["cv_img"]
        upsc = upscaled_images[i]["cv_img"]
        mse_val = mse(orig, upsc)
        psnr_total += mse_val
    
    avg_mse = psnr_total / count
    psnr_value = math.log10(pow(255, 2) / avg_mse) * 10
    
    return psnr_value


def calculate_metrics_summary(originals, upscaled_images):
    """
    Calculate comprehensive metrics summary for image pairs
    
    Args:
        originals: List of original image dictionaries
        upscaled_images: List of upscaled image dictionaries
    
    Returns:
        dict: Summary containing average PSNR, SSIM, and per-image metrics
    """
    if len(originals) == 0 or len(upscaled_images) == 0:
        return {"error": "No images provided"}
    
    count = min(len(originals), len(upscaled_images))
    per_image_metrics = []
    total_psnr = 0.0
    total_ssim = 0.0
    
    for i in range(count):
        orig = originals[i]["cv_img"]
        upsc = upscaled_images[i]["cv_img"]
        
        psnr_val = calculate_psnr(orig, upsc)
        ssim_val = calculate_ssim(orig, upsc)
        mse_val = calculate_mse(orig, upsc)
        
        metrics = {
            "index": i + 1,
            "psnr": float(psnr_val),
            "ssim": float(ssim_val[0]) if isinstance(ssim_val, np.ndarray) else float(ssim_val),
            "mse": float(mse_val)
        }
        
        per_image_metrics.append(metrics)
        total_psnr += metrics["psnr"]
        total_ssim += metrics["ssim"]
    
    return {
        "count": count,
        "average_psnr": total_psnr / count,
        "average_ssim": total_ssim / count,
        "per_image": per_image_metrics
    }


def format_metrics_output(metrics_summary, verbose=False):
    """
    Format metrics summary for console output
    
    Args:
        metrics_summary: Dictionary from calculate_metrics_summary
        verbose: Include per-image details
    
    Returns:
        str: Formatted output string
    """
    if "error" in metrics_summary:
        return f"Error: {metrics_summary['error']}"
    
    output = []
    output.append(f"Metrics Summary ({metrics_summary['count']} image pairs):")
    output.append(f"Average PSNR: {metrics_summary['average_psnr']:.2f} dB")
    output.append(f"Average SSIM: {metrics_summary['average_ssim']:.4f}")
    
    if verbose and metrics_summary.get('per_image'):
        output.append("\nPer-image metrics:")
        for img_metrics in metrics_summary['per_image']:
            output.append(
                f"  Image {img_metrics['index']}: "
                f"PSNR={img_metrics['psnr']:.2f} dB, "
                f"SSIM={img_metrics['ssim']:.4f}, "
                f"MSE={img_metrics['mse']:.2f}"
            )
    
    return "\n".join(output)
