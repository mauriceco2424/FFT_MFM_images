import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from scipy.optimize import least_squares

def compute_fft(image_path, dc_radius, axis_width):
    """Compute the FFT of an image with DC and axis suppression."""
    img = Image.open(image_path)
    img_original = np.array(img)  # Preserve original for display
    if len(img_original.shape) == 3:  # Color image
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
    else:
        img_array = img_original  # Already grayscale
    
    fft = np.fft.fft2(img_array)
    fft_shifted = np.fft.fftshift(fft)
    
    # Suppress DC and axes
    height, width = fft_shifted.shape
    y, x = np.ogrid[-height//2:height//2, -width//2:width//2]
    dc_mask = np.sqrt(x**2 + y**2) <= dc_radius
    axis_mask = (np.abs(x) <= axis_width) | (np.abs(y) <= axis_width)
    suppression_mask = dc_mask | axis_mask
    fft_shifted[suppression_mask] = 0
    
    pow_FFT = np.abs(fft_shifted)**2
    log_FFT = np.log1p(pow_FFT)
    return img_original, pow_FFT, log_FFT

def fit_background(image):
    """Fit a 2D quadratic background to the image using least squares."""
    height, width = image.shape
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    Z = image.flatten()
    
    # Define the quadratic model
    def model(params, X, Y):
        a, b, c, d, e, f = params
        return a*X**2 + b*Y**2 + c*X*Y + d*X + e*Y + f
    
    # Residuals for least squares
    def residuals(params):
        return model(params, X, Y) - Z
    
    # Initial guess
    initial_guess = [0, 0, 0, 0, 0, np.mean(Z)]
    result = least_squares(residuals, initial_guess)
    background = model(result.x, X, Y).reshape(height, width)
    return background

def create_shell_mask(height, width, inner_radius, outer_radius, theta_min, theta_max):
    """Create masks for the shell and angular sectors, adjusted for aspect ratio."""
    center_y, center_x = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    dx = x - center_x
    dy = y - center_y
    radii = np.sqrt(dx**2 + dy**2)
    angles = np.arctan2(-dy * width, dx * height)  # Adjusted for aspect ratio
    
    theta_min_rad = np.deg2rad(theta_min)
    theta_max_rad = np.deg2rad(theta_max)
    theta_min_opp_rad = (theta_min_rad + np.pi) % (2 * np.pi)
    theta_max_opp_rad = (theta_max_rad + np.pi) % (2 * np.pi)
    
    plus_q_mask = (radii >= inner_radius) & (radii <= outer_radius) & (angles >= theta_min_rad) & (angles <= theta_max_rad)
    minus_q_mask = (radii >= inner_radius) & (radii <= outer_radius) & (angles >= theta_min_opp_rad) & (angles <= theta_max_opp_rad)
    shell_mask = plus_q_mask | minus_q_mask
    
    return shell_mask, [theta_min_rad, theta_max_rad, theta_min_opp_rad, theta_max_opp_rad], plus_q_mask, minus_q_mask, angles

def plot_results(img_original, img_subtracted, pow_FFT, log_FFT, shell_mask, theta_rads, inner_radius, outer_radius, window_width, window_height, plus_q_peak, peak_circle_radius, colormap, angle_info, output_filename):
    """Plot the original image, background-subtracted image, power spectrum, and log FFT with shell, peak, and angle info."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img_original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(img_subtracted, cmap='inferno')
    axes[1].set_title('Background Subtracted Image')
    axes[1].axis('off')
    
    center_y, center_x = pow_FFT.shape[0] // 2, pow_FFT.shape[1] // 2
    extent = [-center_x, center_x, -center_y, center_y]
    
    axes[2].imshow(pow_FFT, cmap=colormap, extent=extent)
    axes[2].set_title('Power Spectrum with Shell (DC & Axes Suppressed)')
    axes[2].add_patch(Circle((0, 0), inner_radius, color='white', fill=False))
    axes[2].add_patch(Circle((0, 0), outer_radius, color='white', fill=False))
    for theta in theta_rads:
        axes[2].plot([0, outer_radius * np.cos(theta)], [0, outer_radius * np.sin(theta)], 'w-')
    if plus_q_peak is not None:
        axes[2].add_patch(Circle(plus_q_peak, radius=peak_circle_radius, color='red', fill=False, linewidth=1))
    axes[2].set_xlim(-window_width//2, window_width//2)
    axes[2].set_ylim(-window_height//2, window_height//2)
    axes[2].text(0.05, 0.95, angle_info, transform=axes[2].transAxes, color='white', fontsize=8, verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    
    axes[3].imshow(log_FFT, cmap=colormap, extent=extent)
    axes[3].set_title('Log Power Spectrum with Shell (DC & Axes Suppressed)')
    axes[3].add_patch(Circle((0, 0), inner_radius, color='white', fill=False))
    axes[3].add_patch(Circle((0, 0), outer_radius, color='white', fill=False))
    for theta in theta_rads:
        axes[3].plot([0, outer_radius * np.cos(theta)], [0, outer_radius * np.sin(theta)], 'w-')
    if plus_q_peak is not None:
        axes[3].add_patch(Circle(plus_q_peak, radius=peak_circle_radius, color='red', fill=False, linewidth=1))
    axes[3].set_xlim(-window_width//2, window_width//2)
    axes[3].set_ylim(-window_height//2, window_height//2)
    axes[3].text(0.05, 0.95, angle_info, transform=axes[3].transAxes, color='white', fontsize=8, verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.show()

def main(image_path, inner_radius, outer_radius, theta_min, theta_max, window_width, window_height, dc_radius, axis_width, peak_circle_radius, colormap, output_filename):
    """Main function to process image and find peak with angle error estimation."""
    img_original, pow_FFT, log_FFT = compute_fft(image_path, dc_radius, axis_width)
    height, width = pow_FFT.shape
    center_y, center_x = height // 2, width // 2
    
    # Background subtraction
    if len(img_original.shape) == 3:
        img_gray = np.array(Image.open(image_path).convert('L'))
    else:
        img_gray = img_original
    background = fit_background(img_gray)
    img_subtracted = img_gray - background
    
    shell_mask, theta_rads, plus_q_mask, minus_q_mask, angles = create_shell_mask(height, width, inner_radius, outer_radius, theta_min, theta_max)
    
    # Find peak in +q sector and estimate angle errors
    if np.any(plus_q_mask):
        peak_idx = np.argmax(pow_FFT * plus_q_mask)
        peak_y, peak_x = np.unravel_index(peak_idx, pow_FFT.shape)
        peak_value = pow_FFT[peak_y, peak_x]
        dx = peak_x - center_x
        dy = peak_y - center_y
        r = np.sqrt(dx**2 + dy**2)
        discretization_error = np.rad2deg(0.5 / r) if r > 0 else 0
        peak_angle_rad = np.arctan2(-dy * width, dx * height)  # Adjusted for aspect ratio
        peak_angle_deg = np.rad2deg(peak_angle_rad) % 360
        max_pixel_angle = f"Max Pixel Angle: {peak_angle_deg:.2f} degrees"
        print(max_pixel_angle)
        
        # Define peak region for statistical error estimation
        threshold = 0.3 * peak_value
        peak_region_mask = plus_q_mask & (pow_FFT > threshold)
        
        if np.any(peak_region_mask):
            angles_rad = angles[peak_region_mask]
            intensities = pow_FFT[peak_region_mask]
            
            theta_mean_rad = np.sum(angles_rad * intensities) / np.sum(intensities)
            var_rad = np.sum(intensities * (angles_rad - theta_mean_rad)**2) / np.sum(intensities)
            sigma_rad = np.sqrt(var_rad)
            sigma_deg = np.rad2deg(sigma_rad)
            theta_mean_deg = np.rad2deg(theta_mean_rad) % 360
            
            angle_info = (f"{max_pixel_angle}\n"
                         f"Weighted Mean Angle: {theta_mean_deg:.2f} ± {sigma_deg:.2f} degrees\n"
                         f"(Statistical Error: {sigma_deg:.2f}, Discretization Error: {discretization_error:.2f})")
            print(angle_info)
        else:
            angle_info = (f"{max_pixel_angle} ± {discretization_error:.2f} degrees\n"
                         f"Weighted Mean Angle: N/A\n"
                         f"(Discretization Error: {discretization_error:.2f})")
            print("No pixels above threshold in the +q shell region for statistical error estimation.")
        
        plus_q_peak = (dx + 0.5, -dy - 0.5)  # Center the circle on the pixel
    else:
        angle_info = "No pixels in the +q shell region."
        print(angle_info)
        plus_q_peak = None
    
    plot_results(img_original, img_subtracted, pow_FFT, log_FFT, shell_mask, theta_rads, inner_radius, outer_radius, window_width, window_height, plus_q_peak, peak_circle_radius, colormap, angle_info, output_filename)

if __name__ == "__main__":
    # Adjustable parameters
    image_path = '1to2_outside_bottom.png'
    output_filename = 'fft_1to2_outside_bottom.png'
    inner_radius = 13
    outer_radius = 18
    theta_min = 150
    theta_max = 190
    window_width = 60
    window_height = 60
    dc_radius = 5
    axis_width = 0
    peak_circle_radius = 2
    colormap = 'viridis'
    
    main(image_path, inner_radius, outer_radius, theta_min, theta_max, window_width, window_height, dc_radius, axis_width, peak_circle_radius, colormap, output_filename)