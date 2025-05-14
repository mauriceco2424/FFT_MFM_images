import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from scipy.optimize import least_squares
from matplotlib.patches import Arc



def pad_to_square(img_array):
    """Pad a 2D image to a square shape with zeros, centering the original image."""
    h, w = img_array.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=img_array.dtype)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = img_array
    return padded


def compute_fft(image_path, dc_radius, axis_width):
    """Compute the FFT of an image with DC and axis suppression."""
    img = Image.open(image_path)
    img_original = np.array(img)  # Preserve original for display
    if len(img_original.shape) == 3:  # Color image
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
    else:
        img_array = img_original  # Already grayscale

    # Pad to square to correct anisotropy
    img_array = pad_to_square(img_array)
    
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
    return pow_FFT, log_FFT

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

def plot_results(pow_FFT, log_FFT, shell_mask, theta_rads,
                inner_radius, outer_radius, window_width, window_height, plus_q_peak,
                colormap, angle_info, output_filename,
                theta_mean_deg, label_fontsize=10, box_fontsize=10):
    """Plot the original image, background-subtracted image, power spectrum, and log FFT with shell, peak, and angle info."""
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))

    
    center_y, center_x = pow_FFT.shape[0] // 2, pow_FFT.shape[1] // 2
    extent = [-center_x, center_x, -center_y, center_y]

    #--------- Pow FFT plot ----------
    axes.imshow(pow_FFT, cmap=colormap, extent=extent)
    axes.add_patch(Circle((0, 0), inner_radius, color='white', fill=False))
    axes.add_patch(Circle((0, 0), outer_radius, color='white', fill=False))

    if plus_q_peak is not None and theta_mean_deg is not None:
        px, py = plus_q_peak
        arrow_length = np.sqrt(px**2 + py**2)

        axes.arrow(0, 0, px, py, color='orange', width=0.7, head_width=3, length_includes_head=True)
        angle_label = f"{theta_mean_deg:.0f}°"
        angle_rad = np.deg2rad(theta_mean_deg)
        label_radius = np.sqrt(px**2 + py**2) + 5  

        lx = 1.2*label_radius * np.cos(angle_rad)
        ly = 1.2*label_radius * np.sin(angle_rad)

        #axes.text(lx, ly, angle_label,
        #            color='orange', fontsize=box_fontsize, ha='center', va='center',
        #            bbox=dict(facecolor='black', edgecolor='none', alpha=1))

        # Draw arc for the angle
        arc_radius = 0.3 * arrow_length
        arc = Arc((0, 0), arc_radius*2, arc_radius*2, angle=0,
                theta1=0, theta2=theta_mean_deg, color='orange', linewidth=1.5)
        axes.add_patch(arc)

    axes.axhline(y=0, color='white', linestyle='--', linewidth=1)
    axes.axvline(x=0, color='white', linestyle='--', linewidth=1)

    axes.set_xlim(-window_width//2, window_width//2)
    axes.set_ylim(-window_height//2, window_height//2)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_xlabel('')
    axes.set_ylabel('')

    #axes.text(0.05, 0.95, angle_info, transform=axes.transAxes, color='white', fontsize=box_fontsize, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='black', alpha=1, edgecolor='none'))

    axes.text(0.9, 0.9, angle_info,
          transform=axes.transAxes,
          color='white',
          fontsize=box_fontsize,
          verticalalignment='top',
          horizontalalignment='right',
          bbox=dict(facecolor='black', alpha=1, edgecolor='none'))


    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.show()

def main(image_path, inner_radius, outer_radius, theta_min, theta_max,
        window_width, window_height, dc_radius, axis_width,
        colormap, output_filename,
        threshold_ratio, label_fontsize, box_fontsize):
    """Main function to process image and find peak with angle error estimation."""
    pow_FFT, log_FFT = compute_fft(image_path, dc_radius, axis_width)
    height, width = pow_FFT.shape
    center_y, center_x = height // 2, width // 2
    
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
        threshold = threshold_ratio * peak_value
        peak_region_mask = plus_q_mask & (pow_FFT > threshold)
        
        if np.any(peak_region_mask):
            angles_rad = angles[peak_region_mask]
            intensities = pow_FFT[peak_region_mask]
            
            theta_mean_rad = np.sum(angles_rad * intensities) / np.sum(intensities)
            var_rad = np.sum(intensities * (angles_rad - theta_mean_rad)**2) / np.sum(intensities)
            sigma_rad = np.sqrt(var_rad)
            sigma_deg = np.rad2deg(sigma_rad)
            theta_mean_deg = np.rad2deg(theta_mean_rad) % 360
            
            angle_info = (
                f"$\\theta$ = {theta_mean_deg:.2f}° ± {sigma_deg:.2f}°"
            )


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
    
    plot_results(pow_FFT, log_FFT, shell_mask, theta_rads,
                inner_radius, outer_radius, window_width, window_height, plus_q_peak,
                colormap, angle_info, output_filename,
                theta_mean_deg, label_fontsize, box_fontsize)

if __name__ == "__main__":
    # Adjustable parameters
    image_path = '1x1_top.png'
    output_filename = 'fft_1x1_top_fig.png'
    inner_radius = 20
    outer_radius = 30
    theta_min = 90
    theta_max = 180
    window_width = 70
    window_height = 70
    dc_radius = 5
    axis_width = 5
    colormap = 'viridis'
    threshold_ratio = 0.5  
    label_fontsize = 25
    box_fontsize = 25


    main(image_path, inner_radius, outer_radius, theta_min, theta_max,
        window_width, window_height, dc_radius, axis_width,
        colormap, output_filename,
        threshold_ratio, label_fontsize, box_fontsize)