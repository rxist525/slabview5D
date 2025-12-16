
import argparse
import os
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import fftn, fftshift

# Try to import cppzarr, otherwise mock for dry-run/dev if not present
try:
    import cppzarr
except ImportError:
    print("Warning: cppzarr module not found. Zarr loading will fail unless mocked.", file=sys.stderr)
    cppzarr = None

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create a movie from Zarr images.")
    parser.add_argument("input_dir", type=str, help="Path to the input Zarr directory or parent directory containing timepoints.")
    parser.add_argument("--output", type=str, default="output_movie.mp4", help="Path to save the output movie.")
    parser.add_argument("--slab_size", type=int, default=10, help="Number of Z planes per slab for MIP.")
    parser.add_argument("--num_slabs", type=int, default=None, help="Total number of slabs (overrides slab_size). Extra planes go to first/last slabs.")
    parser.add_argument("--voxel_size", nargs=3, type=float, default=[0.108, 0.108, 0.108], metavar=('X', 'Y', 'Z'), help="Voxel sizes in microns (default: 0.108 0.108 0.108).")
    parser.add_argument("--contrast_percentiles", nargs=2, type=float, default=[1, 99], metavar=('MIN', 'MAX'), help="Percentiles for contrast stretching (default: 1 99).")
    parser.add_argument("--background_offset", type=float, default=0.0, help="Value to subtract from background.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction factor.")
    
    parser.add_argument("--mock", action="store_true", help="Generate mock data instead of loading files (for testing).")
    return parser.parse_args(argv)

def load_zarr_data(path, mock=False):
    if mock:
        # Generate random noise (Z=50, Y=100, X=100)
        return np.random.rand(50, 100, 100)

    if cppzarr:
        # Placeholder for actual cppzarr API
        # expected return: numpy array of shape (Z, Y, X) or similar
        return cppzarr.read_zarr(str(path))
    else:
        raise ImportError("cppzarr is required to load data. Use --mock to test without it.")


def parse_filename_metadata(filename):
    """
    Extracts time (msec) and channel info from filename.
    Pattern: Look for 'CamA'/'CamB', 'ch0'-'ch4', and 'msec'.
    """
    # Channel detection
    channel = None
    if "CamA" in filename:
        cam = "CamA"
    elif "CamB" in filename:
        cam = "CamB"
    else:
        cam = "Unknown"
    
    ch_match = re.search(r'ch(\d)', filename)
    if ch_match:
        ch = f"ch{ch_match.group(1)}"
    else:
        ch = "ch?"

    # Time detection (second msec occurrence)
    # Regex to find all integers followed by 'msec'
    # Example: file_100msec_500msec.zarr
    # We want 500.
    msec_matches = re.finditer(r'(\d+)msec', filename)
    msecs = [int(m.group(1)) for m in msec_matches]
    
    if len(msecs) >= 2:
        time_val = msecs[1]
    elif len(msecs) == 1:
        time_val = msecs[0]
    else:
        time_val = 0 # Default/Fallback
        
    return {
        'cam': cam,
        'ch': ch,
        'time': time_val,
        'full_path': filename
    }

def discover_and_group_files(input_dir):
    """
    Scans input_dir for Zarr files/dirs and groups them by timepoint.
    Returns a sorted list of timepoints, where each timepoint is a dict of channels.
    """
    input_path = Path(input_dir)
    # Assume zarr files are directories ending in .zarr or just files? 
    # Usually zarr is a dir.
    files = sorted([f for f in input_path.iterdir() if f.name.endswith('.zarr') or f.is_dir()]) # Broad check, filter later
    
    grouped_data = {} # time_val -> { 'channel_key': path }
    
    for f in files:
        meta = parse_filename_metadata(f.name)
        if meta['cam'] == "Unknown" and meta['ch'] == "ch?":
            continue # Skip unrelated files
            
        t = meta['time']
        if t not in grouped_data:
            grouped_data[t] = []
        
        grouped_data[t].append(meta)
        
    # Sort timepoints
    sorted_times = sorted(grouped_data.keys())
    
    # Process into structured list
    timeline = []
    for t in sorted_times:
        # Sort channels within timepoint for consistent order
        # Key: CamA_ch0, CamA_ch1... CamB_ch0...
        channels = sorted(grouped_data[t], key=lambda x: (x['cam'], x['ch']))
        timeline.append({
            'time_msec': t,
            'channels': channels
        })
        
    return timeline



def normalize_volume(volume, p_min, p_max, gamma):
    """
    Normalizes volume based on percentiles and applies gamma.
    """
    if volume.size == 0:
        return volume
    
    vmin, vmax = np.percentile(volume, [p_min, p_max])
    
    # Clip and normalize 0-1
    volume = np.clip(volume, vmin, vmax)
    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)
    else:
        volume = np.zeros_like(volume)
        
    # Gamma
    if gamma != 1.0:
        volume = volume ** gamma
        
    return volume

def blend_volumes(volumes, channel_metas):
    """
    Blends multiple single-channel volumes into one RGB volume.
    Simple color mapping based on index or parsing channel name.
    """
    if not volumes:
        return None
        
    z, y, x = volumes[0].shape
    rgb_vol = np.zeros((z, y, x, 3), dtype=np.float32)
    
    # Simple color cycle: R, G, B, C, M, Y
    colors = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [0, 1, 1], [1, 0, 1], [1, 1, 0]
    ]
    
    for i, (vol, meta) in enumerate(zip(volumes, channel_metas)):
        color = np.array(colors[i % len(colors)])
        # Additive blending
        # Resizing color to (1,1,1,3) to broadcast
        # vol is (Z, Y, X) -> (Z, Y, X, 1)
        weighted = vol[..., np.newaxis] * color
        rgb_vol += weighted
        
    # Clip sum to 1.0
    rgb_vol = np.clip(rgb_vol, 0, 1)
    return rgb_vol

def create_slabs_mip(rgb_volume, slab_size=None, num_slabs=None):
    """
    Splits volume into slabs along Z and computes MIP for each.
    If num_slabs is provided, it overrides slab_size.
    Extra planes (remainder) are distributed to first and last slabs.
    Returns: list of RGB images (Y, X, 3).
    """
    slabs = []
    z_dim = rgb_volume.shape[0]
    
    # Determine slab boundaries
    edges = [0]
    
    if num_slabs is not None and num_slabs > 0:
        base_size = z_dim // num_slabs
        remainder = z_dim % num_slabs
        
        # Distribute remainder to first and last
        rem_first = remainder // 2
        rem_last = remainder - rem_first
        
        current_z = 0
        for i in range(num_slabs):
            size = base_size
            if i == 0:
                size += rem_first
            elif i == num_slabs - 1:
                size += rem_last
            
            # Ensure size is at least 1 if z_dim >= num_slabs
            # (If z_dim < num_slabs, some slabs might be empty or handling needed? 
            # Assuming z_dim >= num_slabs for logical slicing)
            if size == 0 and z_dim < num_slabs:
                 # Fallback for edge cases where requested slabs > z thickness?
                 # Just make 1-slice slabs until out of data?
                 pass 
            
            current_z += size
            edges.append(current_z)
            
    else:
        # Fixed slab size
        current_z = 0
        while current_z < z_dim:
            next_z = min(current_z + slab_size, z_dim)
            edges.append(next_z)
            current_z += slab_size
            if current_z >= z_dim and edges[-1] != z_dim:
                 # If loop condition met but last edge wasn't added?
                 # Logic above adds next_z.
                 break

    # Extract MIPs based on edges
    # Edges: [0, z1, z2, ..., z_dim]
    for i in range(len(edges) - 1):
        start = edges[i]
        end = edges[i+1]
        
        if end > start:
            sub_vol = rgb_volume[start:end]
            mip = np.max(sub_vol, axis=0)
            slabs.append(mip)
            
    return slabs

def arrange_mips(mips, voxel_size):
    """
    Juxtaposes MIPs into a single image.
    Strategy: Best-fit grid (sqrt(N)).
    Also draws scale bar (handled at plotting time is easier, but here we prep the image).
    """
    if not mips:
        return None
        
    n = len(mips)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    y, x, c = mips[0].shape
    
    grid_img = np.zeros((rows * y, cols * x, c), dtype=np.float32)
    
    for idx, mip in enumerate(mips):
        r = idx // cols
        c_idx = idx % cols
        grid_img[r*y : (r+1)*y, c_idx*x : (c_idx+1)*x, :] = mip
        
    return grid_img

def compute_bleaching_stats(raw_volumes):
    """
    Sum of intensities for each channel.
    """
    stats = []
    for vol in raw_volumes:
        stats.append(np.sum(vol))
    return stats

def compute_fft_planes(volume):
    """
    Computes 3D FFT and extracts principal central planes (log transform).
    """
    # FFT
    f = fftn(volume)
    fshift = fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-9) # Log magnitude
    
    z, y, x = magnitude.shape
    
    # Principal planes
    xy = magnitude[z//2, :, :]
    xz = magnitude[:, y//2, :]
    yz = magnitude[:, :, x//2]
    
    return xy, xz, yz


def combine_ppt_planes(xy, xz, yz):
    """
    Combines 3 principal planes into one image for display.
    Layout: Stack horizontally [XY, XZ, YZ].
    Pads height to match standard.
    """
    h_xy, w_xy = xy.shape
    h_xz, w_xz = xz.shape
    h_yz, w_yz = yz.shape
    
    target_h = max(h_xy, h_xz, h_yz)
    
    def pad_to_h(img, h):
        c_h, c_w = img.shape
        if c_h < h:
            pad = np.zeros((h - c_h, c_w), dtype=img.dtype)
            return np.vstack([img, pad])
        return img[:h, :]

    full_img = np.hstack([pad_to_h(xy, target_h), pad_to_h(xz, target_h), pad_to_h(yz, target_h)])
    return full_img


def run_movie_generation(args):
    """
    Main execution logic, separated for easier import/usage in notebooks.
    args: Namespace or object with attributes matching command line arguments.
    """
    input_path = Path(args.input_dir)
    timeline = []
    
    if args.mock:
        print("Running in MOCK mode.")
        # Create fake timeline using a loop
        for t in range(0, 500, 100):
            timeline.append({
                'time_msec': t,
                'channels': [
                    {'cam': 'CamA', 'ch': 'ch0', 'full_path': 'mock_path'},
                    {'cam': 'CamB', 'ch': 'ch1', 'full_path': 'mock_path'}
                ]
            })
    elif not input_path.exists():
        print(f"Error: Input path {input_path} does not exist.")
        return
    else:
        timeline = discover_and_group_files(input_path)
    
    if not timeline:
        print("No matching files found.")
        return
        
    print(f"Found {len(timeline)} timepoints.")
    
    # Setup Initialization for Plotting
    bleaching_history = [] 
    
    # Prepare Video Writer
    plt.switch_backend('Agg')
    
    dpi = 100
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3) 
    
    ax_mips = fig.add_subplot(gs[:, 0:2])
    ax_fft = fig.add_subplot(gs[0, 2])
    ax_bleach = fig.add_subplot(gs[1, 2])
    
    ax_mips.axis('off')
    ax_fft.set_title("FFT Principal Planes (XY | XZ | YZ)")
    ax_fft.axis('off')
    
    # Writer
    writer = animation.FFMpegWriter(fps=5, metadata=dict(artist='Antigravity'), bitrate=3000)
    
    # Ensure output directory exists if output path has parents
    out_path_obj = Path(args.output)
    if out_path_obj.parent != Path('.'):
         out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with writer.saving(fig, args.output, dpi=dpi):
        num_channels = 0 
        
        for t_idx, item in enumerate(timeline):
            print(f"Processing timepoint {item['time_msec']} ms ({t_idx+1}/{len(timeline)})...")
            
            # Load Data
            volumes = []
            raw_sums = []
            
            for f_meta in item['channels']:
                try:
                    vol = load_zarr_data(f_meta['full_path'], mock=args.mock)
                    # Process
                    # 1. Background
                    vol_bg = vol.astype(np.float32) - args.background_offset
                    vol_bg[vol_bg < 0] = 0
                    
                    # Store stats on raw-ish data (after BG sub)
                    raw_sums.append(np.sum(vol_bg))
                    
                    # 2. Normalize
                    vol_norm = normalize_volume(vol_bg, args.contrast_percentiles[0], args.contrast_percentiles[1], args.gamma)
                    volumes.append(vol_norm)
                    
                except Exception as e:
                    print(f"Failed to load {f_meta['full_path']}: {e}")
                    # Handle missing channel?
                    volumes.append(np.zeros((10,10,10), dtype=np.float32)) 
                    raw_sums.append(0)

            # Init global stats on first run
            if t_idx == 0:
                num_channels = len(volumes)
                bleaching_history = [[] for _ in range(num_channels)]
                
                # Check shapes
                if volumes:
                    z_shape, y_shape, x_shape = volumes[0].shape
                    print(f"Volume shape: {z_shape, y_shape, x_shape}")

            # Update Bleaching History
            for c_i, val in enumerate(raw_sums):
                if c_i < len(bleaching_history):
                    bleaching_history[c_i].append(val)

            # --- Panel 1: MIPs ---
            rgb_composite = blend_volumes(volumes, item['channels'])
            slabs = create_slabs_mip(rgb_composite, slab_size=args.slab_size, num_slabs=args.num_slabs)
            mips_img = arrange_mips(slabs, args.voxel_size)
            
            ax_mips.clear()
            ax_mips.imshow(mips_img)
            ax_mips.set_title(f"Time: {item['time_msec']} ms")
            ax_mips.axis('off')
            
            # Scale bar
            if args.voxel_size[0] > 0 and mips_img is not None:
                # 10 microns
                bar_len_um = 10
                bar_len_px = bar_len_um / args.voxel_size[0]
                
                # Draw if fits
                if bar_len_px < mips_img.shape[1]:
                    rect = plt.Rectangle((mips_img.shape[1] - bar_len_px - 20, mips_img.shape[0] - 20), 
                                         bar_len_px, 5, color='white')
                    ax_mips.add_patch(rect)
                    ax_mips.text(mips_img.shape[1] - bar_len_px - 20, mips_img.shape[0] - 30, 
                                 f"{bar_len_um} um", color='white', ha='left')

            # --- Panel 2: FFT ---
            sum_vol = np.sum(np.array(volumes), axis=0) 
            fft_xy, fft_xz, fft_yz = compute_fft_planes(sum_vol)
            
            # Combine planes
            fft_combined = combine_ppt_planes(fft_xy, fft_xz, fft_yz)
            
            ax_fft.clear()
            ax_fft.imshow(fft_combined, cmap='inferno')
            ax_fft.set_title("FFT (XY | XZ | YZ)")
            ax_fft.axis('off')
            
            # --- Panel 3: Bleaching ---
            ax_bleach.clear()
            times = [x['time_msec'] for x in timeline[:t_idx+1]]
            for c_i in range(num_channels):
                # Normalize to 1 (first point)
                vals = np.array(bleaching_history[c_i])
                if len(vals) > 0 and vals[0] > 0:
                    vals = vals / vals[0]
                
                label = f"Ch{c_i}"
                if c_i < len(item['channels']):
                    c_meta = item['channels'][c_i]
                    if c_meta['cam'] != "Unknown":
                        label = f"{c_meta['cam']}_{c_meta['ch']}"
                    
                ax_bleach.plot(times, vals, label=label)
            
            ax_bleach.set_title("Bleaching (Norm)")
            ax_bleach.set_xlabel("Time (ms)")
            ax_bleach.legend(fontsize='small')
            
            writer.grab_frame()
            
    print(f"Saved movie to {args.output}")

def main():
    # Only parse args if called directly via script
    args = parse_args()
    run_movie_generation(args)

if __name__ == "__main__":
    main()



