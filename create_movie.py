
import argparse
import os
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
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
        # placeholder for actual cpp_zarr API
        # expected return: numpy array of shape (X, Y, Z)
        # We need (Z, Y, X) for processing
        vol = cppzarr.read_zarr(str(path))
        return np.transpose(vol, (2, 1, 0))
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
        
        # Update full_path to be the actual absolute path found
        meta['full_path'] = str(f.resolve())
            
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
        
    # Scale to 8-bit
    volume = (volume * 255).astype(np.uint8)
    return volume

def blend_channel_mips(channel_mips_list, channel_metas):
    """
    Blends a list of MIPs (one from each channel) into a single RGB image.
    channel_mips_list: [mip_ch0, mip_ch1, ...] where each mip is (Y, X)
    Returns: (Y, X, 3) float32 RGB image (0-1 range).
    """
    if not channel_mips_list:
        return None
        
    y, x = channel_mips_list[0].shape
    rgb_img = np.zeros((y, x, 3), dtype=np.float32)
    
    # Simple color cycle: R, G, B, C, M, Y
    colors = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [0, 1, 1], [1, 0, 1], [1, 1, 0]
    ]
    
    for i, (mip, meta) in enumerate(zip(channel_mips_list, channel_metas)):
        color = np.array(colors[i % len(colors)])
        
        # mip is uint8 (0-255). Convert to float 0-1 for blending
        mip_float = mip.astype(np.float32) / 255.0
        
        # Additive blending
        # Resizing color to (1,1,3) to broadcast
        weighted = mip_float[..., np.newaxis] * color
        rgb_img += weighted
        
    # Clip sum to 1.0
    rgb_img = np.clip(rgb_img, 0, 1)
    return rgb_img

def get_slab_edges(z_dim, slab_size=None, num_slabs=None):
    """
    Calculates Z-positions for slab boundaries.
    """
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
            if size == 0 and z_dim < num_slabs:
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
                 break
    return edges

def create_slabs_mip(volume, slab_size=None, num_slabs=None):
    """
    Splits a single-channel volume into slabs along Z and computes MIP for each.
    volume: (Z, Y, X)
    Returns: list of 2D images (Y, X).
    """
    slabs = []
    z_dim = volume.shape[0]
    
    z_dim = volume.shape[0]
    edges = get_slab_edges(z_dim, slab_size, num_slabs)

    # Extract MIPs based on edges
    # Edges: [0, z1, z2, ..., z_dim]
    for i in range(len(edges) - 1):
        start = edges[i]
        end = edges[i+1]
        
        if end > start:
            sub_vol = volume[start:end]
            mip = np.max(sub_vol, axis=0)
            slabs.append(mip)
            
    return slabs

def arrange_mips(mips, voxel_size, border_width=2):
    """
    Juxtaposes MIPs into a single image with borders.
    Returns: (grid_image, (rows, cols, cell_h, cell_w))
    """
    if not mips:
        return None, (0,0,0,0)
        
    n = len(mips)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    y, x, c = mips[0].shape
    
    cell_h = y + 2 * border_width
    cell_w = x + 2 * border_width
    
    grid_img = np.zeros((rows * cell_h, cols * cell_w, c), dtype=np.float32)
    grid_img.fill(1.0) # White background/border
    
    for idx, mip in enumerate(mips):
        r = idx // cols
        c_idx = idx % cols
        
        y0 = r * cell_h + border_width
        x0 = c_idx * cell_w + border_width
        
        grid_img[y0 : y0+y, x0 : x0+x, :] = mip
        
    return grid_img, (rows, cols, cell_h, cell_w)

def compute_bleaching_stats(raw_volumes):
    """
    Sum of intensities for each channel.
    """
    stats = []
    for vol in raw_volumes:
        stats.append(np.sum(vol))
    return stats

    return stats

def compute_fft_planes_norm(volume):
    """
    Computes 3D FFT, normalizes by DC, and extracts principal planes.
    Returns magnitude normalized to DC=1.
    """
    if volume is None:
        return None, None, None
        
    # FFT
    f = fftn(volume)
    fshift = fftshift(f)
    
    # Magnitude
    magnitude = np.abs(fshift)
    
    # Normalize by max (DC component)
    mx = np.max(magnitude)
    if mx > 0:
        magnitude /= mx
        
    z, y, x = magnitude.shape
    
    # Principal planes
    # XY (at mid Z)
    # XZ (at mid Y)
    # YZ (at mid X) -> We extract (Z, Y) slice at mid X.
    # The plot expects Z vertical, Y horizontal usually?
    # User requested: "ky/kz panel to the right of kx/ky".
    # Standard: kx horizontal, ky vertical.
    # To right: kx becomes kz? No, standard is [x, y], [x, z], [z, y]
    # Let's extract them directly first.
    
    fft_xy = magnitude[z//2, :, :]   # (Y, X) - displays as Ky (vert), Kx (horiz)
    fft_xz = magnitude[:, y//2, :]   # (Z, X) - displays as Kz (vert), Kx (horiz)
    fft_yz = magnitude[:, :, x//2]   # (Z, Y) - displays as Kz (vert), Ky (horiz)
    
    return fft_xy, fft_xz, fft_yz

def get_fft_extent(voxel_size, shape):
    """
    Returns fixed extent [-1, 1] as requested.
    """
    return [-1, 1, -1, 1], [-1, 1, -1, 1], [-1, 1, -1, 1]

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

def format_msec(msec):
    """Formats milliseconds to HH:MM:SS string."""
    seconds = int(msec / 1000)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

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
    
    # Prepare Video Writer
    plt.switch_backend('Agg')
    
    dpi = 100
    # Layout:
    # Row 0: MIPs (spanning 2 cols), FFT XY, FFT YZ
    # Row 1: Bleach (spanning 2 cols), FFT XZ, (Empty or Legend?)
    # Actually user asked for specific FFT layout relative to each other.
    # "kx/kz panel below the kx/ky panel"
    # "ky/kz panel to the right of the kx/ky panel"
    
    # Let's make a custom grid.
    # Col 0-1: Image
    # Col 2: FFT col 1
    # Col 3: FFT col 2
    
    # Layout: 1920x1080 -> 4K 3840x2160
    # Figsize in inches: 38.4 x 21.6 at 100 dpi
    fig = plt.figure(figsize=(38.4, 21.6), dpi=100)
    
    # Grid: Main (Slabs) vs Sidebar (Analysis)
    # Width ratios: Slabs take ~85-90%, Sidebar ~10-15%
    # Using [6, 1] gives 1/7th ~ 14% width for sidebar
    gs_main = fig.add_gridspec(1, 2, width_ratios=[6, 1], wspace=0.02)
    
    # Left: MIPs
    ax_mips = fig.add_subplot(gs_main[0])
    
    # Right: Sidebar with vertical stack
    # 3 FFT panels + 1 Bleaching panel = 4 rows
    gs_side = gs_main[1].subgridspec(4, 1, hspace=0.15)
    
    ax_fft_xy = fig.add_subplot(gs_side[0])
    ax_fft_yz = fig.add_subplot(gs_side[1]) # Stacked vertically
    ax_fft_xz = fig.add_subplot(gs_side[2])
    ax_bleach = fig.add_subplot(gs_side[3])
    
    ax_mips.axis('off')
    
    # Writer
    writer = animation.FFMpegWriter(fps=5, metadata=dict(artist='Antigravity'), bitrate=3000)
    
    # Ensure output directory exists if output path has parents
    out_path_obj = Path(args.output)
    if out_path_obj.parent != Path('.'):
         out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with writer.saving(fig, args.output, dpi=dpi):
        num_channels = 0 
        start_time_msec = timeline[0]['time_msec'] if timeline else 0
        
        for t_idx, item in enumerate(timeline):
            current_msec = item['time_msec']
            rel_msec = current_msec - start_time_msec
            time_str = format_msec(rel_msec)
            
            print(f"Processing timepoint {current_msec} ms ({time_str}) ({t_idx+1}/{len(timeline)})...")
            
            # Load Data
            processed_volumes = [] # Keep for FFT (uint8)
            raw_sums = []
            channel_slabs_list = [] # List of list of mips: [ [ch0_slab0, ch0_slab1], [ch1_slab0...] ]
            
            for f_meta in item['channels']:
                try:
                    vol = load_zarr_data(f_meta['full_path'], mock=args.mock)
                    # Process
                    # 1. Background
                    vol_bg = vol.astype(np.float32) - args.background_offset
                    vol_bg[vol_bg < 0] = 0
                    
                    # Store stats on raw-ish data (after BG sub)
                    raw_sums.append(np.sum(vol_bg))
                    
                    # 2. Normalize and Convert to 8-bit
                    vol_norm = normalize_volume(vol_bg, args.contrast_percentiles[0], args.contrast_percentiles[1], args.gamma)
                    processed_volumes.append(vol_norm)
                    
                    # 3. Create Slabs (MIPs) immediately per channel to save memory? 
                    # Actually we still keep `vol_norm` in memory for FFT.
                    # But we don't build the huge RGB volume anymore.
                    slabs = create_slabs_mip(vol_norm, slab_size=args.slab_size, num_slabs=args.num_slabs)
                    channel_slabs_list.append(slabs)
                    
                except Exception as e:
                    print(f"Failed to load {f_meta['full_path']}: {e}")
                    # Handle missing channel
                    dummy_vol = np.zeros((10,10,10), dtype=np.uint8)
                    processed_volumes.append(dummy_vol)
                    raw_sums.append(0)
                    channel_slabs_list.append([np.zeros((10,10), dtype=np.uint8)])

            # Init global stats on first run
            if t_idx == 0:
                num_channels = len(processed_volumes)
                bleaching_history = [[] for _ in range(num_channels)]
                
                # Check shapes
                if processed_volumes:
                    z_shape, y_shape, x_shape = processed_volumes[0].shape
                    print(f"Volume shape: {z_shape, y_shape, x_shape}")

            # Update Bleaching History
            for c_i, val in enumerate(raw_sums):
                if c_i < len(bleaching_history):
                    bleaching_history[c_i].append(val)

            # --- Panel 1: MIPs ---
            # Blend the MIPs
            final_rgb_mips = []
            if channel_slabs_list:
                num_generated_slabs = len(channel_slabs_list[0])
                for s_i in range(num_generated_slabs):
                    # Gather the s_i-th slab from every channel
                    slabs_to_blend = []
                    for ch_slabs in channel_slabs_list:
                        if s_i < len(ch_slabs):
                            slabs_to_blend.append(ch_slabs[s_i])
                        else:
                            # Should not happen if geometry matches
                            slabs_to_blend.append(None) # blend handles missing? No, assume match
                            
                    rgb_slab = blend_channel_mips(slabs_to_blend, item['channels'])
                    final_rgb_mips.append(rgb_slab)
            
            mips_img, grid_geom = arrange_mips(final_rgb_mips, args.voxel_size, border_width=2)
            rows, cols, cell_h, cell_w = grid_geom
            
            ax_mips.clear()
            ax_mips.imshow(mips_img)
            ax_mips.set_title(f"Time: {time_str} | Frame: {t_idx}", fontsize=24)
            ax_mips.axis('off')

            # Add Z-range labels and scale bar
            if processed_volumes and final_rgb_mips:
                # Recalculate edges for labeling
                z_dim = processed_volumes[0].shape[0]
                edges = get_slab_edges(z_dim, args.slab_size, args.num_slabs)
                z_res = args.voxel_size[2] # Z is index 2
                
                rows, cols, cell_h, cell_w = grid_geom
                
                for s_idx in range(len(final_rgb_mips)):
                    if s_idx < len(edges) - 1:
                        z_start = edges[s_idx] * z_res
                        z_end = edges[s_idx+1] * z_res
                        
                        r = s_idx // cols
                        c_idx = s_idx % cols
                        
                        # Position text above the tile (Top Left)
                        # Left edge + offset
                        text_x = c_idx * cell_w + 40 
                        text_y = r * cell_h + 40 # Top offset
                        
                        ax_mips.text(text_x, text_y, f"{z_start:.1f}-{z_end:.1f} \u00b5m", 
                                     color='white', ha='left', va='top', fontsize=24, 
                                     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
            
            # Scale bar
            if args.voxel_size[0] > 0 and mips_img is not None:
                # 10 microns
                bar_len_um = 10
                bar_len_px = bar_len_um / args.voxel_size[0]
                
                # Draw if fits
                # Position: Bottom Left of FIRST slab (r=0, c=0).
                # x ~ 40, y ~ cell_h - 40
                if bar_len_px < mips_img.shape[1]:
                    rows, cols, cell_h, cell_w = grid_geom
                    # Bar at bottom left of first tile
                    rect = plt.Rectangle((40, cell_h - 40), 
                                         bar_len_px, 15, color='red')
                    ax_mips.add_patch(rect)
                    # Text above bar (Red)
                    ax_mips.text(40 + bar_len_px/2, cell_h - 70, 
                                 f"{bar_len_um} \u00b5m", color='red', ha='center', va='bottom', fontsize=24)

            # --- Panel 2: FFT ---
            if processed_volumes:
                sum_vol = np.sum(np.array(processed_volumes), axis=0)
                # Compute Norm FFT
                f_xy, f_xz, f_yz = compute_fft_planes_norm(sum_vol)
                
                # Extents
                # voxel_size is [x, y, z]
                ext_xy, ext_xz, ext_yz = get_fft_extent(args.voxel_size, sum_vol.shape)
                
                # Labels
                k_label = r"$k / (4\pi n / \lambda_{exc})$"
                
                # XY: kx (H), ky (V)
                ax_fft_xy.clear()
                ax_fft_xy.imshow(f_xy, cmap='jet', extent=ext_xy, 
                                 norm=LogNorm(vmin=0.01, vmax=1), origin='lower')
                ax_fft_xy.set_ylabel("ky")
                ax_fft_xy.set_xticklabels([]) # Hide X labels for compactness in stack
                ax_fft_xy.set_aspect('auto')

                # YZ: kz (H), ky (V)
                # Transposed to match ky vertical if adjacent, but here stacked.
                # Let's keep consistent axes labels.
                f_yz_cj = f_yz.T 
                ax_fft_yz.clear()
                ax_fft_yz.imshow(f_yz_cj, cmap='jet', extent=ext_yz, 
                                 norm=LogNorm(vmin=0.01, vmax=1), origin='lower')
                ax_fft_yz.set_ylabel("ky")
                ax_fft_yz.set_xticklabels([])
                ax_fft_yz.set_aspect('auto')

                # XZ: kx (H), kz (V)
                ax_fft_xz.clear()
                ax_fft_xz.imshow(f_xz, cmap='jet', extent=ext_xz, 
                                 norm=LogNorm(vmin=0.01, vmax=1), origin='lower')
                ax_fft_xz.set_xlabel(k_label)
                ax_fft_xz.set_ylabel("kz")
                ax_fft_xz.set_aspect('auto')
                
                # No colorbar to save sidebar space or add one tight?
                # "include the optimized FFT and bleaching panels"
                # Colorbar might crowd 15% width. Let's omit or make very small inside plot?
                # Let's stick to data for now.
            
            # --- Panel 3: Bleaching ---
            
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
            
            ax_bleach.set_title("Bleaching")
            ax_bleach.set_xlabel("Time (ms)")
            # ax_bleach.legend(fontsize='small') # Might crowd the small subplot
            
            writer.grab_frame()
            
    print(f"Saved movie to {args.output}")

def main():
    # Only parse args if called directly via script
    args = parse_args()
    run_movie_generation(args)

if __name__ == "__main__":
    main()



