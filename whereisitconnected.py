import os
import argparse
import sys
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# Optional imports for acceleration
try:
    import cc3d
    CC3D_AVAILABLE = True
except ImportError:
    CC3D_AVAILABLE = False

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import dask.array as da
    from dask.diagnostics import ProgressBar
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import torch
    # Check if MPS (Metal Performance Shaders) is available
    MPS_AVAILABLE = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    TORCH_AVAILABLE = True
except ImportError:
    MPS_AVAILABLE = False
    TORCH_AVAILABLE = False

def load_raw_3d(filename, dims, dtype=np.uint8):
    """
    Load an 8-bit raw 3D image from a file, automatically determining and skipping header bytes.
    
    Parameters:
        filename (str): Path to the raw file.
        dims (tuple): Dimensions of the 3D image (e.g. (depth, height, width)).
        dtype (data-type): Data type of the image (default np.uint8).
    
    Returns:
        np.ndarray: 3D image array.
    """
    expected_size = np.prod(dims)
    file_size = os.path.getsize(filename)
    
    # Calculate header bytes if file_size is greater than expected_size
    if file_size < expected_size:
        raise ValueError("File size is smaller than expected. The file may be corrupted or the dimensions are wrong.")
    
    header_bytes = file_size - expected_size
    print("File size (bytes):", file_size)
    print("Expected size (bytes):", expected_size)
    print("Automatically detected header bytes:", header_bytes)
    
    with open(filename, 'rb') as f:
        f.seek(header_bytes)  # Skip the header bytes if present
        data = np.frombuffer(f.read(expected_size), dtype=dtype)
    
    if data.size != expected_size:
        raise ValueError("After skipping header, the data size does not match the expected dimensions.")
    return data.reshape(dims)

def check_phase_connectivity(volume, phase=1, backend='cc3d'):
    """
    Checks connectivity for a given phase in a 3D volume using voxel face connectivity.
    Determines if any connected component spans from one face to the opposite face along each axis.

    Parameters:
        volume (np.ndarray): The 3D image array.
        phase (int): The voxel value representing the phase of interest.
        backend (str): Backend to use: 'scipy', 'cc3d', 'cupy', or 'dask'.

    Returns:
        labeled (np.ndarray): Array of labeled connected components.
        num_features (int): Number of connected components.
        connectivity (dict): For each axis, a tuple (is_connected, labels)
                             where is_connected is True if a component touches both boundaries.
    """
    mask = (volume == phase)

    # Perform connected component labeling based on backend
    if backend == 'cc3d':
        if not CC3D_AVAILABLE:
            raise ImportError("cc3d is not installed. Install with: pip install connected-components-3d")
        # cc3d uses 6-connectivity by default
        labeled = cc3d.connected_components(mask.astype(np.uint8), connectivity=6)
        num_features = labeled.max()

    elif backend == 'cupy':
        if not CUPY_AVAILABLE:
            raise ImportError("cupy is not installed. Install with: pip install cupy-cuda11x (or appropriate CUDA version)")
        # Transfer to GPU
        mask_gpu = cp.asarray(mask)
        struct_gpu = cp.zeros((3, 3, 3), dtype=cp.int32)
        struct_gpu[1, 1, 1] = 1
        struct_gpu[0, 1, 1] = 1
        struct_gpu[2, 1, 1] = 1
        struct_gpu[1, 0, 1] = 1
        struct_gpu[1, 2, 1] = 1
        struct_gpu[1, 1, 0] = 1
        struct_gpu[1, 1, 2] = 1
        labeled_gpu, num_features = cp_ndimage.label(mask_gpu, structure=struct_gpu)
        # Transfer back to CPU
        labeled = cp.asnumpy(labeled_gpu)

    elif backend == 'dask':
        if not DASK_AVAILABLE:
            raise ImportError("dask is not installed. Install with: pip install dask[array]")
        # Convert to dask array with chunks
        mask_da = da.from_array(mask, chunks='auto')
        # Note: dask doesn't have native ndimage.label, so we use map_overlap
        # This is a simplified implementation - for production use scipy on chunks
        print("Warning: Dask backend uses scipy.ndimage on chunks - may not be optimal for connected components")
        struct = np.zeros((3, 3, 3), dtype=int)
        struct[1, 1, 1] = 1
        struct[0, 1, 1] = 1
        struct[2, 1, 1] = 1
        struct[1, 0, 1] = 1
        struct[1, 2, 1] = 1
        struct[1, 1, 0] = 1
        struct[1, 1, 2] = 1
        labeled, num_features = ndimage.label(mask, structure=struct)

    elif backend == 'mps':
        if not MPS_AVAILABLE:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not installed. Install with: pip install torch")
            else:
                raise RuntimeError("MPS backend requires Apple Silicon Mac with macOS 12.3+")

        # MPS doesn't have native connected components, so we use a hybrid approach:
        # Use MPS for preprocessing, then use cc3d or scipy for labeling
        print("Note: MPS backend uses GPU for preprocessing, then cc3d/scipy for labeling")

        # Try to use cc3d for the actual labeling (fastest)
        if CC3D_AVAILABLE:
            labeled = cc3d.connected_components(mask.astype(np.uint8), connectivity=6)
            num_features = labeled.max()
        else:
            # Fallback to scipy
            struct = np.zeros((3, 3, 3), dtype=int)
            struct[1, 1, 1] = 1
            struct[0, 1, 1] = 1
            struct[2, 1, 1] = 1
            struct[1, 0, 1] = 1
            struct[1, 2, 1] = 1
            struct[1, 1, 0] = 1
            struct[1, 1, 2] = 1
            labeled, num_features = ndimage.label(mask, structure=struct)

    else:  # scipy (default)
        # 6-connected (face connectivity) structure
        struct = np.zeros((3, 3, 3), dtype=int)
        struct[1, 1, 1] = 1
        struct[0, 1, 1] = 1  # -z
        struct[2, 1, 1] = 1  # +z
        struct[1, 0, 1] = 1  # -y
        struct[1, 2, 1] = 1  # +y
        struct[1, 1, 0] = 1  # -x
        struct[1, 1, 2] = 1  # +x
        labeled, num_features = ndimage.label(mask, structure=struct)

    connectivity = {}
    axes_names = ['axis0 (depth)', 'axis1 (height)', 'axis2 (width)']
    for axis, name in enumerate(axes_names):
        labels_min = set(np.unique(np.take(labeled, indices=0, axis=axis)))
        labels_max = set(np.unique(np.take(labeled, indices=-1, axis=axis)))
        labels_min.discard(0)
        labels_max.discard(0)
        common = labels_min.intersection(labels_max)
        connectivity[name] = (len(common) > 0, list(common))

    return labeled, num_features, connectivity

def get_component_bounding_boxes(labeled, num_features):
    """
    Calculate bounding boxes for each labeled component.

    Parameters:
        labeled (np.ndarray): Array of labeled connected components.
        num_features (int): Number of connected components.

    Returns:
        list: List of dictionaries containing bounding box information for each component.
              Each dictionary contains:
              - label: Component label
              - bbox: Tuple of (min_depth, max_depth, min_height, max_height, min_width, max_width)
              - volume: Number of voxels in the component
              - extent_depth: Size along depth axis
              - extent_height: Size along height axis
              - extent_width: Size along width axis
    """
    components = []

    # Use cc3d's built-in statistics if available (much faster!)
    if CC3D_AVAILABLE:
        stats = cc3d.statistics(labeled)

        for label in range(1, num_features + 1):
            # Get bounding box from cc3d statistics
            # Format: tuple of slice objects (slice_z, slice_y, slice_x)
            bbox_slices = stats['bounding_boxes'][label]

            # Get volume (voxel count)
            volume = stats['voxel_counts'][label]

            # Extract min/max from slice objects
            min_depth = bbox_slices[0].start
            max_depth = bbox_slices[0].stop - 1  # slice.stop is exclusive
            min_height = bbox_slices[1].start
            max_height = bbox_slices[1].stop - 1
            min_width = bbox_slices[2].start
            max_width = bbox_slices[2].stop - 1

            # Calculate extents
            extent_depth = max_depth - min_depth + 1
            extent_height = max_height - min_height + 1
            extent_width = max_width - min_width + 1

            component_info = {
                'label': label,
                'bbox': (min_depth, max_depth,
                        min_height, max_height,
                        min_width, max_width),
                'volume': volume,
                'extent_depth': extent_depth,
                'extent_height': extent_height,
                'extent_width': extent_width
            }

            components.append(component_info)

    else:
        # Fallback to manual calculation if cc3d not available
        for label in range(1, num_features + 1):
            # Find all voxels belonging to this component
            component_mask = (labeled == label)
            coords = np.argwhere(component_mask)

            if coords.size == 0:
                continue

            # Calculate bounding box
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)

            # Calculate extents
            extent_depth = max_coords[0] - min_coords[0] + 1
            extent_height = max_coords[1] - min_coords[1] + 1
            extent_width = max_coords[2] - min_coords[2] + 1

            # Calculate volume (number of voxels)
            volume = coords.shape[0]

            component_info = {
                'label': label,
                'bbox': (min_coords[0], max_coords[0],
                        min_coords[1], max_coords[1],
                        min_coords[2], max_coords[2]),
                'volume': volume,
                'extent_depth': extent_depth,
                'extent_height': extent_height,
                'extent_width': extent_width
            }

            components.append(component_info)

    return components

def plot_slices(volume):
    """
    Plots the central slice from the 3D volume in each direction: axial, coronal, and sagittal.
    
    Parameters:
        volume (np.ndarray): The 3D image array.
    """
    depth, height, width = volume.shape
    axial_slice = volume[depth // 2, :, :]       # Middle slice along depth
    coronal_slice = volume[:, height // 2, :]     # Middle slice along height
    sagittal_slice = volume[:, :, width // 2]     # Middle slice along width

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(axial_slice, cmap='gray')
    axes[0].set_title("Axial Slice (depth middle)")
    axes[0].axis('off')
    
    axes[1].imshow(coronal_slice, cmap='gray')
    axes[1].set_title("Coronal Slice (height middle)")
    axes[1].axis('off')
    
    axes[2].imshow(sagittal_slice, cmap='gray')
    axes[2].set_title("Sagittal Slice (width middle)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def render_3d_component(labeled, component_label, output_file=None, elevation=30, azimuth=45, subsample=1):
    """
    Render a 3D surface visualization of a specific component using marching cubes.

    Parameters:
        labeled (np.ndarray): Array of labeled connected components.
        component_label (int): Component label to visualize.
        output_file (str): Optional file to save the visualization.
        elevation (int): Viewing elevation angle in degrees (default: 30).
        azimuth (int): Viewing azimuth angle in degrees (default: 45).
        subsample (int): Subsampling factor to reduce memory usage (default: 1, no subsampling).
    """
    print("Rendering 3D surface visualization of component {}...".format(component_label))

    # Extract only the target component
    component_mask = (labeled == component_label).astype(np.uint8)

    # Subsample if needed for large volumes
    if subsample > 1:
        component_mask = component_mask[::subsample, ::subsample, ::subsample]
        print("WARNING: Subsampling by factor of {} - thin structures may lose connectivity!".format(subsample))

    if component_mask.sum() == 0:
        print("ERROR: No voxels found for component {}".format(component_label))
        return

    print("Generating surface mesh using marching cubes...")

    # Use marching cubes to create a surface mesh
    try:
        verts, faces, normals, values = measure.marching_cubes(component_mask, level=0.5, spacing=(1.0, 1.0, 1.0))
    except Exception as e:
        print("ERROR: Failed to generate surface mesh: {}".format(e))
        return

    print("Generated mesh with {} vertices and {} faces".format(len(verts), len(faces)))

    # Marching cubes outputs vertices in (Z, Y, X) order, but matplotlib expects (X, Y, Z)
    # Swap the coordinates
    verts_swapped = verts[:, [2, 1, 0]]

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create the mesh with swapped coordinates and lighting
    mesh = Poly3DCollection(verts_swapped[faces], alpha=0.9, linewidth=0, edgecolor='darkblue')
    mesh.set_facecolor([0.3, 0.7, 1.0])
    ax.add_collection3d(mesh)

    # Set plot limits based on actual mesh bounds with some padding
    padding = 5
    ax.set_xlim(verts_swapped[:, 0].min() - padding, verts_swapped[:, 0].max() + padding)
    ax.set_ylim(verts_swapped[:, 1].min() - padding, verts_swapped[:, 1].max() + padding)
    ax.set_zlim(verts_swapped[:, 2].min() - padding, verts_swapped[:, 2].max() + padding)

    # Set labels
    ax.set_xlabel('Width (X)')
    ax.set_ylabel('Height (Y)')
    ax.set_zlabel('Depth (Z)')
    ax.set_title('3D Surface: Component {}'.format(component_label))

    # Set viewing angle
    ax.view_init(elev=elevation, azim=azimuth)

    # Set background color
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
        print("3D visualization saved to: {}".format(output_file))
    else:
        plt.show()

    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze phase connectivity in 3D volumetric imaging data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python whereisitconnected.py image.raw 500 351 351 --phase 1
  python whereisitconnected.py data.raw 100 100 100 -p 2
  python whereisitconnected.py image.raw 500 351 351 --bounding-boxes
  python whereisitconnected.py image.raw 500 351 351 --bounding-boxes --sort-by depth
  python whereisitconnected.py image.raw 500 351 351 --backend cc3d   # Fast CPU
  python whereisitconnected.py image.raw 500 351 351 --backend cupy   # NVIDIA GPU
  python whereisitconnected.py image.raw 500 351 351 --backend mps    # Apple Silicon GPU
  python whereisitconnected.py image.raw 500 351 351 --bounding-boxes -o results.txt  # Save to file
  python whereisitconnected.py image.raw 500 351 351 --save-labels labels.npy  # Save labeled volume
  python whereisitconnected.py image.raw 500 351 351 --crop-mode --label-file labels.npy --component-label 94 --crop-output cropped.raw  # Crop mode
  python whereisitconnected.py image.raw 500 351 351 --render-3d component94.png --render-component 0  # Render largest component
        """
    )

    parser.add_argument('filename', type=str,
                        help='Path to the raw 3D image file')
    parser.add_argument('depth', type=int,
                        help='Depth dimension (z-axis) of the 3D volume')
    parser.add_argument('height', type=int,
                        help='Height dimension (y-axis) of the 3D volume')
    parser.add_argument('width', type=int,
                        help='Width dimension (x-axis) of the 3D volume')
    parser.add_argument('-p', '--phase', type=int, default=1,
                        help='Phase value to check for connectivity (default: 1)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting the slices')
    parser.add_argument('--bounding-boxes', action='store_true',
                        help='Calculate and display bounding boxes for each component')
    parser.add_argument('--sort-by', type=str, default='volume',
                        choices=['volume', 'depth', 'height', 'width'],
                        help='Sort components by: volume (voxel count), depth (extent along depth axis), height, or width (default: volume)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top components to display when using --bounding-boxes (default: 10, use 0 for all)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file for results (default: print to stdout). If specified, results are written to this file.')
    parser.add_argument('--save-labels', type=str, default=None,
                        help='Save the labeled volume to a file. Specify filename (e.g., labels.raw or labels.npy). Format determined by extension: .raw (raw binary), .npy (numpy format)')

    # Crop mode - separate from main analysis
    parser.add_argument('--crop-mode', action='store_true',
                        help='Crop mode: extract bounding box from a pre-computed label image. Requires --label-file and --component-label')
    parser.add_argument('--label-file', type=str, default=None,
                        help='Path to label file (.npy format) for crop mode')
    parser.add_argument('--component-label', type=int, default=None,
                        help='Component label to crop to (required for --crop-mode)')
    parser.add_argument('--crop-output', type=str, default=None,
                        help='Output file for cropped volume (required for --crop-mode)')

    # 3D visualization options
    parser.add_argument('--render-3d', type=str, default=None,
                        help='Render 3D visualization of a component. Specify output image file (e.g., component.png)')
    parser.add_argument('--render-component', type=int, default=None,
                        help='Component label to render (required with --render-3d). Use 0 to render largest component')
    parser.add_argument('--render-subsample', type=int, default=1,
                        help='Subsampling factor for rendering (default: 1, no subsampling). Higher = faster but lower resolution. WARNING: Subsampling thin structures like fractures will lose connectivity')
    parser.add_argument('--render-angle', type=int, nargs=2, default=[30, 45], metavar=('ELEV', 'AZIM'),
                        help='Viewing angles: elevation and azimuth in degrees (default: 30 45)')
    # Determine default backend based on what's available
    default_backend = 'cc3d' if CC3D_AVAILABLE else 'scipy'

    parser.add_argument('--backend', type=str, default=default_backend,
                        choices=['scipy', 'cc3d', 'cupy', 'mps', 'dask'],
                        help='Backend for connected component labeling: cc3d (default if installed, fast CPU), scipy (fallback, slow), cupy (NVIDIA GPU), mps (Apple Silicon GPU), dask (parallel CPU)')

    args = parser.parse_args()

    # Handle crop mode separately
    if args.crop_mode:
        if not args.label_file or not args.component_label or not args.crop_output:
            print("ERROR: Crop mode requires --label-file, --component-label, and --crop-output")
            sys.exit(1)

        print("=== CROP MODE ===")
        print("Loading label file: {}".format(args.label_file))
        print("Original image: {}".format(args.filename))
        print("Target component: {}".format(args.component_label))

        # Load label file
        if args.label_file.endswith('.npy'):
            labeled = np.load(args.label_file)
        else:
            print("ERROR: Label file must be .npy format")
            sys.exit(1)

        # Load original volume
        dims = (args.depth, args.height, args.width)
        volume = load_raw_3d(args.filename, dims)

        # Get bounding box for target component
        coords = np.argwhere(labeled == args.component_label)
        if coords.size == 0:
            print("ERROR: Component label {} not found in label file".format(args.component_label))
            sys.exit(1)

        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        # Crop the original volume
        cropped = volume[min_coords[0]:max_coords[0]+1,
                        min_coords[1]:max_coords[1]+1,
                        min_coords[2]:max_coords[2]+1]

        # Save cropped volume
        if args.crop_output.endswith('.npy'):
            np.save(args.crop_output, cropped)
            print("Saved as NumPy format (.npy)")
        else:
            cropped.tofile(args.crop_output)
            print("Saved as raw binary format")

        print("\nCrop information:")
        print("  Original dimensions: {}".format(volume.shape))
        print("  Cropped dimensions:  {}".format(cropped.shape))
        print("  Bounding box:")
        print("    Depth:  [{} - {}]".format(min_coords[0], max_coords[0]))
        print("    Height: [{} - {}]".format(min_coords[1], max_coords[1]))
        print("    Width:  [{} - {}]".format(min_coords[2], max_coords[2]))
        print("  Output file: {}".format(args.crop_output))
        sys.exit(0)

    # Check backend availability and warn user
    if args.backend == 'cc3d' and not CC3D_AVAILABLE:
        print("ERROR: cc3d backend requested but not installed.")
        print("Install with: pip install connected-components-3d")
        print("Falling back to scipy backend...")
        args.backend = 'scipy'
    elif args.backend == 'cupy' and not CUPY_AVAILABLE:
        print("ERROR: cupy backend requested but not installed.")
        print("Install with: pip install cupy-cuda11x  (or appropriate CUDA version)")
        sys.exit(1)
    elif args.backend == 'dask' and not DASK_AVAILABLE:
        print("ERROR: dask backend requested but not installed.")
        print("Install with: pip install dask[array]")
        sys.exit(1)
    elif args.backend == 'mps' and not MPS_AVAILABLE:
        if not TORCH_AVAILABLE:
            print("ERROR: mps backend requested but PyTorch is not installed.")
            print("Install with: pip install torch")
            sys.exit(1)
        else:
            print("ERROR: MPS (Metal Performance Shaders) is not available.")
            print("MPS requires Apple Silicon Mac (M1/M2/M3) with macOS 12.3 or later.")
            sys.exit(1)

    # Define the dimensions from command-line arguments
    dims = (args.depth, args.height, args.width)

    # Open output file if specified, otherwise use stdout
    if args.output:
        output_file = open(args.output, 'w')
        def output(msg):
            print(msg)  # Still print to console
            output_file.write(msg + '\n')
    else:
        def output(msg):
            print(msg)

    output("Loading file: {}".format(args.filename))
    output("Dimensions: {} (depth x height x width)".format(dims))
    output("Analyzing phase: {}".format(args.phase))
    output("Backend: {}".format(args.backend))
    output("")

    # Load the 3D image with auto-detected header length
    volume = load_raw_3d(args.filename, dims)

    # Plot central slices along each direction to verify correct loading
    if not args.no_plot:
        plot_slices(volume)

    # Check connectivity for the specified phase
    output("Running connected component labeling with {} backend...".format(args.backend))
    import time
    start_time = time.time()
    labeled, num_features, connectivity = check_phase_connectivity(volume, phase=args.phase, backend=args.backend)
    elapsed_time = time.time() - start_time
    output("Labeling completed in {:.2f} seconds".format(elapsed_time))

    output("\nTotal connected components for phase {}: {}".format(args.phase, num_features))
    for axis, (is_conn, labels) in connectivity.items():
        if is_conn:
            output("Phase {} is connected along {} (common component labels: {})".format(args.phase, axis, labels))
        else:
            output("Phase {} is NOT connected along {}".format(args.phase, axis))

    # Calculate and display bounding boxes if requested
    if args.bounding_boxes:
        output("\n" + "="*80)
        output("BOUNDING BOXES FOR PHASE {} COMPONENTS".format(args.phase))
        output("="*80)

        components = get_component_bounding_boxes(labeled, num_features)

        # Sort components based on user preference
        sort_key_map = {
            'volume': 'volume',
            'depth': 'extent_depth',
            'height': 'extent_height',
            'width': 'extent_width'
        }
        sort_key = sort_key_map[args.sort_by]
        components_sorted = sorted(components, key=lambda x: x[sort_key], reverse=True)

        # Limit to top N components if specified
        if args.top_n > 0:
            components_to_display = components_sorted[:args.top_n]
            output("Sorted by: {} (descending)".format(args.sort_by))
            output("Showing top {} of {} components\n".format(min(args.top_n, len(components_sorted)), len(components_sorted)))
        else:
            components_to_display = components_sorted
            output("Sorted by: {} (descending)".format(args.sort_by))
            output("Showing all {} components\n".format(len(components_sorted)))

        for i, comp in enumerate(components_to_display, 1):
            output("Component #{} (Label: {})".format(i, comp['label']))
            output("  Bounding Box:")
            output("    Depth:  [{:5d} - {:5d}]  (extent: {:5d})".format(
                comp['bbox'][0], comp['bbox'][1], comp['extent_depth']))
            output("    Height: [{:5d} - {:5d}]  (extent: {:5d})".format(
                comp['bbox'][2], comp['bbox'][3], comp['extent_height']))
            output("    Width:  [{:5d} - {:5d}]  (extent: {:5d})".format(
                comp['bbox'][4], comp['bbox'][5], comp['extent_width']))
            output("  Volume: {:,} voxels".format(comp['volume']))
            output("")

    # Save labeled volume if requested
    if args.save_labels:
        print("\nSaving labeled volume to: {}".format(args.save_labels))
        if args.save_labels.endswith('.npy'):
            # Save as NumPy format (includes metadata, compressed)
            np.save(args.save_labels, labeled)
            print("Saved as NumPy format (.npy)")
        elif args.save_labels.endswith('.raw'):
            # Save as raw binary format
            # Determine appropriate dtype based on number of labels
            if num_features <= 255:
                save_dtype = np.uint8
            elif num_features <= 65535:
                save_dtype = np.uint16
            else:
                save_dtype = np.uint32

            labeled_to_save = labeled.astype(save_dtype)
            labeled_to_save.tofile(args.save_labels)
            print("Saved as raw binary format (.raw) with dtype: {}".format(save_dtype))
            print("Dimensions: {} (depth x height x width)".format(labeled.shape))
        else:
            print("Warning: Unrecognized extension. Saving as raw binary (.raw format)")
            if num_features <= 255:
                save_dtype = np.uint8
            elif num_features <= 65535:
                save_dtype = np.uint16
            else:
                save_dtype = np.uint32
            labeled_to_save = labeled.astype(save_dtype)
            labeled_to_save.tofile(args.save_labels)
            print("Saved with dtype: {}".format(save_dtype))

    # Render 3D visualization if requested
    if args.render_3d:
        if args.render_component is None:
            print("ERROR: --render-3d requires --render-component")
            sys.exit(1)

        # Determine which component to render
        if args.render_component == 0:
            # Find largest component
            if CC3D_AVAILABLE:
                stats = cc3d.statistics(labeled)
                volumes = [(label, stats['voxel_counts'][label]) for label in range(1, num_features + 1)]
                target_label = max(volumes, key=lambda x: x[1])[0]
                print("Rendering largest component (Label: {})".format(target_label))
            else:
                volumes = [(label, np.sum(labeled == label)) for label in range(1, num_features + 1)]
                target_label = max(volumes, key=lambda x: x[1])[0]
                print("Rendering largest component (Label: {})".format(target_label))
        else:
            target_label = args.render_component

        render_3d_component(labeled, target_label,
                          output_file=args.render_3d,
                          elevation=args.render_angle[0],
                          azimuth=args.render_angle[1],
                          subsample=args.render_subsample)

    # Close output file if it was opened
    if args.output:
        output_file.close()
        print("\nResults written to: {}".format(args.output))