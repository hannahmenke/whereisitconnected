import os
import argparse
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

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

def check_phase_connectivity(volume, phase=1):
    """
    Checks connectivity for a given phase in a 3D volume using voxel face connectivity.
    Determines if any connected component spans from one face to the opposite face along each axis.
    
    Parameters:
        volume (np.ndarray): The 3D image array.
        phase (int): The voxel value representing the phase of interest.
    
    Returns:
        labeled (np.ndarray): Array of labeled connected components.
        num_features (int): Number of connected components.
        connectivity (dict): For each axis, a tuple (is_connected, labels)
                             where is_connected is True if a component touches both boundaries.
    """
    mask = (volume == phase)
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze phase connectivity in 3D volumetric imaging data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python isitconnected.py image.raw 500 351 351 --phase 1
  python isitconnected.py data.raw 100 100 100 -p 2
  python isitconnected.py image.raw 500 351 351 --bounding-boxes
  python isitconnected.py image.raw 500 351 351 --bounding-boxes --sort-by depth
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

    args = parser.parse_args()

    # Define the dimensions from command-line arguments
    dims = (args.depth, args.height, args.width)

    print("Loading file: {}".format(args.filename))
    print("Dimensions: {} (depth x height x width)".format(dims))
    print("Analyzing phase: {}".format(args.phase))
    print()

    # Load the 3D image with auto-detected header length
    volume = load_raw_3d(args.filename, dims)

    # Plot central slices along each direction to verify correct loading
    if not args.no_plot:
        plot_slices(volume)

    # Check connectivity for the specified phase
    labeled, num_features, connectivity = check_phase_connectivity(volume, phase=args.phase)

    print("\nTotal connected components for phase {}: {}".format(args.phase, num_features))
    for axis, (is_conn, labels) in connectivity.items():
        if is_conn:
            print("Phase {} is connected along {} (common component labels: {})".format(args.phase, axis, labels))
        else:
            print("Phase {} is NOT connected along {}".format(args.phase, axis))

    # Calculate and display bounding boxes if requested
    if args.bounding_boxes:
        print("\n" + "="*80)
        print("BOUNDING BOXES FOR PHASE {} COMPONENTS".format(args.phase))
        print("="*80)

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

        print("Sorted by: {} (descending)\n".format(args.sort_by))

        for i, comp in enumerate(components_sorted, 1):
            print("Component #{} (Label: {})".format(i, comp['label']))
            print("  Bounding Box:")
            print("    Depth:  [{:5d} - {:5d}]  (extent: {:5d})".format(
                comp['bbox'][0], comp['bbox'][1], comp['extent_depth']))
            print("    Height: [{:5d} - {:5d}]  (extent: {:5d})".format(
                comp['bbox'][2], comp['bbox'][3], comp['extent_height']))
            print("    Width:  [{:5d} - {:5d}]  (extent: {:5d})".format(
                comp['bbox'][4], comp['bbox'][5], comp['extent_width']))
            print("  Volume: {:,} voxels".format(comp['volume']))
            print()