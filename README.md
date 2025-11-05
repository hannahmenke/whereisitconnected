# IsItConnected - 3D Phase Connectivity Analyzer

A Python tool for analyzing phase connectivity in 3D volumetric imaging data, particularly useful for nanoCT scans and materials science applications.

## Overview

This script determines whether a specific phase (represented by voxel values) forms connected pathways through a 3D volume. It's commonly used to analyze:
- Porosity and permeability in porous materials
- Phase connectivity in composite materials
- Network structures in 3D imaging data

## Features

- **Automatic Header Detection**: Loads raw binary 3D image files with automatic header byte detection
- **6-Connected Connectivity Analysis**: Uses face connectivity (not edge/corner) for accurate pathway detection
- **Multi-Axis Analysis**: Checks connectivity along all three spatial axes
- **Visualization**: Displays central slices in axial, coronal, and sagittal views
- **Component Labeling**: Labels all connected components and reports statistics

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the script with command-line arguments:

```bash
python isitconnected.py <filename> <depth> <height> <width> [options]
```

**Required Arguments:**
- `filename`: Path to the raw 3D image file
- `depth`: Depth dimension (z-axis) of the 3D volume
- `height`: Height dimension (y-axis) of the 3D volume
- `width`: Width dimension (x-axis) of the 3D volume

**Optional Arguments:**
- `-p, --phase`: Phase value to check for connectivity (default: 1)
- `--no-plot`: Skip plotting the slices (useful for batch processing)
- `--bounding-boxes`: Calculate and display bounding boxes for each component
- `--sort-by`: Sort components by `volume`, `depth`, `height`, or `width` (default: volume)
- `-h, --help`: Show help message

**Examples:**

```bash
# Basic usage with default phase (1)
python isitconnected.py image.raw 500 351 351

# Specify a different phase to analyze
python isitconnected.py image.raw 500 351 351 --phase 2

# Skip plotting for faster processing
python isitconnected.py image.raw 500 351 351 -p 1 --no-plot

# Get bounding boxes for each component, sorted by volume
python isitconnected.py image.raw 500 351 351 --bounding-boxes

# Get bounding boxes sorted by extent along depth axis
python isitconnected.py image.raw 500 351 351 --bounding-boxes --sort-by depth

# Analyze phase 2, get bounding boxes sorted by width, skip plotting
python isitconnected.py image.raw 500 351 351 -p 2 --bounding-boxes --sort-by width --no-plot

# Show help
python isitconnected.py --help
```

### Function Reference

#### `load_raw_3d(filename, dims, dtype=np.uint8)`
Loads a raw binary 3D image file.

**Parameters:**
- `filename` (str): Path to the raw file
- `dims` (tuple): Dimensions as (depth, height, width)
- `dtype` (data-type): Data type of the image (default: np.uint8)

**Returns:** 3D numpy array

#### `check_phase_connectivity(volume, phase=1)`
Analyzes connectivity for a specific phase value.

**Parameters:**
- `volume` (np.ndarray): 3D image array
- `phase` (int): Voxel value representing the phase of interest

**Returns:**
- `labeled` (np.ndarray): Labeled connected components
- `num_features` (int): Number of connected components
- `connectivity` (dict): Connectivity information for each axis

#### `get_component_bounding_boxes(labeled, num_features)`
Calculates bounding boxes and statistics for each labeled component.

**Parameters:**
- `labeled` (np.ndarray): Labeled connected components
- `num_features` (int): Number of connected components

**Returns:** List of dictionaries containing:
- `label`: Component label
- `bbox`: Bounding box coordinates (min_depth, max_depth, min_height, max_height, min_width, max_width)
- `volume`: Number of voxels in the component
- `extent_depth`, `extent_height`, `extent_width`: Size along each axis

#### `plot_slices(volume)`
Visualizes the 3D volume with three orthogonal slice views.

**Parameters:**
- `volume` (np.ndarray): 3D image array

## Output

The script provides:
1. **Visual Output**: Three plots showing axial, coronal, and sagittal slices
2. **Console Output**:
   - File size and header information
   - Total number of connected components
   - Connectivity status for each axis with component labels

### Example Output
```bash
$ python isitconnected.py Oxford-28_nanoCT_pore_labels.raw 500 351 351 --phase 1

Loading file: Oxford-28_nanoCT_pore_labels.raw
Dimensions: (500, 351, 351) (depth x height x width)
Analyzing phase: 1

File size (bytes): 61626500
Expected size (bytes): 61625500
Automatically detected header bytes: 1000

Total connected components for phase 1: 42
Phase 1 is connected along axis0 (depth) (common component labels: [5])
Phase 1 is NOT connected along axis1 (height)
Phase 1 is connected along axis2 (width) (common component labels: [5])
```

### Bounding Box Output Example
```bash
$ python isitconnected.py image.raw 500 351 351 --bounding-boxes --sort-by volume

... (connectivity output) ...

================================================================================
BOUNDING BOXES FOR PHASE 1 COMPONENTS
================================================================================
Sorted by: volume (descending)

Component #1 (Label: 5)
  Bounding Box:
    Depth:  [    0 -   499]  (extent:   500)
    Height: [   10 -   340]  (extent:   331)
    Width:  [   12 -   338]  (extent:   327)
  Volume: 1,234,567 voxels

Component #2 (Label: 3)
  Bounding Box:
    Depth:  [   45 -   120]  (extent:    76)
    Height: [  100 -   250]  (extent:   151)
    Width:  [  150 -   200]  (extent:    51)
  Volume: 45,678 voxels

...
```

## Technical Details

### Connectivity Definition
The script uses **6-connected (face) connectivity**, meaning voxels are considered connected only if they share a face, not just an edge or corner. This is the most conservative definition of connectivity.

### Axis Convention
Note that axis ordering may differ from other imaging software (e.g., Fiji, Avizo):
- axis0 = depth (z in some software)
- axis1 = height (y)
- axis2 = width (x in some software)

## Use Cases

- **Pore Network Analysis**: Determine if pores form connected pathways for fluid flow
- **Composite Materials**: Analyze phase continuity in multi-phase materials
- **Percolation Studies**: Identify percolating clusters in 3D structures
- **Quality Control**: Verify connectivity in manufactured porous materials
- **Component Analysis**: Extract individual components by bounding box for detailed study
- **Size Distribution**: Analyze component size distributions by volume or spatial extent
- **Region of Interest**: Identify largest components or components extending along specific directions

## License

This script is provided as-is for research and educational purposes.

## Notes

- Ensure your raw file dimensions are correct to avoid loading errors
- The script automatically handles files with header bytes
- Large volumes may require significant memory
