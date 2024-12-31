#!/bin/bash

# Set base directory for the project
BASE_DIR="$(pwd)/DeepCompress"
echo "Organizing project at: $BASE_DIR"

# Create directories for the new structure
mkdir -p "$BASE_DIR/data"
mkdir -p "$BASE_DIR/models"
mkdir -p "$BASE_DIR/utils"
mkdir -p "$BASE_DIR/experiments"
mkdir -p "$BASE_DIR/coding"
mkdir -p "$BASE_DIR/rendering"
mkdir -p "$BASE_DIR/tests"

# Move files into the appropriate directories
mv "$BASE_DIR/ds_mesh_to_pc.py" "$BASE_DIR/data/" 2>/dev/null
mv "$BASE_DIR/map_color.py" "$BASE_DIR/data/" 2>/dev/null
mv "$BASE_DIR/pc_metric.py" "$BASE_DIR/data/" 2>/dev/null

mv "$BASE_DIR/model_opt.py" "$BASE_DIR/models/" 2>/dev/null
mv "$BASE_DIR/model_transforms.py" "$BASE_DIR/models/" 2>/dev/null
mv "$BASE_DIR/patch_gaussian_conditional.py" "$BASE_DIR/models/" 2>/dev/null

mv "$BASE_DIR/compress_octree.py" "$BASE_DIR/utils/" 2>/dev/null
mv "$BASE_DIR/parallel_process.py" "$BASE_DIR/utils/" 2>/dev/null
mv "$BASE_DIR/colorbar.py" "$BASE_DIR/utils/" 2>/dev/null
mv "$BASE_DIR/pc_error_wrapper.py" "$BASE_DIR/utils/" 2>/dev/null
mv "$BASE_DIR/gpcc_wrapper.py" "$BASE_DIR/utils/" 2>/dev/null

mv "$BASE_DIR/octree_coding.py" "$BASE_DIR/coding/" 2>/dev/null
mv "$BASE_DIR/tmc3" "$BASE_DIR/coding/" 2>/dev/null

mv "$BASE_DIR/ut_run_render.py" "$BASE_DIR/rendering/" 2>/dev/null

mv "$BASE_DIR/ev_run_experiment.py" "$BASE_DIR/experiments/" 2>/dev/null
mv "$BASE_DIR/ev_experiment.yml" "$BASE_DIR/experiments/" 2>/dev/null

# Copy over README and requirements file
mv "$BASE_DIR/README.md" "$BASE_DIR/" 2>/dev/null
mv "$BASE_DIR/requirements.txt" "$BASE_DIR/" 2>/dev/null

# Create a placeholder for the README if not provided
if [ ! -f "$BASE_DIR/README.md" ]; then
    cat > "$BASE_DIR/README.md" <<EOL
# DeepCompress

This project has been restructured for better modularity and maintainability. Below is the new structure:

## Directory Structure
- `/data`: Data loading and preprocessing modules.
- `/models`: Model definitions and optimization.
- `/utils`: Utility functions and helper scripts.
- `/experiments`: Experiment management and configuration.
- `/coding`: Octree coding and related logic.
- `/rendering`: Rendering utilities and scripts.

## Next Steps
- Refactor code to use the new structure.
- Update imports to reflect new file locations.
- Implement unit tests in `/tests`.

EOL
fi

echo "Project restructured successfully."
echo "Next steps: Update imports and begin refactoring files. Refer to README.md for details."
