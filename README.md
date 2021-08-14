# tf-renderer
Simple 3D renderer implemented only with TensorFlow operations.
See test_on_colab.ipynb for usage.

## Requirements
- TensorFlow
- TensorFlow addons (for bilinear interpolation)
- TensorFlow graphics (for BRDF calculation, perspective projection, etc.)

## Notes
- Assigning triangles to pixels costs O(NM) memory space where N is the number of triangles and M is the number of pixels,
so you will need to properly group faces of the mesh (and set face_groups field of the mesh) to maintain the performance.
- The triangle-pixel assignment cannot be differentiated,
whereas other processes (e.g. smooth shading, texture mapping) can be differentiated using tf.GradientTape (maybe, not tested).
- Currently camera position is fixed to the origin and direction is fixed to (0,0,-1). Translate the mesh's vertices to that direction.
