### Task: Implement Automated 2D & 3D Loss Landscape Visualizations

**Objective:** Develop a visualization module that captures the loss landscape of constrained neural networks (Soft Penalty, CVXPY layers, and Theseus layers) to diagnose training dynamics and "chaotic" vs. "convex" behaviors.

**General context:** Use the `loss-landscape` submodule as a reference for building these plots, either copying some code from it or directly reusing it.

**Required Results:**
1. **Filter-Wise Normalization:** All visualizations must use the filter-wise normalization technique from the `loss-landscape` submodule to ensure that sharpness/flatness comparisons across different architectures and constraint types are mathematically meaningful and scale-invariant.
2. **Automated Checkpointing:** Integrate a trigger into the training loop (via Hydra configuration) to generate landscapes every `N` epochs (default: 50).
3. **Dual-Format Output:** For every trigger, generate both:
    * **2D Contour Plots:** High-resolution maps to identify eccentricity, non-convexities, and the "Edge of Stability".
    * **3D Surface Plots:** Visualizations to capture the transition from "nearly convex attractors" to "chaotic" landscapes.
4. **Consistency:** Ensure the same random direction vectors ($\delta, \eta$) and evaluation grid are used for both the 2D and 3D versions of a specific epoch's snapshot.
5. **Experiment Storage:** Save all generated images (e.g., `landscape_2d_epoch_50.png`, `landscape_3d_epoch_100.png`) directly into the project's `results/_experiment_number_/landscapes/` directory alongside existing experiment metrics.

**Technical Constraints:**
* Use the logic found in the `loss-landscape` submodule as the ground-truth for normalization math.
* Ensure the visualization handles the forward pass of differentiable layers (`cvxpy` and `Theseus`) correctly during grid evaluation.
* Maintain constant Batch Normalization statistics during the landscape scan to prevent artifact noise.

Ask additional questions if you encounter amibiguity situations where you want me to confirm what I want exactly.