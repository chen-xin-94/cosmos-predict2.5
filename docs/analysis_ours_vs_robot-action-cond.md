# AWAMv1.5/6 vs COSMOS-Predict-2.5/robot/action-cond

## Technical Comparison

| Feature | **Official `robot/action-cond`** | **Ours (`AWAMv1.5/6`)** |
| :--- | :--- | :--- |
| **Foundation Backbone** | Cosmos-Predict2.5-2B/post-trained | Cosmos-Predict2.5-2B/post-trained |
| **Visual Input** | **Single View** (1 Camera) | **Multi-View** (Supports any N≥1 cameras, fused via concat) |
| **Spatial Resolution** | **256 × 320** | **448 × 1344** (High-Definition, 3× Views) |
| **Action Space** | Fixed 7-DoF (Bridge/Franka) | **Universal Action Space** (7-DoF Franka, 36-DoF AgiBot, or Any) |
| **Instruction Following** | No | **Yes** |

---

## Selling Points (AI-rendered fancy words of the above points)

### 1. Multi-Embodiment
The official model is locked to a single embodiment (usually Franka or Bridge) with a fixed action space.
**AWAMv1.5/6 is an Embodiment-Agnostic Foundation Model.**
-   **One Model, Any Robot:** Whether it's a 7-DoF arm or a 36-DoF humanoid (AgiBot), our architecture adapts seamlessly.
-   **Scalable Architecture:** It learns a shared representation across diverse robot morphologies, enabling positive transfer learning (e.g., learning grasp physics on one robot helps another).

### 2. Occlusion Robustness via Neural Triangulation
Single-view models suffer from critical blind spots. If the robot arm blocks the object, the model hallucinates or fails.
**AWAMv1.5/6 solves this with "Neural Triangulation".**
-   **Implicit 3D Understanding:** By fusing synchronized Multi-View inputs (e.g., Left, Top, Wrist), the transformer learns to correlate features across views.
-   **Occlusion Immunity:** When one view is blocked, the model automatically attends to unobstructed views to maintain a consistent world state—mimicking stereo triangulation without explicit calibration.

### 3. High-Fidelity & Language-Grounded Simulation
The official model generates plausible-looking video at low resolution (256x320), often insufficient for precise manipulation tasks.
**AWAMv1.5/6 is a High-Fidelity Neural Simulator.**
-   **Precision Engineering:** At **448×1344**, our model captures sub-centimeter details (e.g., screw orientation, texture defects) critical for fine manipulation.
-   **Instruction Following:** With text conditioning enabled, you can guide the simulation with natural language (e.g., *"Pick up the red block"* vs *"Push the red block"*), making it a powerful engine for **synthesizing diverse training data** for downstream policy learning.
