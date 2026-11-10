Deskbot is an autonomous dorm desk cleaning robot. It uses a camera and reinforcement learning to navigate the desk, vacuum small crumbs, and push larger debris (like pens) into a designated corner without falling off the edge.

* `hardware/` - Physical hardware controllers (Raspberry Pi Pico 2 H) and communication messaging protocols.
* `shared/` - Global configuration files. All physical dimensions, constants, and shared variables live here.
* `sim/` - The machine learning environment. Includes the Gymnasium environment, Pymunk physics, training loop, and backup driving pattern.
* `vision/` - The camera pipeline. Includes OpenCV scripts for desk tag detection, object sorting, and inference on the Raspberry Pi 5.

* **Branching:** Do not push directly to `main`. Create a feature branch prefixed with your domain (e.g., `vision/tag-detection`, `sim/reward-tuning`).
* **Constants:** If you need a physical measurement or hardware limit, import it from `shared/config.py`. Do not hardcode numbers into your local scripts.