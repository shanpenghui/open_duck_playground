# Open Duck Playground

# Installation 

Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# Training

If you want to use the [imitation reward](https://la.disneyresearch.com/wp-content/uploads/BD_X_paper.pdf), you can generate reference motion with [this repo](https://github.com/apirrone/Open_Duck_reference_motion_generator)

Then copy `polynomial_coefficients.pkl` in `playground/<robot>/data/`

You'll also have to set `USE_IMITATION_REWARD=True` in it's `joystick.py` file

Run: 

```bash
uv run playground/<robot>/runner.py 
```

## Tensorboard

```bash
uv run tensorboard --logdir=<yourlogdir>
```

# Inference 

Infer mujoco

(for now this is specific to open_duck_mini_v2)

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py -o <path_to_.onnx> (-k)
python /home/duck/Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py --onnx_model_path /home/duck/Open_Duck_Mini_Runtime/BEST_WALK_ONNX_2.onnx --duck_config_path /home/duck/duck_config.json --action_scale 0.2
python /home/duck/Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py --onnx_model_path /home/duck/Open_Duck_Mini_Runtime/2025_06_30_225843_151388160.onnx --duck_config_path /home/duck/duck_config.json --action_scale 0.2

```

# Documentation

## Project structure : 

```
.
├── pyproject.toml
├── README.md
├── playground
│   ├── common
│   │   ├── export_onnx.py
│   │   ├── onnx_infer.py
│   │   ├── poly_reference_motion.py
│   │   ├── randomize.py
│   │   ├── rewards.py
│   │   └── runner.py
│   ├── open_duck_mini_v2
│   │   ├── base.py
│   │   ├── data
│   │   │   └── polynomial_coefficients.pkl
│   │   ├── joystick.py
│   │   ├── mujoco_infer.py
│   │   ├── constants.py
│   │   ├── runner.py
│   │   └── xmls
│   │       ├── assets
│   │       ├── open_duck_mini_v2_no_head.xml
│   │       ├── open_duck_mini_v2.xml
│   │       ├── scene_mjx_flat_terrain.xml
│   │       ├── scene_mjx_rough_terrain.xml
│   │       └── scene.xml
```

## Adding a new robot

Create a new directory in `playground` named after `<your robot>`. You can copy the `open_duck_mini_v2` directory as a starting point.

You will need to:
- Edit `base.py`: Mainly renaming stuff to match you robot's name
- Edit `constants.py`: specify the names of some important geoms, sensors etc
  - In your `mjcf`, you'll probably have to add some sites, name some bodies/geoms and add the sensors. Look at how we did it for `open_duck_mini_v2`
- Add your `mjcf` assets in `xmls`. 
- Edit `joystick.py` : to choose the rewards you are interested in
  - Note: for now there is still some hard coded values etc. We'll improve things on the way
- Edit `runner.py`



# Notes

Inspired from https://github.com/kscalelabs/mujoco_playground


## Current win
- train for 300 000 000 steps
- train with backlash (takes longer)
- Flat terrain
- Infer on the robot with kp = 22 instead of 32

## TODO
- Understand why head does not move very much
- low pass filter ? 37.5 hz ? 