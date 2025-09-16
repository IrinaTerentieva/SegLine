import subprocess
import sys
import logging
import os
from omegaconf import DictConfig, OmegaConf
import hydra

def run_step(script_path, config_dir):
    try:
        logging.info(f"Running {script_path} with config directory {config_dir} ...")
        subprocess.run([sys.executable, script_path, "--config-dir", config_dir], check=True)
        logging.info(f"{script_path} completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error in {script_path}: {e}")
        sys.exit(e.returncode)

@hydra.main(config_path="/home/irina/SegLine/src/config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

    # Adjust these paths if needed. Here we assume:
    # - run_pipeline.py is in /home/irina/SegLine/
    # - your utils scripts are in /home/irina/SegLine/src/utils/
    # - your config is in /home/irina/SegLine/src/config/
    # So from a utils script, the config directory is: ../config
    config_dir = "src/config"  # relative path from the utils directory

    if cfg.smoothening.get("perform_smoothing", True):
        steps = [
            "src/utils/assign_id.py",
            "src/utils/smooth_centerline.py",
            "src/utils/split_to_plots.py",
            "src/utils/split_to_sides.py",
            "src/utils/split_to_subplots.py"
        ]
    else:
        steps = [
            "src/utils/assign_id.py",
            "src/utils/split_to_plots.py",
            "src/utils/split_to_sides.py",
            "src/utils/split_to_subplots.py"
        ]

    for step in steps:
        run_step(step, config_dir)

    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()
