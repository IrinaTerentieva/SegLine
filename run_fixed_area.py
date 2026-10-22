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

@hydra.main(config_path="src/config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

    config_dir = "src/config"  # run from the repository root

    if cfg.smoothing.get("perform_smoothing", True):
        steps = [
            "src/fixed_area/assign_id.py",
            "src/fixed_area/clean_centerlines.py",
            "src/common/smooth_centerline.py",
            "src/fixed_area/split_to_plots.py",
            "src/fixed_area/split_to_sides.py",
            "src/fixed_area/split_to_subplots.py"
        ]
    else:
        steps = [
            "src/fixed_area/assign_id.py",
            "src/fixed_area/clean_centerlines.py",
            "src/fixed_area/split_to_plots.py",
            "src/fixed_area/split_to_sides.py",
            "src/fixed_area/split_to_subplots.py"
        ]

    for step in steps:
        run_step(step, config_dir)

    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()
