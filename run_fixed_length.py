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

    # Fixed-length mode splits along the original centerline; smoothing is not supported here.
    if cfg.smoothening.get("perform_smoothing", False):
        logging.warning("smoothening.perform_smoothing is ignored in fixed-length mode.")

    steps = [
        "src/fixed_length/assign_ID_spacing.py",
        "src/fixed_length/split_to_plots_spacing.py",
        "src/fixed_length/split_to_sides_spacing.py",
        "src/fixed_length/split_to_subplots_spacing.py"
    ]

    for step in steps:
        run_step(step, config_dir)

    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()
