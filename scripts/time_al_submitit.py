import hydra
from omegaconf import DictConfig
import time
import traceback


@hydra.main(version_base="1.3.2", config_path="../config", config_name="main_time")
def main(cfg: DictConfig):
    try:
        from scripts.run_al_time import main as al_main
        al_main(cfg)
    except Exception as ex:
        print(ex, flush=True)
        print("Traceback:")
        traceback.print_tb(ex.__traceback__)
        print("AL Crashed")
        time.sleep(5)
        raise ex


if __name__ == "__main__":
    main()
    print("Done with time measurement for active learning experiment.", flush=True)
