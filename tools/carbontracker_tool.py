import subprocess
import logging
from carbontracker import tracker

def run_test(test_case_path, config):
    # Configuration for CarbonTracker
    carbon_tracker = tracker.CarbonTracker(epochs=1, 
                                           log_dir=config["output_path"], 
                                           monitor_epochs=-1, 
                                           verbose=2)

    carbon_tracker.start()
    try:
        # Run the test case script
        subprocess.run(["python", test_case_path], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running test case {test_case_path} with CarbonTracker: {e}")
    finally:
        carbon_tracker.stop()
        logging.info(f"Completed CarbonTracker measurement for {test_case_path}")
