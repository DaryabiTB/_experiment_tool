import logging
import subprocess

import psutil, time


from energat.tracer import EnergyTracer


def run_test(test_case_path, config):
	# Initialize the energy tracker
	print("energat")
	try:
		# Run the test case script
		with EnergyTracer(psutil.Process().pid, output=r'output/Energat/xyz_energy') as tracer:
			subprocess.run(["python", test_case_path], check=True)
	except subprocess.CalledProcessError as e:
		logging.error(f"Error running test case {test_case_path}: {e}")
	finally:
		logging.info(f"Completed energy measurement for {test_case_path}")
