import csv
from codecarbon import EmissionsTracker
import subprocess
import logging
import pandas as pd
import os


def generate_summary(test_case_path, config):
	# Path to the detailed CodeCarbon output data
	data_path = os.path.join(config['output_path'], 'emissions.csv')
	
	# Ensure the output directory exists
	os.makedirs(config['output_path'], exist_ok=True)
	
	# Read the detailed data using Pandas
	data = pd.read_csv(data_path)
	
	# Process the data to extract key metrics
	total_emissions = data['emissions'].sum()
	average_emissions_per_run = data['emissions'].mean()
	total_duration = data['duration'].sum()
	average_duration_per_run = data['duration'].mean()
	total_energy_consumed = data[['cpu_energy', 'gpu_energy', 'ram_energy']].sum().sum()
	average_energy_per_run = total_energy_consumed / len(data)
	num_runs = len(data)
	emissions_rate_stats = data['emissions_rate'].agg(['mean', 'min', 'max'])
	cpu_power_stats = data['cpu_power'].agg(['mean', 'min', 'max'])
	gpu_power_stats = data['gpu_power'].agg(['mean', 'min', 'max'])
	start_date = data['timestamp'].min()
	end_date = data['timestamp'].max()
	
	# # Write the summary
	# summary_path = os.path.join(config['output_path'], f"{test_case_path}_summary.txt")
	# os.makedirs(os.path.dirname(summary_path), exist_ok=True)
	
	# Define CSV file path
	summary_csv_path = os.path.join(config['output_path'], "codecarbon_summary.csv")
	
	# Check if the file exists to decide whether to write headers for append mode
	file_exists = os.path.isfile(summary_csv_path)
	
	# Define the headers and the row data
	row_data = [
		total_emissions, average_emissions_per_run, total_duration,
		average_duration_per_run, total_energy_consumed, average_energy_per_run,
		num_runs,
		emissions_rate_stats['mean'], emissions_rate_stats['min'], emissions_rate_stats['max'],
		cpu_power_stats['mean'], cpu_power_stats['min'], cpu_power_stats['max'],
		gpu_power_stats['mean'], gpu_power_stats['min'], gpu_power_stats['max'],
		f"{start_date} to {end_date}"
	]
	
	# Corresponding headers need to be updated as well
	headers = [
		"Total CO2 Emissions (kg)", "Average Emissions per Run (kg)",
		"Total Duration (sec)", "Average Duration per Run (sec)",
		"Total Energy Consumed (units)", "Average Energy Consumption per Run (units)",
		"Number of Runs",
		"Emissions Rate Mean", "Emissions Rate Min", "Emissions Rate Max",
		"CPU Power Mean", "CPU Power Min", "CPU Power Max",
		"GPU Power Mean", "GPU Power Min", "GPU Power Max",
		"Data Time Frame"
	]
	
	# Write to the CSV file
	with open(summary_csv_path, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(headers)
		
		# Write the header only if the file is new
		# if not file_exists:
		#     writer.writerow(headers)
		
		# Write the data
		writer.writerow(row_data)


def run_test(test_case_path, config):
	tracker = EmissionsTracker(project_name="second_project", output_dir=config["output_path"])
	tracker.start()
	try:
		# Run the test case script
		subprocess.run(["python", test_case_path], check=True)
		generate_summary(test_case_path, config)
	except subprocess.CalledProcessError as e:
		logging.error(f"Error running test case {test_case_path}: {e}")
	finally:
		emissions: float = tracker.stop()
		print(f"Total emissions: {emissions} kg")
		logging.info(f"Completed energy measurement for {test_case_path}")
