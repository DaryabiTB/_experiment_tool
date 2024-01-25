import csv
import os
import subprocess
import logging
from eco2ai import Tracker
import pandas as pd


def generate_summary_eco2ai(tool_config):
	data_path = os.path.join(tool_config['output_path'], 'emission.csv')
	data = pd.read_csv(data_path)
	
	# data columns are :
	# ['id', 'project_name', 'experiment_description',
	# 'epoch', 'start_time', 'duration(s)', 'power_consumption(kWh)',
	# 'CO2_emissions(kg)', 'CPU_name', 'GPU_name', 'OS', 'region/country', 'cost'],
	
	# Process the data to extract key metrics
	total_co2_emissions = data['CO2_emissions(kg)'].sum()
	average_co2_emissions_per_run = data['CO2_emissions(kg)'].mean()
	total_duration = data['duration(s)'].sum()
	average_duration_per_run = data['duration(s)'].mean()
	total_power_consumption = data['power_consumption(kWh)'].sum()
	total_cost = data['cost'].sum()
	num_runs = len(data)
	
	# Define CSV file path for the summary
	summary_csv_path = os.path.join(tool_config['output_path'], "eco2ai_summary.csv")
	
	# Check if the file exists to decide whether to write headers for append mode
	file_exists = os.path.isfile(summary_csv_path)
	
	# Define the headers and the row data
	row_data = [
		total_co2_emissions, average_co2_emissions_per_run,
		total_duration, average_duration_per_run, total_power_consumption, total_cost, num_runs
	]
	
	headers = [
		"Total CO2 Emissions", "Average Emissions per Run",
		"Total Duration", "Average Duration per Run", "Total Power Consumption (kWh)",
		"Total Cost ($)", "Number of Runs"
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


def run_test(test_case_path, tool_config):
	print("eco2ai")
	os.makedirs(tool_config['output_path'], exist_ok=True)
	emissions_file_path = os.path.join(tool_config['output_path'], "emission.csv")
	
	tracker = Tracker(project_name="second project",
	                  experiment_description="Description",
	                  file_name=emissions_file_path, ignore_warnings=True)
	
	tracker.start()
	try:
		subprocess.run(["python", test_case_path], check=True)
		generate_summary_eco2ai(tool_config)
	except subprocess.CalledProcessError as e:
		logging.error(f"Error running test case {test_case_path} with eco2AI: {e}")
	finally:
		tracker.stop()
		logging.info(f"Completed eco2AI measurement for {test_case_path}")
