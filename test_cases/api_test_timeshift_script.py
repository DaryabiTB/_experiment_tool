import requests
import logging
from datetime import datetime, timedelta
import time  # Import for sleep
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# local
# API_URL = "http://127.0.0.1:5000/api/v1/data/forecast/timeshift"
# JWT_TOKEN = os.getenv("CODEGREEN_LOCAL_JWT_TOKEN")

# web
API_URL = "https://codegreen.world/api/v1/data/forecast/timeshift"
JWT_TOKEN = os.getenv("CODEGREEN_WEB_JWT_TOKEN")

SLEEP_DURATION = 5  # Duration to sleep between requests, in seconds


def setup_logger(country_code):
	# Determine the project root directory (assuming this script is inside the project structure)
	project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	
	# Define log file path in 'logs/api_logs' subdirectory
	log_directory = os.path.join(project_root, 'logs', 'api_timeshit_logs')
	os.makedirs(log_directory, exist_ok=True)  # Create the directory if it doesn't exist
	
	log_filename = os.path.join(log_directory, f'{country_code.lower()}_api_test.log')
	
	# Setup logger
	logger = logging.getLogger(country_code)
	logger.setLevel(logging.INFO)
	if not logger.handlers:
		file_handler = logging.FileHandler(log_filename)
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)
	
	return logger


def get_hard_finish_time(additional_time):
	return int((datetime.now() + additional_time).timestamp())


def create_test_body(percent_renewable, hours, minutes, area_code, hard_time):
	return {
		"estimated_runtime_hours": hours,
		"estimated_runtime_minutes": minutes,
		"percent_renewable": percent_renewable,
		"area_code": area_code,
		"process_id": "1",
		"log_request": False,
		"hard_finish_time": hard_time
	}


def run_tests(area_code):
	print(area_code)
	
	test_cases = [
		
		(5, 0, 15, timedelta(minutes=10), "Low renewable energy, short runtime, immediate future"),
		
		(50, 1, 30, timedelta(hours=1), "Moderate renewable energy, moderate runtime, short-term future"),
		
		(90, 3, 0, timedelta(hours=12), "High renewable energy, long runtime, mid-term future"),
		
		(90, 0, 15, timedelta(days=1), "High renewable energy, short runtime, long-term future"),
		
		(50, 3, 0, timedelta(minutes=10), "Moderate renewable energy, long runtime, immediate future"),
		
		(5, 1, 30, timedelta(hours=12), "Low renewable energy, moderate runtime, mid-term future"),
		
		(30, 1, 0, timedelta(hours=1), "Runtime equal to hard finish time"),
		
		(40, 0, 45, timedelta(hours=1), "Runtime shorter than hard finish time"),
		
		(20, 2, 30, timedelta(hours=2),
		 "Runtime longer than hard finish time (should test if API handles this correctly)"),
		
		# invalid inputs
		(-10, 1, 30, timedelta(hours=2), "Percent renewable less than 0 (invalid input)"),
		
		# invalid inputs
		(150, 1, 0, timedelta(hours=1), "Percent renewable more than 100 (invalid input)"),
	]
	
	headers = {'Authorization': f"Bearer {JWT_TOKEN}", 'content-type': 'application/json'}
	
	logger = setup_logger(area_code)
	case_no = 0
	for percent_renewable, hours, minutes, time_delta, description in test_cases:
		body = create_test_body(percent_renewable, hours, minutes, area_code, get_hard_finish_time(time_delta))
		case_no += 1
		print(f"Running test case {case_no}...")
		response = requests.post(API_URL, json=body, headers=headers)
		logger.info(f"Scenario {case_no}: {description} + {time_delta}")
		logger.info(f"Request: {body}")
		try:
			# Try to parse the response as JSON
			json_data = response.json()
			logger.info(f"Response JSON: {json_data}")
		except ValueError:
			# If the response is not JSON, log the first few lines
			response_text_lines = response.text.splitlines()
			preview = response_text_lines[:3]  # Adjust the number of lines as needed
			logger.info(f"{response.status_code}, Response (not JSON, first few lines): {' '.join(preview)}")
		time.sleep(SLEEP_DURATION)  # Sleep after each request


countries_to_test = ['DE', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'AT', 'GR', 'HU', 'IE', 'IT', 'LV',
                     'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 'CH']


def start_test():
	for country in countries_to_test:
		run_tests(country)
		break
	print("Done")


if __name__ == "__main__":
	# Run tests for Germany (DE)
	for country in countries_to_test[:4]:
		run_tests(country)
		break
	print("Done")
