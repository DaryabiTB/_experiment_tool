import json
import logging
import random
import importlib

# Initialize logging
logging.basicConfig(filename='logs/project_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
with open("config.json") as config_file:
	config = json.load(config_file)


def start_all():
	# Randomize the order of tools and log it
	tools = list(config["tools"].keys())
	random.shuffle(tools)
	logging.info(f"Tool execution order: {tools}")
	
	# Run each test with each tool
	for tool_name in tools:
		tool_module = importlib.import_module(f"tools.{tool_name}_tool")
		for test_case in config["test_cases"]:
			logging.info(f"Running {test_case} with {tool_name}")
			tool_module.run_test(test_case, config["tools"][tool_name])
			logging.info("******")


def start_tool(tool_name):
	logging.info(f"Executing tool : {tool_name}")
	tool_module = importlib.import_module(f"tools.{tool_name}_tool")
	for test_case in config["test_cases"]:
		logging.info(f"Running {test_case} with {tool_name}")
		tool_module.run_test(test_case, config["tools"][tool_name])
		logging.info("******")


if __name__ == "__main__":
	import sys
	print("sys.path")
	print(sys.path)
	print("*" * 100)
	avalibe_tool = {
		1: "codecarbon",
		2: "carbontracker",
		3: "eco2ai",
		4: "energat"
	}
	start_tool(avalibe_tool[4])
