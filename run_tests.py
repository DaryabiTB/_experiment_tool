import json
import logging
import random
import importlib
import argparse

# Initialize logging
logging.basicConfig(filename='logs/project_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run emission tracking tools.')
parser.add_argument('--tool', nargs='+', help='Specify tool names separated by space, or "all" to run all tools.')

args = parser.parse_args()

# Load config
with open("config.json") as config_file:
	config = json.load(config_file)


def start_all():
	# Randomize the order of tools and log it
	tools = list(config["tools"].keys())
	random.shuffle(tools)
	logging.info(f"Tool execution order: {tools}")
	
	# Run each test with each tool
	for tool_name in tools:
		start_tool(tool_name)


def start_selected_tools(tools):
	# Run each test with each selected tool
	for tool_name in tools:
		if tool_name in config["tools"]:
			start_tool(tool_name)
		else:
			logging.warning(f"Tool '{tool_name}' not found. Skipping.")


def start_tool(tool_name):
	logging.info(f"Executing tool : {tool_name}")
	tool_module = importlib.import_module(f"tools.{tool_name}_tool")
	for test_case in config["test_cases"]:
		logging.info(f"Running {test_case} with {tool_name}")
		tool_module.run_test(test_case, config["tools"][tool_name])
		logging.info("******")


if __name__ == "__main__":
	if 'all' in args.tool:
		start_all()
	else:
		start_selected_tools(args.tool)
