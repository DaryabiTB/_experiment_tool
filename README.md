# Emission Tracking Toolkit

## Introduction
Welcome to the Emission Tracking Toolkit! This project is designed to utilize a variety of tools to measure and analyze the output of different emission tracking experiments.

## Project Structure
- `/docs`: Contains the documentation for each tool (e.g., `carbontracker_docs.md`).
- `/logs`: Stores logs such as API timeshift logs and project logs.
- `/output`: The directory where tool outputs are saved (e.g., `carbontracker`, `cloud_carbon_footprint`).
- `/test_cases`: Contains different project codes to be tested (e.g., `face_masking`).
- `/tools`: Holds the scripts for each tool (e.g., `carbontracker_tool.py`).
- `/venv`: The virtual environment directory for Python packages.
- `run_tests.py`: The main script to run test cases for all tools or an individual tool.
- `config.json`: Contains the configuration and descriptions for the projects.

## Getting Started

### Setting up the Environment

#### Create a Virtual Environment
For `conda`:
```python
conda create --name emission_env python=3.8
conda activate emission_env
```

For `python virtualenv`:
```python
python -m venv emission_env
source emission_env/bin/activate  # On Windows use `emission_env\Scripts\activate`
```

### Install Required Packages
After activating the environment, install the required packages using:
```python
pip install -r requirements.txt
```

## Running the Toolkit
To run a test case using `run_tests.py`, you can execute the script with the following command:
```bash
python run_tests.py --tool <tool_name>  # Use --tool all to run all tools ex: "python run_tests.py --tool all"
python run_tests.py --tool <tool_name>  # Use single --tool to run a tools ex: "python run_tests.py --tool codecarbon"
python run_tests.py --tool <tool_name>  # Use multiple --tool to run many tools ex: python run_tests.py --tool codecarbon carbontracker
```

## Adding a New Experiment Tool
If you wish to add a new experiment tool to the toolkit, please follow these steps:

1. **Tool Script:**
   - Place your tool's script in the `/tools` directory.
   - Name the file in a consistent manner, e.g., `<newtool>_tool.py`.

2. **Test Case:**
   - Create a new folder within `/test_cases` for your tool's specific experiments. e.g., `new_tool_tests`.

3. **Configuration:**
   - Add your project's configuration to `config.json` following the existing structure. Read below for more information.

4. **Documentation:**
   - Write a Markdown documentation file for your tool.
   - Place the documentation in `/docs` with a name like `<newtool>_docs.md`.

5. **Output Directory:**
   - Ensure your tool script saves its output to the correct folder under `/output`.


## Guide `run_tests.py` 

## Overview
The `run_tests.py` script is the entry point for running test cases across different tools in the Emission Tracking Toolkit. This script allows for executing individual tools or a suite of tools to assess their performance and output.

## Prerequisites
Before running the script, ensure that:
- The virtual environment is activated.
- All dependencies are installed via `requirements.txt`.
- The `config.json` file is properly configured with your test cases and tools.


## Adding a New Test Case
To add a new test case:
1. Place the test script in the `/test_cases` directory.
2. Update the `config.json` file to include the path to the new test script.
3. Ensure the new test case is compatible with the expected input and output structure defined by the toolkit.

## Output
The script will execute the specified test cases and generate logs and output files in their respective directories as configured in `config.json`.

For detailed logs, refer to the `/logs` directory after running the test cases.


## Configuration File `config.json`

The `config.json` file is an essential part of the Emission Tracking Toolkit. It centralizes the configuration for various aspects of the project.

### Structure and Example

The `config.json` file is structured as follows:

- `test_cases`: An array of strings specifying paths to test case scripts.
- `tools`: An object with keys representing tool names and values being settings specific to that tool.
- `log_file`: A string indicating the path to the project's log file.

Here is a brief example showing part of the structure:

```json
{
  "test_cases": ["test_cases/api_test_timeshift_script.py","testcases/facemasking/main.py"],
  "tools": {
    "codecarbon": {
      "output_path": "output/codecarbon",
      "measurement_interval": 2
    },
    "carbontracker": {
      "output_path": "output/carbontracker"
    }
    // Additional tool configurations would be placed here
  },
  "log_file": "logs/project_log.log"
}
```

### Modifying `config.json`

To customize the toolkit, you can modify the `config.json` file. Add new test cases or tool configurations as needed, ensuring that the file maintains valid JSON syntax.

## Contribution
We welcome contributions to the Emission Tracking Toolkit. If you have suggestions or improvements, please follow the standard Github fork and pull request workflow.

## Support
If you encounter any issues or have questions, please file an issue on the project's GitHub repository.

Thank you for participating in our emission tracking efforts!

