from MAGIC import Magic

from pathlib import Path
# import pytest
import subprocess
import sys
import os
import json
import time
import pandas as pd

from prettytable import PrettyTable

from langgraph.graph import END, StateGraph, START

'''for postion in range(len(test_list_endpoints)):
    app = Magic()
    rus = app.run_app(user_endpoint=test_list_endpoints[postion], user_query=test_list_cases[postion])

    print(type(rus["generation"]))

    #print(test_list_endpoints[postion], "\t endpoint")
    #print(test_list_cases[postion], "\t testcase")'''

#app = Magic()
#app.run_app(user_endpoint=ssid_endpoint,user_query=ssid_query)

current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent.absolute()
test_document_path = f'{current_dir}/documents/endpoints.json'

def get_test():
    test_document = {}
    with open(test_document_path, "r") as f:
        test_document = f.read()
        test_document = json.loads(test_document)

    return test_document

def run_app():
    test_document = get_test()

    endpoint_code:dict = {}
    for test_case in test_document:
        print(test_document[test_case], test_case)
        app = Magic()
        response = app.run_app(user_endpoint=test_document[test_case], user_query=test_case)

        print(response.keys())
        print(f"endpoint: {response['endpoint']}")
        endpoint_code[response["code_filename"]] = response["endpoint"]

        print("\n\n\n done with: ",test_case)
        time.sleep(2)

    # execute_generated_tests_dir("./generated_code/")
        #print(test_list_endpoints[postion], "\t endpoint")
        #print(test_list_cases[postion], "\t testcase")
    return endpoint_code


def run_app_for_specific_test(endpoint:str, query:str):
    print(f"Running for: {query}\n Endpoint: {endpoint}")
    endpoint_code:dict = {}

    # Run magic and generate working code
    app = Magic()
    app_response = app.run_app(user_endpoint=endpoint, user_query=query)

    print(f"endpoint: {app_response['endpoint']}")
    endpoint_code[app_response["code_filename"]] = app_response["endpoint"]

    print(f"Done with: {query}")
    time.sleep(2)

    # execute_generated_tests_dir("./generated_code/")
    return app_response


def display(results:dict):
    table = PrettyTable()
    table.field_names = ["Test #", "Filename", "Result"]

    #print(f"Results: {results}")
    test_count = 0
    for test_result in results:
        test_count += 1
        print(type(results[test_result]))
        try:
            if results[test_result]['passed'] != 0:
                r = "P"
            else:
                r = "F"
        except:
            r = "F"
        table.add_row([test_count, os.path.basename(test_result), r])

    print(table)


import re

def _extract_test_results(output: str) -> dict:
    """
    Parses the pytest command output to extract the test results. We are interested
    in knowing the number of test cases that were run and how many of them passed / failed

    Output from the pytest command when test was run correctly look like this:
    ============================= test session starts ==============================
    platform linux -- Python 3.9.16, pytest-8.3.3, pluggy-1.5.0
    rootdir: /home/meraki/magic_intern_proj/magic_streamlit/Magic_PoC/vers/RAG
    plugins: anyio-4.2.0
    collected 2 items

    generated_code/gen_code_rules.py ..                                      [100%]

    ============================== 2 passed in 2.38s ===============================

    When some tests fail and some pass, the output looks like this:
    ========================== 3 failed, 1 passed in 0.12s =========================
    """
    test_results = {
        "total": 0,
        "passed": 0,
        "failed": 0
    }
    output_lines = output.split('\n')
    for line in output_lines:
        if line.startswith('collected'):
            test_results["total"] = int(line.split(' ')[1])
        elif re.search(r'\d+ passed', line) or re.search(r'\d+ failed', line):
            passed_match = re.search(r'(\d+) passed', line)
            failed_match = re.search(r'(\d+) failed', line)
            if passed_match:
                test_results["passed"] = int(passed_match.group(1))
            if failed_match:
                test_results["failed"] = int(failed_match.group(1))

    # simple validation that total is equal to the sum of passed and failed
    if test_results["total"] != test_results["passed"] + test_results["failed"]:
        print(f"Total test cases {test_results['total']} is not matching passed and failed test cases.")
        print(f"Passed: {test_results['passed']}, Failed: {test_results['failed']}")

    #print(f"Test results: {test_results}")
    return test_results

def _run_code(code_file_name_path: str, results:dict) -> dict:
    cmd_output = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            code_file_name_path
        ],
        capture_output=True,
        text=True
    )

    output = cmd_output.stdout
    error = cmd_output.stderr

    if cmd_output.returncode == 0:
        print(f"Output: {cmd_output}")
        test_results = _extract_test_results(output)
        results[code_file_name_path] = test_results
    else:
        print(f"Error: {error}\n Stdout: {output}")
        results[code_file_name_path] = {
        "error":error
    }

    print(f"Results in _run_code: {results}")

def execute_generated_tests_dir(*args:str, debug:bool = False):
    results:dict = {}
    dir_list = []

    for pytest_dir in args:
        try:
            code_dir_object = Path(pytest_dir)
            if not code_dir_object.is_dir():
                raise Exception("Couldn't find file")

        except Exception as e:
            print(e)
            results.update({pytest_dir : [1, e]})
            continue

        files = os.listdir(pytest_dir)
        dir_list = [entry for entry in files if entry.endswith('.py') and os.path.isfile(os.path.join(pytest_dir, entry))]


        execute_tests(*dir_list, results = results, file_path=True)

    #print(f"Results in execute_generated_tests_dir: {results}")
    return results


def execute_tests(*args:str, results:dict = {}, file_path:bool = False, file_name:bool = True, debug:bool = False):
    """
    Execute the generated tests
    """
    for pytest_file in args:
        if file_name:
            code_file_name_path = f'{current_dir}/generated_code/{pytest_file}'
        else:
            code_file_name_path = pytest_file

        try:
            code_file_object = Path(code_file_name_path)
            if not code_file_object.is_file():
                #print(not code_file_object.is_dir())
                raise Exception("Couldn't find file")
        except Exception as e:
            results.update({code_file_name_path : [1, e]})
            continue

        _run_code(results = results, code_file_name_path = code_file_name_path)

    #print(f"Results in execute_tests: {results}")


def execute_single_generated_test(filename:str, debug:bool = False):
    results:dict = {}
    
    try:
        code_dir_object = Path(filename)
        if not code_dir_object.is_file():
            raise Exception("Couldn't find file")

    except Exception as e:
        print(e)
        results.update({filename : [1, e]})

    _run_code(code_file_name_path=filename, results = results)

    #print(f"Results in execute_generated_tests_dir: {results}")
    return(results)

if __name__ == '__main__':
    endpoint_code = run_app()
    results = execute_generated_tests_dir(f"{current_dir}/generated_code/")

    # display the results in the console
    display(results)


# run_app()
