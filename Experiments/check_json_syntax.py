import json
import os
from pathlib import Path
import sys
import subprocess
import platform
from typing import List, Tuple

def check_json_syntax(file_path: str) -> Tuple[bool, str]:

    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"

    if not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}"

    try:

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

                if not content:
                    return False, "File is empty"
                json.loads(content)
                return True, ""
        except UnicodeDecodeError:

            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read().strip()
                if not content:
                    return False, "File is empty"
                json.loads(content)
                return True, ""
    except json.JSONDecodeError as e:
        return False, f"JSONDecodeError: {e.msg} (line {e.lineno}, column {e.colno})"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

def find_all_json_files(root_dir: str) -> List[str]:
    json_files = []

    root_dir = os.path.abspath(root_dir)

    print(f"Searching in directory: {root_dir}")
    print(f"Directory exists: {os.path.exists(root_dir)}")

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.json'):
                full_path = os.path.join(root, file)

                if os.path.exists(full_path) and os.path.isfile(full_path):
                    json_files.append(full_path)
                else:
                    print(f"Warning: Found reference to non-existent file: {full_path}")
    return json_files

def get_relative_path(file_path: str, base_path: str) -> str:

    experiments_index = file_path.find('Experiments')
    if experiments_index != -1:
        return file_path[experiments_index:]
    else:

        try:
            return os.path.relpath(file_path, base_path)
        except ValueError:
            return file_path

def main():

    target_dir = r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection"
    print(f"Searching for JSON files in: {target_dir}")
    print("=" * 60)

    json_files = find_all_json_files(target_dir)

    if not json_files:
        print("No JSON files found in the directory tree.")
        return

    print(f"Found {len(json_files)} JSON file(s)")
    print()

    invalid_files = []
    valid_count = 0

    for i, file_path in enumerate(json_files, 1):

        print(f"[{i}/{len(json_files)}] Checking: {get_relative_path(file_path, target_dir)}")

        is_valid, error_msg = check_json_syntax(file_path)

        if is_valid:
            print(f"  ✓ Valid JSON")
            valid_count += 1
        else:
            print(f"  ✗ Invalid JSON - {error_msg}")
            invalid_files.append((file_path, error_msg))

        print()

    print("=" * 60)
    print("JSON SYNTAX CHECK SUMMARY")
    print("=" * 60)
    print(f"Total JSON files checked: {len(json_files)}")
    print(f"Valid JSON files: {valid_count}")
    print(f"Invalid JSON files: {len(invalid_files)}")
    print()

    if invalid_files:
        print("FILES WITH SYNTAX ERRORS:")
        print("-" * 40)
        for i, (file_path, error_msg) in enumerate(invalid_files, 1):
            relative_path = get_relative_path(file_path, target_dir)
            print(f"{i}. {relative_path}")
            print(f"   Error: {error_msg}")
            print()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        report_filename = os.path.join(script_dir, "json_syntax_error_report.txt")

        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("JSON SYNTAX ERROR REPORT\n")
            f.write("=" * 40 + "\n")
            f.write(f"Generated on: {subprocess.run(['powershell', 'Get-Date'], capture_output=True, text=True).stdout.strip() if platform.system() == 'Windows' else ''}\n")
            f.write(f"Total JSON files checked: {len(json_files)}\n")
            f.write(f"Valid JSON files: {valid_count}\n")
            f.write(f"Invalid JSON files: {len(invalid_files)}\n\n")
            f.write("FILES WITH SYNTAX ERRORS:\n")
            f.write("-" * 40 + "\n")
            for i, (file_path, error_msg) in enumerate(invalid_files, 1):
                relative_path = get_relative_path(file_path, target_dir)
                f.write(f"{i}. {relative_path}\n")
                f.write(f"   Error: {error_msg}\n\n")

        print(f"Detailed report saved to: {report_filename}")
    else:
        print("🎉 All JSON files have valid syntax!")

if __name__ == "__main__":
    main()
