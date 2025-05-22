import os
import subprocess
import sys


def execute_scripts_in_folders(base_path, selected_folders=None):
    passed, failed = [], []

    for root, _, files in sorted(os.walk(base_path), key=lambda x: x[0]):
        if selected_folders:
            rel_root = os.path.relpath(root, base_path)
            if not any(rel_root.startswith(folder) for folder in selected_folders):
                continue
        for file in sorted(files):
            if file.endswith("test.py") and file != os.path.basename(__file__):
                script_path = os.path.join(root, file)
                print(f"\n Executing: {script_path}")
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pytest", script_path],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    print(result.stdout)
                    passed.append(script_path)
                except subprocess.CalledProcessError as e:
                    print(f"\n Error in {script_path}:\n{e.stderr}")
                    failed.append(script_path)

    print("\n==================== TEST SUMMARY ====================")
    print(f" Passed: {len(passed)}")
    for f in passed:
        print(f"    {f}")
    if failed:
        print(f"\n Failed: {len(failed)}")
        for f in failed:
            print(f"    {f}")
    else:
        print("\n All tests passed!")


if __name__ == "__main__":
    base_directory = os.getcwd()
    choice = input(
        "Enter specific folders (comma-separated) or press Enter to scan all: "
    ).strip()
    selected_folders = choice.split(",") if choice else None
    execute_scripts_in_folders(base_directory, selected_folders)
