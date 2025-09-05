"""
       █████  █████ ██████   █████           █████ █████   █████ ██████████ ███████████    █████████  ██████████
      ░░███  ░░███ ░░██████ ░░███           ░░███ ░░███   ░░███ ░░███░░░░░█░░███░░░░░███  ███░░░░░███░░███░░░░░█
       ░███   ░███  ░███░███ ░███   ██████   ░███  ░███    ░███  ░███  █ ░  ░███    ░███ ░███    ░░░  ░███  █ ░ 
       ░███   ░███  ░███░░███░███  ░░░░░███  ░███  ░███    ░███  ░██████    ░██████████  ░░█████████  ░██████   
       ░███   ░███  ░███ ░░██████   ███████  ░███  ░░███   ███   ░███░░█    ░███░░░░░███  ░░░░░░░░███ ░███░░█   
       ░███   ░███  ░███  ░░█████  ███░░███  ░███   ░░░█████░    ░███ ░   █ ░███    ░███  ███    ░███ ░███ ░   █
       ░░████████   █████  ░░█████░░████████ █████    ░░███      ██████████ █████   █████░░█████████  ██████████
        ░░░░░░░░   ░░░░░    ░░░░░  ░░░░░░░░ ░░░░░      ░░░      ░░░░░░░░░░ ░░░░░   ░░░░░  ░░░░░░░░░  ░░░░░░░░░░ 
                 A Collectionless AI Project (https://collectionless.ai)
                 Registration/Login: https://unaiverse.io
                 Code Repositories:  https://github.com/collectionlessai/
                 Main Developers:    Stefano Melacci (Project Leader), Christian Di Maio, Tommaso Guidi
"""
import os
import sys
import uuid
import argparse
import subprocess
from pathlib import Path
from unaiverse.networking.p2p import P2P

# Configuration
DOCKER_IMAGE_NAME = "unaiverse-sandbox"
CONTAINER_NAME_BASE = "unaiverse-sandbox-container"
CONTAINER_NAME = f"{CONTAINER_NAME_BASE}-{uuid.uuid4().hex[:8]}"  # Append a short unique ID
DOCKERFILE_CONTENT = """

# Debian image, automatically guessed architecture
FROM python:3.12-slim-bookworm

# Installing Go compiler
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl git
RUN rm -rf /var/lib/apt/lists/*
RUN ARCH=$(dpkg --print-architecture) && curl -LO https://go.dev/dl/go1.24.5.linux-${ARCH}.tar.gz
RUN ARCH=$(dpkg --print-architecture) && tar -C /usr/local -xzf go1.24.5.linux-${ARCH}.tar.gz
RUN ARCH=$(dpkg --print-architecture) && rm go1.24.5.linux-${ARCH}.tar.gz

# Set Go environment variables
ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH="/go"
RUN mkdir -p /go/bin /go/src /go/pkg

# Setting the working directory inside the container
WORKDIR /unaiverse

# Dependencies
RUN <create_requirements.txt>
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages
"""


def sandbox(file_to_run: str,
            read_only_paths: tuple[str] | list[str] | None = None,
            writable_paths: tuple[str] | list[str] | None = None) -> None:

    # Path of this file
    absolute_path_of_this_file = os.path.abspath(__file__)

    # Folders composing the path (and file name at the end)
    path_components = list(Path(absolute_path_of_this_file).parts)

    # Ensuring the folder/file structure was not manipulated
    assert path_components[-1] == 'sandbox.py', "Major security issue, stopping."
    assert path_components[-2] == 'utils', "Major security issue, stopping."
    assert path_components[-3] == 'unaiverse', "Major security issue, stopping."

    # Main folder of UNaIVERSE
    abspath_of_unaiverse_code = str(Path(*path_components[0:-3]))

    # Clean up any remnants from previous runs first (safety)
    cleanup_docker_artifacts(where=abspath_of_unaiverse_code)

    # Requirements
    echoed_contents_of_requirements = 'printf "'
    with open(os.path.join(abspath_of_unaiverse_code, "requirements.txt"), 'r') as req_file:
        req_lines = req_file.readlines()
    for i, req_line in enumerate(req_lines):
        if i != (len(req_lines) - 1) and len(req_line.strip()) > 0:
            echoed_contents_of_requirements += req_line.strip() + "\\n"
        else:
            echoed_contents_of_requirements += req_line.strip() + "\\n\" > requirements.txt"

    # Create Dockerfile
    print("Creating Dockerfile...")
    with open(os.path.join(abspath_of_unaiverse_code, "Dockerfile"), "w") as f:
        f.write(DOCKERFILE_CONTENT.replace('<create_requirements.txt>', echoed_contents_of_requirements))

    # Building Docker image
    if not build_docker_image(where=abspath_of_unaiverse_code):
        print("Exiting due to Docker image build failure")
        cleanup_docker_artifacts(where=abspath_of_unaiverse_code)  # Try to clean up what was created (if any)
        sys.exit(1)

    # Read only folders from the host machine
    read_only_mount_paths = ([abspath_of_unaiverse_code] +
                             (list(read_only_paths) if read_only_paths is not None else []))

    # Writable folders in host machine
    writable_mount_paths = ([os.path.join(abspath_of_unaiverse_code, 'runners'),
                             os.path.join(abspath_of_unaiverse_code, 'unaiverse', 'library'),
                             os.path.join(abspath_of_unaiverse_code, 'unaiverse', 'networking', 'p2p')] +
                            (list(writable_paths) if writable_paths is not None else []))

    # Running
    if not run_in_docker(file_to_run=os.path.abspath(file_to_run),
                         read_only_host_paths=read_only_mount_paths,
                         writable_host_paths=writable_mount_paths):
        print("Exiting due to Docker container run failure")
        sys.exit(1)

    # Final cleanup
    cleanup_docker_artifacts(where=abspath_of_unaiverse_code)


def build_docker_image(where: str):
    """Builds the Docker image."""
    print(f"Building Docker image '{DOCKER_IMAGE_NAME}'...")

    try:

        # The '.' at the end means build from the current directory
        subprocess.run(["docker", "build", "-t", DOCKER_IMAGE_NAME, where], check=True)
        print(f"Docker image '{DOCKER_IMAGE_NAME}' built successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building Docker image: {e}")
        return False


def cleanup_docker_artifacts(where: str):
    """Cleans up the generated files and Docker image."""
    print("Cleaning...")

    # Stop and remove container if it's still running (e.g., if previous run failed)
    try:
        print(f"Attempting to stop and remove container '{CONTAINER_NAME}' (if running)...")
        subprocess.run(["docker", "stop", CONTAINER_NAME],
                       check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["docker", "rm", CONTAINER_NAME],
                       check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print(f"Error during preliminary container cleanup: {e}")

    # Remove the Docker image
    try:
        print(f"Removing Docker image '{DOCKER_IMAGE_NAME}'...")
        subprocess.run(["docker", "rmi", DOCKER_IMAGE_NAME], check=True)
        print("Docker image removed.")
    except subprocess.CalledProcessError as e:
        print(f"Error removing Docker image '{DOCKER_IMAGE_NAME}': {e}")

    # Remove the generated Dockerfile
    if os.path.exists(os.path.join(where, "Dockerfile")):
        os.remove(os.path.join(where, "Dockerfile"))
        print("Removed Dockerfile.")


def run_in_docker(file_to_run: str, read_only_host_paths: list[str] = None, writable_host_paths: list[str] = None):
    """Runs the code in a Docker container with optional mounts."""
    print(f"\nRunning code in Docker container '{CONTAINER_NAME}'...")

    # Building command (it will continue below...)
    command = ["docker", "run",
               "--rm",  # Automatically remove the container when it exits
               "-e", "PYTHONUNBUFFERED=1",  # Ensure Python output is unbuffered
               "-e", "NODE_STARTING_PORT",
               "--name", CONTAINER_NAME]

    if sys.platform.startswith('linux'):

        # Linux
        command.extend(["--net", "host"]),  # Expose the host network (in macOS and Windows it is still a virtual host)
    else:

        # Not-linux: check ports (adding -p port:port)
        port_int = int(os.getenv("NODE_STARTING_PORT", "0"))
        if port_int > 0:
            command.extend(["-p", str(port_int + 0) + ":" + str(port_int + 0)])
            command.extend(["-p", str(port_int + 1) + ":" + str(port_int + 1) + "/udp"])
            command.extend(["-p", str(port_int + 2) + ":" + str(port_int + 2)])
            command.extend(["-p", str(port_int + 3) + ":" + str(port_int + 3) + "/udp"])

    # Add read-only mount if path is provided
    if read_only_host_paths is not None and len(read_only_host_paths) > 0:
        for path in read_only_host_paths:

            # Ensure the host path exists and is a directory
            if not os.path.isdir(path):
                print(
                    f"Error: Read-only host path '{path}' does not exist or is not a directory. Cannot mount.")
                return False
            else:

                # Augmenting command
                path = os.path.abspath(path)
                command.extend(["-v", f"{path}:{path}:ro"])
                print(f"Mounted host '{path}' as read-only to container")

    # Add writable mount if path is provided
    if writable_host_paths is not None and len(writable_host_paths) > 0:
        for path in writable_host_paths:

            # Ensure the host path exists and is a directory
            if not os.path.isdir(path):
                print(
                    f"Error: Writable host path '{path}' does not exist or is not a directory. Cannot mount.")
                return False
            else:

                # Augmenting command
                path = os.path.abspath(path)
                command.extend(["-v", f"{path}:{path}"])
                print(f"Mounted host '{path}' as writable to container")

    # Completing command
    command.append(DOCKER_IMAGE_NAME)

    try:

        # Running the prepared command... (using Popen to stream output in real-time)
        try:
            command.extend(["python3", file_to_run])
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(line)
            process.wait()  # Wait for the process to finish
            if process.returncode != 0:
                print(f"Container exited with non-zero status code: {process.returncode}")
        except KeyboardInterrupt:
            pass

        print(f"\nContainer '{CONTAINER_NAME}' finished execution.")
        return True
    except FileNotFoundError:
        print("Error: Docker command not found. Is Docker installed and in your PATH?")
        print("Please ensure Docker is installed and running.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error running Docker container: {e}")
        return False


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a Python script adding customizable read-only and writable paths.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
    Examples:
      python utils/sandbox.py my_script.py -r /home/user/data:/opt/app/data -p 1234
      python utils/sandbox.py another_script.py -w /tmp/output:/mnt/results
      python utils/sandbox.py script_with_both.py -r /input:/app/in -w /output:/app/out -p 8082
    """)
    parser.add_argument(help="Path to the Python script to execute.", dest="script_to_run",
                        type=str)
    parser.add_argument("-p", "--port", dest="port",
                        help="The starting port of the node(s) (each node uses 4 ports, consecutive port numbers)",
                        type=str, required=True)
    parser.add_argument("-r", "--read-only", dest="read_only_folders",
                        help="One or multiple paths to mount as read-only. "
                             "Use a colon to separate multiple paths (e.g., /path/a:/path/b).",
                        type=str, default=None)
    parser.add_argument("-w", "--writable", dest="writable_folders",
                        help="One or multiple paths to mount as writable. "
                             "Use a colon to separate multiple paths (e.g., /path/c:/path/d).",
                        type=str, default=None)
    args = parser.parse_args()

    if not args.script_to_run.endswith(".py"):
        parser.error(f"The script '{args.script_to_run}' must be a Python file (e.g., ending with .py)")
    script_to_run = args.script_to_run
    if not int(args.port) > 0:
        parser.error(f"Invalid port")

    read_only_folders = None
    if args.read_only_folders:
        read_only_folders = args.read_only_folders.split(':')
    writable_folders = None
    if args.writable_folders:
        writable_folders = args.writable_folders.split(':')

    print("\n Running in sandbox...")
    print(f"- Script to run: {script_to_run}")
    print(f"- Starting port (+0, +1, +2, +3): {args.port}")
    print(f"- Read only paths to mount (the UNaIVERSE code folder will be automatically mounted): {read_only_folders}")
    print(f"- Writable paths to mount: {writable_folders}\n")

    # Marking
    os.environ["NODE_STARTING_PORT"] = args.port

    # Running the sandbox and the script
    sandbox(script_to_run, read_only_paths=read_only_folders, writable_paths=writable_folders)
