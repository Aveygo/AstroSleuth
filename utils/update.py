import requests
import subprocess

def get_current_branch(repo_path):
    # Get the current branch of the local repository
    result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=repo_path, capture_output=True, text=True)
    return result.stdout.strip()

def get_latest_commit_sha(branch):
    url = f'https://api.github.com/repos/aveygo/astrosleuth/commits/{branch}'
    response = requests.get(url)
    
    if response.status_code == 200:
        commit_data = response.json()
        return commit_data['sha'], commit_data["commit"]["message"]
    else:
        raise Exception(f"Error fetching latest commit: {response.status_code} - {response.text}")

def update_repository(repo_path):
    subprocess.run(['git', 'pull'], cwd=repo_path)

def main():
    repo_path = '.'

    # Get the latest commit SHA from the GitHub repository
    branch_name = get_current_branch(repo_path)
    latest_commit_sha = get_latest_commit_sha(branch_name)

    # Check the current local commit SHA
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo_path, capture_output=True, text=True)
    local_commit_sha, message = result.stdout.strip()

    # Compare local and remote commit SHAs
    if local_commit_sha != latest_commit_sha:
        print(f"Found an update: {message}")
        if input("Would you like to update? [y/n]") == "y":
            update_repository(repo_path)
            print("Repository updated successfully.")
        else:
            print("Update canceled")
    else:
        print("Repository is already up to date.")

if __name__ == "__main__":
    print("WARNING! This script is experimental! Errors are very much expected!")
    input("Press enter to continue...")
    main()