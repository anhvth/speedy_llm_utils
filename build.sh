#!/bin/bash

# Get the current version before making any changes
CURRENT_VERSION=$(bump2version --dry-run --allow-dirty --list patch | grep current_version | cut -d'=' -f2 | xargs)
NEW_VERSION=$(bump2version --dry-run --allow-dirty --list patch | grep new_version | cut -d'=' -f2 | xargs)

echo "Attempting to bump version from $CURRENT_VERSION to $NEW_VERSION..."

# Check if working directory has uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "Git working directory is not clean. Please commit or stash your changes first."
    echo "Changes detected:"
    git status --short
    
    # Ask user if they want to commit these changes
    read -p "Do you want to commit these changes now? (y/n): " choice
    if [[ "$choice" == [Yy]* ]]; then
        echo "Committing pending changes..."
        read -p "Enter a commit message: " commit_message
        if [ -z "$commit_message" ]; then
            echo "No commit message entered. Aborting."
            exit 1
        fi
        git add -A && git commit -m "$commit_message" || { echo "Commit failed"; exit 1; }
    else
        echo "Aborting version bump. Please clean your working directory first."
        exit 1
    fi
fi

# Now bump the version (requires clean working directory)
if ! bump2version patch; then
    echo "Error: bump2version failed"
    exit 1
fi


git add -A && git commit -m "Bumped version to $NEW_VERSION" || { echo "Commit failed"; exit 1; }
git push