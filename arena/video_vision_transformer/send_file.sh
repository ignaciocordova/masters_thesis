
#!/bin/bash

# Define the remote server login credentials and destination folder
remote_server="icordova@login3.ccc.uam.es"
remote_folder="/home/icordova/tfm"

# Check if a file name is specified as a command-line argument
if [ $# -eq 0 ]; then
  echo "Error: no file specified."
  exit 1
fi

# Copy the specified file to the remote server using scp
scp "$1" "$remote_server:$remote_folder"
echo "File $1 sent to remote server."

echo "File transfer complete."
