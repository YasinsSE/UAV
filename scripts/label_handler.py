import os

label_folder = "C:/Users/Yasins/Desktop/UAV/dataset"


def update_labels_to_zero(label_folder):
    print("Process started: Updating label IDs to 0...")

    total_files = 0
    updated_files = 0

    # Loop for every txt file inside the folder
    for filename in os.listdir(label_folder):
        if filename.endswith(".txt"):
            total_files += 1
            file_path = os.path.join(label_folder, filename)
            print(f"Processing file: {filename}")

            # Read file and update every line by processing
            with open(file_path, "r") as file:
                lines = file.readlines()

            # Flag to save updated lines
            changes_made = False
            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                old_id = parts[0]

                # If clas ID !=0 , make it 0
                if old_id != '0':
                    parts[0] = '0'
                    changes_made = True
                    print(f"Updated label ID from {old_id} to 0 in {filename}")

                updated_lines.append(" ".join(parts))

            # If changes has made, update the file
            if changes_made:
                with open(file_path, "w") as file:
                    file.write("\n".join(updated_lines))
                updated_files += 1
                print(f"File {filename} updated successfully.")

    print(f"Process completed: {total_files} files processed, {updated_files} files updated.")


update_labels_to_zero(label_folder)
