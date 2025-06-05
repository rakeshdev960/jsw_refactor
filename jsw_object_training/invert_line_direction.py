# Simple script to modify the combined_detection.py file to invert direction for Line 1
# Run this script from the command line

file_path = 'f:/jsw20042025/jsw_object_training/combined_detection.py'

# Read the file
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Find the positions where we need to insert the inversion code
upload_video_insert_pos = None
rtsp_camera_insert_pos = None

# Look for the direction calculation line followed by # Get current time
for i in range(len(lines) - 1):
    if "line['start'], line['end']" in lines[i] and ")" in lines[i] and i < len(lines) - 2:
        if "# Get current time" in lines[i+2]:
            if upload_video_insert_pos is None:
                upload_video_insert_pos = i + 2
            else:
                rtsp_camera_insert_pos = i + 2

# Insert our code at the found positions
if upload_video_insert_pos:
    inversion_code = [
        "                                # Invert direction for Line 1 only\n",
        "                                if line['name'] == 'Line 1':\n",
        "                                    direction = \"OUT\" if direction == \"IN\" else \"IN\"\n",
        "\n"
    ]
    lines[upload_video_insert_pos:upload_video_insert_pos] = inversion_code

if rtsp_camera_insert_pos:
    # If we've already inserted code, adjust position
    if upload_video_insert_pos and rtsp_camera_insert_pos > upload_video_insert_pos:
        rtsp_camera_insert_pos += len(inversion_code)
    
    lines[rtsp_camera_insert_pos:rtsp_camera_insert_pos] = inversion_code

# Write the file back
with open(file_path, 'w', encoding='utf-8') as file:
    file.writelines(lines)

print("Direction inversion for Line 1 has been added successfully.")
