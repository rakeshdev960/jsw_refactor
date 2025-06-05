# This is a code snippet with the fix to invert IN/OUT direction for Line 1 only
# Insert after the direction calculation in the upload video section:
                                direction = calculate_direction(
                                    previous_positions[id], center, 
                                    line['start'], line['end']
                                )
                                # Invert direction for Line 1 only
                                if line['name'] == 'Line 1':
                                    direction = "OUT" if direction == "IN" else "IN"

# Insert after the direction calculation in the RTSP section:
                                direction = calculate_direction(
                                    previous_positions[id], center, 
                                    line['start'], line['end']
                                )
                                # Invert direction for Line 1 only
                                if line['name'] == 'Line 1':
                                    direction = "OUT" if direction == "IN" else "IN"
