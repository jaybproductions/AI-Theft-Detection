import cv2

def draw_prediction(frame, label="Suspicious", confidence=0.0, color=(0, 0, 255)):
    """
    Draws a prediction label on the video frame.

    Args:
        frame (np.array): The current frame from the video stream.
        label (str): The classification label (e.g., "Suspicious", "Normal").
        confidence (float): Confidence score between 0 and 1.
        color (tuple): BGR color of the label text (default red).
    
    Returns:
        np.array: Annotated frame
    """
    text = f"{label}: {confidence * 100:.1f}%"
    position = (10, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # Draw a filled rectangle behind the text
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (position[0] - 5, position[1] - 25),
                  (position[0] + text_width + 5, position[1] + 5), (0, 0, 0), -1)

    # Draw the text
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return frame
