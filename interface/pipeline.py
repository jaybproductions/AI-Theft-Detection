import cv2

def run_inference(config):
    """
    Starts the real-time video inference loop based on config settings.

    Args:
        config (dict): Configuration dictionary loaded from config.yaml
    """
    input_config = config.get("input", {})
    output_config = config.get("output", {})

    source = input_config.get("source", 0)  # 0 = webcam
    show_video = output_config.get("show_video", True)

    # Open video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video source {source}")
        return

    # Set resolution if provided
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_config.get("frame_width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_config.get("frame_height", 480))
    cap.set(cv2.CAP_PROP_FPS, input_config.get("fps", 30))

    print("✅ Starting video stream... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Unable to read frame. Exiting.")
            break

        # Placeholder: process frame (e.g., action recognition here)

        # Display video
        if show_video:
            cv2.imshow("Theft Detection", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()