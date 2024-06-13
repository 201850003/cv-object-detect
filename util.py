import cv2
import os
import glob


def create_video_from_images(image_folder, output_video_path, fps=30, frame_size=None):
    """
    Create a video from a series of images.

    :param image_folder: The path to the folder containing images.
    :param output_video_path: The path where the output video will be saved.
    :param fps: Frames per second for the video.
    :param frame_size: The size of each frame (width, height).
    """
    # Get all images in the folder
    images = sorted(glob.glob(os.path.join(image_folder, '*.jpg')), key=os.path.getmtime)
    if not images:
        raise ValueError("No images found in the specified directory")

    # Define the codec and create VideoWriter object
    if not frame_size:
        # Read the first image to get the frame size automatically
        test_img = cv2.imread(images[0])
        frame_size = (test_img.shape[1], test_img.shape[0])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Read each image and write it to the video
    for image_file in images:
        img = cv2.imread(image_file)
        if img is None:
            continue
        img = cv2.resize(img, frame_size)  # Resize image if necessary
        out.write(img)

    # Release everything when job is finished
    out.release()
    print(f"Video saved as {output_video_path}")


# Example usage
if __name__ == "__main__":
    create_video_from_images('./q4t-output', 'output_video3.mp4', fps=30)
