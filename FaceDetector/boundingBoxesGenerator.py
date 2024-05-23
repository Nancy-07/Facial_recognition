import os
import cv2
import pandas as pd
from mtcnn import MTCNN

def numeric_sort_key(s):
    """ Helper function to sort strings with numeric values correctly. """
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def process_batch(detector, image_folder, batch):
    data = []
    for img_name in batch:
        img_path = os.path.join(image_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image {img_path}")
            continue
        try:
            faces = detector.detect_faces(image)
            for face in faces:
                x, y, width, height = face['box']
                data.append([img_name, x, y, width, height])
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            raise  # Re-raise the exception to handle it in the outer function
    return data

def generate_bounding_boxes(image_folder, output_csv):
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
    else:
        df = pd.DataFrame(columns=['Image', 'X', 'Y', 'Width', 'Height'])

    detector = MTCNN()
    images = os.listdir(image_folder)
    images.sort(key=numeric_sort_key)  # Sort images numerically

    for i in range(0, len(images), 10):
        batch = images[i:i+10]
        successful = False
        while not successful:
            try:
                batch_data = process_batch(detector, image_folder, batch)
                successful = True
            except Exception as e:
                print(f"Error processing batch starting at image {images[i]}: {e}. Retrying...")

        # Convert batch data to DataFrame and append to the main DataFrame
        batch_df = pd.DataFrame(batch_data, columns=['Image', 'X', 'Y', 'Width', 'Height'])
        df = pd.concat([df, batch_df])
        df.to_csv(output_csv, index=False)  # Save after each successful batch
        print(f"Saved up to image {batch[-1]}")

if __name__ == "__main__":
    dataset = "lfw"
    csvFile = "bounding_boxes.csv"
    generate_bounding_boxes(dataset, csvFile)