import sys
import logging
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch
import cv2
import glob
import os
import time
from tqdm import tqdm
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
batch_size = 200
resize_scale = 0.25
resize_scale_for_sharpness = 0.25
sharpness_threshold = 100
min_width, min_height = 300 , 300 
margin_factor = 1
similarity_threshold = 1
sharpness_threshold = False 
extract_from_video = False
export_full_frame = True
enable_target_face_detection = False

target_face_path = r"Target_face_path"
input_path = r"Input_Path"
export_path = r"Export_Path"


enable_logging = False  
logging.basicConfig(level=logging.WARNING, handlers=[logging.StreamHandler(sys.stdout)])

mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(model, face_image):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])

    if face_image.mode == 'RGBA':
        face_image = face_image.convert('RGB')

    face_image = transform(face_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(face_image)
    return embedding

def is_target_face(detected_face_embedding, target_face_embedding, threshold):
    dist = torch.nn.functional.pairwise_distance(detected_face_embedding, target_face_embedding)
    logging.info(f"Distance: {dist.item()}")
    return dist.item() < threshold

def is_blurry(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

def process_image(image, filename, mtcnn, export_path, resize_scale, resize_scale_for_sharpness, sharpness_threshold, min_width, min_height, export_full_frame, margin_factor, enable_target_face_detection, target_face_embedding=None):


    logging.info(f"Processing image: {filename}")

    original_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)

    logging.info("Detecting faces...")
    boxes, _ = mtcnn.detect([image])

    if boxes is not None and boxes[0] is not None:
        logging.info(f"Detected {len(boxes[0])} faces in {filename}")
        for box in boxes[0]:
            if box is not None:
                logging.info(f"Original Box: {box}")
                width = box[2] - box[0]
                height = box[3] - box[1]
                margin_x = width * margin_factor / 2
                margin_y = height * margin_factor / 2

                x1 = max(0, int((box[0] - margin_x) / resize_scale))
                y1 = max(0, int((box[1] - margin_y) / resize_scale))
                x2 = min(original_image.shape[1], int((box[2] + margin_x) / resize_scale))
                y2 = min(original_image.shape[0], int((box[3] + margin_y) / resize_scale))
                
                logging.info(f"Adjusted Box: [{x1}, {y1}, {x2}, {y2}]")

                if x2 <= x1 or y2 <= y1:
                    logging.info("Invalid box dimensions, skipping box.")
                    continue

                cropped_face = original_image[y1:y2, x1:x2]
                if cropped_face.size == 0 or cropped_face.shape[1] < min_width or cropped_face.shape[0] < min_height:
                    logging.info("Face too small, skipping.")
                    continue

                resized_face_for_sharpness = cv2.resize(cropped_face, (0, 0), fx=resize_scale_for_sharpness, fy=resize_scale_for_sharpness)
                # Perform sharpness check if sharpness_threshold is not False
                if sharpness_threshold is not False and is_blurry(resized_face_for_sharpness, sharpness_threshold):
                    logging.info("Face is blurry, skipping.")
                    continue

                cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
                detected_face_embedding = get_embedding(model, cropped_face_pil)

                if enable_target_face_detection and target_face_embedding is not None:
                    if is_target_face(detected_face_embedding, target_face_embedding, similarity_threshold):
                        logging.info(f"Target face detected in {filename}.")
                        save_face(original_image, cropped_face, export_path, filename, export_full_frame)
                    else:
                        logging.info(f"No target face match in {filename}.")
                else:
                    save_face(original_image, cropped_face, export_path, filename, export_full_frame)

    else:
        logging.info(f"No faces detected in {filename}.")

def process_video(video_path, mtcnn, export_path, resize_scale, resize_scale_for_sharpness, sharpness_threshold, min_width, min_height, export_full_frame, margin_factor, enable_target_face_detection):
    v_cap = cv2.VideoCapture(video_path)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    original_frames = []
    frames_processed = 0
    start = time.time()

    for j in tqdm(range(v_len), total=v_len, desc="Processing Video", unit="frame"):
        success, frame = v_cap.read()
        if not success:
            continue
        original_frames.append(frame.copy())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
        frames.append(frame)

        if len(frames) >= batch_size or j == v_len - 1:
            boxes, _ = mtcnn.detect(frames)
            frames_processed += len(frames)

            if boxes is not None:
                for i, box in enumerate(boxes):
                    if box is not None:
                        for b in box:
                            logging.info(f"Original Box: {b}")
                            x1, y1, x2, y2 = [max(0, int(b[k] / resize_scale)) for k in range(4)]
                            width = x2 - x1
                            height = y2 - y1
                            margin_x = width * margin_factor / 2
                            margin_y = height * margin_factor / 2

                            x1 = max(0, x1 - int(margin_x))
                            y1 = max(0, y1 - int(margin_y))
                            x2 = min(original_frames[i].shape[1], x2 + int(margin_x))
                            y2 = min(original_frames[i].shape[0], y2 + int(margin_y))

                            logging.info(f"Adjusted Box: [{x1}, {y1}, {x2}, {y2}]")

                            if x2 <= x1 or y2 <= y1:
                                logging.info("Invalid box dimensions, skipping box.")
                                continue
                            cropped_face = original_frames[i][y1:y2, x1:x2]
                            if cropped_face.size == 0 or cropped_face.shape[1] < min_width or cropped_face.shape[0] < min_height:
                                continue

                            resized_face_for_sharpness = cv2.resize(cropped_face, (0, 0), fx=resize_scale_for_sharpness, fy=resize_scale_for_sharpness)
                            if is_blurry(resized_face_for_sharpness, sharpness_threshold):
                                continue

                            cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
                            detected_face_embedding = get_embedding(model, cropped_face_pil)

                            if enable_target_face_detection:
                                if is_target_face(detected_face_embedding, target_face_embedding, similarity_threshold):
                                    logging.info(f"Target face detected in frame {frames_processed + i}.")
                                    save_face(original_frames[i], cropped_face, export_path, f'frame_{frames_processed + i}', export_full_frame)
                                else:
                                    logging.info(f"No target face match in frame {frames_processed + i}.")
                            else:
                                save_face(original_frames[i], cropped_face, export_path, f'frame_{frames_processed + i}', export_full_frame)

            frames = []
            original_frames = []

    v_cap.release()

def save_face(original_image, cropped_face, export_path, filename, export_full_frame):
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    if export_full_frame:
        save_path = os.path.join(export_path, f'{base_filename}.png')
    else:
        save_path = os.path.join(export_path, f'face_{base_filename}.png')
    
    Image.fromarray(cv2.cvtColor(cropped_face if not export_full_frame else original_image, cv2.COLOR_BGR2RGB)).save(save_path)
    logging.info(f"Saved face to {save_path}")

def detect_and_crop_face(image, mtcnn):
    image_rgb = image.convert('RGB')
    boxes, _ = mtcnn.detect(image_rgb)
    if boxes is not None:
        box = boxes[0]
        margin = 20
        x1, y1, x2, y2 = [int(b) for b in box]
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image_rgb.width, x2 + margin)
        y2 = min(image_rgb.height, y2 + margin)
        return image_rgb.crop((x1, y1, x2, y2))
    return image_rgb


def run_detection(mtcnn, input_path, export_path, resize_scale, resize_scale_for_sharpness, sharpness_threshold, min_width, min_height, export_full_frame, margin_factor, enable_target_face_detection, extract_from_video, target_face_path=None):
    target_face_embedding = None

    if enable_target_face_detection:
        if target_face_path is None or not os.path.exists(target_face_path):
            logging.error("Target face detection is enabled, but no valid target face path was provided.")
            return
        else:
            target_face_image = Image.open(target_face_path)
            target_face_image = detect_and_crop_face(target_face_image, mtcnn)
            target_face_embedding = get_embedding(model, target_face_image)
            logging.info("Target face embedding loaded.")
    else:
        logging.info("Target face detection is disabled. Processing all detected faces without similarity check.")

    if os.path.isdir(input_path):
        filenames = glob.glob(os.path.join(input_path, '*.*'))
        for filename in tqdm(filenames, desc="Processing Files", unit="file"):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image = cv2.imread(filename)
                if image is not None:
                    process_image(image, filename, mtcnn, export_path, resize_scale, resize_scale_for_sharpness, sharpness_threshold, min_width, min_height, export_full_frame, margin_factor, enable_target_face_detection, target_face_embedding if enable_target_face_detection else None)

            elif extract_from_video and filename.lower().endswith(('.mp4', '.avi', '.mkv')):
                process_video(filename, mtcnn, export_path, resize_scale, resize_scale_for_sharpness, sharpness_threshold, min_width, min_height, export_full_frame, margin_factor, enable_target_face_detection)
    elif extract_from_video and os.path.isfile(input_path) and input_path.lower().endswith(('.mp4', '.avi', '.mkv')):
        process_video(input_path, mtcnn, export_path, resize_scale, resize_scale_for_sharpness, sharpness_threshold, min_width, min_height, export_full_frame, margin_factor, enable_target_face_detection)



if not os.path.exists(export_path):
    os.makedirs(export_path)

if enable_target_face_detection:
    if os.path.exists(target_face_path):
        target_face_image = Image.open(target_face_path)
        target_face_image = detect_and_crop_face(target_face_image, mtcnn)
        target_face_embedding = get_embedding(model, target_face_image)
        logging.info("Target face embedding loaded.")
    else:
        logging.info(f"Target face image file not found at {target_face_path}")
        exit()
else:
    logging.info("Target face detection is disabled.")

if not os.path.exists(export_path):
    os.makedirs(export_path)

if __name__ == "__main__":
    run_detection(mtcnn, input_path, export_path, resize_scale, resize_scale_for_sharpness, sharpness_threshold, min_width, min_height, export_full_frame, margin_factor, enable_target_face_detection, extract_from_video, target_face_path)

