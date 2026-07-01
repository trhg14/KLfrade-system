import cv2
import os
import random
from pathlib import Path
from tqdm import tqdm


def preprocess_image_clahe(
    image_path, target_size=(640, 640), use_blur=True, clahe_clip_limit=2.0
):
    """
    Preprocess image with CLAHE and optional Gaussian Blur.
    """
    # Read image as grayscale
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image from {image_path}")

    # Store original size for label scaling
    original_size = image.shape[:2]

    # Resize image
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)

    if use_blur:
        # Apply Gaussian Blur (noise reduction)
        image_processed = cv2.GaussianBlur(image_resized, (5, 5), 0)
    else:
        image_processed = image_resized

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image_processed)

    return enhanced_image, original_size


def scale_bounding_box(bbox, original_size, new_size=(640, 640)):
    """
    Scale bounding box coordinates from original size to new size.
    """
    x_center, y_center, width, height = bbox

    # Convert to pixel coordinates
    x_center_pixel = x_center * original_size[1]  # width
    y_center_pixel = y_center * original_size[0]  # height
    width_pixel = width * original_size[1]
    height_pixel = height * original_size[0]

    # Scale to new size
    x_scale = new_size[0] / original_size[1]
    y_scale = new_size[1] / original_size[0]

    x_center_scaled = x_center_pixel * x_scale
    y_center_scaled = y_center_pixel * y_scale
    width_scaled = width_pixel * x_scale
    height_scaled = height_pixel * y_scale

    # Convert back to normalized coordinates
    x_center_new = x_center_scaled / new_size[0]
    y_center_new = y_center_scaled / new_size[1]
    width_new = width_scaled / new_size[0]
    height_new = height_scaled / new_size[1]

    return [x_center_new, y_center_new, width_new, height_new]


def process_labels(label_path, original_size, new_size=(640, 640)):
    """
    Process and scale labels for resized images.
    """
    with open(label_path, "r") as file:
        lines = file.readlines()

    new_labels = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])

        scaled_bbox = scale_bounding_box(
            [x_center, y_center, width, height], original_size, new_size
        )

        new_labels.append(f"{class_id} {' '.join(map(str, scaled_bbox))}")

    return new_labels


def flip_image_and_labels(image, labels):
    """
    Flip image horizontally and adjust labels.
    """
    flipped_image = cv2.flip(image, 1)

    flipped_labels = []
    for label in labels:
        parts = label.split()
        class_id = parts[0]
        x_center, y_center, width, height = map(float, parts[1:5])

        # Flip x_center
        x_center_flipped = 1.0 - x_center

        flipped_labels.append(
            f"{class_id} {x_center_flipped:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return flipped_image, flipped_labels


def count_class_distribution(label_dir, num_classes=5):
    """
    Count class distribution in dataset.
    """
    class_counts = {i: 0 for i in range(num_classes)}

    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, "r") as f:
                for line in f:
                    try:
                        class_id = int(line.split()[0])
                        if class_id in class_counts:
                            class_counts[class_id] += 1
                    except (ValueError, IndexError):
                        pass

    return class_counts


def balance_dataset_with_flip(
    img_dir,
    label_dir,
    output_img_dir,
    output_label_dir,
    aux_label_dir=None,
    output_aux_label_dir=None,
    num_classes=5,
    target_size=(640, 640),
    use_blur=True,
    clahe_clip_limit=2.0,
    augment_minority=True,
):
    """
    Balance dataset by flipping augmentation for minority classes.
    Supports processing an auxiliary label directory (e.g., knee labels) in sync.
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    if aux_label_dir and output_aux_label_dir:
        os.makedirs(output_aux_label_dir, exist_ok=True)
        print(f"✅ Auxiliary labels enabled: {aux_label_dir} -> {output_aux_label_dir}")

    # Count class distribution
    print("\n📊 Analyzing class distribution...")
    class_counts = count_class_distribution(label_dir, num_classes)

    print("\nOriginal class distribution:")
    for class_id, count in class_counts.items():
        print(f"  Class {class_id}: {count} instances")

    if not any(class_counts.values()):
        print("Warning: No labels found or empty distribution.")
        return

    max_count = max(class_counts.values())
    print(f"\nTarget count: {max_count} instances per class")

    # Process all images
    image_files = [
        f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    print(
        f"\n🔄 Processing {len(image_files)} images (Blur={use_blur}, CLAHE={clahe_clip_limit})..."
    )

    for img_file in tqdm(image_files):
        img_path = os.path.join(img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            continue

        # Preprocess image with CLAHE
        enhanced_image, original_size = preprocess_image_clahe(
            img_path, target_size, use_blur=use_blur, clahe_clip_limit=clahe_clip_limit
        )

        # Process primary labels
        new_labels = process_labels(label_path, original_size, target_size)

        # Process auxiliary labels if provided
        new_aux_labels = None
        if aux_label_dir and output_aux_label_dir:
            aux_label_path = os.path.join(aux_label_dir, label_file)
            if os.path.exists(aux_label_path):
                new_aux_labels = process_labels(
                    aux_label_path, original_size, target_size
                )

        # Save processed image (as grayscale PNG)
        base_name = os.path.splitext(img_file)[0]
        output_img_path = os.path.join(output_img_dir, base_name + ".png")
        cv2.imwrite(output_img_path, enhanced_image)

        # Save processed labels
        output_label_path = os.path.join(output_label_dir, label_file)
        with open(output_label_path, "w") as f:
            f.write("\n".join(new_labels))

        # Save processed aux labels
        if new_aux_labels is not None and output_aux_label_dir:
            output_aux_label_path = os.path.join(output_aux_label_dir, label_file)
            with open(output_aux_label_path, "w") as f:
                f.write("\n".join(new_aux_labels))

    if not augment_minority:
        print("\n🚫 Skipping minority class augmentation (Natural Distribution).")
        return

    # Augment minority classes
    print("\n🔄 Augmenting minority classes with flip...")

    current_counts = count_class_distribution(output_label_dir, num_classes)

    for class_id in range(num_classes):
        needed = max_count - current_counts[class_id]

        if needed <= 0:
            print(f"  Class {class_id}: Already balanced")
            continue

        print(f"  Class {class_id}: Need {needed} more instances")

        # Find images with this class
        images_with_class = []
        for label_file in os.listdir(output_label_dir):
            if not label_file.endswith(".txt"):
                continue

            label_path = os.path.join(output_label_dir, label_file)
            with open(label_path, "r") as f:
                labels = f.readlines()
                if any(int(line.split()[0]) == class_id for line in labels):
                    img_file = os.path.splitext(label_file)[0] + ".png"
                    img_path = os.path.join(output_img_dir, img_file)
                    if os.path.exists(img_path):
                        images_with_class.append((img_path, label_path))

        if not images_with_class:
            print(f"    ⚠️  No images found for class {class_id}")
            continue

        # Augment by flipping
        random.shuffle(images_with_class)
        augmented = 0
        idx = 0

        while augmented < needed and idx < len(images_with_class) * 10:  # Max 10 rounds
            img_path, label_path = images_with_class[idx % len(images_with_class)]
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_file_name = base_name + ".txt"
            idx += 1

            # Read image and labels
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            with open(label_path, "r") as f:
                labels = [line.strip() for line in f.readlines()]

            # Filter labels for current class only
            class_labels = [l for l in labels if int(l.split()[0]) == class_id]

            if not class_labels:
                continue

            # Flip image and labels
            flipped_image, flipped_labels = flip_image_and_labels(image, class_labels)

            # Handle Aux Labels Flip
            flipped_aux_labels = None
            if aux_label_dir and output_aux_label_dir:
                aux_label_path = os.path.join(
                    output_aux_label_dir, label_file_name
                )  # Use output dir as source since strictly syncing
                if os.path.exists(aux_label_path):
                    with open(aux_label_path, "r") as f:
                        aux_labels_content = [line.strip() for line in f.readlines()]
                    # Flip aux labels (using same image flip logic, just different labels)
                    _, flipped_aux_labels = flip_image_and_labels(
                        image, aux_labels_content
                    )

            # Save augmented data
            aug_base_name = f"{base_name}_flip_{augmented}"
            aug_img_name = f"{aug_base_name}.png"
            aug_label_name = f"{aug_base_name}.txt"

            aug_img_path = os.path.join(output_img_dir, aug_img_name)
            aug_label_path = os.path.join(output_label_dir, aug_label_name)

            cv2.imwrite(aug_img_path, flipped_image)
            with open(aug_label_path, "w") as f:
                f.write("\n".join(flipped_labels))

            if flipped_aux_labels is not None and output_aux_label_dir:
                aug_aux_path = os.path.join(output_aux_label_dir, aug_label_name)
                with open(aug_aux_path, "w") as f:
                    f.write("\n".join(flipped_aux_labels))

            augmented += len(flipped_labels)

        print(f"    ✅ Augmented {augmented} instances")
