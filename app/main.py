import numpy as np
from PIL import Image
from flask import Blueprint, request, jsonify
from pathlib import Path
import boto3
from datetime import datetime
from .feature_extractor import FeatureExtractor
from config import AWS_BUCKET_NAME, AWS_REGION

app = Blueprint('app', __name__)

# Initialize AWS S3 client
s3 = boto3.client('s3', region_name=AWS_REGION)

# Initialize feature extractor
fe = FeatureExtractor()

# Load primary dataset features and metadata
features, img_paths, img_names = [], [], []
primary_features_s3_prefix = "datasets/primary/features/"
primary_images_s3_prefix = "datasets/primary/images/"

# Load secondary dataset for deep_search
deep_features, deep_img_paths, deep_img_names = [], [], []
secondary_features_s3_prefix = "datasets/secondary/features/"
secondary_images_s3_prefix = "datasets/secondary/images/"

def load_s3_dataset(s3_prefix, features_list, img_paths_list, img_names_list):
    """Load dataset features and metadata from S3."""
    for obj in s3.list_objects_v2(Bucket=AWS_BUCKET_NAME, Prefix=s3_prefix).get('Contents', []):
        if obj['Key'].endswith('.npy'):
            feature_local_path = f"/tmp/{Path(obj['Key']).name}"
            s3.download_file(AWS_BUCKET_NAME, obj['Key'], feature_local_path)
            features_list.append(np.load(feature_local_path))

            img_key = obj['Key'].replace("features", "images").replace(".npy", ".jpg")
            img_paths_list.append(img_key)
            img_names_list.append(Path(img_key).name)

load_s3_dataset(primary_features_s3_prefix, features, img_paths, img_names)
load_s3_dataset(secondary_features_s3_prefix, deep_features, deep_img_paths, deep_img_names)

features = np.array(features)
deep_features = np.array(deep_features)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API is running"}), 200

@app.route('/search', methods=['POST'])
def search():
    return handle_search(features, img_paths, img_names, "uploads/search/")

@app.route('/deep_search', methods=['POST'])
def deep_search():
    return handle_search(deep_features, deep_img_paths, deep_img_names, "uploads/deep_search/")

def handle_search(feature_set, img_paths, img_names, s3_upload_prefix):
    if 'query_img' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['query_img']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save query image to /tmp
        img = Image.open(file.stream)
        local_img_path = f"/tmp/{file.filename}"
        img.save(local_img_path)

        # Upload query image to S3
        s3_upload_path = f"{s3_upload_prefix}{datetime.now().isoformat().replace(':', '.')}_{file.filename}"
        s3.upload_file(local_img_path, AWS_BUCKET_NAME, s3_upload_path)

        # Generate query image S3 URL
        query_image_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_upload_path}"

        # Run feature extraction
        query = fe.extract(img)
        dists = np.linalg.norm(feature_set - query, axis=1)

        # Calculate similarities
        similarities = [100 / (1 + dist) for dist in dists]
        results = [
            {
                'similarity': float(similarity),
                'image_name': img_names[id],
                'image_url': f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{img_paths[id]}"
            }
            for id, similarity in enumerate(similarities)
        ]

        # Sort results by similarity
        results.sort(reverse=True, key=lambda x: x['similarity'])
        results = results[:50]

        return jsonify({'query_image_url': query_image_url, 'results': results}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
