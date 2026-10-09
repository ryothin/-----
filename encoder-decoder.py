from flask import Flask, render_template, request, send_from_directory, jsonify, redirect, url_for
import os
import time
import base64
from PIL import Image
from diffusers.utils import load_image
from controlnet_aux import PidiNetDetector
import logging
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import torch
import asyncio
import aiohttp
import re

# ここからSDXL＋ControlNetのライブラリ読み込み
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

# Flaskアプリケーションの設定
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['CONTROLNET_OUTPUT_FOLDER'] = 'controlnet_outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['CONTROLNET_OUTPUT_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)
executor = ThreadPoolExecutor(max_workers=8)

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# PidiNetDetectorの初期化
processor = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
processor.to(device)

# ログをexcelに出力するための関数
def log_to_excel(file_name, proccessing_time, original_size, lineart_size, excel_path='processing_log.xlsx'):
    log_data = {
        "File Name": [file_name],
        "Processing Time (s)": [proccessing_time],
        "Original Size (bytes)": [original_size],
        "Lineart Size (bytes)": [lineart_size]
    }
    df = pd.DataFrame(log_data)
    if not os.path.exists(excel_path): 
        #　ファイルがなければ新規作成して書き込む
        df.to_excel(excel_path, index=False, engine='openpyxl')
    else:
        # ファイルが存在する場合は追記する
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, index=False, header=False, startrow=writer.sheets.get('Sheet1') is None)

def encode_image(image_path):
    """画像をbase64にエンコードする関数"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def resize_image(image_path, target_pixels = 1024 * 1024):
    """画像を指定されたピクセル数にリサイズする関数"""
    with Image.open(image_path) as img:
        original_width, original_height = img.size
        original_pixels = original_width * original_height
        # 面積がtarget_pixelsになるようなリサイズ比率を計算
        scaling_factor = (target_pixels / original_pixels) ** 0.5
        new_width = max(1, int(original_width * scaling_factor))
        new_height = max(1, int(original_height * scaling_factor))
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img.save(image_path) # 元のファイルに上書き保存

def check_image_format(image_path):
    """画像の形式をチェックする関数"""
    try:
        with Image.open(image_path) as img:
            img_format = img.format
            if img_format not in ['JPEG', 'PNG']:
                # JPEGまたはPNG以外の形式の場合、PNGに変換
                raise ValueError("対応していない画像形式です。JPEGまたはPNGを使用してください。")
    except Exception as e:
        raise ValueError(f"無効な画像ファイルです: {e}")
    return img_format

def process_image_manual(image_path, prompt, negative_prompt, result):
    """画像の前処理を行う関数"""
    try:
        check_image_format(image_path)
        resize_image(image_path)

        if not prompt:
            prompt = "a photo, high quality, realistic, detailed"
        if not negative_prompt: 
            negative_prompt = (
                "(worst quality, low quality:2), (painting, sketch, flat color:1.3), "
                "monochrome, grayscale, bad anatomy, unrealistic, blurry, "
                "too dark, too bright, poor composition"
            )

            result['prompt'] = prompt
            result['negative_prompt'] = negative_prompt

    except Exception as e:
        logging.error(f"前処理中にエラー: {e}")
        result['error'] = str(e)

def generate_lineart(image_path, output_path, prompt_info, negative_prompt_info):
    try:
        # 現画像の読み込み
        init_image = load_image(image_path)
        control_image = processor(init_image, safe = False)

        # 生成された線画をpng形式で一時保存して、RGBのjpgに変換する
        temp_output_path = output_path.replace('.jpg', '_temp.png')
        control_image.save(temp_output_path, format='PNG')

        with Image.open(temp_output_path) as img:
            rgb_img = img.convert('RGB')
            rgb_img.save(output_path, format ='JPEG')

        # EXIFにプロンプトを埋め込む
        exif_data = rgb_img.getexif()
        exif_data[0x9286] = (
            f"Prompt: {prompt_info}\n"
            f"Negative Prompt: {negative_prompt_info}"
        )
        rgb_img.save(output_path, format="JPEG", exif=exif_data)

        # 一時的なファイルを削除
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

        
    except Exception as e:
        logging.error(f"線画生成中にエラー: {e}")
        raise e

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start_time = time.time()

        # ファイルのアップロード処理と検証
        if 'image' not in request.files:
            return "ファイルが選択されていません。"
        file = request.files['image']
        if file.filename == '':
            return "ファイルが選択されていません。"

        if file:
            filename = secure_filename(file.filename)
            if not filename:
                return "無効なファイル名です。"
            name, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext not in ["jpg", ".jpeg", "png"]:
                return "対応していない画像形式です。JPEGまたはPNGを使用してください。"

            # アップロードされたファイルを保存
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path) 

            try:
                # 現画像のサイズを取得  
                original_size = os.path.getsize(image_path)

                # フォームからプロンプトを取得
                prompt_in = request.form.get('prompt', '')
                negative_in = request.form.get('negative_prompt', '')
                result = {}
                process_image_manual(image_path, prompt_in, negative_in, result)
                if 'error' in result:
                    return f"前処理中にエラーが発生しました: {result['error']}"
                prompt = result["prompt"]
                negative_prompt = result["negative_prompt", ""]

                # 線画の生成とプロンプトをexifに埋め込む
                output_filename = f"{name}_lineart.jpg"
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                generate_lineart(image_path, output_path, prompt, negative_prompt)

                # 処理ログをエクセルに記録
                elapsed_time = time.time() - start_time
                lineart_size = os.path.getsize(output_path)
                log_to_excel(output_filename, elapsed_time, original_size, lineart_size)

                # 結果ページを表示
                return render_template(
                    'result.html',
                    lineart_image=output_filename,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    total_time=f"{elapsed_time:.2f}"
                )
            except Exception as e:
                logging.error(f"エラーが発生しました: {e}")
                return str(e)

    return render_template('index.html')

#　線画をwebに返す
@app.route("/outputs/<path:filename>")
def output_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)