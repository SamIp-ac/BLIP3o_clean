# # simplified_image_understanding_mps.py
# import sys
# import os
# from PIL import Image
# import torch
# from transformers import AutoProcessor

# # --- 檢查 MPS 可用性 ---
# if not torch.backends.mps.is_available():
#     if not torch.backends.mps.is_built():
#         print("MPS not available because the current PyTorch install was not built with MPS enabled.")
#     else:
#         print("MPS not available because the current macOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
#     sys.exit(1)
# else:
#     print("MPS is available. Using device: mps")

# # 假設 blip3o 和 qwen_vl_utils 在你的 Python 路徑或同一目錄下
# # 你可能需要調整 import 路徑
# try:
#     from blip3o.model.builder import load_pretrained_model
#     from blip3o.utils import disable_torch_init
#     from qwen_vl_utils import process_vision_info
# except ImportError as e:
#     print(f"Error importing custom modules: {e}")
#     print("Please ensure 'blip3o' and 'qwen_vl_utils' are correctly installed or in the Python path.")
#     sys.exit(1)

# # --- 設定目標設備 ---
# DEVICE = 'mps' # 在 macOS 上使用 MPS

# def process_image(prompt: str, img: Image.Image, model, processor) -> str:
#     """
#     使用 Qwen-VL 模型處理圖片和提示文字，進行圖像理解。
#     """
#     # 1. 準備訊息格式
#     messages = [{
#         "role": "user",
#         "content": [
#             {"type": "image", "image": img},
#             {"type": "text", "text": prompt},
#         ],
#     }]
    
#     # 2. 使用 processor 處理訊息
#     # 注意：apply_chat_template 通常在 CPU 上執行，不需要特別移到 MPS
#     text_prompt_for_qwen = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
    
#     # 3. 處理視覺資訊 (圖片/影片)
#     # process_vision_info 也可能涉及張量操作，但通常輸出是 CPU list
#     # 如果內部有模型運算，可能需要修改原始碼。這裡先假設它兼容。
#     image_inputs, video_inputs = process_vision_info(messages)
    
#     # 4. 將處理後的輸入轉為模型所需的張量
#     inputs = processor(
#         text=[text_prompt_for_qwen],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     ) # 先在 CPU 上建立張量
    
#     # 5. ***關鍵修改*** 將張量移動到 MPS 設備
#     inputs = inputs.to(DEVICE) 
#     print(f"Input tensors moved to device: {inputs.input_ids.device}")

#     # 6. 生成回應 (在 MPS 上執行)
#     print("Generating response...")
#     generated_ids = model.generate(**inputs, max_new_tokens=1024)
#     print(f"Generated IDs device: {generated_ids.device}")
    
#     # 7. 移除輸入部分，只保留生成的部分
#     input_token_len = inputs.input_ids.shape[1]
#     generated_ids_trimmed = generated_ids[:, input_token_len:]
    
#     # 8. 解碼生成的 token 為文字 (通常在 CPU 上執行)
#     # 將結果移回 CPU 進行解碼是常見做法
#     generated_ids_trimmed_cpu = generated_ids_trimmed.to('cpu')
#     output_text = processor.batch_decode(
#         generated_ids_trimmed_cpu, skip_special_tokens=True,
#         clean_up_tokenization_spaces=False
#     )[0]
    
#     return output_text

# def main():
#     """
#     主函數：載入模型，獲取輸入，調用處理函數並輸出結果。
#     """
#     if len(sys.argv) != 2:
#         print("Usage: python simplified_image_understanding_mps.py <path_to_blip3o_model>")
#         sys.exit(1)

#     model_path = os.path.expanduser(sys.argv[1])
    
#     # --- 初始化模型 ---
#     print("Loading model and processor...")
#     disable_torch_init() # 根據原始腳本
#     try:
#         # 載入 BLIP3o 模型組件 (tokenizer, multi_model)
#         tokenizer, multi_model, _ = load_pretrained_model(model_path)
        
#         # ***關鍵修改*** 將模型移動到 MPS 設備
#         multi_model.to(DEVICE)
#         print(f"Model moved to device: {next(multi_model.parameters()).device}")
        
#         # 載入 Qwen-VL processor (processor 本身通常不包含大型模型權重)
#         processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        
#         print("Model and processor loaded successfully.")
#     except Exception as e:
#         print(f"Error loading model or processor: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)

#     # --- 獲取用戶輸入 ---
#     print("\n--- Image Understanding (MPS) ---")
#     prompt = input("Enter your prompt/question about the image: ")
    
#     image_path = input("Enter the path to your image file: ").strip()
#     if not os.path.exists(image_path):
#         print(f"Error: File not found at {image_path}")
#         sys.exit(1)

#     try:
#         # 打開圖片
#         img = Image.open(image_path).convert('RGB') 
#         print(f"Image loaded: {image_path}")
#     except Exception as e:
#         print(f"Error opening image: {e}")
#         sys.exit(1)

#     # --- 處理並獲取結果 ---
#     print("\nProcessing image and generating response...")
#     try:
#         result_text = process_image(prompt, img, multi_model, processor)
#         print("\n--- Model Response ---")
#         print(result_text)
#     except Exception as e:
#         print(f"Error during processing or generation: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)

# if __name__ == "__main__":
#     main()

# image_understanding_gradio_mps.py
import sys
import os
from PIL import Image
import torch
import gradio as gr
from transformers import AutoProcessor
import gc

# === 在模組層級宣告全局變數 ===
model = None
processor = None

# --- 檢查 MPS 可用性 ---
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        error_msg = "MPS not available because the current PyTorch install was not built with MPS enabled."
    else:
        error_msg = "MPS not available because the current macOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."
    
    print(error_msg)
    with gr.Blocks() as demo_error:
        gr.Markdown(f"## Error\n\n{error_msg}")
    demo_error.launch(share=True)
    sys.exit(1)
else:
    print("MPS is available. Using device: mps")

# --- 設定目標設備 ---
DEVICE = 'mps'

# --- 匯入自定義模組 ---
try:
    from blip3o.model.builder import load_pretrained_model
    from blip3o.utils import disable_torch_init
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    error_msg = f"Error importing custom modules: {e}\nPlease ensure 'blip3o' and 'qwen_vl_utils' are correctly installed or in the Python path."
    print(error_msg)
    with gr.Blocks() as demo_import_error:
        gr.Markdown(f"## Import Error\n\n{error_msg}")
    demo_import_error.launch(share=True)
    sys.exit(1)

# --- 圖像理解核心函數 ---
def process_image(prompt: str, img: Image.Image) -> str:
    """
    使用 Qwen-VL 模型處理圖片和提示文字，進行圖像理解。
    """
    global model, processor

    if model is None or processor is None:
        return "Error: Model or processor not loaded yet. Please check server initialization."

    if img is None:
        return "Error: No image uploaded."
    if not prompt.strip():
        return "Error: Please enter a prompt/question."

    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }]

        text_prompt_for_qwen = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text_prompt_for_qwen],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(DEVICE)
        print(f"[Processing] Input tensors moved to device: {inputs.input_ids.device}")

        print("[Processing] Generating response...")
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        print(f"[Processing] Generated IDs device: {generated_ids.device}")

        input_token_len = inputs.input_ids.shape[1]
        generated_ids_trimmed = generated_ids[:, input_token_len:]

        generated_ids_trimmed_cpu = generated_ids_trimmed.to('cpu')
        output_text = processor.batch_decode(
            generated_ids_trimmed_cpu, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[Error] Processing failed: {e}")
        print(error_details)
        return f"Error during processing: {str(e)}\nDetails:\n{error_details}"
    finally:
        # 無論成功或失敗，都嘗試釋放資源
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


# --- 初始化模型 ---
def initialize_model():
    global model, processor

    if len(sys.argv) != 2:
        raise ValueError("Usage: python image_understanding_gradio_mps.py <path_to_blip3o_model>")

    model_path = os.path.expanduser(sys.argv[1])

    print("Loading model and processor...")
    disable_torch_init()  # 禁用不必要的初始化

    try:
        # 載入模型（注意：確認 load_pretrained_model 支援 device='mps'）
        tokenizer, multi_model, _ = load_pretrained_model(model_path, device=DEVICE)

        # 確保模型在 MPS 上
        multi_model = multi_model.to(DEVICE)

        # 載入 Qwen-VL processor
        processor_local = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        # 賦值給全局變數
        model = multi_model
        processor = processor_local

        print("✅ Model and processor loaded successfully.")
        return "Model loaded successfully!"

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


# --- Gradio 介面 ---
with gr.Blocks(title="BLIP3-o Image Understanding (MPS)") as demo:
    gr.Markdown("## BLIP3-o Image Understanding (MPS)")
    
    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.Image(label="Upload Image", type="pil")
            prompt_input = gr.Textbox(
                label="Question/Prompt",
                placeholder="Describe the image you want...",
                lines=2
            )
            run_btn = gr.Button("Analyze Image")
            clear_btn = gr.Button("Clear")

            gr.Examples(
                examples=[
                    [None, "What is the main object in this image?"],
                    [None, "Describe the scene in this image."],
                    [None, "Are there any people in this image? If so, what are they doing?"],
                ],
                inputs=[image_input, prompt_input],
                outputs=None,
                cache_examples=False,
                label="Example Prompts"
            )

        with gr.Column(scale=3):
            output_text = gr.Textbox(label="Model Response", lines=10, interactive=False)

    # 事件綁定
    run_btn.click(fn=process_image, inputs=[prompt_input, image_input], outputs=output_text)
    prompt_input.submit(fn=process_image, inputs=[prompt_input, image_input], outputs=output_text)

    def clear_inputs():
        return None, "", ""

    clear_btn.click(fn=clear_inputs, inputs=[], outputs=[image_input, prompt_input, output_text])


# === 主程式入口 ===
if __name__ == "__main__":
    print("🚀 Starting BLIP3-o Gradio App...")

    try:
        initialize_model()
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        sys.exit(1)

    if model is None or processor is None:
        print("❌ Model or processor is still None after initialization.")
        sys.exit(1)

    print("✅ Ready to launch Gradio interface.")
    demo.launch(server_name="0.0.0.0", server_port=7861)