import os
import json
import random
import datetime
import time
import re
import uuid
from urllib import request, error
from dotenv import load_dotenv

# 加载.env配置
load_dotenv()

# ================== 从环境变量读取配置 ==================
api_workflow_dir = os.getenv("API_WORKFLOW_DIR")
lora_dir = os.getenv("LORA_DIR")
api_workflow_file = os.getenv("API_WORKFLOW_FILE")
api_endpoint = os.getenv("API_ENDPOINT")
comfyui_output_dir = os.getenv("COMFYUI_OUTPUT_DIR")
total_images = int(os.getenv("TOTAL_IMAGES", 3000))  # 默认3000张

# 拼接API地址（http://+地址+/prompt）
api_endpoint = f"http://{api_endpoint}/prompt"

# 工作流文件完整路径
workflow_file_path = os.path.join(api_workflow_dir, api_workflow_file)
# 加载工作流JSON
with open(workflow_file_path, "r", encoding="utf-8") as f:
    workflow = json.load(f)

# 输出目录（按当前时间创建子目录）
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
relative_output_path = current_datetime  # 相对ComfyUI输出目录的路径

# 超时设置（秒）
directory_creation_timeout = 3000  # 等待目录创建超时
image_generation_timeout = 30000   # 等待图片生成超时

# ================== 提示词配置 ==================
# 缺陷描述
DEFECT_DETAIL = [
    "insulator miss_1",
    "insulator miss_2",
    "insulator miss_3",
    "insulator miss_more",
]

# 背景环境
BACKGROUND = [
    "brown earthy outdoor backdrop",
    "lush green outdoor backdrop",
    "mountainous outdoor backdrop",
    "overcast grey outdoor backdrop",
]

# 固定前缀
PROMPT_PREFIX = "k4s4"

# ================== 工具函数 ==================
def queue_prompt(workflow_dict):
    """提交工作流到ComfyUI API"""
    global api_endpoint
    url = api_endpoint

    # 构建请求数据（包含client_id，新版ComfyUI必需）
    payload = {
        "prompt": workflow_dict,
        "client_id": "insulator-generator-" + str(uuid.uuid4())
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    # 禁用代理，避免连接问题
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    proxy_handler = request.ProxyHandler({})
    opener = request.build_opener(proxy_handler)

    try:
        req = request.Request(url, data=data, headers=headers, method="POST")
        with opener.open(req, timeout=60) as response:
            status = response.getcode()
            if status == 200:
                print("[API] 工作流提交成功")
                return True
            else:
                print(f"[API] 提交失败，状态码：{status}")
                return False
    except error.HTTPError as e:
        print(f"[API错误] HTTP错误 {e.code}：{e.reason}")
        return False
    except error.URLError as e:
        print(f"[API错误] 连接错误：{e.reason}")
        return False
    except Exception as e:
        print(f"[API错误] 未知错误：{str(e)}")
        return False


def wait_for_directory_creation(directory, timeout):
    """等待输出目录创建"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(directory):
            print(f"输出目录已创建：{directory}")
            return True
        time.sleep(5)
    print(f"超时！未找到目录：{directory}")
    return False


def wait_for_images(image_folder, expected_count, timeout):
    """等待所有图片生成"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(image_folder):
            # 统计png图片数量
            image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
            if len(image_files) >= expected_count:
                print(f"已找到所有{expected_count}张图片")
                return True
        time.sleep(5)
    print(f"超时！未生成所有{expected_count}张图片")
    return False


# ================== 核心生成逻辑 ==================
def generate_images():
    target_step = 300  # 目标LoRA步数
    print(f"开始生成图片：目标LoRA步数={target_step}，总数量={total_images}")

    # 查找300步的LoRA文件
    target_lora = None
    for filename in os.listdir(lora_dir):
        if filename.endswith("_lora.safetensors") and f"checkpoint-{target_step}" in filename:
            target_lora = filename
            break

    if not target_lora:
        print(f"错误：未在{lora_dir}找到{target_step}步的LoRA文件！")
        return 0

    # 构建LoRA绝对路径（关键：使用绝对路径避免ComfyUI路径解析问题）
    lora_absolute_path = os.path.join(lora_dir, target_lora)
    print(f"使用LoRA文件：{lora_absolute_path}")

    # 更新工作流中的LoraLoader节点（276号）
    workflow["276"]["inputs"]["lora_name"] = lora_absolute_path
    # 确保允许自定义路径（工作流中已设置，再次确认）
    workflow["276"]["_meta"]["allow_custom_paths"] = True

    # 配置CLIP和VAE的dtype（避免精度问题）
    workflow["11"]["inputs"]["dtype"] = "torch.float16"
    workflow["500"]["inputs"]["dtype"] = "torch.float16"

    # 循环生成图片
    for i in range(total_images):
        # 随机生成提示词后缀
        random_defect = random.choice(DEFECT_DETAIL)
        random_bg = random.choice(BACKGROUND)
        prompt_suffix = f"{random_defect}, connected to wires against {random_bg},"

        # 更新提示词节点（285号是动态文本部分）
        workflow["285"]["inputs"]["text"] = prompt_suffix

        # 配置采样参数
        workflow["291"]["inputs"]["steps"] = 30  # 采样步数
        workflow["291"]["inputs"]["cfg"] = 4.0   # 引导系数
        random_seed = random.randint(1, 10**18)  # 随机种子
        workflow["291"]["inputs"]["seed"] = random_seed

        # 配置输出文件名
        filename_prefix = f"insulator_step{target_step}_img{i+1:04d}_seed{random_seed}"
        workflow["314"]["inputs"]["output_path"] = relative_output_path
        workflow["314"]["inputs"]["filename_prefix"] = filename_prefix

        # 打印当前进度
        full_prompt = f"{PROMPT_PREFIX}, {prompt_suffix}"
        print(f"\n===== 第{i+1}/{total_images}张 =====")
        print(f"种子：{random_seed}")
        print(f"提示词：{full_prompt}")
        print(f"输出文件名：{filename_prefix}")

        # 提交工作流到ComfyUI
        success = queue_prompt(workflow)
        if not success:
            print(f"警告：第{i+1}张图片提交失败，将继续尝试下一张")

    return total_images


# ================== 主入口 ==================
if __name__ == "__main__":
    # 执行生成
    expected_count = generate_images()

    # 等待输出目录和图片生成
    absolute_output_path = os.path.join(comfyui_output_dir, relative_output_path)
    print(f"\n输出目录：{absolute_output_path}")

    # 等待目录创建
    if wait_for_directory_creation(absolute_output_path, directory_creation_timeout):
        # 等待所有图片生成
        wait_for_images(absolute_output_path, expected_count, image_generation_timeout)
    else:
        print("输出目录未创建，无法继续等待图片生成")

    print("生成任务结束")