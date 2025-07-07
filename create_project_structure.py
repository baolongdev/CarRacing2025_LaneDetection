import os

base_path = "D:/UIT_Cuoc Thi/CarRacing2025"

folders_and_files = {
    "logs": ["log.txt"],
    "models": ["__init__.py", "controller.py"],
    "utils": ["__init__.py", "image_utils.py"],
    "data": ["__init__.py", "dataset_loader.py"],
    "images": [],
}

template_files = {
    "controller.py": "# Đây là nơi khai báo controller (MPC, PID, v.v.)\n\nclass Controller:\n    def __init__(self):\n        pass\n",
    "image_utils.py": "# Hàm tiện ích xử lý ảnh\n\ndef normalize_image(img):\n    pass\n",
    "dataset_loader.py": "# Load dữ liệu mô phỏng\n\ndef load_data():\n    pass\n",
    "log.txt": "",
    "__init__.py": "",
}

def create_project_structure(base):
    for folder, files in folders_and_files.items():
        folder_path = os.path.join(base, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                content = template_files.get(file_name, "")
                f.write(content)
            print(f"  Created file: {file_path}")

    # Đảm bảo có main.py
    main_path = os.path.join(base, "main.py")
    if not os.path.exists(main_path):
        with open(main_path, "w", encoding="utf-8") as f:
            f.write("# Main entry point\n\nif __name__ == \"__main__\":\n    print(\"Starting simulation...\")\n")
        print(f"Created file: {main_path}")

if __name__ == "__main__":
    create_project_structure(base_path)
