import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from services.restoration_service import (
    RestorationError,
    RestorationOptions,
    get_model_status,
    restore_photo,
)


PROJECT_ROOT = Path(__file__).resolve().parent


class OldPhotoRestorationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("基于生成对抗网络的老旧照片修复系统")
        self.root.minsize(1080, 720)

        self.source_image: Image.Image | None = None
        self.restored_image: Image.Image | None = None
        self.source_preview: ImageTk.PhotoImage | None = None
        self.result_preview: ImageTk.PhotoImage | None = None
        self.face_enabled = tk.BooleanVar(value=True)
        self.upscale_enabled = tk.BooleanVar(value=True)
        self.scale = tk.StringVar(value="2")
        self.status_text = tk.StringVar(value=self._model_summary())

        self._build_layout()

    def _build_layout(self) -> None:
        header = tk.Frame(self.root)
        header.pack(fill="x", padx=28, pady=(20, 10))
        tk.Label(
            header,
            text="基于生成对抗网络的老旧照片修复系统",
            font=("Microsoft YaHei", 20, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="GFPGAN 人脸修复 + Real-ESRGAN 图像清晰化",
            font=("Microsoft YaHei", 11),
            fg="#555555",
        ).pack(anchor="w", pady=(4, 0))

        settings = ttk.LabelFrame(self.root, text="修复设置", padding=12)
        settings.pack(fill="x", padx=28, pady=(0, 12))
        ttk.Checkbutton(settings, text="启用 GFPGAN 人脸修复", variable=self.face_enabled).grid(
            row=0, column=0, padx=(0, 22), sticky="w"
        )
        ttk.Checkbutton(settings, text="启用 Real-ESRGAN 清晰化", variable=self.upscale_enabled).grid(
            row=0, column=1, padx=(0, 12), sticky="w"
        )
        ttk.Label(settings, text="清晰化倍率:").grid(row=0, column=2, padx=(0, 6), sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.scale,
            values=("2", "4"),
            state="readonly",
            width=5,
        ).grid(row=0, column=3, sticky="w")

        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=28, pady=(0, 12))
        ttk.Button(controls, text="选择照片", command=self.select_photo).pack(side="left")
        self.restore_button = ttk.Button(controls, text="开始修复", command=self.run_restoration)
        self.restore_button.pack(side="left", padx=10)
        ttk.Button(controls, text="保存结果", command=self.save_result).pack(side="left")
        tk.Label(controls, textvariable=self.status_text, fg="#245c32", anchor="e").pack(
            side="right", fill="x", expand=True
        )

        images = tk.Frame(self.root)
        images.pack(fill="both", expand=True, padx=28, pady=(0, 28))
        self.source_label = self._create_image_panel(images, "原始照片", 0)
        self.result_label = self._create_image_panel(images, "修复结果", 1)

    @staticmethod
    def _create_image_panel(parent: tk.Frame, title: str, column: int) -> tk.Label:
        panel = tk.Frame(parent, bd=1, relief="solid")
        panel.grid(row=0, column=column, padx=8, sticky="nsew")
        parent.grid_columnconfigure(column, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        tk.Label(panel, text=title, font=("Microsoft YaHei", 13, "bold")).pack(pady=(10, 8))
        label = tk.Label(panel, text="暂无图片", bg="#f4f4f4")
        label.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        return label

    def select_photo(self) -> None:
        selected_path = filedialog.askopenfilename(
            title="选择待修复照片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.webp"), ("所有文件", "*.*")],
        )
        if not selected_path:
            return
        try:
            self.source_image = Image.open(selected_path).convert("RGB")
        except OSError as error:
            messagebox.showerror("无法打开图片", str(error))
            return

        self.restored_image = None
        self._show_image(self.source_image, self.source_label, "source")
        self.result_label.configure(image="", text="等待修复")
        self.status_text.set("照片已载入，请选择修复模块。")

    def run_restoration(self) -> None:
        if self.source_image is None:
            messagebox.showwarning("提示", "请先选择一张照片。")
            return

        options = RestorationOptions(
            face_enhancement=self.face_enabled.get(),
            upscaling=self.upscale_enabled.get(),
            scale=int(self.scale.get()),
        )
        self.restore_button.configure(state="disabled")
        self._set_progress("正在准备模型...")
        try:
            self.restored_image, steps = restore_photo(
                self.source_image,
                options,
                progress_callback=self._set_progress,
            )
        except RestorationError as error:
            self.status_text.set("修复未完成。")
            messagebox.showerror("模型运行提示", str(error))
            return
        except Exception as error:
            self.status_text.set("修复未完成。")
            messagebox.showerror("修复失败", f"发生未预期错误：\n{error}")
            return
        finally:
            self.restore_button.configure(state="normal")

        self._show_image(self.restored_image, self.result_label, "result")
        self.status_text.set("已完成：" + " + ".join(steps))

    def save_result(self) -> None:
        if self.restored_image is None:
            messagebox.showwarning("提示", "请先完成照片修复。")
            return

        output_dir = PROJECT_ROOT / "data" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = filedialog.asksaveasfilename(
            title="保存修复结果",
            initialdir=output_dir,
            initialfile="restored_photo.png",
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("JPEG 图片", "*.jpg")],
        )
        if save_path:
            self.restored_image.save(save_path)
            self.status_text.set("修复结果已保存。")

    def _model_summary(self) -> str:
        status = get_model_status()
        weights = "权重已就绪" if status["gfpgan_weight"] and status["realesrgan_weight"] else "模型权重缺失"
        packages = "推理依赖已就绪" if status["gfpgan_package"] and status["realesrgan_package"] else "待安装推理依赖"
        return f"{weights}；{packages}"

    def _set_progress(self, message: str) -> None:
        self.status_text.set(message)
        self.root.update_idletasks()

    def _show_image(self, image: Image.Image, label: tk.Label, kind: str) -> None:
        preview = image.copy()
        preview.thumbnail((500, 500))
        photo = ImageTk.PhotoImage(preview)
        label.configure(image=photo, text="")
        if kind == "source":
            self.source_preview = photo
        else:
            self.result_preview = photo


if __name__ == "__main__":
    window = tk.Tk()
    OldPhotoRestorationApp(window)
    window.mainloop()
