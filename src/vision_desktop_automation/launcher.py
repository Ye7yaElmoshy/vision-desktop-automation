"""
Tkinter launcher GUI for Vision Desktop Automation.

Provides a simple form to override configuration defaults (target, max posts,
search mode, and template matching image) before launching the main automation
process.

Run this file as a script to launch the GUI:
    python -m vision_desktop_automation.launcher
"""

import os
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


class LauncherApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Vision Desktop Automation")
        self.root.minsize(600, 420)
        
        # Form variables
        self.api_key_var = tk.StringVar()
        self.target_type_var = tk.StringVar(value="notepad")
        self.custom_target_var = tk.StringVar()
        self.posts_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="default")
        self.template_path_var = tk.StringVar()
        
        self._build_ui()
        
    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding="16")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        current_row = 0
        
        # ========== ROW 0: API Key Label ==========
        ttk.Label(
            main_frame, 
            text="Gemini API key :"
        ).grid(row=current_row, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 0))
        current_row += 1
        
        # ========== ROW 1: API Key Entry ==========
        ttk.Entry(main_frame, textvariable=self.api_key_var, show="*").grid(
            row=current_row, column=0, columnspan=2, sticky="we", padx=4, pady=(0, 4)
        )
        current_row += 1
        
        
        # ========== ROW 3: Target Icon Label & Radio Buttons ==========
        ttk.Label(main_frame, text="Target icon:").grid(
            row=current_row, column=0, sticky="w", padx=4, pady=4
        )
        target_frame = ttk.Frame(main_frame)
        target_frame.grid(row=current_row, column=1, sticky="w", padx=4, pady=4)
        ttk.Radiobutton(
            target_frame, 
            text="Notepad", 
            variable=self.target_type_var, 
            value="notepad"
        ).pack(side="left")
        ttk.Radiobutton(
            target_frame, 
            text="Others", 
            variable=self.target_type_var, 
            value="others"
        ).pack(side="left", padx=8)
        current_row += 1
        
        # ========== ROW 4: Custom Target Entry (conditional) ==========
        ttk.Label(main_frame, text="Icon name / target description:").grid(
            row=current_row, column=0, sticky="w", padx=4, pady=4
        )
        self.custom_target_entry = ttk.Entry(
            main_frame, 
            textvariable=self.custom_target_var, 
            width=45
        )
        self.custom_target_entry.grid(row=current_row, column=1, sticky="we", padx=4, pady=4)
        current_row += 1
        
        # ========== ROW 5: Template Image ==========
        ttk.Label(main_frame, text="Template image:").grid(
            row=current_row, column=0, sticky="w", padx=4, pady=4
        )
        tmpl_frame = ttk.Frame(main_frame)
        tmpl_frame.grid(row=current_row, column=1, sticky="we", padx=4, pady=4)
        ttk.Button(tmpl_frame, text="Browse...", command=self._browse_template).pack(side="left")
        ttk.Button(tmpl_frame, text="Clear", command=self._clear_template).pack(side="left", padx=4)
        self.template_lbl = ttk.Label(tmpl_frame, text="No file selected")
        self.template_lbl.pack(side="left", padx=8)
        current_row += 1
        
        # ========== ROW 6: Max Posts ==========
        ttk.Label(main_frame, text="Max posts:").grid(
            row=current_row, column=0, sticky="w", padx=4, pady=4
        )
        posts_frame = ttk.Frame(main_frame)
        posts_frame.grid(row=current_row, column=1, sticky="w", padx=4, pady=4)
        self.posts_entry = ttk.Entry(posts_frame, textvariable=self.posts_var, width=15)
        self.posts_entry.pack(side="left")
        ttk.Label(posts_frame, text="/ 10").pack(side="left", padx=(4, 0))
        current_row += 1
        
        # ========== ROW 7: Search Mode ==========
        ttk.Label(main_frame, text="Search mode:").grid(
            row=current_row, column=0, sticky="w", padx=4, pady=4
        )
        mode_frame = ttk.Frame(main_frame)
        mode_frame.grid(row=current_row, column=1, sticky="w", padx=4, pady=4)
        ttk.Radiobutton(
            mode_frame, 
            text="Default", 
            variable=self.mode_var, 
            value="default"
        ).pack(side="left")
        ttk.Radiobutton(
            mode_frame, 
            text="Fast (top candidate only)", 
            variable=self.mode_var, 
            value="fast"
        ).pack(side="left", padx=8)
        ttk.Radiobutton(
            mode_frame, 
            text="Robust (multiple regions)", 
            variable=self.mode_var, 
            value="robust"
        ).pack(side="left", padx=8)
        current_row += 1
        
        # ========== ROW 8: Run / Cancel Buttons ==========
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=current_row, column=0, columnspan=2, pady=(24, 0))
        ttk.Button(btn_frame, text="Run", command=self._run).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Cancel", command=self._cancel).pack(side="left", padx=4)
        
        # Setup tracing for conditional field enable/disable
        self.target_type_var.trace_add("write", self._on_target_type_change)
        self._on_target_type_change()

    def _on_target_type_change(self, *args) -> None:
        """Enable/disable fields based on target type selection."""
        if self.target_type_var.get() == "notepad":
            self.custom_target_entry.configure(state="disabled")
            self.posts_entry.configure(state="normal")
        else:
            self.custom_target_entry.configure(state="normal")
            self.posts_entry.configure(state="disabled")

    def _browse_template(self) -> None:
        """Open file dialog to select a template image."""
        file_path = filedialog.askopenfilename(
            title="Select Template Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        if file_path:
            self.template_path_var.set(file_path)
            self.template_lbl.config(text=f"✓ {Path(file_path).name}")

    def _clear_template(self) -> None:
        """Clear the selected template image."""
        self.template_path_var.set("")
        self.template_lbl.config(text="No file selected")
        
    def _run(self) -> None:
        """Validate inputs, apply overrides, and launch automation."""
        
        # 1. Set API Key from UI if provided
        api_key_val = self.api_key_var.get().strip()
        if api_key_val:
            os.environ["GEMINI_API_KEY"] = api_key_val

        # 2. Validate API Key is available
        if not os.getenv("GEMINI_API_KEY"):
            messagebox.showerror(
                "Configuration Error",
                "GEMINI_API_KEY is not set in the environment.\n\n"
                "Please set it before running the automation."
            )
            return

        # 3. Gather and validate target description
        target_str = ""
        if self.target_type_var.get() == "others":
            target_str = self.custom_target_var.get().strip()
            if not target_str:
                messagebox.showerror(
                    "Invalid Input", 
                    "Please enter a target description or select Notepad."
                )
                return

        # 4. Gather and validate max posts
        posts_str = ""
        if self.target_type_var.get() == "notepad":
            posts_str = self.posts_var.get().strip()
            if posts_str:
                if not posts_str.isdigit() or int(posts_str) <= 0:
                    messagebox.showerror(
                        "Invalid Input", 
                        "Max posts must be a positive integer."
                    )
                    return
        
        # 5. Copy template image to templates directory if selected
        template_src = self.template_path_var.get()
        if template_src:
            try:
                src_path = Path(template_src)
                project_root = Path(__file__).resolve().parents[2]
                templates_dir = project_root / "templates"
                templates_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, templates_dir / src_path.name)
            except Exception as e:
                messagebox.showerror(
                    "File Error", 
                    f"Failed to copy template image:\n{e}"
                )
                return

        # 6. Apply configuration overrides
        try:
            import vision_desktop_automation.config as cfg
            
            if target_str:
                cfg.TARGET_DESCRIPTION = target_str
                
            if posts_str:
                cfg.POST_LIMIT = int(posts_str)
                
            cfg.PLANNER_SEARCH_MODE = self.mode_var.get()
        except ImportError:
            messagebox.showerror(
                "Import Error",
                "Could not import vision_desktop_automation.config module."
            )
            return

        # 7. Destroy GUI and launch main automation
        self.root.destroy()
        
        try:
            from vision_desktop_automation.main import main
            main()
        except ImportError:
            messagebox.showerror(
                "Import Error",
                "Could not import vision_desktop_automation.main module."
            )

    def _cancel(self) -> None:
        """Close the GUI without running automation."""
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop()