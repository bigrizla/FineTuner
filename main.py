import tkinter as tk
import importlib.util
from tkinter import ttk, filedialog, messagebox
import json
import os
import re
import ast
import pyperclip
import optuna
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HYPERPARAM_DIR = "hyperparams"
os.makedirs(HYPERPARAM_DIR, exist_ok=True)

PREDEFINED_HYPERPARAMS = sorted([
    "Batch Size", "Beta1", "Beta2", "Dropout", "Epsilon", "Epoch Number", "Gamma", "Gradient Clipping",
    "Learning Rate", "L1 Regularization", "L2 Regularization", "Momentum", "Number of Layers", "Patience", "Weight Decay"
])

def load_script_as_module(script_path):
    spec = importlib.util.spec_from_file_location("user_model_module", script_path)
    if spec is None:
        raise ImportError(f"Could not load spec from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class HyperparameterManager:
    def __init__(self, master):
        self.master = master
        self.master.title("ML Fine-Tuner")
        
        self.params = {}
        self.param_entries = {}
        self.model_path = None
        
        self.left_frame = tk.Frame(master)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.right_frame = tk.Frame(master)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Configure grid resizing for the right frame
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)  # Allow right side to expand

        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        
        self.model_title = tk.Label(self.left_frame, text="No model selected", font=("Arial", 12, "bold"))
        self.model_title.pack(pady=5)
        
        self.model_buttons_frame = tk.Frame(self.left_frame)
        self.model_buttons_frame.pack(pady=5)
        
        self.browse_button = tk.Button(self.model_buttons_frame, text="Load Model", command=self.load_model)
        self.browse_button.grid(row=0, column=0, padx=(0, 5))
        
        self.extract_button = tk.Button(self.model_buttons_frame, text="Extract Hyperparameters", command=self.extract_hyperparams)
        self.extract_button.grid(row=0, column=1)
        
        self.param_label = tk.Label(self.left_frame, text="Hyperparameter:")
        self.param_label.pack()
        
        self.param_container = tk.Frame(self.left_frame)
        self.param_container.pack()
        
        self.param_var = tk.StringVar()
        self.param_dropdown = ttk.Combobox(
            self.param_container,
            textvariable=self.param_var,
            values=PREDEFINED_HYPERPARAMS,
            state='readonly'
        )
        self.param_dropdown.grid(row=0, column=0, padx=5, pady=5)
        
        self.add_button = tk.Button(self.param_container, text="+", command=self.add_param, width=3)
        self.add_button.grid(row=0, column=1, padx=5, pady=5)
        
        self.params_frame = tk.Frame(self.left_frame)
        self.params_frame.pack(pady=5)
        
        self.training_settings_frame = tk.Frame(self.left_frame)
        self.training_settings_frame.pack(pady=5)
        
        optimiser_label = tk.Label(self.training_settings_frame, text="Optimiser:")
        optimiser_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.optimiser_var = tk.StringVar()
        self.optimiser_dropdown = ttk.Combobox(
            self.training_settings_frame,
            textvariable=self.optimiser_var,
            state='readonly',
            values=["Adam", "SGD", "RMSprop"]
        )
        self.optimiser_dropdown.current(0)
        self.optimiser_dropdown.grid(row=0, column=1, padx=5, pady=2)
        
        loss_label = tk.Label(self.training_settings_frame, text="Loss Function:")
        loss_label.grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.loss_var = tk.StringVar()
        self.loss_dropdown = ttk.Combobox(
            self.training_settings_frame,
            textvariable=self.loss_var,
            state='readonly',
            values=["MSELoss", "CrossEntropyLoss", "SmoothL1Loss"]
        )
        self.loss_dropdown.current(0)
        self.loss_dropdown.grid(row=1, column=1, padx=5, pady=2)

        # Storage for final per-trial data:
        # self.all_trial_data[trial_number] = {
        #     'epochs': [...],
        #     'train_accs': [...],
        #     'val_accs': [...],
        #     'val_losses': [...]
        # }
        self.all_trial_data = {}

        # Slider for number of trials (3-20)
        self.trial_label = tk.Label(self.left_frame, text="Number of Trials:")
        self.trial_label.pack()

        self.trial_var = tk.IntVar(value=3)  # Default value
        self.trial_slider = tk.Scale(self.left_frame, from_=3, to=20, orient="horizontal", variable=self.trial_var)
        self.trial_slider.pack()
        
        self.copy_button = tk.Button(self.left_frame, text="Copy Template", command=self.copy_template)
        self.copy_button.pack(pady=5)

        # Frame for Tune and Terminate buttons
        self.tuning_buttons_frame = tk.Frame(self.left_frame)
        self.tuning_buttons_frame.pack(pady=5)

        # Tune Button
        self.tune_button = tk.Button(self.tuning_buttons_frame, text="Tune", command=self.tune_hyperparams)
        self.tune_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Terminate Button
        self.terminate_button = tk.Button(self.tuning_buttons_frame, text="Terminate", command=self.terminate_tuning, state=tk.DISABLED)
        self.terminate_button.grid(row=0, column=1, padx=5, sticky="ew")

        self.stop_tuning = False  # Flag to stop tuning

        self.figure, axes = plt.subplots(
            3, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 3, 2]}, constrained_layout=True
        )
        self.ax1, self.ax2, self.ax3 = axes
        self.ax1.set_title("Loss Curve")
        self.ax1.grid(True)
        self.ax2.set_title("Validation Accuracy")
        self.ax2.grid(True)
        self.ax3.set_title("Hyperparameter Sensitivity")
        self.ax3.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")  # Expands in both directions
        
        self.clear_graphs()
    
    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Model Script",
            filetypes=[("Python Files", "*.py")]
        )
        if file_path:
            self.model_path = file_path
            self.model_title.config(text=os.path.basename(file_path))
    
    def extract_hyperparams(self):
        if not self.model_path:
            messagebox.showwarning("Warning", "Please load a model file first.")
            return
        
        try:
            with open(self.model_path, 'r') as f:
                code = f.read()
            
            match = re.search(r'HYPERPARAMS\s*=\s*(\{.*?\})', code, re.DOTALL)
            if not match:
                messagebox.showerror("Error", "No HYPERPARAMS dictionary found in the script.")
                return
            
            dict_str = match.group(1)
            hyperparams = ast.literal_eval(dict_str)
            
            self.params.clear()
            self.param_entries.clear()
            
            for k, v in hyperparams.items():
                self.params[k] = v
            
            self.rebuild_params()
            messagebox.showinfo("Success", "Hyperparameters extracted successfully!")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract hyperparameters.\n{e}")
    
    def add_param(self):
        name = self.param_var.get()
        if name and name not in self.params:
            self.params[name] = 0.0
            self.rebuild_params()
    
    def remove_param(self, param_name):
        if param_name in self.params:
            del self.params[param_name]
            del self.param_entries[param_name]
        self.rebuild_params()
    
    def rebuild_params(self):
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        for i, (param, value) in enumerate(self.params.items()):
            label = tk.Label(self.params_frame, text=param)
            label.grid(row=i, column=0, padx=5, pady=2, sticky="w")
            
            entry = tk.Entry(self.params_frame, width=10)
            entry.grid(row=i, column=1, padx=5, pady=2)
            entry.insert(0, str(value))
            
            self.param_entries[param] = entry
            
            remove_button = tk.Button(
                self.params_frame,
                text="-",
                command=lambda p=param: self.remove_param(p)
            )
            remove_button.grid(row=i, column=2, padx=5, pady=2)
    
    def copy_template(self):
        template = "HYPERPARAMS = {\n"
        for param, entry in self.param_entries.items():
            value = entry.get()
            if value.replace('.', '', 1).isdigit():
                value = float(value) if '.' in value else int(value)
            template += f'    "{param}": {value},\n'
        template = template.rstrip(',\n') + "\n}"
        pyperclip.copy(template)
        messagebox.showinfo("Copied", "Hyperparameter template copied to clipboard!")

    def tune_hyperparams(self):
        if not self.model_path:
            messagebox.showwarning("Warning", "Please load a model file first.")
            return

        self.stop_tuning = False  # Reset stop flag
        self.terminate_button.config(state=tk.NORMAL)  # Enable Terminate button

        num_trials = self.trial_var.get()
        user_hyperparams = {}

        for param, entry in self.param_entries.items():
            try:
                value = float(entry.get()) if "." in entry.get() else int(entry.get())
                user_hyperparams[param] = value
            except ValueError:
                messagebox.showwarning("Invalid Input", f"Invalid value for {param}. Using default.")
                user_hyperparams[param] = self.params.get(param, 0)

        tuning_thread = threading.Thread(target=self.run_tuning, args=(num_trials, user_hyperparams), daemon=True)
        tuning_thread.start()

    def terminate_tuning(self):
        self.stop_tuning = True  # Set stop flag
        self.terminate_button.config(state=tk.DISABLED)  # Disable button immediately
        self.clear_graphs()  # Clear graphs immediately
        print("Tuning terminated; graphs cleared.")

    def run_tuning(self, num_trials, user_hyperparams):
        """Runs the Optuna study with user-defined hyperparameters."""
        try:
            user_module = load_script_as_module(self.model_path)
            if not hasattr(user_module, "SimpleNet") or not hasattr(user_module, "train_model"):
                raise Exception("The model script is missing required functions/classes.")

            SimpleNet = user_module.SimpleNet
            train_model = user_module.train_model

            def objective(trial):
                if self.stop_tuning:
                    raise optuna.TrialPruned()  # Abort trial if termination is requested
                
                """Objective function for Optuna to minimize validation loss."""
                hyperparams = user_hyperparams.copy()  # Use user values as defaults

                # Allow Optuna to tune around the user's input
                hyperparams["Learning Rate"] = trial.suggest_float(
                    "learning_rate",
                    max(1e-6, hyperparams["Learning Rate"] * 0.5),  # 50% lower
                    hyperparams["Learning Rate"] * 2,  # 2x higher
                    log=True
                )
                hyperparams["Epoch Number"] = trial.suggest_int(
                    "epochs",
                    max(1, hyperparams["Epoch Number"] // 2),
                    hyperparams["Epoch Number"] * 2
                )

                model = SimpleNet(input_dim=10, hidden_dim=20, output_dim=10).to(device)
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["Learning Rate"])

                train_loader = user_module.train_loader
                val_loader = user_module.val_loader

                train_accs, val_accs, val_losses, epochs = train_model(
                    model, train_loader, val_loader, criterion, optimizer, device,
                    hyperparams["Epoch Number"], hyperparams.get("Patience", 3)
                )

                self.all_trial_data[trial.number] = {
                    'epochs': epochs,
                    'train_accs': train_accs,
                    'val_accs': val_accs,
                    'val_losses': val_losses
                }

                return val_losses[-1]  # Return last validation loss

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=num_trials)  # Use user-defined number of trials

            best_params = study.best_params
            self.master.after(0, lambda: messagebox.showinfo("Best Params", f"Optimised Hyperparameters: {best_params}"))
            self.master.after(0, lambda: self.plot_results(study))

        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Error", f"Failed during tuning:\n{e}"))

    def plot_results(self, study):
        """
        Plot the final results after all trials finish:
        - Loss curve (per trial)
        - Accuracy curve (per trial)
        - Hyperparameter Sensitivity (bar chart with only integer x-axis ticks)
        """
        self.clear_graphs()

        # Plot each trial's loss & accuracy curves
        for trial_number, data in self.all_trial_data.items():
            if 'epochs' in data and len(data['epochs']) > 0:
                self.ax1.plot(data['epochs'], data['val_losses'], marker='o', linestyle='-', label=f"Trial {trial_number}")
                self.ax2.plot(data['epochs'], data['train_accs'], marker='o', linestyle='-', label=f"Train {trial_number}")
                self.ax2.plot(data['epochs'], data['val_accs'], marker='x', linestyle='--', label=f"Val {trial_number}")

        self.ax1.set_title("Loss Curve")
        self.ax1.set_xlabel("Epochs")
        self.ax1.set_ylabel("Loss")
        self.ax1.grid(True)
        # Only add legend if there are labels (lines plotted)
        if self.ax1.has_data():
            self.ax1.legend()

        self.ax2.set_title("Validation Accuracy")
        self.ax2.set_xlabel("Epochs")
        self.ax2.set_ylabel("Accuracy (%)")
        self.ax2.grid(True)
        if self.ax2.has_data():
            self.ax2.legend()

        # Extract trial objective values
        values = [t.value for t in study.trials]
        trials_range = range(len(values))  # Ensure x-axis is a sequence of integers

        # Convert Hyperparameter Sensitivity plot to a bar chart
        self.ax3.bar(trials_range, values, label='Objective Value', color='steelblue')

        # Force x-axis to show only integer ticks (whole numbers)
        self.ax3.set_xticklabels([str(t) for t in trials_range])  # Convert to string labels

        self.ax3.set_title("Hyperparameter Sensitivity")
        self.ax3.set_xlabel("Trials")
        self.ax3.set_ylabel("Objective Value")
        self.ax3.grid(False)
        self.ax3.legend()
        self.figure.tight_layout()  # Ensure subplots resize properly 
        self.canvas.draw()

    def clear_graphs(self):
        self.figure.clear()
        self.ax1, self.ax2, self.ax3 = self.figure.subplots(3, 1)
        self.ax1.set_title("Loss Curve")
        self.ax1.set_xlabel("Epochs")
        self.ax1.set_ylabel("Loss")
        self.ax1.grid(True)

        self.ax2.set_title("Validation Accuracy")
        self.ax2.set_xlabel("Epochs")
        self.ax2.set_ylabel("Accuracy (%)")
        self.ax2.grid(True)

        self.ax3.set_title("Hyperparameter Sensitivity")
        self.ax3.set_xlabel("Trials")
        self.ax3.set_ylabel("Value")
        self.ax3.grid(False)

        self.figure.tight_layout()  # Ensure subplots resize properly
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = HyperparameterManager(root)
    root.mainloop()
