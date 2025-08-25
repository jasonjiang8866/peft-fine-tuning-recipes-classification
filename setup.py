#!/usr/bin/env python3
"""
Setup script for PEFT Fine-tuning Recipes Classification project.
Provides automated setup commands for different use cases.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_command(cmd, description=None):
    """Run a shell command and handle errors."""
    if description:
        print(f"📦 {description}")
    
    if isinstance(cmd, list):
        cmd = " && ".join(cmd)
    
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        return False
    
    print("✅ Command completed successfully")
    return True


def setup_full():
    """Install all dependencies including dev/jupyter dependencies."""
    commands = [
        "uv pip install -e .[dev]",
    ]
    
    success = run_command(commands, "Setting up full environment with Jupyter support")
    
    if success:
        print("\n🎉 Full setup complete! You can now run:")
        print("  - python setup.py notebook  # Start Jupyter notebook")
        print("  - python setup.py lab       # Start Jupyter lab")
        print("  - python setup.py train     # Run training")


def setup_training():
    """Install only training dependencies (no jupyter)."""
    commands = [
        "uv pip install -e .",
    ]
    
    success = run_command(commands, "Setting up training-only environment")
    
    if success:
        print("\n🎉 Training setup complete! You can now run:")
        print("  - python setup.py train     # Run training")
        print("  - python training.py        # Run training directly")


def setup_hf():
    """Set up environment with HuggingFace authentication."""
    commands = [
        "uv pip install -e .[dev]",
        "python jupyter_setup.py",
    ]
    
    success = run_command(commands, "Setting up environment with HuggingFace authentication")
    
    if success:
        print("\n🎉 HuggingFace setup complete!")


def run_train():
    """Run the training script."""
    success = run_command("python training.py", "Running training script")
    
    if success:
        print("\n🎉 Training completed! Check train_loss.html and eval_metrics.html for results.")


def run_notebook():
    """Start Jupyter notebook."""
    # First ensure dev dependencies are installed
    run_command("uv pip install -e .[dev]", "Ensuring Jupyter dependencies are installed")
    
    success = run_command("jupyter notebook finetuning.ipynb", "Starting Jupyter notebook")
    
    if not success:
        print("❌ Failed to start Jupyter notebook. Make sure Jupyter is installed.")


def run_lab():
    """Start Jupyter lab."""
    # First ensure dev dependencies are installed
    run_command("uv pip install -e .[dev]", "Ensuring Jupyter dependencies are installed")
    
    success = run_command("jupyter lab", "Starting Jupyter lab")
    
    if not success:
        print("❌ Failed to start Jupyter lab. Make sure Jupyter is installed.")


def visualize():
    """Generate visualization from training logs."""
    success = run_command("python viz.py", "Generating visualizations")
    
    if success:
        print("\n🎉 Visualizations generated!")


def clean():
    """Clean up generated files."""
    commands = [
        "rm -f *.html",
        "rm -rf ag_news_lora/",
        "rm -rf __pycache__/",
        "rm -rf .ipynb_checkpoints/",
        "find . -name '*.pyc' -delete",
        "find . -name '*.pyo' -delete",
    ]
    
    for cmd in commands:
        run_command(cmd)
    
    print("\n🧹 Cleanup complete!")


def main():
    """Main entry point for the setup script."""
    parser = argparse.ArgumentParser(
        description="Setup script for PEFT Fine-tuning Recipes Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  setup       - Install all dependencies (including Jupyter)
  setup-train - Install only training dependencies
  setup-hf    - Setup with HuggingFace authentication
  train       - Run training script
  notebook    - Start Jupyter notebook
  lab         - Start Jupyter lab
  visualize   - Generate training visualizations
  clean       - Clean up generated files
  
Examples:
  python setup.py setup       # Full setup
  python setup.py train       # Run training
  python setup.py notebook    # Start Jupyter notebook
        """
    )
    
    parser.add_argument(
        "command",
        choices=["setup", "setup-train", "setup-hf", "train", "notebook", "lab", "visualize", "clean"],
        help="Command to execute"
    )
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"🚀 Running command: {args.command}")
    
    command_map = {
        "setup": setup_full,
        "setup-train": setup_training,
        "setup-hf": setup_hf,
        "train": run_train,
        "notebook": run_notebook,
        "lab": run_lab,
        "visualize": visualize,
        "clean": clean,
    }
    
    command_func = command_map.get(args.command)
    if command_func:
        command_func()
    else:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()