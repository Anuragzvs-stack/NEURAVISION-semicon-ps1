import os
import sys
import torch
import numpy as np
from model import SRUNet

def main():
    # 1. Setup hardware & folders from command line arguments
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- INITIATING BATCH INFERENCE ---")
    print(f"Compute Engine: {device.type.upper()}")

    # Create the output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load the trained model pointing to models/ folder
    model = SRUNet().to(device)
    weights_path = os.path.join("models", "srunet_overnight_weights.pth")

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.eval() # CRITICAL: Lock the weights for inference
        print("Firmware loaded successfully!")
    else:
        print(f"ERROR: Could not find {weights_path}.")
        sys.exit(1)

    # 3. Process the test images
    test_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    total_files = len(test_files)
    print(f"Found {total_files} test images. Starting production line...\n")

    with torch.no_grad(): # Turn off gradient tracking
        for i, file_name in enumerate(test_files):
            # A. Load the noisy image
            file_path = os.path.join(input_dir, file_name)
            noisy_np = np.load(file_path)
            
            # B. Format for the GPU: add Batch and Channel dimensions -> (1, 1, H, W)
            noisy_tensor = torch.from_numpy(noisy_np).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # C. Run it through the U-Net
            clean_tensor = model(noisy_tensor)
            
            # D. Pull it back to CPU, remove extra dimensions, and convert to numpy
            clean_np = clean_tensor.squeeze().cpu().numpy()
            
            # E. Enforce compliance: Clean NaN/Inf and clamp strictly to [0, 1]
            clean_np = np.nan_to_num(clean_np, nan=0.0, posinf=1.0, neginf=0.0)
            clean_np = np.clip(clean_np, 0.0, 1.0)
            
            # F. Save to the target output folder with the same filename
            save_path = os.path.join(output_dir, file_name)
            np.save(save_path, clean_np)
            
            # Print a progress update every 100 images
            if (i + 1) % 100 == 0 or (i + 1) == total_files:
                print(f"Processed [{i + 1}/{total_files}] files...")

    print("\n--- BATCH INFERENCE COMPLETE ---")
    print(f"All cleaned files are saved in the '{output_dir}' folder!")

if __name__ == "__main__":
    main()