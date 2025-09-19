from pathlib import Path

# List all files in the train directory
train_dir = Path("balloon/train")
print("Files in train directory:")
for file in train_dir.glob("*.jpg"):
    print(f"- {file.name}")

# List all files in the val directory
val_dir = Path("balloon/val")
print("\nFiles in val directory:")
for file in val_dir.glob("*.jpg"):
    print(f"- {file.name}")
