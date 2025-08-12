#!/usr/bin/env python3
import os
import random
import shutil

# Set random seed for reproducibility
random.seed(42)

# Calculate 20% of each dataset
sam_files = os.listdir('embedded_sam')
non_sam_files = os.listdir('embedded_non_sam')

sam_test_count = int(len(sam_files) * 0.2)  # 20% of embedded_sam
non_sam_test_count = int(len(non_sam_files) * 0.2)  # 20% of embedded_non_sam

print(f"Total files in embedded_sam: {len(sam_files)}")
print(f"Total files in embedded_non_sam: {len(non_sam_files)}")
print(f"Moving {sam_test_count} files from embedded_sam to embedded_test")
print(f"Moving {non_sam_test_count} files from embedded_non_sam to embedded_test")

# Randomly select files from embedded_sam
sam_test_files = random.sample(sam_files, sam_test_count)

# Randomly select files from embedded_non_sam
non_sam_test_files = random.sample(non_sam_files, non_sam_test_count)

# Move files from embedded_sam to embedded_test
for file in sam_test_files:
    src = os.path.join('embedded_sam', file)
    dst = os.path.join('embedded_test', file)
    shutil.move(src, dst)
    print(f"Moved {file} from embedded_sam to embedded_test")

# Move files from embedded_non_sam to embedded_test
for file in non_sam_test_files:
    src = os.path.join('embedded_non_sam', file)
    dst = os.path.join('embedded_test', file)
    shutil.move(src, dst)
    print(f"Moved {file} from embedded_non_sam to embedded_test")

print(f"\nTest set creation complete!")
print(f"Files in embedded_test: {len(os.listdir('embedded_test'))}")
print(f"Files remaining in embedded_sam: {len(os.listdir('embedded_sam'))}")
print(f"Files remaining in embedded_non_sam: {len(os.listdir('embedded_non_sam'))}") 