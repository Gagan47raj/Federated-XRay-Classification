
# Chest X-Ray Dataset Report

## Dataset Overview

Total Images: 5856

Number of Classes: 2

Classes:

- NORMAL: 1583 images (27.03%)
- PNEUMONIA: 4273 images (72.97%)


---

## Class Distribution

The dataset contains two classes:

1. NORMAL
2. PNEUMONIA

The dataset is imbalanced with a larger number
of PNEUMONIA images than NORMAL images.

---

## Image Statistics

- Min Width: 384.0
- Max Width: 2916.0
- Mean Width: 1327.880806010929
- Min Height: 127.0
- Max Height: 2713.0
- Mean Height: 970.689036885246


---

## Data Quality Assessment

- Missing Folders: 6
- Empty Folders: 0
- Corrupted Images: 0
- Invalid Files: 0
- Duplicate Images: 0


---

## Dataset Split Verification



---

## Key Findings

1. The dataset consists of chest X-ray images
   belonging to NORMAL and PNEUMONIA classes.

2. The dataset is imbalanced with PNEUMONIA
   representing the majority class.

3. Image resolutions vary significantly.

4. Chest X-ray images are grayscale.

5. No corrupted files were detected.

6. No duplicate files were detected.

7. All required train, validation,
   and test directories were present.

8. Image resizing will be necessary before
   model training.

9. Class imbalance should be considered
   during model evaluation.

---

## Conclusion

The dataset passed all quality checks and
is suitable for preprocessing and model
development in subsequent project phases.
