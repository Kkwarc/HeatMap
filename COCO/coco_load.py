'''
Marlena Podleśna
'''
import fiftyone as fo
import fiftyone.zoo as foz

# Download, load and export the train split of COCO-2017 for persons, tables and chairs
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    classes=["person", "chair", "dining table"],
    max_samples=1000,
    shuffle=True,
)

dataset.export(
   export_dir="./Images",
   dataset_type=fo.types.COCODetectionDataset,
   label_field="ground_truth",
   classes=["person", "chair", "dining table"],
)

# Checking data using fiftyone in theFiftyOne app
if __name__ == "__main__":
    # Ensures that the App processes are safely launched on Windows
    session = fo.launch_app(dataset)
    session.wait()