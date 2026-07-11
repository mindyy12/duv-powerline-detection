## Phase 1

### Setup 27/6/26
- Created virtual environment and installed ultralytics which contains YOLOv8 and PyTorch
- Testing first detection

## Phase 2
### Model Training 10/7/26
- Found public data set to train model 
- Trained 60 epochs and got mAP50 to 0.493 and mAP50-95 to 0.302
### Model Validation and Testing 10/7/26
- Some of the images in Roboflows dataset are mosaic -> dataset limitation  

## Phase 3 
### Connected to flask 11/7/26
- Made app.py and .html for dashboard
- Prototyped design on figma then took inspiration from web templates

## Notes and Learning 
- Epoch: one full pass through entire traning set. epochs=10 means model sees all images 10 times. 
- Batch size: number of images being processed at once before updating weights
- Loss: measure of incorrectness. Look for down trend across epochs. 
- Learning rate: how big a step it takes when correcting itself. Kind of like proportional gain correction in control systems (most real training uses momentum)

- Yolov8n.pt already knows general shapes. We are fine tuning with 
* - Train: what the model learns from 
* - Val: checked during training to catch overfitting. If train drops the model is memorising rather than generalising.  
* - Test: for final performance check

- We are after metric mAP (mean average precision)
* - mAP50 correct if overlap is >50% - detection 
* - mAP50-95 takes average across multiple thressholds from 50% to 95% in 5% increments - drawn box precision 