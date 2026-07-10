## Phase 1

### Setup 27/6/26
- Created virtual environment and installed ultralytics which contains YOLOv8 and PyTorch
- Testing first detection

### Training 10/7/26
- Found public data set to train model 

## Notes and Learning 
- Epoch: one full pass through entire traning set. epochs=10 means model sees all images 10 times. 
- Batch size: number of images being processed at once before updating weights
- Loss: measure of incorrectness. Look for down trend across epochs. 
- Learning rate: how big a step it takes when correcting itself. Kind of like proportional gain correction in control systems (most real training uses momentum)

- Yolov8n.pt already knows general shapes. We are fine tuning with 
> - Train: what the model learns from 
> - Val: checked during training to catch overfitting. If train drops the model is memorising rather than generalising.  
> - Test: for final performance check

- We are after metric mAP (mean average precision)