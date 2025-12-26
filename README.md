# bubble-flow-hybrid-segmentation-and-completion
This repository provides a hybrid framework for bubble flow image segmentation and completion. The pipeline includes three stages: CycleGAN-based image-to-mask translation, watershed-based instance segmentation of individual bubbles, and a U-net model for completing occluded or incomplete single-bubble images.

## Bubble-CycleGAN
> Training:
```bash
python main.py --training
```
> Testing:
```bash
python main.py --testing
```

## Watershed

## Unet-based bubble complement

## Notes:
It is worth noting that this method does not require additional annotated data. We use the bubble dataset provided by [BubGAN] (https://github.com/ycfu/BubGAN) as the training data to train both the CycleGAN image-to-mask model and the U-Net-based bubble completion model.

For the implementation of CycleGAN, reference is made to: [https://www.youtube.com/watch?v=4LktBHGCNfw](https://www.youtube.com/watch?v=4LktBHGCNfw)