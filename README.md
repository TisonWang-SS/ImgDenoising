# ImgDenoising

## Reference
This project mainly depends on this paper.
```
Yue, Zongsheng, Qian Zhao, Lei Zhang, and Deyu Meng. "Dual adversarial network: Toward real-world noise removal and noise generation." In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part X 16, pp. 41-58. Springer International Publishing, 2020.
```


## Datasets and Trained model file link:
### Datasets
This project use SIDD-Medium-srgb as training dataset and SIDD-validation as validation dataset.

- You can refer to the SIDD official site(https://www.eecs.yorku.ca/~kamel/sidd/index.php) download the image data and use the scripts inside `./Datasets` to generate h5 dataset file.

- Or simply download processed h5 Dataset from the google drive:
    - training Dataset h5: https://drive.google.com/file/d/1N8n3eaHm-RrI5yx7J1sICVmqv7MP4XFT/view?usp=sharing
    - validation Dataset h5: https://drive.google.com/file/d/1WwSam6Xf_qPWEOzTF-zjBc5LycowgNV7/view?usp=sharing

### Trained Model
Download from the google drive: https://drive.google.com/file/d/1Kl4QCt7y8hCFl0cT1AqHtXOYHusyq7KU/view?usp=sharing and put the model checkpoint in the `./checkpoints` folder and test.

## Model Training
To train your own network, simply run with command
```
python main.py
```
You can adjust the training parameters inside the `params` of the main script.

## Model Testing
You can view the image result on validation dataset with the command
```
python test_script_on_validation.py
```