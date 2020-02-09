
Group project for CS4243 Computer Vision and Pattern Recognition
-----------
To run the detection script, use:
```
python test.py [-h] [--visual] [--target all/waldo/wenda/wizard] [--loc file]

-h : Brings up the help list

--visual : Run this argument to display visual output of each detection

--target : Selects which target to run the detection script on, inputting <all> as an argument runs the detection script on all 3 targets. For example, --target waldo
     
--loc : Specifies the location of .txt file containing which information on which images to run detection on. For example: datasets/ImageSets/val.txt
```

Detection results will be saved in 'results/'. There are 4 sub-folders:
```
'results/eval': Contains bounding box information of detected results as per VOC format. Detections on Waldo, Wenda and Wizard are stored in separate .txt files in this folder.

'results/<waldo/wenda/wizard>': Visual results of the detection on <waldo/wenda/wizard>
```

To train the SVM, use:
```
python train.py  [-h] [--target waldo/wenda/wizard]

-h: Brings up the help list

--target : Selects which target to train a SVM model on.

To adjust the HOG features that is used to train the SVM, please configure the code manually inside train.py via the SVM constructor.
```

Training data for our models are stored in 'datasets/Templates'.
Other relevant files used in the project are as follows:
```
get_template.py: Used to extract ground truth templates for training, from the image dataset based on their annotations
haar.py: Contains the haar_cascade detection code
svm.py: Contains code pertaining to the SVM
sift_match.py: Contains code pertaining to SIFT matching
pred_<waldo/wenda/wizard>: Contains the detection pipeline for the corresponding target
```

Environment:
```
numpy
opencv-python
sklearn
skimage
pickle
cyvlfeat
```

In case the code fails, please contact:
```
Liu Zhaoyu: e0253678@u.nus.edu
Fong Wei Zheng: e0035221@u.nus.edu
```
