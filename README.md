# Brain-MRI-segmentation

## Business Objective:
To create segmentation masks for tumors in the human brain using MRI scans.

<h2 id="table-of-contents"> :book: Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#dataset"> ➤ Dataset</a></li>
    <li><a href="#sample-example"> ➤ Sample example</a></li>
    <li><a href="#How-to-use-the-trained-model"> ➤ How to use the trained model</a></li>
    <li><a href="#Sample-runtime-output of application"> ➤ Sample runtime output of application</a></li>
   
  </ol>
</details>


<!-- DATASET -->
<h3 id="dataset"> :floppy_disk: Dataset</h3>

Source of dataset is kaggle.

The dataset is publicly available. Please refer to the 
[Brain MRI segmentation](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)

<!-- Sample example -->
<h3 id="sample-example"> :spiral_notepad: Sample example</h3>

![Brain and its segmentation mask](https://imgur.com/x3XlzNm.png)



<!-- How to use the trained model -->
<h3 id="How-to-use-the-trained-model"> :dart: How to use the trained model</h3>
1. Install following required libraries
  <ol>
    <li>cv2</li>
    <li>streamlit</li>
    <li>PIL</li>
    <li>tensorflow</li>
  </ol>
    
  For more details use requirements.txt file. 
  
    `pip install -r /path/to/requirements.txt`
    
2. Link to trained final model : [EfficientNet](https://drive.google.com/drive/folders/10o-7xrqrRWyfmSQaszlO8BPCbryB2VVl?usp=sharing)
3. Download this repo and the trained model folder.
4. change your directory to the file containing application.py and then run the command in terminal. 

        `streamlit run application.py`

5. Sample images for trying out model  [Inference Images](https://drive.google.com/drive/folders/1-3Wz-Og_Fb91zPAFXUkRHWbc3avW-3EJ?usp=sharing)



<!-- Sample example -->
<h3 id="Sample-runtime-output of application"> :spiral_notepad: Sample runtime output of application</h3>

![Sample application output](https://imgur.com/U9eBEOL.png)

