# Image-Classification-on-Covid-Chest-X-ray-Scan

Five times more deadly than the flu, COVID-19 causes significant morbidity and mortality. Like other pneumonias, pulmonary infection with COVID-19 results in inflammation and fluid in the lungs. COVID-19 looks very similar to other viral and bacterial pneumonias on chest radiographs, which makes it difficult to diagnose. Computer vision models to detect COVID-19 could help doctors provide a quick and confident diagnosis. As a result, patients could get the right treatment before the most severe effects of the virus take hold.
#### So, the goal of this project is the classification of chest X-ray images to find out whether a person's lungs is healthy or infected. The system has been trained to recognize a chest X-ray scan as one of the following : 
- Normal i.e lungs are healthy
- Covid i.e lungs show presence of covid
- Viral Pneumonia i.e lungs show presence of viral-pneumonia

### Simple Web App available at: https://share.streamlit.io/amancrackpot/imageclassification_covid_x_ray_scans/main/src/app_streamlit.py

<p>
  <img alt="Webapp" src="https://github.com/tripathiGithub/ImageClassification_Covid_X_Ray_Scans/raw/main/Results/covid2.gif">
</p>

### Sample Analysis Report

<p>
  <img alt="Report" src="https://github.com/tripathiGithub/ImageClassification_Covid_X_Ray_Scans/raw/main/Results/covid_sample.png" width="60%">
</p>



#### The code needed to train the model is detailed in here https://github.com/amancrackpot/ImageClassification_Covid_X_Ray_Scans/blob/main/covid-classification-resnet50.ipynb
Deep Learning has been used to create this model. I have used Transfer Learning which involves loading a generic well trained image classification model for feature extraction and then adding a few layers as head so that it can be trained for our specific task. Apart from this, to train the system and get better results, modern deep learning practices have been used like data-augmentation , one-cycle-policy, discriminative-learning-rate, etc

#### Dataset available at https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
## Results on test data
![image](https://github.com/amancrackpot/ImageClassification_Covid_X_Ray_Scans/blob/main/Results/test_stats.png)
