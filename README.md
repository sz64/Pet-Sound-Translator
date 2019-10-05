# Pet-Sound-Translator (Cats)

## Product Definition
* Product Mission

The goal of this project is to create a cat sound processing software that can identify their positive and negative emotions. Our product can be used by cat owners to have a better understanding of their pets. It can also be used to monitor cat's mood automatically.
* Target Users

    1. new pet owner
    2. pet store/pet clinic staff
    3. anyone who want to have a better understanding of domestic cats
* User Stories

    1. I, a new pet owner, should be able to figure out the mood of my cat(s) using this product.
    2. I, as a pet owner, should be able to compare the recognition result of this product with the understanding of my cat(s) based on my own experience and make a comment on the result.
    3. I, as a pet store/clinic staff, shoud be able to use this product to monitor cat(s)' mood when I leave the cattery.
    4. I, as a pet store/clinic staff, should be able to look at records of cats' mood to see when the cat had bad emotions, how serious it was and how long it lasted.
    5. I, as a ML researcher, should have access to the code of this model and thus can use it as a pretrained model for other animals' mood detection.

* MVP

    1. Uer should be able to upload cat sound audio file via a GUI
    2. User shoud be able to get a cat mood recognition result.
* User Iterface Design

Our product should be an API or a computer/mobile app that:

  1. User can upload cat sound recordings using our interfaces/ User can record live cat sound and upload it to the app.
  2. Once the record is uploaded, the user will get a generalized category of cat’s emotion corresponding to its sound. 
 
 
 ## Product Survey
 * Existing Similar Products
 * Related Papers
 
 We can only find two papers directly related to this task which were done by the same team.
 
 ***Domestic Cat Sound Classification Using Transfer Learning***, Yagya Raj Pandeya, Dongwhoon Kim, Joonwhooan Lee, 2018
 ***Domestic Cat Sound Classification Using Learned Features from Deep Neural Network***, Yagya Raj Pandeya, Dongwhoon Kim, Joonwhooan Lee, 2018

According to these two papers, the state-of-art method is as follow:
   1. Feature Extraction:
      - Extract mel bank features from original audio signal
      - Use a pre-trained CNN (with Frequency Divison Average Pooling) to extract features from mel bank features.
   2. Classification:
      - Use five different machine learnign algorithms (Random Forest, KNN, Extra Tree, Linear Discriminant Analysis, Support Vector Machine) for the classification and ensemble all of them using majority voting.
   3. Categories: There are ten categories of cat mood (Happy, Angry, Resting, Defence, etc).

## System Design
* Major Components

Pet Sound -> Feature Extraction -> Training -> Classification -> Output
  
  `User Interface` Module: 
  
  `Feature Extractor` Module: 
  
  `Classification` Module: 
  
  `OS Interface` Module: 
* Technology Selection
  `Feature Exytaction`: 
  
     First Step: Mel bank features
     
     Second Step:
     1. PCA (Principal Component Analysis)
     2. ICA (Independent Component Analysis)
     3. CNN (Convolutional Neural Network)
     4. CDBN (Convolutional Deep Belief Network)
   `Classification`:
   
     Classifier:
     1. KNN
     2. Linear Deiscriminant Analysis
     3. Random Forest
     4. Extra Tree
     5. Support Vector Machine
     6. CNN
     7. RNN
     
     Ensembler:
     1. Majority Voting
     2. bagging
     3. boosting
* Test Programs

 `Feature Exytaction`: 
  
   First Step: Mel bank features
     
   Second Step:
   1. PCA (Principal Component Analysis)
   2. ICA (Independent Component Analysis)
   3. CNN (Convolutional Neural Network)
   4. CDBN (Convolutional Deep Belief Network)
   `Classification`:
   
   Classifier:
   1. KNN
   2. Linear Deiscriminant Analysis
   3. Random Forest
   4. Extra Tree
   5. Support Vector Machine
   6. CNN
   7. RNN
     
   Ensembler:
   1. Majority Voting
   2. bagging
   3. boosting
