# I-Farm

This is **I-Farm** Project repo

## Repo details

- All ml and dl code is in Jupyter Notebooks directory
- Datsets used for this are in datsets folder
  - Except this one used for DL [dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)
- The web app is only created from the Web_App directory by using the git subtree method and it runs independently prior to parent directory
  - App is created using Flask and frontend is done using HTML, CSS, Bootstrap(v5.x) and Javascript
  - All asscociated web app files are in this directory
- To Run the code use requirements.txt in parent directory to create the project development environment
- Check **Project Report** document for detailed description

## Precision Farming and Our Motive:

It is an information and technology-based farm management system that identifies, analyses and manages variability in fields by conducting crop production practices at the right place and time and in the right way, for optimum profitability, sustainability and protection of the land resource.

Machine Learning has emerged together with big data technologies and high-performance computing to create new opportunities to unravel, quantify, and understand data intensive processes in agricultural operational environments. So we want to use power of new age technologies in improving agriculture and solving problems.

In a country like India, which has ever increasing demand of food due to rising population, advances in agriculture sector are required to meet the needs.

# Our Project:

This project is done as part of [Griet Epitome 2023](http://epitome23.griet.ac.in/), under AgroTech Domain. The title of the project is I-Fram (Intellect)

We have used data from trusted sources and built some ML/DL models tackling 5 problem-statements:

- Recommending crop to be grown based on soil and climatic conditions.
- Predicting the Yield of the crop based on the geographical aspects
- Predicting the modal price of the crop based on the historic data.
- Identifying the health status of soil.
- Classifying the health status of the crop based on the images of leaves.

#### 1. Crop Recommendation

The system requires inputs soil composition and its Ph, average annual rainfall(mm), temperature, from the user.
Based on the given parameters the system will predict the Production (in quintals) for the crop and yield of the
crop per acre.

#### 2. Yield Prediction:

The system requires inputs such as state, year, crop name, area, average annual rainfall(mm) from the user. Based on the given parameters the system will predict the Production (in quintals) for the crop and yield of the crop per acre.

#### 3. Price Prediction:

Price Prediction nowadays, has become a very important agricultural problem. The aim of this project is to predict the crop price for the next rotation. This provides the farmer with an insight of what the future price (per quintal) of the crop that farmer is going to harvest.

#### 4. Soil Health Prediction:

Fertilizers are being used to increase crop productivity, but for producing more crop, nutrient level of soil and crop monitoring is more important. This is one of the main factors for the production of rich and good quality crops and also helps the farmers to determine the type of crops to be planted on his land. Based on the output of optimal N-P-K values further action can be taken.

#### 5. Fertilizer Suggestion

Soil fertility refers to the ability of soil to sustain agricultural plant growth, i.e. to provide plant habitat
and result in sustained and consistent yields of high quality. Based on the current composition of soil we
suggest what to do to improve soil fertility

#### 6. Disease Detection:

Crop diseases are a major threat to food security, but their rapid identification remains difficult in many parts of the world due to the lack of the necessary infrastructure. Plant diseases are not only a threat to food security at the global scale, but can also have disastrous consequences for smallholder farmers whose livelihoods depend on healthy crops. Detect Them.

We classify the leaf images of the following crops and predict the health the status and provide the details of the crop disease and some cures.

<details>
<summary>Crops list</summary>
<br>

- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Orange
- Pepper
- Peach
- Potato
- Raspberry
- Soyabean
- Strawberry
- Squash
- Tomato

</details>

<br>

**Note** : Due to limited data available we could train our models confinig to it, so there is comparitively little info to check. Kindly upload the files and input in proper format and units respectively.

#### Team Members:

- Abhijna
- Deekshitha
- Mohit
- Nitin
- Sreeja
