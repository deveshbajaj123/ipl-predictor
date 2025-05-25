IPL Match Winner Prediction Report  

1. **Introduction to Problem**

The objective of this project is to predict the winner between two IPL teams, specifically  whether Team 1 wins at the halfway point, using historical match data. The motivation  stems from the growing popularity of T20 cricket analytics and the desire to make data driven predictions by incorporating match metadata, player performances, and contextual  variables like playoffs or home advantage. Furthermore, while we use data from every match played in the IPL, we give more weightage to recent games. 

2. **Dataset Sources and Snippet of Dataset** 

Data Source: Kaggle - IPL Complete Dataset (2008–2024)  

Files Used:  

- matches.csv: Summary of every IPL match  
- deliveries.csv: Ball-by-ball delivery-level details of each match  

Example Snippet from deliveries.csv:  

![image](https://github.com/user-attachments/assets/d1a49d10-a9e7-4d4b-a66c-4d6c9082da36)


Example Snippet from matches.csv:  

![image](https://github.com/user-attachments/assets/3e1a854c-f782-41a4-895d-42a51212b3d6)


3. **Data Exploration and Data Processing**  

We take first innings data from the deliveries dataset, like calculating the total number of runs, dot balls , boundaries etc, and send it to the model alongside the team that won the match and the venue from the matches dataset to train the model. More details on data processing are given in the bullet points below: 

Exploration:  

- Removed rows with missing match outcomes.  
- Identified categorical features like team names and venue.  
- Verified consistency in the season field and standardized it.  

Feature Engineering:  

Total of 23 input features were engineered, including:  

- Team & Venue IDs  
- First Innings Stats: total\_runs, wickets, balls  
- Batter & Bowler Stats (best bowler economy, best batter score, fours and sixes) 
- Contextual Variables: team1\_home, is\_after\_half, is\_playoff, is\_final  
- Season Weighting (3 for 2024, 2 for 2023 and 2022, and 1 for the rest, keeping in mind  that recent data is more valuable)   

. 

4. **Methodology**  

Techniques Used:  

1. Random Forest Classifier  
1. Multi-layer Perceptron (MLP)  

Pipeline:  

- Train-test split using recent seasont (2024) – the entire data of 2008 + 50% of 2024 is  used for training and 50% of 2024 is used for testing. We have this uniques split keeping in mind the chronological nature of matches. It is unrealistic to predict the outcome of a match in 2008 using 2024 data, which is what might have happened if the split was random. 
- Standard feature scaling before MLP  
- RandomizedSearchCV for hyperparameter optimization of both models  

Parameters such as n\_estimators, max\_depth (RF) and hidden\_layer\_sizes, activation (MLP)  were tuned using random search.   

5. **Results**  

![image](https://github.com/user-attachments/assets/816ec8e1-420d-452a-9c76-cc55162417cb)

![image](https://github.com/user-attachments/assets/98d00c2c-6fb1-45ee-8d85-b56c539d369d)

6. **Code Highlights / Appendix**  

Feature Importance:  

![image](https://github.com/user-attachments/assets/e05ac34f-6016-4b90-b62a-203308e75413)

**Conclusion**  

This model has extensive real world use and gives us insight into what variables known at halfway stage are predictive of the final outcome. MLPs outperform random forests. But both models achieve a significant level of accuracy. This model can be used for betting and match analysis in general. 

