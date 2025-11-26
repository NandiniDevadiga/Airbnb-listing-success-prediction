# Airbnb-listing-success-prediction 

Demand-Based Success Classification using O_score, Elastic Net, and Decision Trees
________________________________________
# Project Overview ⭐
This project predicts whether an Airbnb listing is “Good” or “Bad” using a demand-based success measure called O_score, created from number_of_reviews_ltm.<br>
>
Listings in the top 25% of demand are labelled Good.
________________________________________
# Objectives 🎯
•	Build a realistic demand-based success metric<br>
•	Prevent all forms of data leakage<br>
•	Compare linear vs. non-linear models<br>
•	Provide interpretable, business-friendly insights
________________________________________
# Success Metric: O_score 📊
•	O_score = scaled number_of_reviews_ltm (0–1), using training-only min/max<br>
•	Listings with O_score ≥ Q3 (75th percentile) of training set → Good<br>
This ensures success is purely based on real market demand.
________________________________________
# Data Preprocessing 🧹
All steps use train-only statistics.<br>
✔ Cleaning<br>
•	Converted price, host_response_rate, host_acceptance_rate, bathrooms_text → numeric<br>
•	Simplified categories (e.g., “Entire home/apt” → “Entire home”)<br>
✔ Missing Values<br>
•	Median imputation (training median only)<br>
✔ Outlier Handling<br>
•	Capped price, minimum_nights using IQR method<br>
✔ Feature Engineering<br>
•	Created capacity = beds + bedrooms + bathrooms<br>
✔ Train/Test Split<br>
•	80/20 chronological split using host_since<br>
✔ Encoding<br>
•	One-Hot Encoding using dummyVars<br>
________________________________________
# Hypotheses 🧠
1.	🏠 Entire homes outperform private rooms<br>
2.	👍 100% acceptance rate increases success<br>
3.	⏳ Short minimum stays (1–2) outperform long stays<br>
4.	⭐ Superhost + ⚡ Instant Bookable = highest success<br>
5.	💲 Moderate prices (Q2/Q3) perform best<br>
6.	🛁 More bathrooms → higher success<br>
________________________________________
# Models Built 🤖
1️⃣ Elastic Net (glmnet)<br>
•	Tuned alpha<br>
•	Produces Odds Ratios for interpretation<br>
2️⃣ Decision Tree (CART)<br>
•	Tuned cp<br>
•	Captures non-linear patterns<br>
________________________________________
# Model Performance 🏆

| Model                     | Accuracy |   AUC   |
|---------------------------|----------|---------|
| Decision Tree (Tuned)     | 0.7087   | 0.8385  |
| Decision Tree (Baseline)  | 0.7350   | 0.8349  |
| Elastic Net (Tuned)       | 0.7069   | 0.7352  |
| Elastic Net (Baseline)    | 0.7251   | 0.7350  |

➡️ Decision Tree performed best.<br>
________________________________________
# Model Interpretation (with Business Meaning) 🔍
⭐ Superhost (TRUE)<br>
•	Odds Ratio: 2.49<br>
•	Superhosts are 2.5x more likely to be successful.<br>
⏳ Minimum Nights<br>
•	OR: 0.804<br>
•	Higher minimum stays reduce success.<br>
🛏 Beds<br>
•	OR: 1.44<br>
•	Each extra bed increases success by 44%.<br>
🧮 Capacity<br>
•	OR: 0.846<br>
•	Shows non-linear interactions with other features.<br>
💲 Price<br>
•	OR: 0.999<br>
•	Marginal effect; demand depends more on convenience/quality.<br>
________________________________________
# Key Takeaways 💡
•	⭐ Superhost status has the largest positive impact<br>
•	⏳ Allowing short stays increases bookings<br>
•	🛏 Increase bed capacity where possible<br>
•	💲 Price plays a smaller role than expected<br>
•	🔄 Host acceptance/response rates remain critical
________________________________________
# Limitations & Future Improvements 🧭
•	Trees may overfit → use Gradient Boosting / Random Forest<br>
•	Add neighbourhood/seasonal features<br>
•	Use better imputation (e.g., KNN, model-based)<br>
•	Deploy as Success Score Dashboard for hosts
________________________________________
# Productization Potential 🚀
This system can be deployed as:<br>
📈 Host Dashboard<br>
Real-time success score + optimisation suggestions<br>
(e.g., “Reduce minimum_nights to 2”.)<br>
🧠 Internal Airbnb Recommender<br>
Identify low-performing listings and recommend actions.<br>
________________________________________
# Technologies Used 🛠 
•	R<br>
•	dplyr, ggplot2<br>
•	caret<br>
•	glmnet<br>
•	rpart, rpart.plot<br>
•	pROC<br>

