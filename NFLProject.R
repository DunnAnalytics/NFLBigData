install.packages("nflfastR")
library(tidyverse)
library(ggrepel)
library(nflreadr)
library(nflplotR)
library(caret)
library(pROC)

#what years should we use? 

data <- load_pbp(2012:2022)

D_was <- subset(data, home_team == "WAS"| away_team == "WAS")

#colnames(data)

#A model to predict Play type based on game statistics
#Outcome Variable is Play type play_type
#Predictors: Game Seconds Remaining, Distance from Endzone, yards to go, timeouts remaining, score differential, Win Probability

#all plays where Washington is on offense
D_was_off <- subset(D_was, posteam == "WAS")


D_was_off<- D_was_off[, c("game_id","play_type","game_seconds_remaining","yardline_100","ydstogo","posteam_timeouts_remaining","score_differential","wp", "half_seconds_remaining")]

#Specifying levels of outcome variable
subset_levels <- c("pass","run")
D_was_off <- D_was_off[D_was_off$play_type %in% subset_levels, ]

#Predictor 1: game_seconds_remaining
#Predictor 2: yardline_100
#Predictor 3: ydstogo
#predictor 4: posteam_timeouts_remaining
#predictor 5: score_differential
#predictor 6: wp (win probability)
#predictor 7: half_seconds_remaining

#visualize data
#score differential and win probability
plot1 <- D_was_off[, c("score_differential", "wp")]

plot(plot1$wp, # x-axis variable
     plot1$score_differential,  # y-axis variable
     xlim=c(0,1),
     ylim=c(-50,50), xlab = "Win Probability", ylab = "Score Differential", main = "Win Probability in Relation to Score Differential", col = "#5A1414") 
abline(h=0, # slope
       col="#FFB612", # color
       lty=1) # dashed line

#Plays per yardline
hist(D_was_off$yardline_100, xlab = "Yards From Opponent Endzone", main = "Frequency of Plays by Yardline", col = "#FFB612") 

#Play type frequency
barplot(table(D_was_off$play_type), col = "#5A1414", xlab = "Play Type", ylab = "Frequency", main = "Play Frequency")

#score differential distribution
boxplot(D_was_off$score_differential, col = "#5A1414", ylab = "Score Differential", main = "Score Differential Distribution")

#Win Probability throughout the game
plot(D_was_off$game_seconds_remaining, D_was_off$wp, col = "#5A1414", xlab = "Game Seconds Remaining", ylab = "Win Probability", main = "Win Probability Trends")
abline(h = 0.5, col = "#FFB612")

#Remove Na
D_was_off <- na.omit(D_was_off)

#train data 
train.rows <- sample(1:nrow(D_was_off), 0.8*nrow(D_was_off))

D_was_off_train <- D_was_off[train.rows,]


#holdout data
D_was_off_holdout <- D_was_off[-train.rows,]

#Model 1 Decision Tree Model
set.seed(474)
rpartGrid <- expand.grid(cp=seq(from=0.001, to=0.1, length=10))

fitControl <- trainControl(method="cv", number=5, classProbs = TRUE, summaryFunction = twoClassSummary)

D_was_rpart <- caret::train(play_type ~ game_seconds_remaining + yardline_100 + ydstogo + posteam_timeouts_remaining + score_differential + wp + half_seconds_remaining, data=D_was_off_train, 
                     method="rpart",
               trControl= fitControl,
               tuneGrid=rpartGrid,
               preProc=c("center","scale"))

D_was_rpart$results
D_was_rpart$bestTune
D_was_rpart$results[rownames(D_was_rpart$bestTune),]




#Model 2 Logistic Regression
set.seed(2024)
glmnetGrid <- expand.grid(alpha = seq(0, 1, by=0.25),
                          lambda = 10^seq(-3,0,by=0.5)) 
D_was_glm <- caret::train( play_type ~ game_seconds_remaining + yardline_100 + ydstogo + posteam_timeouts_remaining + score_differential + wp + half_seconds_remaining, data=D_was_off_train, 
                    method="glmnet", tuneGrid=glmnetGrid, trControl=fitControl, preProc=c("center", "scale"))


D_was_glm$results
D_was_glm$bestTune
D_was_glm$results[rownames(D_was_glm$bestTune),] 


#Model 3 Gradient Boosting Machine
set.seed(2024)

gbmGrid <- expand.grid(n.trees=c(2000),
                       shrinkage=c(.002),
                       interaction.depth=c(1,2),
                       n.minobsinnode=c(10))

D_was_GBM <- caret::train(play_type ~ game_seconds_remaining + yardline_100 + ydstogo + posteam_timeouts_remaining + score_differential + wp + half_seconds_remaining , data=D_was_off_train, method="gbm",
                   tuneGrid=gbmGrid,
                   trControl=fitControl,
                   preProc=c("center","scale"),
                   verbose=FALSE)

D_was_GBM$results[rownames(D_was_GBM$bestTune),]



#Choosing model using 1-SD Rule

#Model 1
#ROC:0.6984304  SD: 0.01799961

#Model 2
#ROC: 0.6442865  SD: 0.01365671

#Model 3
#ROC: 0.7177319 SD: 0.01024882

(0.7177319 - 0.6984304)/0.01799961

#Predict using Holdout set

Was_predict <- pROC::roc(D_was_off_holdout$play_type, predict(D_was_GBM, newdata = D_was_off_holdout, type = "prob")[,2])

Was_predict
plot(Was_predict, col = "#5A1414")

#creates new column as the predicted play type (pass or run)
D_was_off_holdout$Was_predicted   <- predict(D_was_GBM, newdata = D_was_off_holdout, type = "raw")

#confusion matrix comparing play type and predicted play type

D_was_off_holdout$play_type <- factor(D_was_off_holdout$play_type, levels = c("pass","run"))

wasConfusion <- caret::confusionMatrix(data = D_was_off_holdout$Was_predicted, reference = D_was_off_holdout$play_type)

wasConfusion





