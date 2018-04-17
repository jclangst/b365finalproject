## Missing Value Visualization in R

library(dbplyr)
library(ggplot2)
library(tidyverse)


mydata <- read.csv(file = "c:/Users/Christian/Documents/train.csv", header = TRUE)



#Prop of Non-Claims (0) versus Claims (1)
prop.table(table(mydata$target))


#Missing Value Plot by Category in Descending Order
data.frame(feature = names(mydata),
           per_miss = map_dbl(mydata, function(x) { sum(x == -1) / length(x) })) %>%
  ggplot(aes(x = reorder(feature,-per_miss), y = per_miss))+
  geom_bar(stat= "identity", color = "white", fill = "#5a64cd")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  labs(x = "", y ="% missing", title = "Missing Attributes by Feature")+
  scale_y_continuous(labels = scales::percent)
