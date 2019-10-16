# Water Source Identification from Water Quality Parameters using Fully Connected Neural Networks


**OBJECTIVE :** Water samples at various points of interest from all over the country are collected throughout the year. These data consist of around 30-45 water quality parameters per observation. The quality parameter values differ both spatially and temporally. They are also non-linearly dependent on each other and the water source. This project tries to understand the relationship of water quality parameters with the water source using a fully connected deep neural network.


**METHODS :** The data came from various organizations and so needed to be curated. As, there was a significantly large number of missing values for most of the quality parameters, among the 45 parameters only 8 of them were used. The final water quality parameters were temperature, PH level, electrical conductivity, chloride content level, alkali level, turbidity, dissolved oxygen and biochemical oxygen demand.  A simple fully connected network is used. The input to the model are the eight water quality parameters and the output is a probability distribution over possible water sources. 


**RESULTS :** Different variations of FCNs were experimented with to see which architecture gave the highest accuracy. An architecture where the number of neurons per layer increased gradually and then decreased to the number of sources for the last layer, gave the best performance. With a learning rate of 0.001, minibatch size of 64, for 1000 epochs gave an accuracy of 98.33% on the train set and 79% on the test set. Further regularization will most likely improve the results.


**CONCLUSION :** A water source identification model like this one can give some idea about the water source of a sample taken downstream of a river. Adding this sort of information to water quality reports will shed more light on how the water quality parameters are changing downstream of a water source.

