# JusticeBot Project: Classifying sentences as facts or analysis

Our current task is to produce a binary classification system of sentences as **facts** or **analysis**. We use a bi-directional Recurrent Neural Network (RNN), specifically, a Gate Recurrent Unit model (GRU) to capture the various linguistic relations between the constituents of a sentence and output a two-dimensional vector representing the probability that the sentence belongs to either category.
