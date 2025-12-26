# Problem Description

In this challenge, you will construct ship tracks from observations that only include a ship’s position and velocity at a given time, but no information about the ship’s identity.

This problem has a couple of applications. Some sources of movement data, such as [GMTI](https://en.wikipedia.org/wiki/Moving_target_indication), tell you that something is moving with a certain velocity at a given time and place, but not whether observations at different times correspond with the same mover. In other situations, we combine movement data from different sources without knowing how to map movers observed in one data source to movers observed in the other.

You will work with publicly available [Automatic Identification System (AIS) data](https://coast.noaa.gov/digitalcoast/tools/ais.html) that has been stripped of identifying information and subsampled.

An individual observation of a ship is called a *posit*. Each AIS posit includes the time of observation, a unique identifier for the ship, the ship's geolocation (position), the ship's velocity, and other features such as the ship's name and size. In this challenge, all identifying information (including the ship's unique ID, name, and size) will be stripped from the posits. You will only get the time of observation, ship position, and ship velocity.

Constructing ship tracks amounts to assigning sets of posits to an unknown number of ships. We’ll ask you to do that for 24 hours of data covering most of the Gulf Coast and the Florida Straits.

Typically, AIS data is highly sampled, but to make this problem more like real applications, we significantly down-sample the number of observations for each ship. To increase difficulty, we also add a small amount of normally distributed error to positions and a small amount of uniformly distributed error to the observation times.

We will determine the challenge winner by evaluating predictions for a dataset that we
will provide on August 5. Before then, we will provide similar datasets over the same
region. You may optionally submit your predictions for those datasets to get our
evaluations. We also provide you the code we use to download and process the
AIS data to make the anonymized sets. Using this code, you can make as much data as
you’d like to test your ideas. You can measure your performance with the evaluation
script that we will use to determine the challenge winner.

See [Data Description](data.md) for an explanation of the formatting we use for the AIS
dataset.

See [Submitting Predictions](submission.md) for an explanation of the format your 
predictions must have.

See [Evaluation of Predictions](metric.md) for an explanation of the metric we’ll
use to determine the challenge winner.

See the [Tutorials Section](cookbook/predictions.md) for examples of how to use the code we’ve provided.
