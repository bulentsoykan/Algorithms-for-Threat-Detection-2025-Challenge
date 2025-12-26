# Data Description

The ATD 2025 Challenge uses ship [Automatic Identification System (AIS)](https://hub.marinecadastre.gov/pages/vesseltraffic)
data, obtained from the U.S. Coast Guard. Each
row in the dataset consists of a ship's information, such as unique identification, course, and
speed, at a precise location and time. Each row represents a posit -- a 
point on a ship's trajectory. 

For the challenge, we will focus on posits in the Gulf Coast area. Using our data processor, we
subsampled the data, removed invalid values and stationary points, and added a small amount of
random noise the posit time and location. We discard each posit's ship ID, 
resulting in anonymized AIS data.

In this challenge you need to construct tracks from 24 hours worth of anonymous posits near the 
southern US coast. 

## Schema

The schema of the *unlabeled* dataset is shown below:

| Name     | Description                                                     | Example             | Units           | Format                       | Type       |
|----------|-----------------------------------------------------------------|---------------------|-----------------|------------------------------|------------|
| point_id | 0-indexed unique identifier for each *posit* in the day         | 246                 |                 |                              | `int`      |
| time     | Full UTC date and time in ISO 8601 format                       | 2024-01-01T00:00:01 |                 | `yyyy-MM-dd'T'HH:mm:ss.SSSZ` | `datetime` |
| lat      | Latitude                                                        | 30.0455172          | decimal degrees | `XX.XXXXXXX`                 | `float`    |
| lon      | Longitude                                                       | -90.5421262         | decimal degrees | `XXX.XXXXXXX`                | `float`    |
| speed    | Speed Over Ground <br> The vessel's true speed of progress      | 15.5                | knots           | `XXX.X`                      | `float`    |
| course   | Course Over Ground <br> The vessel's true direction of progress | 101.8               | degrees         | `XXX.X`                      | `float`    |
