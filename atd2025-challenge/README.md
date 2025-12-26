# atd2025

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![platform](
https://img.shields.io/badge/platform-linux--64%20|%20win--64%20|%20osx--64-blue.svg)](
https://img.shields.io/badge/platform-linux--64%20|%20win--64%20|%20osx--64-blue)
[![python](
https://img.shields.io/badge/python-3.9|3.10|3.11|3.12|3.13-blue.svg)](
https://img.shields.io/badge/python-3.9|3.10|3.11|3.12|3.13-blue)
[![ATD2025 Docs](
https://img.shields.io/badge/atd2025_docs-online-green)](
https://algorithms-for-threat-detection.gitlab.io/2025/atd2025/)
[![slack](
https://img.shields.io/badge/slack-online-green)](
http://atdchallenge.slack.com)

Welcome to the ATD2025 challenge!

In this README, we briefly define this year's [challenge problem](#problem-description).
Next, we outline the [contents of this repository](#repository-contents).
After, we provide some links to our online documentation that will help you [get started](#getting-started).
Finally, we provide a link to the challenge's [slack](#chat-on-slack) and a [schedule of upcoming events and deadlines](#calendar-of-events).

## Problem Description

In this challenge, you will construct ship tracks from individual observations that
include a ship’s position and velocity at a given time, but no information about the
ship’s identity.

You will work with publicly
available [Automatic Identification System (AIS) data](https://coast.noaa.gov/digitalcoast/tools/ais.html)
that has been stripped of identifying information and subsampled.

An individual observation of a ship is called a *posit*. Each AIS posit includes the
time of observation, a unique identifier for the ship, the ship's geolocation (
position), the ship's velocity, and other features such as the ship's name and size. In
this challenge, all identifying information (including the ship's unique ID, name, and
size) will be stripped from the posits. You will only get the time of observation, ship
position, and ship velocity.

Constructing ship tracks amounts to assigning sets of posits to an unknown number of
ships. We’ll ask you to do that for 24 hours of data covering most of the Gulf Coast and
the Florida Straits.

Typically, AIS data is highly sampled, but to make this problem more like real
applications, we significantly down-sample the number of observations for each ship. To
increase difficulty, we also add a small amount of normally distributed error to
positions and a small amount of uniformly distributed error to the observation times.

## Repository Contents

This repository contains

* `src/atd2025`:  Python library that includes tools for reading, writing, processing,
  and visualizing ATD 2025 data; baseline algorithms; and the metric we will use to
  determine the challenge winner.
* `src/atd2025/data`: ATD2025 Datasets.
* `docs/`: A static site containing both narrative and automatically generated API
  documentation ([hosted online here](https://algorithms-for-threat-detection.gitlab.io/2025/atd2025/))
* `scripts/`: Examples of using the `atd2025` library for common tasks.

## Getting Started

First visit
the ["Getting Started" section](https://algorithms-for-threat-detection.gitlab.io/2025/atd2025/getting-started/dependencies/)
of our documentation to configure your ATD2025 development environment. Then try our 
[tutorials](https://algorithms-for-threat-detection.gitlab.io/2025/atd2025/cookbook/predictions/).

For the `atd2025` Python library's API reference, visit
our [API docs](https://algorithms-for-threat-detection.gitlab.io/2025/atd2025/api/io/).

Go [here](docs/submission.md) to read how to submit your predictions for evaluation.

## Chat on Slack

You can use the
ATD2025 [![slack](https://img.shields.io/badge/slack-online-green)](http://atdchallenge.slack.com)
to chat with other participants and the ATD2025 administrative team in a persistent
chat. As part of the the ATD2025 onboarding process, you should have received an email
inviting you to join the `atd2025` slack group. Please reach out to challenge organizers
through your PI if you did not receive this invitation.

## Calendar of Events

The ATD2025 challenge consists of two webinars, three problem set deadlines, and an
evaluation deadline.

| Date        | Time        | Description                                                  |
|-------------|-------------|--------------------------------------------------------------|
| February 26 | 2:00 PM EST | Initial Webinar                                              |
| May 14      | 1:00 PM EDT | Kickoff Webinar for all Participants and PIs                 |
| June 9      | 8:00 AM EDT | *Optional* Predictions for Dataset 1 Deadline                |
| June 10     |             | Dataset 2 and Solution for Dataset 1 Posted                  |
| July 14     | 8:00 AM EDT | *Optional* Predictions for Dataset 2 Deadline                |
| July 15     |             | Dataset 3 and Solution for Dataset 2 Posted                  |
| August 4    | 8:00 AM EDT | *Optional* Predictions for Dataset 3 Deadline                |
| August 5    |             | Evaluation Dataset and Solution for Dataset 3 Posted         |
| August 18   | 8:00 AM EDT | **Required** Predictions for Evaluation Dataset Deadline     |
| August 25   |             | Solution for Evaluation Dataset Posted and Winner Determined |

# We hope you enjoy the ATD 2025 Challenge!
