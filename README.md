# WNBA Prediction Project

## Overview

The WNBA Prediction project is a predictive modeling project aimed at forecasting the outcomes of WNBA basketball games. Using machine learning techniques, the project involves data cleaning, feature selection, and model training to predict whether the home team will win or lose a game. The model achieved an accuracy of approximately 70.4%.

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
4. [Technologies Used](#technologies-used)

## Installation

To run this project, ensure you have the required Python packages installed:

```bash
pip install pandas scikit-learn
```

## Project Structure

The project is organized into several parts:

- **Part 1: Cleaning Data:** Involves importing and cleaning schedule and advanced statistics data, merging datasets, and preparing the data for modeling.

- **Part 2: Determining Predictors:** Utilizes Sequential Feature Selector to identify the top 10 predictors for the model.

- **Part 3: Creating and Testing Model:** Implements a logistic regression model and evaluates its performance using Monte Carlo simulation.

- **Part 4: Predicting Finals:** Predicts the outcome of a specific WNBA final matchup between the Las Vegas Aces and New York Liberty.

## Usage

To replicate the project's steps and results, follow the instructions in the Jupyter Notebook files provided. Make sure to have Python and the required packages installed.

```bash
jupyter notebook WNBA_Prediction.ipynb
```

## Technologies Used

- Python
- Pandas
- Scikit-learn
