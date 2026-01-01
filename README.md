# Santa 2025 - Christmas Tree Packing Challenge

## Overview
**Competition:** [Santa 2025 - Christmas Tree Packing Challenge](https://www.kaggle.com/competitions/santa-2025)  
**Goal:** Pack a set of Christmas tree polygons into the smallest possible square bounding box.  
**Context:** A classic optimization problem with a festive twist, continuing Kaggle's annual holiday tradition.

## Problem Description
You are given a set of identical Christmas tree shapes (defined as polygons with 15 vertices). For each problem instance `n` (where `n` ranges from 1 to 200), you must arrange `n` trees such that:
1.  **No Overlaps:** Trees must not overlap with each other.
2.  **Square Bounds:** All trees must fit within a square bounding box.
3.  **Minimization:** The side length of the bounding box should be minimized.

### Output Format
For each tree in a configuration `n`, you must specify:
-   `x`: X-coordinate of the tree's center.
-   `y`: Y-coordinate of the tree's center.
-   `deg`: Rotation of the tree in degrees.

## Evaluation Metric
The score for a submission is the sum of the normalized area of the square bounding box for each puzzle size `n`.

$$ \text{Score} = \sum_{n=1}^{200} \frac{s_n^2}{n} $$

Where:
-   $s_n$ is the side length of the smallest square bounding box that contains all $n$ trees.
-   $n$ is the number of trees.

Lower scores are better.


