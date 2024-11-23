# Galaxy Projection along LOS and Rotation Curves

This repository contains a collection of functions and interactive tools designed to manage and analyze data from cosmological simulations, mainly to study kinematics and dynamics from spiral galaxies. The primary focus is on handling positions, velocities, and masses of particles from galaxies, transforming coordinate systems, creating mock observations of velocity fields and constructing rotation curves.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Acknowledgments](#acknowledgments)

## Features

- **Coordinate Transformations**: Convert positions and velocities between Cartesian and cylindrical coordinate systems.
- **Inertia Tensor Construction**: Calculate the inertia tensor and determine the angular momentum direction of a system.
- **System Rotation**: Rotate the system to align with a specific line of sight.
- **Line-of-Sight Velocity Calculation**: Compute the velocity of particles along the line of sight using imposed angles (inclination and position angle).
- **Tilted Ring Projection**: Project a body along the line of sight using the tilted ring method.
- **Data Decomposition**: Decompose an object at a fixed orientation to extract and analyze its components.
- **Mock Observation Generation**: Create mock observations of velocity fields and generate images.
- **Rotation Curve Construction**: Build rotation curves of galaxies using curve fitting to model velocities and errors along the radius.
- **Interactive Visualization**: Use interactive widgets to visualize various rotation curves and velocity fields with adjustable parameters.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/LevithanW/Galaxy_Projection.git

## Acknowledgments

Feel free to customize and expand upon this README to suit your project's specific needs. If you use this material, please acknowledge the source.
