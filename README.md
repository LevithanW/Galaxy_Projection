# Galaxy Projection along LOS and Rotation Curves

This repository contains a collection of Python functions and interactive tools designed to manage and analyze data from cosmological simulations. The primary focus is on handling positions, velocities, and masses of particles (e.g., from galaxies), transforming coordinate systems, projecting observations, and constructing rotation curves.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Functions Module](#functions-module)
  - [Interactive Rotation Curves](#interactive-rotation-curves)
- [Dependencies](#dependencies)
- [Data](#data)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Coordinate Transformations**: Convert positions and velocities between Cartesian and cylindrical coordinate systems.
- **Inertia Tensor Construction**: Calculate the inertia tensor and determine the angular momentum direction of a system.
- **System Rotation**: Rotate the system to align with a specific line of sight.
- **Line-of-Sight Velocity Calculation**: Compute the velocity of particles along the line of sight using imposed angles (inclination and position angle).
- **Tilted Ring Projection**: Project a body along the line of sight using the tilted ring method.
- **Data Decomposition**: Decompose an object at a fixed orientation to extract and analyze its components.
- **Mock Observation Generation**: Create mock observations of velocity fields and generate velocity field images.
- **Rotation Curve Construction**: Build rotation curves of galaxies using curve fitting to model velocities and errors along the radius.
- **Interactive Visualization**: Use interactive widgets to visualize various rotation curves and velocity fields with adjustable parameters.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/yourrepository.git
