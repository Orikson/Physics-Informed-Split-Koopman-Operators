# Physics-informed Split Koopman Operators for Data-efficient Soft Robotic Simulation

<!-- Insert figure that shows frames of rendered soft robot moving around -->

This repository contains the code for the paper "Physics-informed Split Koopman Operators for Data-efficient Soft Robotic Simulation" by [Eron Ristich](https://eron.ristich.com), [Lei Zhang](https://scholar.google.com/citations?user=5Fbl9l8AAAAJ&hl=zh-CN), [Yi Ren](https://search.asu.edu/profile/2422024), and [Jiefeng Sun](https://jiefengsun.github.io/). The paper is currently under review.

Check out the [paper's website](https://sunrobotics.lab.asu.edu/blog/2024/ristich-icra-2025/) for more information.

## Abstract
Koopman operator theory provides a powerful data-driven technique for modeling nonlinear dynamical systems in a linear framework, in comparison to computationally expensive and highly nonlinear physics-based simulations. However, Koopman operator-based models for soft robots are very high dimensional and require considerable amounts of data to properly resolve. Inspired by physics-informed techniques from machine learning, we present a novel physics-informed Koopman operator identification method that improves simulation accuracy for small dataset sizes. Through Strang splitting, the method takes advantage of both continuous and discrete Koopman operator approximation to obtain information both from trajectory and phase space data. The method is validated on a tendon-driven soft robotic arm, showing orders of magnitude improvement over standard methods in terms of the shape error. We envision this method can significantly reduce the data requirement of Koopman operators for systems with partially known physical models, and thus reduce the cost of obtaining data.

## Details of this repository
This repository contains implementations of standard linear, bilinear, and nonliear EDMDc.

In addition, it contains an implementation of the proposed method, PI-EDMDc.

Jacobian matrices are computed using automatic differentiation using the `pytorch` library, which is slow, but allows for very general dictionaries of functions.

