# Modified Global K-Means Clustering Project

This project implements the Modified Global K-Means (MGKM) algorithm, based on the work of Adil Bagirov, and compares its performance with several other K-Means variants. The analysis is performed on various real-world and synthetic datasets.

This work was completed as a Bachelor's project under the supervision of Dr. Najmeh Hosseini Monjezi.

## 📜 Description

The primary goal of this project is to provide a Python implementation of the Modified Global K-Means algorithm and evaluate its effectiveness. The project explores how MGKM performs in terms of clustering quality (measured by an objective function and error rate against known optima) and computational time.

The Jupyter Notebook (`Project.ipynb`) includes:
* Implementations of various K-Means algorithms.
* Functions for testing these algorithms on different datasets.
* Visualization tools for 2D and 3D cluster representation.
* Comparative analysis across datasets: Iris, Liver Disorders, TSPLIB1060, Heart Disease, and Breast Cancer.

## ⚙️ Algorithms Implemented

The notebook contains implementations for the following clustering algorithms:

1.  **Standard K-Means**: A custom implementation of the classic K-Means algorithm.
    * Includes functions for random center initialization (`initialize_centers`), cluster assignment (`assign_clusters`), center updates (`update_centers`), and inertia calculation (`calculate_inertia`).
2.  **Modified Global K-Means (MGKM)**: The core algorithm of this project (`mgkm`). It iteratively adds new cluster centers by utilizing `algorithm_3` and refines them using K-Means steps.
3.  **Algorithm 3**: A helper algorithm (`algorithm_3`) crucial for MGKM, which determines the next best candidate for a cluster center based on an objective function `compute_fk`.
4.  **Global K-Means (GKM)**: An implementation of the Global K-Means algorithm (`global_kmeans`).
5.  **Custom K-Means**: A K-Means variant (`custom_kmeans_best_inertia`) with a specific initialization strategy: iteratively adding the farthest point to the existing centers.
6.  **Fast Global K-Means (Fast GKM)**: An implementation of the Fast Global K-Means algorithm (`fast_gkm`).
7.  **Algorithm 4**: An algorithm (`algorithm_4`) to determine a "good" number of clusters (k) by iteratively adding centers (using `algorithm_3`) and stopping when the improvement in the objective function `fk` falls below a specified tolerance.

Key helper functions include:
* `compute_fk(X, centers)`: Calculates the objective function value, defined as the mean of the minimum squared Euclidean distances from each data point to the nearest cluster center.
* `compute_d(X, centers)`: Computes the minimum squared Euclidean distances of each data point to any of the existing cluster centers.
* `compute_S2(point, X, d)`: Identifies neighboring points used within `algorithm_3`.
* `compute_b_j(data, d)`: A helper function for the Fast GKM algorithm.

## 📊 Datasets Used

The algorithms are evaluated on the following datasets:
* **Iris Dataset**
* **Liver Disorders Dataset**
* **TSPLIB1060 Dataset**
* **Heart Disease Dataset**
* **Breast Cancer Dataset**

Data for these datasets is loaded within the notebook, primarily from text files.

## 📈 Project Structure and Usage

The project is contained within a single Jupyter Notebook: `Project.ipynb`.

1.  **Imports and Algorithm Definitions**:
    * Standard Python libraries such as NumPy, Matplotlib, Seaborn, Scikit-learn, and Pandas are imported.
    * All clustering algorithms and their helper functions are defined in the initial sections of the notebook.
    * Test functions (`test_mgkm`, `test_km`, `test_gkm`, `test_custom`, `test_fast`) are defined to evaluate each algorithm. These functions measure execution time and the percentage error `E` relative to pre-defined optimal objective function values (`f_opt`) for various `k` values.
    * Plotting utilities (`d2_plot`, `d3_plot`) are included for 2D and 3D visualization of the clustering results.

2.  **Dataset Analysis**:
    * For each dataset, the various algorithms are executed for a range of `k` values.
    * Performance metrics (time, error `E`, and objective function value) are tabulated.
    * Plots comparing `E` and `time` versus `k` are generated for MGKM and an averaged standard K-Means (referred to as MS K-Means in the notebook outputs).
    * Algorithm 4 is used to suggest an optimal `k` for some datasets, and the resulting clusters using MGKM with this `k` are visualized.
    * Elbow method plots are generated using the results from GKM to visually inspect potential optimal `k` values.

3.  **Experimental Section (CIRCULAR)**:
    * The notebook includes a function `distribute_centers_cosine_distance`, which appears to be an experimental approach for center initialization using cosine distances, though it's not integrated into the main comparative tests.

To run this project:
1.  Ensure all dependencies are installed.
2.  Open and run the `Project.ipynb` notebook in a Jupyter environment.
3.  Data files are expected to be located in a Google Drive directory as per the paths in the notebook (e.g., `/content/drive/My Drive/Modified Global Kmeans/Data sets/Real Data Sets/`). You may need to adjust these paths or mount your Google Drive if running in Google Colab.

## 🛠️ Dependencies

The project requires the following Python libraries:
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn` (sklearn)
* `pandas`

## ✍️ Author

MohammadReza Zaheri

## 🧑‍🏫 Supervisor

Dr. Najmeh Hosseini Monjezi

## 🙏 Acknowledgments

* The Modified Global K-Means algorithm implemented in this project is based on the research by Adil Bagirov. (It would be good to add a specific citation to the paper here if you have it).
* The datasets used are standard datasets available for clustering research.
