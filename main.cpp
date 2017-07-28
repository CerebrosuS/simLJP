/* Copyright 2017 <Christian Krippendorf>
 *
 * Permission is hereby granted, free of
 * charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#include <iostream>
#include <mkl.h>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <random>
#include <sys/stat.h>
#include <ctime>
#include <fstream>

#define EIGEN_USE_MKL_ALL

// Cofficients for the Lennard-Jones potential.
#define SIGMA 1.0e-1
#define EPSILON 1.0

// The mass of an atom. /kg
#define MASS 1

// Total number of particles to simulate.
#define TOTAL_PARTICLE 64

// Total number of simulation loops.
#define TOTAL_TIMESTEPS 100

// Single timestep for integration. /s
#define TIMESTEP 1e-7

using namespace Eigen;

// Define csv format for eigen
const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");

// Constant variables and information.
const char * const __version__ = "1.0";
const char * const __author__ = "Christian Krippendorf";
const char * const __email__ = "Coding@Christian-Krippendorf.de";

/** 
 * \brief Manipulate the position and velocity matrices for border conditions.
 *
 * \param[in] mp Reference to the position matrix of all particles /m.
 * \param[in] mv Reference to the velocity matrix of all particles /(m/s).
 * \param[in] closed True if a limited and closed box should be simulated, 
 *            else false. If it is not closed an algorithm put every particle 
 *            on the opposit site on reaching the border.
 * \param[in] left Left border of the box /m.
 * \param[in] right Right border of the box /m.
 * \param[in] top Top border of the box /m.
 * \param[in] bottom Bottom border of the box /m.
 * \param[in] front Front border of the box /m.
 * \param[in] back Back border of the box /m. */
void border_handling(MatrixXd &mp, MatrixXd &mv, bool closed, double left,
  double right, double top, double bottom, double front, double back) {
  if (closed) {
    // If one of the particles reaches the end of the box, the velocity has to
    // be reverted (multiplication with -1). The only problem is to decide with
    // component of the vector has to be inverted.

    // Go throught all particle and search for a position which is outside the
    // box.
    for (int pi = 0; pi < mp.cols(); pi++) {
      if (mp(0, pi) > right || mp(0, pi) < left)
        mv(0, pi) *= -1;

      if (mp(1, pi) > top || mp(1, pi) < bottom)
        mv(1, pi) *= -1;

      if (mp(2, pi) > back || mp(2, pi) < front)
        mv(2, pi) *= -1;
    }
  }
}

/** 
 * \brief Initialize the velocities of the particles.
 *
 * The velocities of the particles follow the Boltzmann-Maxwell distribution.
 * This is just another version of component-wise normal distribution, which
 * will be implemented here.
 *
 * \param[out] mv Reference to the velocity matrix of all particles. */
void init_velocities(MatrixXd &mv) {
  // Total number of columns (particles).
  int co = mv.cols();

  // Create the normal distribution object for generating random velocity
  // numbers.
  std::default_random_engine generator;
  std::normal_distribution<double> dist(0.0, 2.0);

  // Calculate velocity components for every particle.
  for (int pi = 0; pi < co; pi++) {
    mv(0, pi) = dist(generator);
    mv(1, pi) = dist(generator);
    mv(2, pi) = dist(generator);
  }
}

/** 
 * \brief Initialize the positions of all particles.
 *
 * The particles will be positioned like equal distanced particles in a
 * cube. Therefore the number of total particles should be the third power of
 * any natural number.
 *
 * \param[out] mp Reference to the position matrix of all particles. */
void init_grid(MatrixXd &mp) {
  // Position variables for counting over the loops.
  int px = 0, py = 0, pz = 0;

  // Total number of columns (particles).
  int co = mp.cols();

  // The number of rows gives the dimension. The number of columns gives the
  // number of all particles. The number of particles per dimension side
  // should be the dimension root of the particle number. Otherwise the number
  // of particles is wrong.
  double po = cbrt(co);
  if (fmod(po, 1) != 0)
    std::cout << "Error: Wrong size of particles." << std::endl;

  // Got through all particle postitions and give them a position number.
  for (int pi = 0; pi < co; pi++) {
    mp(0, pi) = px;
    mp(1, pi) = py;
    mp(2, pi) = pz;

    // If the x position is a multiple of po value, reset the px value to
    // zero and increase the y position. The same calculation follows with
    // y and z position. */
    px++;
    if ((px % (int)po) == 0) {
      px = 0;
      py++;
    }
    if (py != 0 && (py % (int)po) == 0) {
      py = 0;
      pz++;
    }
  }
}

/** 
 * \brief Calculate the Lennard-Jones potential energy force for all particles.
 *
 * \param[in] vp Reference to the vector object of the particle to calculate the
 *               final force for.
 * \param[in] mp Reference to the matrix object of all surrounding particles.
 * \param[out] mpo Reference to the matrix object where the final force will be
 *                 stored. */
void calc_lenjon_force(const VectorXd &vp, const MatrixXd &mp, MatrixXd &mpo) {
  // Get distance between the main particle and all surrounding particles.
  MatrixXd rp = mp-vp.replicate(1, mp.cols());

  // Calculate the distance of the particles by the norm.
  VectorXd rpn0 = rp.colwise().norm().cwiseInverse();

  // Calculate the resulting forces as magnitude.
  VectorXd rpn1 = rpn0*SIGMA;
  rpn1 = 24*EPSILON*(2*rpn1.array().pow(7.0)-rpn1.array().pow(13.0));

  // Go back to the component wise view.
  mpo = rp.array().rowwise()*rpn1.cwiseProduct(rpn0).transpose().array();
}

/** 
 * \brief Calculation of the particle accelerations based on the resulting 
 *        forces.
 *
 * \param[in] mp Matrix object for the positions with 3 rows and n columns.
 * \param[out] ma Matrix object for accelerations with 3 rows and n columns. */
void calc_accel(const MatrixXd &mp, MatrixXd &ma) {
  // Number of columns (particles).
  int co = mp.cols();

  // Temporary vector/matrix objectes for calculation.
  VectorXd vf(3, 1);
  MatrixXd mpo(3, co);

  for (int pi = 0; pi < co; pi++) {
    VectorXd vp = mp.col(pi);
    calc_lenjon_force(vp, mp.block(0, pi+1, 3, co-pi-1), mpo);
    ma.col(pi) = mpo.rowwise().sum();
  }
}

/** 
 * \brief Test whether a path exist or not.
 *
 * \return True if path exist, else false. */
bool path_exist(const char* path) {
  struct stat my_stat;
  return (stat(path, &my_stat) == 0);
}

/** 
 * \brief Initialize serialization.
 *
 * Search for a saving path and create it if neccessary. This method should be
 * optimized throught a configuration file.
 *
 * \return Name of the output path. */
std::string init_serialize() {
  // Time data object for getting the raw data.
  time_t rawtime;
  struct tm *timeinfo;

  // String containing the time information in the right format.
  char tbuf[80];

  // Get current datetime from the time object.
  time(&rawtime);
  timeinfo = localtime(&rawtime);

  // Convert the datetime information to string.
  strftime(tbuf, sizeof(tbuf), "%d-%m-%Y_%I-%M-%S", timeinfo);

  // Create final path as string with prefix.
  std::string path = std::string("mds-") + std::string(tbuf) + std::string("/");
  mkdir(path.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IWGRP |
    S_IXGRP);

  return path;
}

/** 
 * \brief Write the given matrices to file.
 *
 * Get all references to the matrices and write them into a separate csv file
 * in the given path.
 *
 * \param[in] mp Matrix object for the positions with 3 rows and n columns.
 * \param[in] ma Matrix object for accelerations with 3 rows and n columns.
 * \param[in] mv Matrix object for velocties with 3 rows and n columns.
 * \param[in] count Number of loop; This gives information about the number of 
 *                  file to write in. */
void write(const MatrixXd &mp, const MatrixXd &mv, const MatrixXd &ma,
	   const std::string &path, const int &count) {
  // Open the output stream.
  std::ofstream out((path + std::string("/mds-") + std::to_string(count) +
		     std::string(".csv")).c_str());

  // Write data into the stream in an appropriate data format.
  out << mp.transpose().format(CSVFormat);

  // Close the output stream.
  out.close();
}

/** 
 * \brief Simulate the system by calculation with velocity verlet algorithm.
 *
 * \param[in] mp Reference to the position matrix of all particles.
 * \param[in] mv Reference to the velocity matrix of all particles.
 * \param[in] ma Reference to the acceleration matrix of all particles. 
 * \param[in] serialize True if serialization wanted, else false. */
void simulate(MatrixXd &mp, MatrixXd &mv, MatrixXd &ma, bool serialize) {
  std::cout << "\nStart simulatin process..." << std::flush;

  // If serialization is wanted. Initialize the system to do so.
  std::string path;
  if (serialize)
    path = init_serialize();

  // Calculate box borders from number of particles.
  double po = cbrt(mp.cols());
  if (fmod(po, 1) != 0)
    std::cout << "Error: Wrong size of particles." << std::endl;

  // Calculate accelerations
  calc_accel(mp, ma);

  // Temporary calculations that will be done here once instead of multiple
  // times inside the loop.
  double td205 = 0.5 * std::pow(TIMESTEP, 2);
  double td05 = 0.5 * TIMESTEP;

  // Main timestep loop
  for (int ts = 0; ts < TOTAL_TIMESTEPS; ts++) {
    // Implementation of the Störmer-Velocity-Verlet algorithm.
    mp = mp + mv*TIMESTEP + ma*td205;
    calc_accel(mp, ma);
    mv += ma*td05;

    // Correct the velocities and/or positions related to the way of handling
    // border conditions.
    border_handling(mp, mv, true, 0, po, 0, po, 0, po);

    // Write current state to file.
    if (serialize)
      write(mp, mv, ma, path, ts);
  }

  std::cout << "finish!" << std::endl << std::flush;
}

/** 
 * \brief Write short information about hte application. */
void info() {
  std::cout << "Molecular Dynamic Simulation (Ver. " << __version__ << ")" <<
    std::endl << "by " << __author__ << " <" << __email__ << ">" << std::endl;
}

/** 
 * \brief Main entry function. */
int main(int argc, char **argv) {
    // Print application starting information
    info();

    // Define matrices of position, velocity and acceleration.
    MatrixXd mp(3, TOTAL_PARTICLE), mv(3, TOTAL_PARTICLE),
      ma(3, TOTAL_PARTICLE);

    // Initialize the position and velocity matrices to solve initial condition
    // problem. 
    init_grid(mp);
    init_velocities(mv);

    // Start timer.
    std::clock_t stime;
    double duration;
    stime = std::clock();
    
    // Start the main simulation process.
    simulate(mp, mv, ma, true);

    // End timer and show result.
    duration = (std::clock() - stime) / (double) CLOCKS_PER_SEC;
    std::cout << "Time needed for simulation: " << duration << "s" << std::endl;

    return 0;
}
