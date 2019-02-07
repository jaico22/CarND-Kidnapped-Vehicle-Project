/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen;

  num_particles = 100;  
  // Generate random number for gaussian noise addition

  // Define gaussian distributions
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  // Populate particles
  for (int i = 0; i < num_particles; i++){
    Particle newParticle; 
    newParticle.id = i;
    newParticle.x = dist_x(gen);
    newParticle.y = dist_y(gen);
    newParticle.theta = dist_theta(gen);
    newParticle.weight = 1.0f;
    particles.push_back(newParticle);
    weights.push_back(1.0f);
  }
  // Set initilization flag
  is_initialized = true; 
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  normal_distribution<double> dist_x(0.0,std_pos[0]);
  normal_distribution<double> dist_y(0.0,std_pos[1]);
  normal_distribution<double> dist_theta(0.0,std_pos[2]);
  // Update each particle
  for (int i = 0; i < num_particles; i++){
    // Update measurements with noise added
    if (fabs(yaw_rate) < 0.00001){
      particles[i].x += velocity*delta_t * cos(particles[i].theta);
      particles[i].y += velocity*delta_t * sin(particles[i].theta);
    }else{
      particles[i].x += velocity/yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) 
        - sin(particles[i].theta));
      particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) -
        cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);    
    // Store back in particle vector
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (int i = 0; i < observations.size(); i++){
    // Find nearest predicted position
    LandmarkObs obs = observations.at(i);
    //std::cout << " X = " << obs.x << " Y = " << obs.y << std::endl;
    double lowestDist = 10000000.0;
    for (int j = 0; j < predicted.size(); j++){
      LandmarkObs prd = predicted.at(j);
      double distance = dist(prd.x,prd.y,obs.x,obs.y);
      if (distance < lowestDist){
        lowestDist = distance;
        observations[i].id = prd.id;
      }
    }
    // Update observation's matching id
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  ////////////////////////////////////////////////////
  // 1. Transform observations into map coordinates //
  ////////////////////////////////////////////////////

  for (int i = 0; i < num_particles; i++){
    // Retreive particle
    // Coordinate transforms
    vector<LandmarkObs> tObs; 
    for (int j = 0; j < observations.size(); j++){
      LandmarkObs obs = observations.at(j);
      // Update coordinates
      obs.x = cos(particles[i].theta)*observations[j].x - sin(particles[i].theta)*observations[j].y + particles[i].x;
      obs.y = sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y + particles[i].y;
      tObs.push_back(obs);
    }
    ///////////////////////////
    // 2. Setup Associations //
    ///////////////////////////

    // Convert map to vector of landmarks. 
    vector<LandmarkObs> map_landmarks_landmarkObs;
    for (int j = 0;j < map_landmarks.landmark_list.size(); j++){
      LandmarkObs tmp_landmark;
      tmp_landmark.id = map_landmarks.landmark_list.at(j).id_i;
      tmp_landmark.x = map_landmarks.landmark_list.at(j).x_f;
      tmp_landmark.y = map_landmarks.landmark_list.at(j).y_f;
      // Only accept landmarks withing range of the sensor
      if (dist(tmp_landmark.x,tmp_landmark.y,particles[i].x,particles[i].y) < sensor_range){
        map_landmarks_landmarkObs.push_back(tmp_landmark);
      }
    }
    // Data Association
    dataAssociation(map_landmarks_landmarkObs,tObs);
    
    ////////////////////////////////
    // 3. Calculate Probabilities //
    ////////////////////////////////
    double p_weight = 1.0; 
    for (int j = 0; j < tObs.size(); j++){
      // Find correspond map id
      for (int k = 0; k < map_landmarks_landmarkObs.size(); k++){
        if (map_landmarks_landmarkObs.at(k).id == tObs.at(j).id){
          double newWeight = multiv_prob(std_landmark[0],std_landmark[1],tObs.at(j).x,tObs.at(j).y,
            map_landmarks_landmarkObs.at(k).x,map_landmarks_landmarkObs.at(k).y);
          p_weight *= newWeight;
        }
      }
    }
    // Update weight
    particles[i].weight = p_weight; 
    weights[i] = p_weight;
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // Generate discrete distribution of weights
  std::discrete_distribution<double> discrete_weights(weights.begin(),weights.end());
  std::vector<Particle> new_particles;
  // Populate with random weights
  std::random_device randDev;
  std::mt19937 gen(randDev());
  for (int i =0; i < num_particles; i++){
    new_particles.push_back(particles[discrete_weights(gen)]);
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}