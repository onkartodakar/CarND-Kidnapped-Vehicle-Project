/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  std::default_random_engine gen;
  num_particles = 100;

  // define normal distributions for input GPS data
  normal_distribution<double> N_x(x, std[0]);
  normal_distribution<double> N_y(y, std[1]);
  normal_distribution<double> N_theta(theta, std[2]);

    // Initialize 
  for(unsigned int i=0; i<num_particles; i++){
    Particle p;
    p.id = i;
    p.x = N_x(gen);
    p.y = N_y(gen);
    p.theta = N_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
    weights.push_back(1.0);
  }

  is_initialized = true;

  //  cout << "Initialized" << endl;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  default_random_engine gen;
  
  // This line creates a normal (Gaussian) noise for x, y and theta around zero. 
  //Noise has been set propotional to the time step delta_t
  normal_distribution<double> dist_x(0, std_x*delta_t);
  normal_distribution<double> dist_y(0, std_y*delta_t);
  normal_distribution<double> dist_theta(0, std_theta*delta_t);
  
  for (unsigned int i = 0; i < num_particles; ++i){
    Particle &particle = particles[i]; //create a pointer to the ith particle
    if (yaw_rate ==0){
      particle.x += velocity*delta_t*cos(particle.theta) + dist_x(gen);
      particle.y += velocity*delta_t*sin(particle.theta) + dist_y(gen);
      particle.theta= particle.theta + dist_theta(gen);
    }else{
      particle.x += (velocity/yaw_rate)*(sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta)) + dist_x(gen);
      particle.y += (velocity/yaw_rate)*(cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t)) + dist_y(gen);
      particle.theta += yaw_rate*delta_t + dist_theta(gen);
    }
  }  
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.
  double min_distance, dist, dx, dy;
  int min_i;

  for(unsigned i = 0; i < observations.size(); i++){
    auto obs = observations[i];
    min_distance = INFINITY;
    min_i = -1;
    for(unsigned j = 0; j < predicted.size(); j++){
      auto pred_lm = predicted[j];
      dx = (pred_lm.x - obs.x);
      dy = (pred_lm.y - obs.y);
      dist = dx*dx + dy*dy;
      if(dist < min_distance){
        min_distance = dist;
        min_i = j;
      }
    }
    observations[i].id = min_i; // Use index of landmark as the ID (rather than the id field)
  } 

}

inline const double gaussian_2d(const LandmarkObs& obs, const LandmarkObs &lm, const double sigma[])
{
  auto cov_x = sigma[0]*sigma[0];
  auto cov_y = sigma[1]*sigma[1];
  auto normalizer = 2.0*M_PI*sigma[0]*sigma[1];
  auto dx = (obs.x - lm.x);
  auto dy = (obs.y - lm.y);
  return exp(-(dx*dx/(2*cov_x) + dy*dy/(2*cov_y)))/normalizer;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
  std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  //cout << "In updateWeights" << endl;
  for(unsigned int i=0; i<num_particles ; i++){

    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    //Consider landmarks within the sensor range

    vector<LandmarkObs> inRangeLandmarks;

    for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++){

      double landmark_x = map_landmarks.landmark_list[j].x_f;
      double landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;

      double distance = sqrt(pow((p_x - landmark_x),2) + pow((p_y - landmark_y),2));

      if(distance <= sensor_range)
        inRangeLandmarks.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});

    }

    vector<LandmarkObs> transformedObservations;

    for(unsigned int j = 0 ; j < observations.size(); j++){
      double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
      double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
      transformedObservations.push_back(LandmarkObs{observations[j].id, t_x, t_y});
    }

    dataAssociation(inRangeLandmarks, transformedObservations);

    double total_prob = 1.0f;
    double sigma_landmark [2] = {0.3, 0.3}; // Landmark measurement uncertainty [x [m], y [m]]

    for(unsigned int j = 0; j<transformedObservations.size(); j++){
      auto current_obs = transformedObservations[j];
      auto associated_location = inRangeLandmarks[current_obs.id];
      double pdf = gaussian_2d(current_obs, associated_location, sigma_landmark);
      total_prob *= pdf;
    }

    particles[i].weight = total_prob;
    weights[i] = total_prob;

  }
}



void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  std::discrete_distribution<int> d(weights.begin(), weights.end());
  std::vector<Particle> new_particles;

  for(unsigned i = 0; i < num_particles; i++)
  {
    auto ind = d(gen);
    new_particles.push_back(std::move(particles[ind]));
  }
  particles = std::move(new_particles);

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
  }
  string ParticleFilter::getSenseX(Particle best)
  {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
  }
  string ParticleFilter::getSenseY(Particle best)
  {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
  }
