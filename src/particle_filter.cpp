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

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /* Set the number of particles. Initialize all particles to first position (based on estimates of
     * x, y, theta and their uncertainties from GPS) and all weights to 1.
     * Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method (and others in this file).
     */
    num_particles = 50;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    for (unsigned int i = 0; i < num_particles; ++i) {
        Particle p;
        p.id = int(i);
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std[], double velocity, double yaw_rate) {
    /* Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
     * http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     * http://www.cplusplus.com/reference/random/default_random_engine
     */
    for (auto &p:particles) {
        if (abs(yaw_rate) < 1e-5) {
            p.x += velocity * cos(p.theta) * delta_t;
            p.y += velocity * sin(p.theta) * delta_t;
        } else {
            p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
            p.y += velocity / yaw_rate * (-cos(p.theta + yaw_rate * delta_t) + cos(p.theta));
            p.theta += yaw_rate * delta_t;
        }
        normal_distribution<double> dist_x(p.x, std[0]);
        normal_distribution<double> dist_y(p.y, std[1]);
        normal_distribution<double> dist_theta(p.theta, std[2]);
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    /* Find the predicted measurement that is closest to each observed measurement and assign the
     * observed measurement to this particular landmark.
     * NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
     * implement this method and use it as a helper during the updateWeights phase.
     */

    for (auto &observed_meas : observations) {
        double min_distance = numeric_limits<double>::max();
        for (const auto &predict_meas : predicted) {
            double distance = dist(predict_meas.x, predict_meas.y, observed_meas.x, observed_meas.y);
            if (distance < min_distance) {
                observed_meas.id = predict_meas.id;
                min_distance = distance;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    /* Update the weights of each particle using a mult-variate Gaussian distribution. You can read
     *   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
     *  according to the MAP'S coordinate system. You will need to transform between the two systems.
     * Keep in mind that this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement (look at equation 3.33
     *   http://planning.cs.uiuc.edu/node99.html
     */
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];

    double norm_factor = 0.0;

    // Iterate over all particles
    for (unsigned int i = 0; i < num_particles; ++i) {
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;

        // List all landmarks within sensor range of each particle
        vector<LandmarkObs> predicted_landmarks;

        for (const auto &map_landmark : map_landmarks.landmark_list) {
            int l_id = map_landmark.id_i;
            double l_x = (double) map_landmark.x_f;
            double l_y = (double) map_landmark.y_f;

            double d = dist(p_x, p_y, l_x, l_y);
            if (d < sensor_range) {
                LandmarkObs l_pred;
                l_pred.id = l_id;
                l_pred.x = l_x;
                l_pred.y = l_y;
                predicted_landmarks.push_back(l_pred);
            }
        }

        // List all observations in map coordinates
        vector<LandmarkObs> observed_landmarks_map_ref;
        for (size_t j = 0; j < observations.size(); ++j) {

            // Convert observation from particle(vehicle) to map coordinate system
            LandmarkObs transformed_obs;
            transformed_obs.x = cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y + p_x;
            transformed_obs.y = sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y + p_y;

            observed_landmarks_map_ref.push_back(transformed_obs);
        }

        // Find which observations correspond to which landmarks (associate ids)
        dataAssociation(predicted_landmarks, observed_landmarks_map_ref);

        // Compute the likelihood for each particle, that is the probablity of obtaining
        // current observations being in state (particle_x, particle_y, particle_theta)
        double particle_likelihood = 1.0;

        double mu_x, mu_y;
        for (const auto &obs : observed_landmarks_map_ref) {
            // Find corresponding landmark on map for centering gaussian distribution
            for (const auto &land: predicted_landmarks) {
                if (obs.id == land.id) {
                    mu_x = land.x;
                    mu_y = land.y;
                    break;
                }
            }
            double norm_factor = 2 * M_PI * std_x * std_y;
            double prob = exp(
                    -(pow(obs.x - mu_x, 2) / (2 * std_x * std_x) + pow(obs.y - mu_y, 2) / (2 * std_y * std_y)));

            particle_likelihood *= prob / norm_factor;
        }
        particles[i].weight = particle_likelihood;
        norm_factor += particle_likelihood;
    }

    // Normalize weights
    for (auto &particle : particles)
        particle.weight /= (norm_factor + numeric_limits<double>::epsilon());

}

void ParticleFilter::resample() {
    /* Resample particles with replacement with probability proportional to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     * http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     * std::discrete_distribution produces random integers on the interval [0, n),
     * where the probability of each individual integer i is defined as wi/S,
     * that is the weight of the ith integer divided by the sum of all n weights.
     */
    vector<double> particle_weights;
    for (const auto &particle : particles)
        particle_weights.push_back(particle.weight);

    discrete_distribution<int> weighted_distribution(particle_weights.begin(), particle_weights.end());

    vector<Particle> resampled_particles;
    for (size_t i = 0; i < num_particles; ++i) {
        int k = weighted_distribution(gen);
        resampled_particles.push_back(particles[k]);
    }

    particles = resampled_particles;

    for (auto &particle : particles)
        particle.weight = 1.0;

}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
                                         const std::vector<double> &sense_x, const std::vector<double> &sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
