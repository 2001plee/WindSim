#define TINYOBJLOADER_IMPLEMENTATION
#include <iostream>
#include <vector>
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include <limits>
#include "tiny_obj_loader.h"
#include "cmath"
#include <thread>
#include <mutex>
#include <unordered_map>
#include <chrono>
#include <random> 
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
GLuint compileShaders();
struct MaterialColor {
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
};
struct Arrow {//particle/wind struct
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 direction = glm::normalize(velocity); // Assuming this is a normalized velocity vector
    float mass = 0;
    float density = 0;
    float pressure = 0;
};
class Vertex {
public:
    glm::vec3 position; //position of vert
    glm::vec3 normal;
    glm::vec2 texCoord; //text coord to paint textures. not needed tbh
};
struct Triangle {
    Vertex v0, v1, v2;
};

//relic from a darker time with floating point errors everywhere before index search
//std::vector<Arrow> neighbor_particles; 


glm::vec3 eye = glm::vec3(0.0f, 140.0f, 50.0f);
//glm::vec3 cameraPos = glm::vec3(17.2895, 0, 6.25519); for the vertical one laying down
glm::vec3 origin = glm::normalize(glm::vec3(0.0f, 0.0f, 0.0f) - eye); // Direction from camera to origin
glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
float y_axis = -90.0f;
float x_axis = 0.0f;
float camera_x = 800 / 2.0f;
float camera_y = 600 / 2.0f;
bool mouse = true;
float cameraSpeed = 2.5f;
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (mouse) {
        camera_x = xpos;
        camera_y = ypos;
        mouse = false;
    }
    float xoffset = xpos - camera_x;
    float yoffset = camera_y - ypos;
    camera_x = xpos;
    camera_y = ypos;
    xoffset *= 0.1f;
    yoffset *= 0.1f;
    y_axis += xoffset;
    x_axis += yoffset;
    if (x_axis > 89.0f) x_axis = 89.0f;
    if (x_axis < -89.0f) x_axis = -89.0f;
    glm::vec3 origin2;
    origin2.x = cos(glm::radians(y_axis)) * cos(glm::radians(x_axis));
    origin2.y = sin(glm::radians(x_axis));
    origin2.z = sin(glm::radians(y_axis)) * cos(glm::radians(x_axis));
    origin = glm::normalize(origin2);
}
//mouse move with keyboard
void key_callback(GLFWwindow* window, int key, int code, int movement, int md) {
    if (movement == GLFW_PRESS || movement == GLFW_REPEAT) {
        float cameraSpeed = 2.5f;
        if (key == GLFW_KEY_W)
            eye += glm::vec3(0.0f, 0.0f, 1.0f) * cameraSpeed; //dont really use this tbh
        if (key == GLFW_KEY_S)
            eye += glm::vec3(0.0f, 0.0f, -1.0f) * cameraSpeed; //same dont use this It doesnt really work
        if (key == GLFW_KEY_A)
            eye -= glm::normalize(glm::cross(origin, up)) * cameraSpeed; //left -x
        if (key == GLFW_KEY_D)
            eye += glm::normalize(glm::cross(origin, up)) * cameraSpeed; //right x
        if (key == GLFW_KEY_SPACE)
            eye += glm::vec3(0.0f, 1.0f, 0.0f) * cameraSpeed; //y up
        if (key == GLFW_KEY_E)
            eye += glm::vec3(0.0f, -1.0f, 0.0f) * cameraSpeed; //y down
    }
}



//https://pysph.readthedocs.io/en/latest/reference/kernels.html Cubic Spline Kernel: [Monaghan1992] // try this if it doesn't work out for this kernel
//3/2pi*h^2 is 2d and 1/pi * h^3 is 3d.
float kernel_function2(float q) { //one from paper
    if (q < 1.0f) {
        return (2.0f / 3.0f - pow(q, 2.0f) + 0.5f * pow(q, 3.0f)) * (1.0f / (M_PI));
    }
    else if (q < 2.0f) {
        float a = 2.0f - q;
        return (1.0f / 6.0f) * pow(a, 3.0f) * (1.0f / (M_PI));
    }
    else {
        return 0.0f;
    }
}
//https://cg.informatik.uni-freiburg.de/publications/2014_EG_SPH_STAR.pdf the bible text for SPH for us
float kernel_function(float q) { //Cubic Spline Kernel : [Monaghan1992] version (claims its a 3d sph so using this) other paper didnt claim so...
    if (q < 1.0f) {
        return (1.0f - 1.5f * q * q + 0.75f * q * q * q) / M_PI;
    }
    else if (q < 2.0f) {
        return (0.25f * pow(2.0f - q, 3)) / M_PI;
    }
    else {
        return 0.0f;
    }
}

float smoothing_kernel(float radius, float distance) { //equation 4
    if (distance < 0.0f) {
        distance = distance * -1.0f;
    }
    float q = distance / radius;
    return (1.0f / pow(radius, 3.0f)) * kernel_function(q);
}

void set_mass(float radius, float p0, std::vector<Arrow>& all_particles_i) {
    for (int i = 0; i < all_particles_i.size(); i++) {
        all_particles_i[i].mass = (pow(radius, 3) * p0);
    }
}


int hash_maker2(int x_cord, int y_cord, int z_cord, int m) { //not used
    int p1 = 73856093;
    int p2 = 19349663;
    int p3 = 83492791;
    int d = 3;
    //this is the claimed hash on paper, but it freezes the particles in same spot it seems so im not using it.
    return (((x_cord / d) * p1) ^ ((y_cord / d) * p2) ^ ((z_cord / d) * p3)) % m;
}


unordered_map<int, vector<int>> hashmap;
int hash_maker(int x_cord, int y_cord, int z_cord) { //used
    int p1 = 73856093; //on 22 page doc recommend this prime
    int p2 = 19349663;
    int p3 = 83492791;
    return ((x_cord)*p1 + (y_cord)*p2 + (z_cord)*p3);
}

void maker_hash(float radius, std::vector<Arrow>& particles) { //used 
    hashmap.clear();
    hashmap.reserve(particles.size());
    for (int i = 0; i < particles.size(); ++i) {
        int x = floor(particles[i].position.x / (radius * 2));
        int y = floor(particles[i].position.y / (radius * 2));
        int z = floor(particles[i].position.z / (radius * 2));
        int hash_key = hash_maker(x, y, z);
        hashmap[hash_key].push_back(i);
    }
}
void find_neighborhood(float radius, std::vector<Arrow>& all_particles_j, 
    std::vector<std::vector<int>>& neighbor_indices, float k, float p0) { //used
    neighbor_indices.clear();
    neighbor_indices.resize(all_particles_j.size());
    maker_hash(radius, all_particles_j); //get hash of particles
#pragma omp parallel for
    for (int i = 0; i < all_particles_j.size(); ++i) {
        int count = 0;
        all_particles_j[i].mass = (pow(radius, 3) * p0);
        float density = 0.0f;
        int x = floor(all_particles_j[i].position.x / (radius * 2));//particle i x,y,z index for hashing
        int y = floor(all_particles_j[i].position.y / (radius * 2));
        int z = floor(all_particles_j[i].position.z / (radius * 2));
        for (int neighbor_x = -1; neighbor_x <= 1; ++neighbor_x) { //neighbors
            for (int neighbor_y = -1; neighbor_y <= 1; ++neighbor_y) {
                for (int neighbor_z = -1; neighbor_z <= 1; ++neighbor_z) {
                    int key = hash_maker(x + neighbor_x, y + neighbor_y, z + neighbor_z);
                    if (hashmap.find(key) != hashmap.end()) { // check if in hashma
                        for (int index_j = 0; index_j < hashmap[key].size(); index_j++) {//check in hash for index_j
                            int j = hashmap[key][index_j];
                            if (i != j) { //make sure not itself is neighbor
                                const auto& particle_j = all_particles_j[j];
                                float distance = glm::distance(all_particles_j[i].position,
                                    all_particles_j[j].position);
                                if (distance < 0.0f) {
                                    distance = distance * -1.0f;
                                }
                                if (distance < radius) {
                                    neighbor_indices[i].push_back(j);
                                    //std::cout << "distance: " << distance << std::endl;
                                    //std::cout << "mass_j " << particle_j.mass << std::endl;
                                    float influence = smoothing_kernel(radius, distance); 
                                    //std::cout << "influence: " << influence << "from particle j: " << j << std::endl;
                                    density += particle_j.mass * influence;
                                    count++;
                                }
                            }
                        }
                    }
                }
            }
        }
        // Update the particle's density and pressure
        //std::cout << "Index: " << i << " density is: " << density << " neighbor #: " << count << std::endl;
        all_particles_j[i].density = density;
        all_particles_j[i].pressure = k * (pow(density / p0, 7.0f) - 1.0f);
    }
}
//old neighbor not optimzied with spatial hash
void find_neighborhood2(float radius, std::vector<Arrow>& all_particles_j, std::vector<std::vector<int>>& neighbor_indices, float k, float p0) {
#pragma omp parallel for
    for (int i = 0; i < all_particles_j.size(); ++i) {
        //std::cout << "Index I: " << i <<" Position of I: " << all_particles_j[i].position.x << ", " << all_particles_j[i].position.y << ", " << all_particles_j[i].position.z << std::endl;
        float density = 0.0f;
        for (int j = 0; j < all_particles_j.size(); ++j) {
            if (i != j) {
                float distance = glm::distance(all_particles_j[i].position, all_particles_j[j].position);
                if (distance < radius) {
                    neighbor_indices[i].push_back(j); // Store index instead of particle
                    //std::cout << "Current Index: " << i << " neighbor index is: " << j << " Position is: " << all_particles_j[i].position.x << ", " << all_particles_j[i].position.y << ", " << all_particles_j[i].position.z << std::endl;
                    float influence = smoothing_kernel(radius, distance);
                    density += all_particles_j[j].mass * influence;
                }
            }
        }
        // Update the particle's density and pressure
    }
}

//old pressure and density
/*
    void compute_density(std::vector<Arrow>& particles, float radius) {//calling it particles instead normal all_particles_j so dont confuse on second for loop
        for (int i = 0; i < particles.size(); ++i) {
            float density = 0;
            Arrow& particle_i = particles[i];
            find_neighborhood(radius, particles, neighbor_particles);//get neighborhood of particle i
            for (int j = 0; j < neighbor_particles.size(); ++j) { //use neighborhood
                Arrow particle_j = neighbor_particles[j]; //a particle j that is in neighborhood
                float distance = glm::distance(particle_i.position, particle_j.position); //distance from particle i(position x) to j
                float influence = smoothing_kernel(radius, distance); //influence based off smoothing kernel
                density += particle_j.mass * influence; //add that to density
            }
            //std::cout << "particle_i.density_old: " << particle_i.density << std::endl;
            particle_i.density = density; //particle_i density based off neighborhood particles j
            //std::cout << "particle_i.density_new: " << particle_i.density << std::endl;
        }
    }

void compute_pressure(std::vector<Arrow>& particles, float k, float p0) { //p0 seems to be p(rho) from slides this means it should be set to 1.3f
    for (int i = 0; i < particles.size(); ++i) { //go through all the particles.
        Arrow& particle_i = particles[i];
        //pi = k * ((ρi/ρ0)^7 - 1)
        //pi = pressure of particle i
        //k is stiffness constant (high k means less compress water and visa versa)
        //ρi (difference) is density of particle i
        //idk why 7
        //-1 set to 0 i think when other part (ρi/ρ0)^7 is 1 cuz equal?
        particle_i.pressure = k * (glm::pow(particle_i.density / p0, 7.0f) - 1.0f);
    }
}
*/
/*
glm::vec3 compute_gradient_wij(float radius, float distance, glm::vec3 position_i, glm::vec3 position_j) {
    float constant = 1.0f / pow(radius, 3.0f);
    float wijx = constant * kernel_function((position_i.x - position_j.x) / (radius * distance));
    float wijy = constant * kernel_function((position_i.y - position_j.y) / (radius * distance));
    float wijz = constant * kernel_function((position_i.z - position_j.z) / (radius * distance));
    return glm::vec3(wijx, wijy, wijz);
}
*/

float kernel_function_derivative(float q) {
    if (q < 1.0f) {
        return (-3.0f * q + 2.25f * q * q) / M_PI;
    }
    else if (q < 2.0f) {
        float term = 2.0f - q;
        return (-0.75f * term * term) / M_PI;
    }
    else {
        return 0.0f;
    }
}

glm::vec3 compute_gradient_wij(float radius, glm::vec3 position_i, glm::vec3 position_j) {
    glm::vec3 r = position_i - position_j; // Distance vector from j to i
    float distance = glm::length(r);
    if (distance == 0.0f) {
        return glm::vec3(0.0f);
    }
    float q = distance / radius;
    glm::vec3 n = r / distance;
    float derivative = kernel_function_derivative(q);
    return n * derivative * (1.0f / pow(radius, 3.0f));
}
//not used old compute that used too many for loops so I just combined all into one function. Saving incase maybe used in future.
/*
glm::vec3 compute_pressure_force(float radius, int current_index_i, const std::vector<int>& neighbor_indices, const std::vector<Arrow>& arrows) { //const keep it same
    glm::vec3 gradient = glm::vec3(0.0f);
    const Arrow& particle_i = arrows[current_index_i]; //i swear god we're not changing it somehow through some bs somewhere lol
    float density_i = particle_i.density;
    float mass_i = particle_i.mass;
    float denominator_squared_i = density_i * density_i; //pi^2
    for (int neighbor_index : neighbor_indices) {//just for loop take index and iterates through neighbor_indiices
        const Arrow& particle_j = arrows[neighbor_index];
        float density_j = particle_j.density;
        float mass_j = particle_j.mass;
        float denominator_squared_j = density_j * density_j; // ρj^2
        float distance = glm::length(particle_i.position - particle_j.position);
        glm::vec3 gradient_W_ij = compute_gradient_wij(radius, particle_i.position, particle_j.position);
        gradient += mass_j * (particle_i.pressure / denominator_squared_i + particle_j.pressure / denominator_squared_j) * gradient_W_ij;
    }
    glm::vec3 pressure_force = (-1.0f) * mass_i * gradient;
    return pressure_force;
}
//not used old compute
glm::vec3 compute_viscosity_force(float radius, int current_index_i, const std::vector<int>& neighbor_indices, const std::vector<Arrow>& arrows) {
    float kinematic_viscosity = 1.48e-5;  // 1.48 x 10^-5 m^2/s is usually for air google
    glm::vec3 viscosity_force(0.0f);
    const Arrow& particle_i = arrows[current_index_i];
    for (int index = 0; index < neighbor_indices.size(); ++index) {
        int j = neighbor_indices[index];
        const Arrow& particle_j = arrows[j];
        glm::vec3 xij = particle_i.position - particle_j.position; // Position vector from j to i
        glm::vec3 v_ij = particle_i.velocity - particle_j.velocity; // vij that is triangle^2 A = triangle^2 V
        float distance = glm::length(xij);
        //if (distance > 0.0f) {  // no 0 divide since no collision stuff divideb y 0
        glm::vec3 wij = compute_gradient_wij(radius, particle_i.position, particle_j.position); ///entire right part equation 8
        float numerator = glm::dot(xij, wij);
        float denominator = glm::dot(xij, xij) + 0.01f * pow(radius, 2);
        viscosity_force += (particle_j.mass / particle_j.density) * v_ij * (numerator / denominator); //equation 8 entirety
        //}
    }

    // Apply kinematic viscosity and the factor of 2 from the equation
    viscosity_force *= particle_i.mass * 2.0f * kinematic_viscosity;

    return viscosity_force;
}
*/

//compute all forces needed in SPH
glm::vec3 compute_viscosity_and_pressure_force_other(float radius, int current_index_i,
    const std::vector<int>& neighbor_indices, const std::vector<Arrow>& arrows) {
    glm::vec3 gradient = glm::vec3(0.0f);
    float kinematic_viscosity = 1.48e-5;  // 1.48 x 10^-5 m^2/s 
    glm::vec3 viscosity_force(0.0f);
    const Arrow& particle_i = arrows[current_index_i];
    float density_i = particle_i.density;
    //std::cout << "index in force: " << current_index_i << std::endl;
    //std::cout << "density_i: " << density_i << std::endl;
    float mass_i = particle_i.mass;
    float denominator_squared_i = density_i * density_i; //pi^2
    //std::cout << "Denominator_i: " << denominator_squared_i << std::endl;
    for (int neighbor_index : neighbor_indices) {//just for loop take index and iterates through neighbor_indiices
        const Arrow& particle_j = arrows[neighbor_index];
        //std::cout << "particle_j: " << particle_j. << " " << particle_j.y << " " << particle_j.z << std::endl;
        float density_j = particle_j.density;
        float mass_j = particle_j.mass;
        float denominator_squared_j = density_j * density_j; // ρj^2
        float distance = glm::length(particle_i.position - particle_j.position);
        //std::cout << "\nNeighbor Index: " << neighbor_index << std::endl;
        //std::cout << "Density_j: " << density_j << std::endl;
        //std::cout << "Mass_j: " << mass_j << std::endl;
        //std::cout << "Denominator Squared_j: " << denominator_squared_j << std::endl;
        //std::cout << "Distance: " << distance << std::endl;
        glm::vec3 gradient_W_ij = compute_gradient_wij(radius, particle_i.position, particle_j.position);
        //std::cout << "Gradient_W_ij: " << gradient_W_ij.x << ", " << gradient_W_ij.y << ", " << gradient_W_ij.z << std::endl;
        gradient += mass_j * (particle_i.pressure / denominator_squared_i + particle_j.pressure / denominator_squared_j) * gradient_W_ij;
        //std::cout << "Gradient: " << gradient.x << " " << gradient.y << " " << gradient.z << std::endl;
        //pressure is above
        //viscocity 
        glm::vec3 xij = particle_i.position - particle_j.position; // Position vector from j to i
        glm::vec3 v_ij = particle_i.velocity - particle_j.velocity; // vij that is triangle^2 A = triangle^2 V
        float numerator = glm::dot(xij, gradient_W_ij);
        float denominator = glm::dot(xij, xij) + 0.01f * pow(radius, 2);
        viscosity_force += (particle_j.mass / particle_j.density) * v_ij * (numerator / denominator); //equation8
    }
    glm::vec3 pressure_force = (-1.0f) * mass_i * gradient;
    //std::cout << "pressure_force: " << pressure_force.x << " " << pressure_force.y << " " << pressure_force.z << std::endl;
    viscosity_force *= particle_i.mass * 2.0f * kinematic_viscosity;
    //std::cout << "viscosity_force: " << viscosity_force.x << " " << viscosity_force.y << " " << viscosity_force.z << std::endl;
    float speed = particle_i.mass * 9.8;
    glm::vec3 gravity = glm::vec3(0.0f, -1.0f, 0.0f) * speed;
    glm::vec3 combine_force = pressure_force + viscosity_force + gravity;
    return combine_force;
}

//old code used combined (not used anymore have a big do all SPH function) keeping to show runtime of optimized vs unoptimized
/*
glm::vec3 compute_other_force(Arrow particle_i) {
    //float speed = particle_i.mass * 9.8;
    float speed = particle_i.mass * 9.8;
    glm::vec3 gravity = glm::vec3(0.0f, -1.0f, 0.0f) * speed;
    return gravity;
}

glm::vec3 compute_resultant_force(glm::vec3 p_force, glm::vec3 v_force, glm::vec3 o_force) {
    return p_force + v_force + o_force;
}
glm::vec3 compute_resultant_force_combine(glm::vec3 combine_force, glm::vec3 o_force) {
    return combine_force + o_force;
}
*/

void update_particle_position(Arrow& particle_i, glm::vec3 resultant_force, float delta_t) {
    //std::cout << "old_velocity: " << particle_i.velocity.x << " " << particle_i.velocity.y << " " << particle_i.velocity.z << std::endl;
    //std::cout << "resultant_force: " << resultant_force.x << " " << resultant_force.y << " " << resultant_force.z << std::endl;
    //std::cout << "delta time: " << delta_t << std::endl;
    //std::cout << "mass: " << particle_i.mass << std::endl;
    glm::vec3 new_velocity = particle_i.velocity + delta_t * resultant_force / particle_i.mass;
    //std::cout << "new_velocity: " << new_velocity.x << " " << new_velocity.y << " " << new_velocity.z << std::endl;
    glm::vec3 new_position = particle_i.position + delta_t * new_velocity;
    particle_i.velocity = new_velocity;
    particle_i.position = new_position;
    //std::cout << "resultant_Force: " << resultant_force.x << " " << resultant_force.y << " " << resultant_force.z << std::endl;
    //std::cout << "particle_i.density: " << particle_i.density << std::endl;
    //std::cout << "Reflect arrow particle_i.velocity: " << particle_i.velocity.x << " " << particle_i.velocity.y << " " << particle_i.velocity.z << std::endl;
    //std::cout << "particle position: " << particle_i.position.x << " " << particle_i.position.y << " " << particle_i.position.z << std::endl;
}

std::vector<Arrow> arrows = {//main vector for particles
};

glm::vec3 calculateNormal(const Vertex& v0, const Vertex& v1, const Vertex& v2) {
    glm::vec3 edge_1 = v1.position - v0.position;
    glm::vec3 edge_2 = v2.position - v0.position;
    return glm::normalize(glm::cross(edge_1, edge_2));
}

void reflect(std::vector<Arrow>& arrows, Arrow& reflect_arrow, glm::vec3& intersectionPoint, glm::vec3& normal, float excess_energy) {
    glm::vec3 normal_2 = glm::normalize(normal);
    glm::vec3 reflectionVector = glm::normalize(reflect_arrow.velocity) - 2 * glm::dot(glm::normalize(reflect_arrow.velocity), normal_2) * normal_2;
    float dampingFactor = 0.75f; //energy loss (just so bouncing isnt like water spewing out so crazy)
    reflectionVector *= dampingFactor;
    Arrow old_reflect_arrow = reflect_arrow;
    reflect_arrow.position = intersectionPoint;
    reflect_arrow.direction = glm::normalize(reflectionVector);
    float velocityMagnitude = glm::length(reflect_arrow.velocity) * dampingFactor;
    reflect_arrow.velocity = reflectionVector * velocityMagnitude;
    float energyAdjustmentFactor = 0.8f; //reduce energy to lower velocity hopefully so doesn't shoot to stratephere
    reflect_arrow.position += reflect_arrow.direction * (excess_energy * energyAdjustmentFactor);
    // Update the arrow in the arrows vector
    for (int i = 0; i < arrows.size(); ++i) {
        if (arrows[i].position == old_reflect_arrow.position) {
            arrows[i] = reflect_arrow;
            break;
        }
    }
}


bool rayIntersectsTriangle(const glm::vec3& rayOrigin, const glm::vec3& rayEnd, const Triangle& triangle, glm::vec3& IntersectionPoint) {
    //std::cout << "rayOrigin " << rayOrigin.x << " " << rayOrigin.y  << " " << rayOrigin.z << std::endl;
    //std::cout << "rayEnd " << rayEnd.x << " " << rayEnd.y << " " << rayEnd.z << "\n" << std::endl;
    constexpr float epsilon = std::numeric_limits<float>::epsilon();
    glm::vec3 rayVector = rayEnd - rayOrigin;
    glm::vec3 edge1 = triangle.v1.position - triangle.v0.position;
    glm::vec3 edge2 = triangle.v2.position - triangle.v0.position;
    glm::vec3 h = glm::cross(rayVector, edge2);
    float a = glm::dot(edge1, h);

    if (a > -epsilon && a < epsilon)
        return false;

    float f = 1.0 / a;
    glm::vec3 s = rayOrigin - triangle.v0.position;
    float u = f * glm::dot(s, h);

    if (u < 0.0 || u > 1.0)
        return false;

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(rayVector, q);

    if (v < 0.0 || u + v > 1.0)
        return false;

    float t = f * glm::dot(edge2, q);
    if (t > epsilon && t < 1.0) { // intersection between current and new step
        IntersectionPoint = rayOrigin + rayVector * t;
        return true;
    }
    return false;
}

struct intersectchunk { //IntersectionResult had bool
    bool hit;
    glm::vec3 intersectionPoint;
    Triangle triangle;
    int index_arrow;
    float intersect_step_point;
};

//old and one works rn for old code but slower ofc then parallel threads
bool checkArrowObjectIntersections(std::vector<Vertex>& objVertices, std::vector<Arrow>& arrows, 
    float deltaTime, glm::vec3& intersectionPoint, Triangle& intersect_triangle,
    Arrow& reflect_arrow, float& intersect_step_point) {
    for (auto& arrow : arrows) {
        glm::vec3 arrowStep = arrow.velocity * deltaTime; // Use velocity directly, scaled by deltaTime
        glm::vec3 arrowNewPos = arrow.position + arrowStep;
        for (int i = 0; i + 2 < objVertices.size(); i += 3) {
            Triangle triangle = { objVertices[i], objVertices[i + 1], objVertices[i + 2] };
            if (rayIntersectsTriangle(arrow.position, arrowNewPos, triangle, intersectionPoint)) {
                reflect_arrow = arrow;
                intersect_triangle = triangle;
                intersect_step_point = glm::distance(intersectionPoint, arrowNewPos);
                return true;
            }
        }
    }
}
//dont use test trial for openmp see difference between my code and Ian recommendation for MP
intersectchunk checkArrowObjectIntersections_openmp(std::vector<Vertex>& objVertices, std::vector<Arrow>& arrows, int arrow_index, float deltaTime) {
    glm::vec3 intersectionPoint;
    Triangle triangle;
    glm::vec3 arrowStep = arrows[arrow_index].velocity * deltaTime; // Use velocity directly, scaled by deltaTime
    glm::vec3 arrowNewPos = arrows[arrow_index].position + arrowStep; //small arrowspeed causes float values will be fixed by velocity i believe though
    for (int i = 0; i + 2 < objVertices.size(); i += 3) {
        triangle = { objVertices[i], objVertices[i + 1], objVertices[i + 2] };
        if (rayIntersectsTriangle(arrows[arrow_index].position, arrowNewPos, triangle, intersectionPoint)) {
            return { true, intersectionPoint, triangle, arrow_index, glm::distance(intersectionPoint, arrowNewPos) };
        }
    }
    return { false, intersectionPoint, triangle, 0, 0.0f };
}
//threads that check based on start-end of number arrows this parallel thread suppose to check then put in struct
void checkIntersectionsThread(const std::vector<Vertex>& objVertices, const std::vector<Arrow>& arrows, 
    float deltaTime, int start, int end, std::vector<intersectchunk>& results, std::mutex& resultsMutex) {
    //resize results in the render loop
    for (int i = start; i < end && i < arrows.size(); ++i) {
        const auto& arrow = arrows[i];
        glm::vec3 arrowStep = arrows[i].velocity * deltaTime;
        glm::vec3 arrowNewPos = arrows[i].position + arrowStep;

        for (int j = 0; j + 2 < objVertices.size(); j += 3) {
            Triangle triangle = { objVertices[j], objVertices[j + 1], objVertices[j + 2] };
            glm::vec3 intersectionPoint;
            if (rayIntersectsTriangle(arrow.position, arrowNewPos, triangle, intersectionPoint)) {
                std::lock_guard<std::mutex> guard(resultsMutex);
                results.push_back({ true, intersectionPoint, triangle, i,
                    glm::distance(intersectionPoint, arrowNewPos) }); //faster to allocate the spaces needed
                break; // Assuming we only care about the first intersection
            }
        }
    }
}
//get # parallel threads and divide for particles between each one and then combine the struct together at end for reflect later
std::vector<intersectchunk> parallelCheckArrowObjectIntersections(const std::vector<Vertex>& objVertices,
    const std::vector<Arrow>& arrows, float deltaTime) {
    int numThreads = std::thread::hardware_concurrency();
    //std::cout << "cores avaliable: " << std::thread::hardware_concurrency() << std::endl;
    std::vector<std::thread> threads;
    std::vector<intersectchunk> results;
    std::mutex resultsMutex;

    int arrowsPerThread = arrows.size() / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int start = i * arrowsPerThread;
        int end = (i + 1) * arrowsPerThread;
        threads.push_back(std::thread(checkIntersectionsThread, std::ref(objVertices), std::ref(arrows),
            deltaTime, start, end, std::ref(results), std::ref(resultsMutex)));
    }

    for (auto& thread : threads) { //join the threads
        thread.join();
    }
    return results;
}
//using this generate when testing not used anymore
/*
void generateCubeOfArrows(int arrowsPerLayer, const std::vector<float>& yLayers, float xMin, float xMax, float zMin, float zMax, std::vector<Arrow>& arrows) {
    float x_dist = (xMax - xMin) / sqrt(arrowsPerLayer); //trying space them
    float z_dist = (zMax - zMin) / sqrt(arrowsPerLayer);
    //std::cout << "X" << x_dist << std::endl;
    //std::cout << "Z" << z_dist << std::endl;
    // Iterate through each y layer
    for (auto y : yLayers) {
        // Generate a grid of arrows for this layer
        for (int i = 0; i < sqrt(arrowsPerLayer); ++i) {
            for (int j = 0; j < sqrt(arrowsPerLayer); ++j) {
                float x = xMin + i * x_dist + x_dist / 2; //x cord
                float z = zMin + j * z_dist + z_dist / 2; //y cord //i default y to be based off below in the vector y Layers in sets of 5

                // Add the new arrow to the collection
                arrows.push_back(Arrow{ glm::vec3(x, y, z), glm::vec3(0.0f, -5.0f, 0.0f) });
            }
        }
    }
}
*/

//used in small_step find highest velocity of all particles
float calculateMaxVelocity(const std::vector<Arrow>& arrows) { //dont want modify it accidently so const (float point trauma be like)
    float maxVelocity = 0.0f;
    for (int i = 0; i < arrows.size(); ++i) {
        float velocityMagnitude = glm::length(arrows[i].velocity);
        if (velocityMagnitude > maxVelocity) {
            maxVelocity = velocityMagnitude;
        }
    }
    return maxVelocity;
}

//The time step t is governed by the Courant-Friedrich-Levy (CFL) condition Page 3
void smaller_step(std::vector<Arrow>& arrows, float& deltaTime, float radius, float lambda = 0.4) {
    float velocity_highest = calculateMaxVelocity(arrows);
    float CFL = lambda * radius / velocity_highest;
    //get smaller one
    deltaTime = std::min(deltaTime, CFL); //time step <= lamba * h/|vmax|
}

//rectangle one using for checkpoint week 2 
void cube_surround_other_cube3(std::vector<Arrow>& arrows, const std::vector<float>& yLayers, float arrowCubeSize, int arrowsPerLayer) {
    float cubeCenterX = 0.0f, cubeCenterZ = -50.0f; // Assuming cubeCenterZ is the intended starting point, adjust as necessary.

    float x_start = cubeCenterX - arrowCubeSize / 2;
    // Adjust z_start based on the range [0, 40] requirement
    float z_start = 20.0f; // Start at the beginning of the range
    float x2 = arrowCubeSize / (sqrt(arrowsPerLayer) + 1);
    // Calculate z2 to ensure that z values fall within [0, 40]
    float z2 = 40.0f / (sqrt(arrowsPerLayer) + 1); // Adjust division to distribute arrows within [0, 40]
    std::cout << "X" << x2 << std::endl;
    for (float y : yLayers) {
        for (int i = 0; i < sqrt(arrowsPerLayer); ++i) {
            for (int j = 0; j < sqrt(arrowsPerLayer); ++j) {
                float x = x_start + x2 * (i + 1);
                float z = z_start + z2 * (j + 1); // No need to ensure range here, calculation adjusted

                arrows.push_back(Arrow{ glm::vec3(x, y, z), glm::vec3(0, 0, 0) });
            }
        }
    }
}
//randomize generate better verison of old one that deals with random gen one in the demo used
void cube_surround_other_cube(std::vector<Arrow>& arrows, const std::vector<float>& yLayers, float arrowCubeSize, int arrowsPerLayer) {
    float cubeCenterX = 0.0f, cubeCenterZ = -50.0f; // Assuming cubeCenterZ is the intended starting point, adjust as necessary.

    float x_start = cubeCenterX - arrowCubeSize / 2;
    float z_start = 20.0f; // Start at the beginning of the range

    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_real_distribution<> disX(x_start, x_start + arrowCubeSize); // Define the distribution for x within the allowed range
    std::uniform_real_distribution<> disZ(z_start, z_start + 40.0f); // Define the distribution for z within the allowed range

    for (float y : yLayers) {
        for (int i = 0; i < arrowsPerLayer; ++i) {
            float x = disX(gen); // Generate a random x within the allowed range
            float z = disZ(gen); // Generate a random z within the allowed range

            arrows.push_back(Arrow{ glm::vec3(x, y, z), glm::vec3(0, -10, 0) });
        }
    }
}
//old cube that like actually makes a cube not a rectangle.
void cube_surround_other_cube2(std::vector<Arrow>& arrows, const std::vector<float>& yLayers, float arrowCubeSize, int arrowsPerLayer) {
    float cubeCenterX = 0.0f, cubeCenterZ = -45.0f;

    float x_start = cubeCenterX - arrowCubeSize / 2;
    float z_start = cubeCenterZ - arrowCubeSize / 2;
    float x2 = arrowCubeSize / (sqrt(arrowsPerLayer) + 1);
    float z2 = arrowCubeSize / (sqrt(arrowsPerLayer) + 1);
    std::cout << "X" << x2 << std::endl;
    //std::cout << "Z" << z2 << std::endl; //distance from each other
    for (float y : yLayers) {
        for (int i = 0; i < sqrt(arrowsPerLayer); ++i) {
            for (int j = 0; j < sqrt(arrowsPerLayer); ++j) {
                float x = x_start + x2 * (i + 1);
                float z = z_start + z2 * (j + 1);
                arrows.push_back(Arrow{ glm::vec3(x, y, z), glm::vec3(0, 0, 0) });
            }
        }
    }
}


int main() {
    // Initialize GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    //Window
    GLFWwindow* window = glfwCreateWindow(800, 600, "Top down View", nullptr, nullptr);
    if (!window) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Load GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    //compile the shaders
    GLuint shaderProgram = compileShaders();
    glUseProgram(shaderProgram);
    glDisable(GL_LIGHTING);

    //VAO VBO for ARROW
    GLuint arrow_VAO, arrow_VBO;
    float scale = 10.0f;
    //what makes the arrow image (old so it used based off bottom arrow shaft)
    /*
    float arrowVertices[] = { //default looks right make angles easier
        //arrow line
        //0.0f, 0.0f, 0.0f, 0.5f * scale, 0.0f, 0.0f,
        0.5f * scale, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        //head of arrow
        0.5f * scale, 0.0f, 0.0f,
        0.45f * scale,0.05f * scale, 0.0f,
        0.5f * scale, 0.0f, 0.0f,
        0.45f * scale, -0.05f * scale, 0.0f,
    };
    */


    float arrowVertices[] = {
        // Arrow line
        -0.5f * scale, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        // Head of arrow
        0.0f, 0.0f, 0.0f, -0.05f * scale, 0.05f * scale, 0.0f, // Left side of arrow head
        0.0f, 0.0f, 0.0f, -0.05f * scale, -0.05f * scale, 0.0f, // Right side of arrow head
    };

    glGenVertexArrays(1, &arrow_VAO);
    glBindVertexArray(arrow_VAO);

    glGenBuffers(1, &arrow_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, arrow_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(arrowVertices), arrowVertices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    //LOADOBJ

        std::string inputfile = "cube_upward_demo6.obj"; //where change obj file 
        std::string modelMtlBaseDir = "cube_upward_demo6.mtl";
        std::vector<glm::vec4> materialColors;
        tinyobj::attrib_t attributes;

        std::vector<tinyobj::shape_t> shapes_dont_use;
        std::vector<tinyobj::material_t> materials_use;
        std::cout << "Before loading, materials size: " << materials_use.size() << std::endl;

        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = "./"; //reads in same directory for the obj 
        tinyobj::ObjReader reader;
        if (!reader.ParseFromFile(inputfile, reader_config)) {
            if (!reader.Error().empty()) {
                std::cerr << "TinyObjReader: " << reader.Error();
            }
            exit(1);
        }
        if (!reader.Warning().empty()) {
            std::cout << "TinyObjReader Warning: " << reader.Warning();
        }
        shapes_dont_use = reader.GetShapes();
        materials_use = reader.GetMaterials(); // Now this should be correctly filled with material data
        attributes = reader.GetAttrib();
        std::cout << "After loading, materials size: " << materials_use.size() << std::endl;

        GLuint obj_VAO, obj_VBO; //vao and vbo for an obj file
        std::vector<Vertex> OBJ_vertices;
        //if (load_obj) { //true thus do the obj stuff
        std::cout << "obj did load" << std::endl;
        //tldr this basically process all the vertex in the obj file
        //int face_number = 1; //#1
        for (int i = 0; i < shapes_dont_use.size(); i++) {
            tinyobj::shape_t& shape = shapes_dont_use[i];
            tinyobj::mesh_t& mesh = shape.mesh;
            for (int j = 0; j < mesh.indices.size(); j++) {
                tinyobj::index_t mesh_index = mesh.indices[j];
                glm::vec3 position(
                    attributes.vertices[mesh_index.vertex_index * 3],
                    attributes.vertices[mesh_index.vertex_index * 3 + 1],
                    attributes.vertices[mesh_index.vertex_index * 3 + 2]
                );
                glm::vec3 normal(0.0f, 0.0f, 0.0f);
                if (mesh_index.normal_index >= 0) {
                    normal = glm::vec3(
                        attributes.normals[mesh_index.normal_index * 3],
                        attributes.normals[mesh_index.normal_index * 3 + 1],
                        attributes.normals[mesh_index.normal_index * 3 + 2]
                    );
                }
                glm::vec3 up(0.0f, 1.0f, 0.0f);
                float dotProduct = glm::dot(normal, up);
                if (dotProduct < 0.0f) {
                    normal = -normal;
                }
                Vertex vert = { position, normal, glm::vec2(0.0f, 0.0f) }; //text coords unneeded
                OBJ_vertices.push_back(vert);
            }
            for (const auto& material : materials_use) {
                glm::vec4 color(material.diffuse[0], material.diffuse[1], material.diffuse[2], 1.0f);
                std::cout << "Color: " << color.r << ", " << color.g << ", " << color.b << ", " << color.a << std::endl;
                materialColors.push_back(color); //unused cuz material id is bugging out 
            }
            /* //checks if its reading correct # vert and faces. Its right so commented out till it breaks again...
            int numVertices = 0;
            int numFaces = 0;
            for (const auto& shape : shapes_dont_use) {
                const tinyobj::mesh_t& mesh = shape.mesh;
                numVertices += mesh.indices.size();
                numFaces += mesh.indices.size() / 3;//triangulation so /3
            }

            std::cout << "Number of vertices: " << numVertices << std::endl;
            std::cout << "Number of faces: " << numFaces << std::endl;
            */

        // Create another VAO and VBO for the model

        //based off blog technically same thing

        glGenVertexArrays(1, &obj_VAO);
        glBindVertexArray(obj_VAO);

        glGenBuffers(1, &obj_VBO);
        glBindBuffer(GL_ARRAY_BUFFER, obj_VBO);

        glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * OBJ_vertices.size(), &OBJ_vertices[0], GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, nullptr);
        
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)(sizeof(float) * 3));
        
        glBindVertexArray(0);
    }
    int numArrows = 0;
    float sph_radius = 2.6f; //2.6 is with 5000 arrow per layer and ylayer is 30 layers
    float k_constant = 5.0f;
    float air_density = 1.3f; 
    int numArrowsPerLayer = 2000; //# of arrows per layer
    //kinda hard coded layers as x and z are the layer size randomly generating # arrows in that constraint and ylayer is the elevation of each layer
    std::vector<float> yLayers = { 200, 202.5, 205, 207.5, 210, 212.5, 215, 217.5, 220, 222.5, 225, 227.5, 230, 232.5, 235, 237.5, 240, 242.5, 245,247.5, 250.0f, 252.5f, 255.0f, 257.5f, 260.0f, 262.5f, 265.0f, 267.5f, 270.0f, 272.5f }; //4 layers
    //std::vector<float> yLayers = { 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370,380,390,400,410,420,430,440,450,460, 470, 480, 490, 500 }; //32
    cube_surround_other_cube(arrows, yLayers, 220, numArrowsPerLayer); //function calls the ^
    //mouse movement for camera (NOTE IT SEEMS THAT IMGUI DOESNT WORK WITH this and im blanking on how cancel it out so just test with function above reminder
    //imgui only use debug anyway so its fine.
    glfwSetCursorPosCallback(window, mouse_callback); 
    glfwSetKeyCallback(window, key_callback); //key movement
    float old_time = glfwGetTime(); //delta time stuff (we get the frame's time)
    // Main loop Render loop
    while (!glfwWindowShouldClose(window)) {
        float new_time = glfwGetTime(); //current
        float deltaTime = new_time - old_time;//delta time ( REALLLYY IMPORTANT MULTIPLY DELTATIME onto all aspects of trying change position by velocity and also do multiply deltaTiem when we add a force to velocityto 
        old_time = new_time; //set previous time from new time
        //float arrowSpeed = 1.0f * deltaTime; //shouldnt affect arrowspeed with delta time
        smaller_step(arrows, deltaTime, sph_radius);
        processInput(window);
        //Start Imgui
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            // ImGui window for simulation parameters
            ImGui::Begin("Wind Sim Parameters");
            //add the stuff below for parameters later------------------------------------------
            if (ImGui::SliderFloat("Arrow Scale", &scale, 1.0f, 100.0f)) {
                // Recalculate arrow vertices only if scale has changed
                float arrowVertices[] = {
                    // Arrow line (adjust these vertices based on the new 'scale' value)
                    -0.5f * scale, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    // Head of arrow (adjust these vertices too)
                    0.0f, 0.0f, 0.0f, -0.05f * scale, 0.05f * scale, 0.0f, // Left side of arrow head
                    0.0f, 0.0f, 0.0f, -0.05f * scale, -0.05f * scale, 0.0f, // Right side of arrow head
                };

                // Update the VBO with the new vertices
                glBindBuffer(GL_ARRAY_BUFFER, arrow_VBO);
                glBufferData(GL_ARRAY_BUFFER, sizeof(arrowVertices), arrowVertices, GL_STATIC_DRAW);
            }
            if (ImGui::SliderFloat("Radius", &sph_radius, 0.0f, 50.0f, "%.00f")) {
                sph_radius = (int)(sph_radius); //so its an int
            }
            if (ImGui::SliderInt("Arrows in each Layer", &numArrowsPerLayer, 0, 50000)) {
                arrows.clear(); // Clear existing arrows to regenerate
                std::cout << "SWAP ARROW" << std::endl;
                //generateCubeOfArrows(numArrowsPerLayer, yLayers, 10.0f, 40.0f, 10.0f, 40.0f, arrows);
                //std::vector<float> yLayers = { 385.0f, 380.0f, 375.0f, 370.0f, 365.0f, 360.0f, 355.0f, 350.0f, 345.0f, 335.0f, 330.0f, 325.0f, 320.0f, 315.0f, 310.0f, 305.0f, 300.0f, 295.0f, 290.0f, 280.0f, 270.0f }; // Two layers at heights 70 and 65
                std::vector<float> yLayers = { 200.0f, 195.0f,190.0f,185.0f, 180.0f, 175.0f, 170.0f }; // Two layers at heights 70 and 65
                cube_surround_other_cube(arrows, yLayers, 180, numArrowsPerLayer);
            }

            ImGui::End();
            ImGui::Render();
        
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.50f, 0.50f, 0.50f, 1.00f); //COLOR OF THE BACKGROUND || currently gray
        glEnable(GL_DEPTH_TEST);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDepthFunc(GL_LESS);
        glUseProgram(shaderProgram);
        glDisable(GL_LIGHTING);
        int screenWidth, screenHeight;
        glfwGetWindowSize(window, &screenWidth, &screenHeight);

        float aspectRatio = static_cast<float>(screenWidth) / static_cast<float>(screenHeight);
        float proj_size = 100.0f; // Adjust this value as needed for imageplane
        glm::mat4 projection = glm::ortho(-proj_size * aspectRatio, proj_size * aspectRatio, -proj_size, proj_size, -100.0f, 100.0f);
        /* //old view with no camera movement hard coded
        glm::mat4 view = glm::lookAt( //trdition not look up (u got change more stuff then just swapping just a note for if we do)
            glm::vec3(10.0f, 100.0f, 50.0f), // Eye  //change to 5,5,5
            //glm::vec3(20.0f, 80.0f, 70.0f),
            glm::vec3(0.0f, 0.0f, 0.0f),   // origin
            glm::vec3(0.0f, 1.0f, 0.0f)    // Up vector
        );
        */
        glm::mat4 view = glm::lookAt(eye, eye + origin, up);

        //commented out Very first version of our code with a removed find neighbor search above for loop tried throwing openmp and stuff. its way slower keep this code for old showcase
        /*
        std::vector<std::vector<int>> neighbor_hood; //holds index to the arrows
        std::vector<intersectchunk> local_results;
        local_results.clear();
        set_mass(sph_radius, air_density, arrows);
        find_neighborhood(sph_radius, arrows, neighbor_hood, k_constant, air_density);
        std::vector<glm::vec3> forces(arrows.size());
        std::vector<intersectchunk> results;
        for (int i = 0; i < arrows.size(); i++) {
            std::vector<int>& neighbor_indices = neighbor_hood[i];
            glm::vec3 pressure_force = compute_pressure_force(sph_radius, i, neighbor_indices, arrows);
            glm::vec3 viscosity_force = compute_viscosity_force(sph_radius, i, neighbor_indices, arrows);
            glm::vec3 other_force = compute_other_force(arrows[i]);
            forces[i] = compute_resultant_force(pressure_force, viscosity_force, other_force);
            //std::cout << "Force: (" << forces[i].x << ", " << forces[i].y << ", " << forces[i].z << ")\n" << std::endl;
        }
#pragma omp parallel for
        for (int i = 0; i < arrows.size(); i++) {
            intersectchunk x = checkArrowObjectIntersections_openmp(OBJ_vertices, arrows, i, deltaTime);
#pragma omp critical
            {
                if (x.hit == true) {
                    results.push_back(x);
                }
            }
        }
        for (auto& result : results) {
            glm::vec3 normal = calculateNormal(result.triangle.v0, result.triangle.v1, result.triangle.v2);
            if (result.hit) {
                //std::cout << "hello" << std::endl;
                glm::vec3 normal = calculateNormal(result.triangle.v0, result.triangle.v1, result.triangle.v2);
                //std::cout << "hello2" << std::endl;
                //std::cout << result.index_arrow << std::endl;
                Arrow reflect_arrow = arrows[result.index_arrow];
                //std::cout << "hello3" << std::endl;
                glm::vec3 originalPosition = reflect_arrow.position;
                //std::cout << "hello4" << std::endl;
                glm::vec3 reflectionVector = glm::normalize(reflect_arrow.velocity) - 2 * glm::dot(glm::normalize(reflect_arrow.velocity), glm::normalize(normal)) * glm::normalize(normal);
                float dampingFactor = 0.75f;
                reflectionVector *= dampingFactor;
                float velocityMagnitude = glm::length(reflect_arrow.velocity) * dampingFactor;
                reflect_arrow.velocity = reflectionVector * velocityMagnitude;
                float energyAdjustmentFactor = 0.8f;
                reflect_arrow.position = result.intersectionPoint + reflectionVector * (result.intersect_step_point * energyAdjustmentFactor);
                reflect_arrow.direction = glm::normalize(reflectionVector);
                float threshold = 0.001f;
                //std::cout << "hello" << std::endl;
                for (int i = 0; i < arrows.size(); ++i) {
                    if (glm::length(arrows[i].position - originalPosition) < threshold) { //floating point errors. go REEEEEEEEEEEEEEEEEEEEEEEE
                        //std::cout << glm::length(arrows[i].position - originalPosition) << std::endl;
                        arrows[i] = reflect_arrow;
                        break;
                    }
                }
            } //reflect
        }

        for (int i = 0; i < arrows.size(); ++i) {
            update_particle_position(arrows[i], forces[i], deltaTime);
            arrows[i].direction = glm::normalize(arrows[i].velocity);
        }
        */

        //glm::vec3 intersectionPoint;
        //Triangle triangle;
        //Arrow reflect_arrow;
        //float remaining_energy = 0;


        //our render code with parallel processing
        std::vector<std::vector<int>> neighbor_hood; //holds index to the arrows
        const auto start = std::chrono::steady_clock::now();
        set_mass(sph_radius, air_density, arrows);
        find_neighborhood(sph_radius, arrows, neighbor_hood, k_constant, air_density); //spatial hashing 10x faster 13k ms to 1.3k
        std::vector<glm::vec3> forces(arrows.size());
        //#pragma omp parallel for //its 4x faster with pragma still like 40-20 ms so really small in scheme of things
        for (int i = 0; i < arrows.size(); i++) {
            std::vector<int>& neighbor_indices = neighbor_hood[i];
            forces[i] = compute_viscosity_and_pressure_force_other(sph_radius, i, neighbor_indices, arrows); //we're just doing everything at once to save on for loops
        }
        std::vector<intersectchunk> results;
        results.resize(arrows.size() / 20);
        results = parallelCheckArrowObjectIntersections(OBJ_vertices, arrows, deltaTime); //multi thread (the entire thing + reflect at 300 with radius 6 is like 20ms then 50 ms when reflect and intersetion occur
        //reflection code in for loop
        for (auto& result : results) { //results is the struct gets all stuff from parallel thread of intersection to do reflection
            if (result.hit) {
                glm::vec3 normal = calculateNormal(result.triangle.v0, result.triangle.v1, result.triangle.v2);
                Arrow reflect_arrow = arrows[result.index_arrow];
                glm::vec3 originalPosition = reflect_arrow.position;
                glm::vec3 reflectionVector = glm::normalize(reflect_arrow.velocity) - 2 * 
                    glm::dot(glm::normalize(reflect_arrow.velocity), glm::normalize(normal)) * glm::normalize(normal);
                float dampen = 0.8f; //try make it not so extreme
                reflectionVector *= dampen;
                reflect_arrow.velocity = reflectionVector * (glm::length(reflect_arrow.velocity) * dampen);
                float energy_adjust = 1.0f; //not that big deal dont need change position we're in small floats already
                reflect_arrow.position = result.intersectionPoint + reflectionVector * 
                    (result.intersect_step_point * energy_adjust);
                reflect_arrow.direction = glm::normalize(reflectionVector);
                float epsilon = 0.001f;
                for (int i = 0; i < arrows.size(); ++i) {
                    if (glm::length(arrows[i].position - originalPosition) < epsilon) { 
                        arrows[i] = reflect_arrow;
                        break;
                    }
                }
            } //reflect
        }
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 modelview = projection * view * model;
        GLint modelviewLoc = glGetUniformLocation(shaderProgram, "modelview");
        glUniformMatrix4fv(modelviewLoc, 1, GL_FALSE, glm::value_ptr(modelview));
        glBindVertexArray(obj_VAO);

        int count = 0;
        //generating color
        for (const auto& shape : shapes_dont_use) {
            for (int faceIndex = 0; faceIndex < shape.mesh.indices.size() / 3; faceIndex++) {
                glm::vec4 color;
                if (count % 2 == 0) { //i could do count % 6 == 2 for 2 triangle at time but square becomes orange on orange T.T so dont -plee
                    color = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f); // White color
                }
                else {
                    color = glm::vec4(1.0f, 0.624f, 0.0f, 1.0f); // orange
                }
                glUniform4fv(glGetUniformLocation(shaderProgram, "color"), 1, glm::value_ptr(color));
                count++;
                glDrawArrays(GL_TRIANGLES, faceIndex * 3, 3);
            }
        }
        
        glm::vec4 arrowColor = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);//Blue
        GLint colorLoc = glGetUniformLocation(shaderProgram, "color");
        glUniform4fv(colorLoc, 1, glm::value_ptr(arrowColor));

        glBindVertexArray(arrow_VAO); //bind vao to arrow
        //#pragma omp parallel for not worth doing its 200ms slower (also bugs out it seems probably cuz of model[] call)
        //its not worth trying parallel thread the first two lines as its just longer with pragma op. and u cant prallel process the draw i believe?
        for (int i = 0; i < arrows.size(); i++) { //for loop for the two arrows (translate and rotate it from origin to the pos it needs to be)
            update_particle_position(arrows[i], forces[i], deltaTime);
            arrows[i].direction = glm::normalize(arrows[i].velocity);
            glm::mat4 model = glm::mat4(1.0f); //translate then rotate so its correct
            model = glm::translate(model, arrows[i].position);
            float angle = glm::degrees(atan2(arrows[i].direction.y, arrows[i].direction.x)); //check might be x y but basically use y isntead of z
            float rad_angle = glm::radians(angle);
            model = glm::rotate(model, angle, glm::vec3(0.0f, 0.0f, 1.0f));
            glm::mat4 modelview = projection * view * model; //just grab it all at once instead sending view and proj and model.
            GLint modelviewLoc = glGetUniformLocation(shaderProgram, "modelview"); //hold for vertex shader
            glUniformMatrix4fv(modelviewLoc, 1, GL_FALSE, glm::value_ptr(modelview));  //send modelview to vertex shader

            glDrawArrays(GL_LINES, 0, 6); // the arrow being drawn, it needs 6 since 6 vertex for arrow 2 for each 3 lines (VAO DRAW ARROW)
        }
        const auto end = std::chrono::steady_clock::now();

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        //ms for entire render loop
        std::cout << "process takes: " << ms << "ms to finish" << std::endl;

        //swap color stuff for faces
        glBindVertexArray(0); //unbind cuz done

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glDeleteVertexArrays(1, &obj_VAO);
    glDeleteBuffers(1, &obj_VBO);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

GLuint compileShaders() {
    // Vertex Shader
    const char* vsSrc = "#version 330 core\n"
        "layout (location = 0) in vec4 iPos;\n"
        "uniform mat4 modelview;\n"
        "void main()\n"
        "{\n"
        "   vec4 oPos=modelview* iPos;\n"
        "   gl_Position = vec4(oPos.x, oPos.y, oPos.z, oPos.w);\n"
        "}\0";
    //"layout (location = 0) in vec3 iPos;\n"
    //"   vec4 oPos = modelview * vec4(iPos, 1.0);\n"
    // Fragment Shader
    const char* fsSrc = "#version 330 core\n"
        "out vec4 col;\n"
        "uniform vec4 color;\n"
        "void main()\n"
        "{\n"
        "   col = color;\n"
        "}\0";

    //Create VS object
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    //Attach VS src to the Vertex Shader Object
    glShaderSource(vs, 1, &vsSrc, NULL);
    //Compile the vs
    glCompileShader(vs);

    //The same for FS
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsSrc, NULL);
    glCompileShader(fs);

    //Get shader program object
    GLuint shaderProg = glCreateProgram();
    //Attach both vs and fs
    glAttachShader(shaderProg, vs);
    glAttachShader(shaderProg, fs);
    //Link all
    glLinkProgram(shaderProg);

    //Clear the VS and FS objects
    glDeleteShader(vs);
    glDeleteShader(fs);
    return shaderProg;
}
