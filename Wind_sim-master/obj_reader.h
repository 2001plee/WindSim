#ifndef OBJ_READER_H
#define OBJ_READER_H

#include <vector>
#include <string>
#include <glm/glm.hpp>

// Function to load OBJ file
bool loadOBJ(const std::string& filename, std::vector<glm::vec3>& vertices);

#endif // OBJ_READER_H