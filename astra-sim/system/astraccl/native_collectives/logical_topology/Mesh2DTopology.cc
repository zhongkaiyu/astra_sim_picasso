/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/logical_topology/Mesh2DTopology.hh"
#include "astra-sim/common/Logging.hh"

#include <cassert>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace AstraSim;

Mesh2DTopology::Mesh2DTopology(int id, int width, int height)
    : BasicLogicalTopology(BasicLogicalTopology::BasicTopology::Mesh2D),
      id(id), width(width), height(height), total_nodes(width * height) {
    
    assert(width > 0);
    assert(height > 0);
    assert(id >= 0 && id < total_nodes);
    
    auto logger = LoggerFactory::get_logger("system::topology::Mesh2D");
    logger->info("Mesh2D logical topology: id={}, width={}, height={}, total_nodes={}",
                 id, width, height, total_nodes);
}

int Mesh2DTopology::get_num_of_nodes_in_dimension(int dimension) {
    // For basic topologies, return total nodes
    return total_nodes;
}

int Mesh2DTopology::get_neighbor(int node_id, Direction direction) {
    assert(node_id >= 0 && node_id < total_nodes);
    
    auto [x, y] = node_id_to_coords(node_id);
    
    switch (direction) {
        case Direction::East:  // Right
            if (x + 1 < width) {
                return coords_to_node_id(x + 1, y);
            }
            break;
        case Direction::West:  // Left
            if (x - 1 >= 0) {
                return coords_to_node_id(x - 1, y);
            }
            break;
        case Direction::North:  // Up
            if (y - 1 >= 0) {
                return coords_to_node_id(x, y - 1);
            }
            break;
        case Direction::South:  // Down
            if (y + 1 < height) {
                return coords_to_node_id(x, y + 1);
            }
            break;
    }
    
    // No neighbor exists (edge boundary)
    return -1;
}

int Mesh2DTopology::coords_to_node_id(int x, int y) {
    validate_coords(x, y);
    return y * width + x;
}

std::pair<int, int> Mesh2DTopology::node_id_to_coords(int node_id) {
    assert(node_id >= 0 && node_id < total_nodes);
    int y = node_id / width;
    int x = node_id % width;
    return {x, y};
}

int Mesh2DTopology::manhattan_distance(int src, int dest) {
    auto [src_x, src_y] = node_id_to_coords(src);
    auto [dest_x, dest_y] = node_id_to_coords(dest);
    
    int dx = abs(dest_x - src_x);
    int dy = abs(dest_y - src_y);
    
    return dx + dy;
}

bool Mesh2DTopology::are_neighbors(int src, int dest) {
    return manhattan_distance(src, dest) == 1;
}

void Mesh2DTopology::validate_coords(int x, int y) {
    assert(x >= 0 && x < width);
    assert(y >= 0 && y < height);
}
