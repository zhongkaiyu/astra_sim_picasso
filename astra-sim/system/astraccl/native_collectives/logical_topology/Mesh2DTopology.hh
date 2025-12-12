/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __MESH2D_TOPOLOGY_HH__
#define __MESH2D_TOPOLOGY_HH__

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "astra-sim/system/astraccl/native_collectives/logical_topology/BasicLogicalTopology.hh"

namespace AstraSim {

/**
 * Mesh2DTopology implements a 2D mesh logical arrangement of nodes.
 * 
 * This arranges nodes in a logical 2D grid (e.g., for organizing collective
 * communication patterns). The mesh has width × height = total_nodes.
 * 
 * Example: Mesh2DTopology(16 nodes, width=4, height=4):
 *     0 --- 1 --- 2 --- 3
 *     |     |     |     |
 *     4 --- 5 --- 6 --- 7
 *     |     |     |     |
 *     8 --- 9 --- 10--- 11
 *     |     |     |     |
 *     12--- 13--- 14--- 15
 * 
 * This is different from the physical Mesh2D topology - this logical arrangement
 * is used to organize nodes for collective communication algorithms.
 */
class Mesh2DTopology : public BasicLogicalTopology {
  public:
    enum class Direction { East, West, North, South };
    enum class Dimension { X, Y, NA };

    /**
     * Constructor for Mesh2D logical topology.
     * 
     * @param id the node ID (0-indexed)
     * @param width number of nodes in X dimension
     * @param height number of nodes in Y dimension
     * @param total_nodes should be width × height
     */
    Mesh2DTopology(int id, int width, int height);

    /**
     * Get number of nodes in a given dimension (always returns total_nodes for basic topology).
     */
    int get_num_of_nodes_in_dimension(int dimension) override;

    /**
     * Get neighbor node ID in a given direction.
     * Returns -1 if no neighbor exists (edge boundary).
     * 
     * @param node_id source node ID
     * @param direction direction to get neighbor (East, West, North, South)
     * @return neighbor node ID or -1 if on edge
     */
    int get_neighbor(int node_id, Direction direction);

    /**
     * Get node ID at given 2D coordinates.
     * 
     * @param x column coordinate (0 to width-1)
     * @param y row coordinate (0 to height-1)
     * @return node ID at those coordinates
     */
    int coords_to_node_id(int x, int y);

    /**
     * Convert node ID to 2D coordinates.
     * 
     * @param node_id the node ID
     * @return pair of (x, y) coordinates
     */
    std::pair<int, int> node_id_to_coords(int node_id);

    /**
     * Get Manhattan distance between two nodes.
     * 
     * @param src source node ID
     * @param dest destination node ID
     * @return Manhattan distance = |x2-x1| + |y2-y1|
     */
    int manhattan_distance(int src, int dest);

    /**
     * Check if two nodes are neighbors (distance = 1).
     */
    bool are_neighbors(int src, int dest);

    int get_width() { return width; }
    int get_height() { return height; }
    int get_total_nodes() { return total_nodes; }

  private:
    int id;
    int width;
    int height;
    int total_nodes;
    
    /**
     * Validate that coordinates are within bounds.
     */
    void validate_coords(int x, int y);
};

}  // namespace AstraSim

#endif /* __MESH2D_TOPOLOGY_HH__ */

