/* 
 * File:   TriNElementDescription.C
 * Author: Raphael Lindegger
 *
 * Created on November 2, 2019, 2:02 PM
 */

#include <element_promotion/TriNElementDescription.h>
#include <element_promotion/LagrangeBasis.h>
#include <element_promotion/QuadratureRule.h>
#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/ElementDescription.h>
#include <NaluEnv.h>
#include <nalu_make_unique.h>

#include <stk_util/util/ReportHandler.hpp>
#include <stk_topology/topology.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <array>
#include <numeric>


namespace sierra {
namespace nalu {

TriNElementDescription::TriNElementDescription(std::vector<double> in_nodeLocs)
: ElementDescription()
{
  nodeLocs1D = scaleNodeLocs(in_nodeLocs);
  polyOrder = nodeLocs1D.size()-1;
  nodes1D = nodeLocs1D.size();
  nodesPerSide = nodes1D;
  nodesPerElement = 0.5*(nodes1D*(nodes1D+1)); // Triangular number

  baseTopo = stk::topology::TRI_3_2D;
  dimension = 2;
  numEdges = 3;
  numFaces = 0;
  numBoundaries = numEdges;
  nodesInBaseElement = baseTopo.num_nodes();
  nodesPerSubElement = baseTopo.num_nodes();
  baseEdgeConnectivity = { {0,1}, {1,2}, {2,0} };

  //first 3 nodes are base nodes.  Rest have been added.
  baseNodeOrdinals = {0,1,2};
  promotedNodeOrdinals.resize(nodesPerElement-nodesInBaseElement);
  std::iota(promotedNodeOrdinals.begin(), promotedNodeOrdinals.end(), 3);

  newNodesPerEdge = polyOrder - 1;
  newNodesPerVolume = nodesPerElement - (3*nodes1D - 3); // (3*nodes1D - 3) = no. of nodes on the edges and corners

  set_edge_node_connectivities();
  set_volume_node_connectivities();
  set_tensor_product_node_mappings();
  set_boundary_node_mappings();
  set_side_node_ordinals();
  set_isoparametric_coordinates();
  set_subelement_connectivites();
}
//--------------------------------------------------------------------------
// Convert the isoparametric range from the quad element (-1..1) to the range of the triangle (0..1)
std::vector<double> TriNElementDescription::scaleNodeLocs(std::vector<double> in_nodeLocs)
{
  std::vector<double> nodeLocs(in_nodeLocs.size());
  
  for (std::size_t i = 0; i < in_nodeLocs.size(); ++i) {
    nodeLocs[i] = 0.5 * (in_nodeLocs[i] + 1);
  }
  
  return nodeLocs;
}
//--------------------------------------------------------------------------
std::vector<int> TriNElementDescription::edge_node_ordinals()
{
  // base nodes -> edge nodes for node ordering
  int numNewNodes = newNodesPerEdge * numEdges;
  std::vector<ordinal_type> edgeNodeOrdinals(numNewNodes);

  ordinal_type firstEdgeNodeNumber = nodesInBaseElement;
  std::iota(edgeNodeOrdinals.begin(), edgeNodeOrdinals.end(), firstEdgeNodeNumber);

  return edgeNodeOrdinals;
}
//--------------------------------------------------------------------------
void TriNElementDescription::set_edge_node_connectivities()
{
  std::array<ordinal_type,3> edgeOrdinals = {{0, 1, 2}};
  auto edgeNodeOrdinals = edge_node_ordinals();

  int edgeOffset = 0;
  for (const auto edgeOrdinal : edgeOrdinals) {
    std::vector<ordinal_type> newNodesOnEdge(polyOrder-1);
    for (int j = 0; j < polyOrder-1; ++j) {
      newNodesOnEdge.at(j) = edgeNodeOrdinals.at(edgeOffset + j);
    }
    edgeNodeConnectivities.insert({edgeOrdinal, newNodesOnEdge});
    edgeOffset += newNodesPerEdge;
  }
}
//--------------------------------------------------------------------------
std::vector<ordinal_type> TriNElementDescription::volume_node_ordinals()
{
  // 2D volume
  int numNewNodes = newNodesPerVolume;
  std::vector<ordinal_type> volumeNodeOrdinals(numNewNodes);

  ordinal_type firstVolumeNodeNumber = edgeNodeConnectivities.size() * (polyOrder-1) + nodesInBaseElement;
  std::iota(volumeNodeOrdinals.begin(), volumeNodeOrdinals.end(), firstVolumeNodeNumber);

  return volumeNodeOrdinals;
}
//--------------------------------------------------------------------------
void TriNElementDescription::set_volume_node_connectivities()
{
  // Only 1 volume: just insert.
  volumeNodeConnectivities.insert({0, volume_node_ordinals()});
}
//--------------------------------------------------------------------------
std::pair<ordinal_type,ordinal_type>
TriNElementDescription::get_edge_offsets(
  ordinal_type i, ordinal_type j,
  ordinal_type edge_ordinal)
{
  // index of the "left"-most node along an edge
  ordinal_type il = 0;
  ordinal_type jl = 0;

  // index of the "right"-most node along an edge
  ordinal_type ir = nodes1D - 1;
  ordinal_type jr = nodes1D - 1;

  // output
  ordinal_type ix = -1;
  ordinal_type iy = -1;
  ordinal_type stk_index = -1;

  // just hard-code
  switch (edge_ordinal) {
    case 0:
    {
      ix = il + (i + 1);
      iy = jl;
      stk_index = i;
      break;
    }
    case 1:
    {
      ix = ir - (j + 1);
      iy = jl + (j + 1);
      stk_index = j;
      break;
    }
    case 2:
    {
      ix = il;
      iy = jr - (j + 1);
      stk_index = j;
      break;
    }
  }
  int a = iy-1;
  ordinal_type tensor_index = ( ix + (nodes1D * iy) - 0.5*( a*(a+1) ) );
  return {tensor_index, stk_index};
}
//--------------------------------------------------------------------------
void TriNElementDescription::set_base_node_maps()
{
  // The node map shows the relation between the ordinal and the node
  // numbering starting from bottom left to top right (isoparametric
  // coordinate frame)
  //
  // Ordinals: (P=2 element)
  //  2
  //   o
  //   ¦  \.
  //   ¦    \.
  // 5 o      o  4
  //   ¦        \.
  //   ¦          \.
  //   o-----o-----o
  //  0      3       1
  //
  // Node numbering: (P=2 element)
  //  5
  //   o
  //   ¦  \.
  //   ¦    \.
  // 3 o      o  4
  //   ¦        \.
  //   ¦          \.
  //   o-----o-----o
  //  0      1       2
  //
  
  nodeMap.resize(nodesPerElement);
  inverseNodeMap.resize(nodesPerElement);
  
  nodeMap[0] = 0;                 // node 0 is the first
  nodeMap[polyOrder] = 1;         // node 1 comes polyOrder no. of nodes after node 0
  nodeMap[nodesPerElement-1] = 2; // node 2 is the last node
  
  inverseNodeMap[0] = {0, 0};
  inverseNodeMap[1] = {polyOrder, 0};
  inverseNodeMap[2] = {0, polyOrder};
}
void TriNElementDescription::set_boundary_node_mappings()
{
  std::vector<ordinal_type> bcNodeOrdinals(polyOrder-1);
  std::iota(bcNodeOrdinals.begin(), bcNodeOrdinals.end(), 2);

  nodeMapBC.resize(nodes1D);
  nodeMapBC[0] = 0;
  for (int j = 1; j < polyOrder; ++j) {
    nodeMapBC.at(j) = bcNodeOrdinals.at(j-1);
  }
  nodeMapBC[nodes1D-1] = 1;

  inverseNodeMapBC.resize(nodes1D);
  for (int j = 0; j < nodes1D; ++j) {
    inverseNodeMapBC[nodeMapBC.at(j)] = { j };
  }
}
//--------------------------------------------------------------------------
void TriNElementDescription::set_tensor_product_node_mappings()
{
  set_base_node_maps();

  if (polyOrder > 1) {
    std::array<ordinal_type,3> edgeOrdinals = {{0, 1, 2}};
    for (auto edgeOrdinal : edgeOrdinals) {
      auto newNodeOrdinals = edgeNodeConnectivities.at(edgeOrdinal);
      for (int j = 0; j < newNodesPerEdge; ++j) {
        for (int i = 0; i < newNodesPerEdge; ++i) {
          auto offsets = get_edge_offsets(i,j,edgeOrdinal);
          nodeMap.at(offsets.first) = newNodeOrdinals.at(offsets.second);
        }
      }
    }

    auto newVolumeNodes = volumeNodeConnectivities.at(0);
    int nodeCount = 0;
    for (int j = 0; j < polyOrder-2; ++j) {
      for (int i = 0; i < polyOrder-2-j; ++i) {
        int offset = (j+1)*nodes1D + 1 - 0.5*( j*(j+1) ) + i;
        nodeMap.at(offset) = newVolumeNodes.at(nodeCount);
        nodeCount++;
      }
    }
  }

  //inverse map
  inverseNodeMap.resize(nodesPerElement);
  int nodeCount = 0;
  for (int j = 0; j < nodes1D; ++j) {
    for (int i = 0; i < nodes1D-j; ++i) {
      inverseNodeMap[nodeMap.at(nodeCount)] = {i, j};
      nodeCount++;
    }
  }
}
//--------------------------------------------------------------------------
void
TriNElementDescription::set_isoparametric_coordinates()
{
  int count = 0;
  for (int j = 0; j < nodes1D; ++j) {
    for (int i = 0; i < (nodes1D - j); ++i) { // adopted to triangle, every level one node less gets iterated
      std::vector<double> nodeLoc = { nodeLocs1D.at(i), nodeLocs1D.at(j) };
      int offset = count;// j*nodes1D - 0.5*( (j-1)*((j-1)+1) ) + i;
      nodeLocs.insert({nodeMap.at(offset), nodeLoc});
      count++;
    }
  }
}
//--------------------------------------------------------------------------
void
TriNElementDescription::set_subelement_connectivites()
{
  subElementConnectivity.resize((nodes1D-1)*(nodes1D-1));
  int count = 0;
  int a = 2;
  for (int j = 0; j < nodes1D-1; ++j) {
    int offset = 2*(nodes1D - 2 - j) + 1;
    a = a - 2;
    for (int i = 0; i < offset; ++i) {
      if (i % 2 == 0) {
        int offset1 = count - a;
        int offset2 = offset1 + 1;
        int offset3 = offset1 + nodes1D - j;
        subElementConnectivity[count]= {
            nodeMap.at(offset1),
            nodeMap.at(offset2),
            nodeMap.at(offset3)
        };
        a++;
      }
      else {
        int offset1 = count - a + 1;
        int offset2 = offset1 + nodes1D - j;
        int offset3 = offset2 - 1;
        subElementConnectivity[count]= {
            nodeMap.at(offset1),
            nodeMap.at(offset2),
            nodeMap.at(offset3)
        };
      }
      count++;
    }
  }
}
//--------------------------------------------------------------------------
void
TriNElementDescription::set_side_node_ordinals()
{
  faceNodeMap.resize(numBoundaries);
  for (int face_ord = 0; face_ord < numBoundaries; ++face_ord) {
     faceNodeMap.at(face_ord).resize(nodesPerSide);
  }

  // bottom
  for (int m = 0; m < nodes1D; ++m) {
    faceNodeMap.at(0).at(m) = nodeMap.at(m);
  }

  // right
  for (int m = 0; m < nodes1D; ++m) {
    int offset = (m+1)*nodes1D - 1 - 0.5*( m*(m+1) );
    faceNodeMap.at(1).at(m) = nodeMap.at(offset);
  }

  //left
  for (int m = 0; m < nodes1D; ++m) {
    int offset = 0.5*( nodes1D*(nodes1D +1) ) - 1 - m - 0.5*( m*(m+1) );
    faceNodeMap.at(2).at(m) = nodeMap.at(offset);
  }

  sideOrdinalMap.resize(3);
  for (int face_ordinal = 0; face_ordinal < 3; ++face_ordinal) {
    sideOrdinalMap[face_ordinal].resize(nodesPerSide);
    for (int j = 0; j < nodesPerSide; ++j) {
      const auto& ords = inverseNodeMapBC[j];
      sideOrdinalMap.at(face_ordinal).at(j) = faceNodeMap.at(face_ordinal).at(ords[0]);
    }
  }
}

} // namespace nalu
}  // namespace sierra
